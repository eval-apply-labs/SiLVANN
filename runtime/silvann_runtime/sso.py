"""SINGLE SIGN-ON — A USER SIGNED IN BY THEIR ORGANISATION'S ACCOUNT, THROUGH OPENID CONNECT (NN-157).

⚖ 2026-11-06 *"would it be mad to add single sign on? certainly not now... but it would be good to know what is
needed... also it would be important to keep folder separation"*. A sign-on changes only how a sign-in's token is got:
the provider says who the account is, this server checks that itself — the ID token's signature against the provider's
keys, its issuer, its audience, its time and its nonce — and the account is a user as a password's user is (▶ users.py):
their folder, their REPL's fence, the token the server hands out.

The provider is the data folder's `sso.json`, the machine's — read by the server alone (the REPL's fence closes it, as
it closes the rest of the data folder), since it holds the client's secret:
```
  {"issuer": "https://login.example.com/realms/acme",      the provider: its /.well-known/openid-configuration
   "client_id": "silvann", "client_secret": "…",            this server, as the provider knows it
   "redirect_uri": "https://chat.example.com/_silvann/sso",  the UI's page the browser comes back to, as registered
   "label": "Acme",                                           the UI's button: "Sign in with Acme"
   "allow": {"domains": ["example.com"], "emails": []},       who becomes a user at their first sign-on
   "scopes": "openid email profile"}                          optional: these are the default
```
Who gets in: an account linked to a user (`silvann users link <name> <issuer> <subject>`) as that user; else one whose
verified e-mail the `allow` names — its domain or itself — as a user made then, named from the e-mail
(`users.add_signed_on`); else no one. `"verified_email": false` lets in an e-mail the provider does not mark verified
(Microsoft Entra marks none). The issuer is HTTPS, but on this machine's loopback.

Two flows: the browser's — the authorization code with PKCE (`start`, the provider's page, `finish`), the UI's "Sign in
with …" — and a terminal's — the device authorization grant (`device`, `device_poll`): a code the user types at the
provider's page on any device.
"""
import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse

import httpx

from .users import Refused

PENDING_SECONDS = 600            # a sign-on begun and not finished in this long is forgotten
PENDING_MOST = 1000              # and no more than this many at once: the routes are open to anyone
ASYMMETRIC = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA")
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class Unreachable(Exception):
    """The provider did not answer, or answered what is not OpenID Connect."""


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _loopback(url):
    return urllib.parse.urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")


class SignOn:
    """The data folder's provider and the sign-ons under way in this process."""

    def __init__(self, users, data):
        self.users = users
        self.path = os.path.join(os.path.abspath(str(data)), "sso.json")
        self.lock = threading.Lock()
        self.pending = {}            # state -> (nonce, verifier, when): the browser's flow
        self.devices = {}            # handle -> (device_code, when, expires): a terminal's
        self._discovery = None       # (issuer, document, when)
        self._keys = None            # (jwks_uri, PyJWKClient)

    # ── the provider ──────────────────────────────────────────────────────────
    def config(self):
        """`sso.json`, or None where there is none."""
        try:
            with open(self.path) as f:
                c = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as ex:
            raise Refused("sso.json cannot be read: %s" % ex)
        for k in ("issuer", "client_id", "client_secret", "redirect_uri"):
            if not c.get(k):
                raise Refused("sso.json names no %s" % k)
        if not c["issuer"].startswith("https://") and not _loopback(c["issuer"]):
            raise Refused("sso.json's issuer is not HTTPS: %s" % c["issuer"])
        return c

    def discovery(self, c):
        """The provider's OpenID configuration, read once an hour."""
        got = self._discovery
        if got and got[0] == c["issuer"] and time.time() - got[2] < 3600:
            return got[1]
        url = c["issuer"].rstrip("/") + "/.well-known/openid-configuration"
        try:
            r = httpx.get(url, timeout=10)
            r.raise_for_status()
            doc = r.json()
        except (httpx.HTTPError, ValueError) as ex:
            raise Unreachable("the provider's configuration (%s): %s" % (url, ex))
        if doc.get("issuer") != c["issuer"]:
            raise Unreachable("the provider names itself %r, sso.json %r" % (doc.get("issuer"), c["issuer"]))
        for k in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not doc.get(k):
                raise Unreachable("the provider's configuration names no %s" % k)
        self._discovery = (c["issuer"], doc, time.time())
        return doc

    def public(self):
        """What a client is told: whether there is a provider, its button's label, whether a terminal can sign on."""
        try:
            c = self.config()
        except Refused:
            return {"on": False}
        if c is None:
            return {"on": False}
        try:
            device = bool(self.discovery(c).get("device_authorization_endpoint"))
        except Unreachable:
            device = False
        return {"on": True, "label": c.get("label") or urllib.parse.urlsplit(c["issuer"]).hostname, "device": device}

    def _need(self):
        c = self.config()
        if c is None:
            raise Refused("this server has no provider to sign on with (sso.json)")
        return c, self.discovery(c)

    @staticmethod
    def _as_client(c, doc, form):
        """A request to the provider's back channel with this server authenticated as its client: its secret in the form
        (client_secret_post) where the provider takes that — the device flow's providers read it there (Dex, Google) —
        else in HTTP Basic, the method a provider that names none takes. Answers (form, headers)."""
        methods = doc.get("token_endpoint_auth_methods_supported") or ["client_secret_basic"]
        headers, form = {"Accept": "application/json"}, dict(form)
        if "client_secret_post" in methods:
            form.update(client_id=c["client_id"], client_secret=c["client_secret"])
        else:
            pair = "%s:%s" % (urllib.parse.quote(c["client_id"], safe=""), urllib.parse.quote(c["client_secret"], safe=""))
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        return form, headers

    def _token_request(self, c, doc, form):
        """The provider's token endpoint, as its client: (status, the JSON it answered)."""
        form, headers = self._as_client(c, doc, form)
        try:
            r = httpx.post(doc["token_endpoint"], data=form, headers=headers, timeout=15)
        except httpx.HTTPError as ex:
            raise Unreachable("the provider's token endpoint: %s" % ex)
        try:
            body = r.json()
        except ValueError:
            raise Unreachable("the provider's token endpoint answered %d, not JSON" % r.status_code)
        return r.status_code, body

    # ── the ID token, checked here ───────────────────────────────────────────────
    def identity(self, c, doc, id_token, nonce=None):
        """The ID token's claims, once its signature (the provider's keys), issuer, audience, time and — for the
        browser's flow — nonce are checked; Refused otherwise."""
        try:
            import jwt
        except ImportError:
            raise Refused("single sign-on needs PyJWT with its cryptography: pip install 'PyJWT[crypto]'")
        if self._keys is None or self._keys[0] != doc["jwks_uri"]:
            self._keys = (doc["jwks_uri"], jwt.PyJWKClient(doc["jwks_uri"], cache_keys=True, lifespan=3600, timeout=10))
        algorithms = [a for a in (doc.get("id_token_signing_alg_values_supported") or ["RS256"]) if a in ASYMMETRIC]
        try:
            key = self._keys[1].get_signing_key_from_jwt(id_token)
            claims = jwt.decode(id_token, key.key, algorithms=algorithms, audience=c["client_id"], issuer=c["issuer"],
                                leeway=60, options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        except jwt.PyJWKClientConnectionError as ex:
            raise Unreachable("the provider's keys: %s" % ex)
        except jwt.PyJWTError as ex:
            raise Refused("the provider's ID token does not check: %s" % ex)
        aud = claims.get("aud")
        if isinstance(aud, list) and len(aud) > 1 and claims.get("azp") != c["client_id"]:
            raise Refused("the provider's ID token is for another client (azp %r)" % claims.get("azp"))
        if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise Refused("the provider's ID token is not this sign-on's (its nonce)")
        return claims

    def user_for(self, c, claims):
        """The user the account signs on as: linked, or made now where `allow` lets its e-mail in; Refused otherwise."""
        iss, sub = claims["iss"], str(claims["sub"])
        name = self.users.linked(iss, sub)
        if name is not None:
            return name
        email = str(claims.get("email") or "").strip().lower()
        allow = c.get("allow") or {}
        verified = claims.get("email_verified") is True or c.get("verified_email") is False
        domain = email.rpartition("@")[2] if "@" in email else ""
        if email and verified and (email in [e.lower() for e in allow.get("emails", [])]
                                   or domain in [d.lower() for d in allow.get("domains", [])]):
            return self.users.add_signed_on(email.partition("@")[0], iss, sub, email)
        raise Refused("this account (%s, subject %s) is not one this server lets in — its administrator can: "
                      "silvann users link <name> %s %s" % (email or "no e-mail", sub, iss, sub))

    # ── the browser's flow: the authorization code with PKCE ─────────────────────────
    def _forget_old(self):
        """The sign-ons begun more than ten minutes ago, and the device codes expired, forgotten; then the oldest, while
        there are PENDING_MOST of either."""
        now = time.time()
        for k in [k for k, v in self.pending.items() if now - v[2] > PENDING_SECONDS]:
            del self.pending[k]
        for k in [k for k, v in self.devices.items() if now > v[2]]:
            del self.devices[k]
        for table in (self.pending, self.devices):
            while len(table) >= PENDING_MOST:
                table.pop(next(iter(table)))

    def start(self):
        """The provider's page to send the browser to, and the state the UI keeps to know the browser coming back."""
        c, doc = self._need()
        state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        with self.lock:
            self._forget_old()
            self.pending[state] = (nonce, verifier, time.time())
        query = {"response_type": "code", "client_id": c["client_id"], "redirect_uri": c["redirect_uri"],
                 "scope": c.get("scopes") or "openid email profile", "state": state, "nonce": nonce,
                 "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
                 "code_challenge_method": "S256"}
        sep = "&" if "?" in doc["authorization_endpoint"] else "?"
        return {"url": doc["authorization_endpoint"] + sep + urllib.parse.urlencode(query), "state": state}

    def finish(self, code, state):
        """The browser back with the provider's code: exchanged for the ID token, checked, and its user answered."""
        c, doc = self._need()
        with self.lock:
            got = self.pending.pop(state or "", None)
        if got is None or time.time() - got[2] > PENDING_SECONDS:
            raise Refused("this sign-on was not begun here, or took longer than ten minutes: begin it again")
        nonce, verifier, _ = got
        status, body = self._token_request(c, doc, {"grant_type": "authorization_code", "code": code or "",
                                                    "redirect_uri": c["redirect_uri"], "code_verifier": verifier})
        if status != 200 or not body.get("id_token"):
            raise Refused("the provider refused the sign-on: %s" % (body.get("error_description") or body.get("error")
                                                                     or "no ID token"))
        return self.user_for(c, self.identity(c, doc, body["id_token"], nonce))

    # ── a terminal's flow: the device authorization grant ───────────────────────────
    def device(self):
        """A code for the user to type at the provider's page, and the handle the terminal asks after it with."""
        c, doc = self._need()
        if not doc.get("device_authorization_endpoint"):
            raise Refused("the provider gives no device codes: sign in from the UI")
        form, headers = self._as_client(c, doc, {"client_id": c["client_id"],
                                                 "scope": c.get("scopes") or "openid email profile"})
        try:
            r = httpx.post(doc["device_authorization_endpoint"], data=form, headers=headers, timeout=15)
            body = r.json()
        except (httpx.HTTPError, ValueError) as ex:
            raise Unreachable("the provider's device endpoint: %s" % ex)
        if r.status_code != 200 or not body.get("device_code"):
            raise Refused("the provider gave no device code: %s" % (body.get("error_description") or body.get("error")))
        handle, now = secrets.token_urlsafe(32), time.time()
        with self.lock:
            self._forget_old()
            self.devices[handle] = (body["device_code"], now, now + float(body.get("expires_in") or PENDING_SECONDS))
        return {"handle": handle, "user_code": body.get("user_code"), "verification_uri": body.get("verification_uri"),
                "verification_uri_complete": body.get("verification_uri_complete"),
                "interval": int(body.get("interval") or 5), "expires_in": int(body.get("expires_in") or PENDING_SECONDS)}

    def device_poll(self, handle):
        """None while the user has not finished at the provider's page ("authorization_pending", "slow_down"); then the
        user; Refused when they declined, or the code expired."""
        c, doc = self._need()
        with self.lock:
            got = self.devices.get(handle or "")
        if got is None or time.time() > got[2]:
            raise Refused("this device code is not one begun here, or it expired: begin again")
        status, body = self._token_request(c, doc, {"grant_type": DEVICE_GRANT, "device_code": got[0],
                                                    "client_id": c["client_id"]})
        if status != 200:
            if body.get("error") in ("authorization_pending", "slow_down"):
                return None
            with self.lock:
                self.devices.pop(handle, None)
            raise Refused("the provider refused the sign-on: %s" % (body.get("error_description") or body.get("error")))
        with self.lock:
            self.devices.pop(handle, None)
        if not body.get("id_token"):
            raise Refused("the provider gave no ID token")
        return self.user_for(c, self.identity(c, doc, body["id_token"]))
