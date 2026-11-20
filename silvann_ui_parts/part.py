"""What every part of a page stands on."""


class Part:
    """A part of a page. It holds the page — another part's names are read through it — and, as its own, what
    every part reads of the page: its state, its client, who it is for."""

    def __init__(self, page):
        self.page = page
        self.admin = page.admin
        self.browser_id = page.browser_id
        self.client = page.client
        self.me = page.me
        self.multi = page.multi
        self.page_client = page.page_client
        self.state = page.state
        self.web_user_id = page.web_user_id

    def meet(self):
        """The other parts, once the page has made them all."""
        page = self.page
        self.shell = page.shell
        self.chat = page.chat
        self.options = page.options
        self.models = page.models
        self.users_mode = page.users_mode
        self.users = page.users
        self.account = page.account
        self.help = page.help
        self.connection = page.connection
        self.conversations = page.conversations
        self.turn = page.turn
        self.remote = page.remote

    def safe_notify(self, *args, **kwargs):
        return self.page.safe_notify(*args, **kwargs)
