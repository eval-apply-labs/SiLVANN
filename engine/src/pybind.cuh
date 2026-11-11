#ifndef SILVANN__PYBIND_CUH
#define SILVANN__PYBIND_CUH
/* ══ THE PYTHON MODULE — THE SECOND TIER, AND IT REACHES NOTHING BUT THE FIRST ═══════════════════════
 * ⭐⭐ EVERY LINE BELOW CALLS A C FUNCTION AND NOTHING ELSE. No device symbol is named here and no kernel
 *   is launched here. That is the whole point of there being two tiers rather than one: the C interface
 *   is what a kernel can be reached through, and this is one consumer of it — so a second consumer in
 *   another language costs nothing this file already spent.
 * ⛳ IT DOES NAME ONE TYPE, AND ONLY AS ROOM TO PUT CELLS IN. A list crosses in one call, so something has
 *   to hold the cells while they are handed over — and they are MADE by the C interface's own verb, so
 *   how a node is laid out stays on the other side of the door.
 * ⛳ AND IT ASKS THE SYSTEM THROUGH THE OS LAYER. Loading a family's library and asking after a path are
 *   the system's (`sys__os__*`, `operating_systems/`), not a way into the engine: ⚖ 2026-10-18 the
 *   architect, *"the rule takes sys__os__ as the system's"*.
 * ⛳ AND IT LIVES IN THE TRANSLATION UNIT RATHER THAN IN THE ENGINE ON PURPOSE. The engine is checked
 *   for naming any binding symbol at all, because a package publishes words and the interpreter must not
 *   be written against one particular host. This file is where that rule stops applying.
 * ⛳ THE BOOT IS NOT DONE AT IMPORT. Importing a module should not claim a device, and a caller that
 *   wants a different geometry has to be able to say so before anything is taken. The families ARE
 *   loaded at import: that opens their libraries and counts nothing, so no device is claimed.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <vector>
#include "operating_systems/os.cuh"              /* loading a library, a path: the system's */
#include "silicon_families/cards_present.cuh"   /* the cards present, before any family loads */
namespace py = pybind11;

#ifndef SILVANN_MODULE_NAME
#define SILVANN_MODULE_NAME silvann_engine_hip
#endif

/* ══ THE SILICON FAMILIES THE WHEEL LOADS ═════════════════════════════════════════════════════════
 * ⚖ *"the gpu side becomes a separate binary that is loaded at the start of the cpu session"*, and the
 * program never loads one itself — so the wheel does, here, when it is imported. A family is a
 * `libsilvann_<family>` with the system's ending (`SYS__OS__LIBRARY_SUFFIX`); the wheel opens it, asks `silvann_silicon_family_get` for the table of the
 * version this program reads, and offers that table to the engine.
 *
 * ⭐⭐ WHICH ONES: THE FAMILIES THAT TAKE A CARD THIS MACHINE HAS, AND NO OTHERS. ⚖ *"find the device name
 * and check if it is included inside that family … this way we move from arch to arch until we find one
 * that works"* · *"and we drop the optimistic setup"*. The wheel names the cards present
 * (`silicon_families/cards_present.cuh`) and offers each to the families that include it, highest number
 * first; the first whose binary opens — its runtime is installed — and sees its cards takes it
 * (`silicon_families/choose.cuh`). A card no family includes is UNSUPPORTED and said so by name in
 * `cards()` — it is never handed to the newest binary to see what happens.
 * ⛳ So a family is loaded only where it has a card, and a second family of one vendor is never opened in
 *   the same process: that card is reported as needing a process of its own.
 *
 * ⛳ WHERE IT LOOKS: the directory in `SILVANN_SILICON_FAMILY_DIR` when it is set, and only that one — a
 * caller who named a directory and got another would be told it had what it asked for. Otherwise
 * `silicon_families/` beside the wheel's own file, then `build/silicon_families/` beside it, where a
 * development build puts them; the first that exists is the only one read, so a bundle's own families
 * never mix with a development tree's.
 * ⛳ A FAMILY THAT WILL NOT LOAD IS RECORDED, NOT RAISED, and so is one whose binary is missing: that must
 * not stop the module importing, and `silicon_families()` says what happened to each.
 * ⛳ A LOADED FAMILY IS NEVER CLOSED: the engine keeps its table, and the table's doors are code inside it. */
struct silvann_wheel_silicon_family {
    std::string family;     /* the family it was opened for; empty for one loaded by path */
    std::string path;
    std::string silicon;    /* empty when it did not load */
    std::string outcome;
};
static std::vector<silvann_wheel_silicon_family> silvann_wheel_silicon_families;
static std::string silvann_wheel_silicon_family_dir;

/* Each card present, the family that took it — empty when none did — and every family it was offered to,
 * highest first, with what came of it. */
struct silvann_wheel_card {
    std::string name;
    std::string family;
    std::vector<std::pair<std::string, std::string>> tried;    /* family, what came of it */
};
static std::vector<silvann_wheel_card> silvann_wheel_cards;

/* Load one family and offer its card. Answers the silicon's spelling, or an empty string with `why`
 * saying what refused. */
static std::string silvann_wheel_load_silicon_family(const std::string& path, std::string* why) {
    const sys__os__library lib = sys__os__library_open(path.c_str());
    if (lib == 0) { const char* e = sys__os__library_why(); *why = e ? e : "the system would not load it"; return std::string(); }
    typedef const void* (*entry_fn)(unsigned int);
    const entry_fn open = (entry_fn)sys__os__library_find(lib, eng_abi_silicon_family_entry());
    if (open == 0) { *why = std::string("no ") + eng_abi_silicon_family_entry() + " in it"; return std::string(); }
    const void* table = open(eng_abi_silicon_family_version());
    if (table == 0) { *why = "it does not speak family version " +
                             std::to_string(eng_abi_silicon_family_version()); return std::string(); }
    if (!eng_abi_offer_silicon_family(table)) { *why = "its table was refused — a second card for one silicon, "
                                                  "or a table too small"; return std::string(); }
    const char* name = eng_abi_silicon_family_silicon(table);
    *why = "loaded";
    return name ? std::string(name) : std::string("?");
}

static bool silvann_wheel_is_dir(const std::string& p) {
    return !p.empty() && sys__os__path_is_directory(p.c_str());
}

/* The directory the wheel's own file is in, found from the address of this function. */
static std::string silvann_wheel_own_dir(void) {
    const char* file = sys__os__library_path_of((const void*)&silvann_wheel_own_dir);
    if (file == 0) return std::string();
    const std::string f(file);
    const size_t slash = f.rfind('/');
    return slash == std::string::npos ? std::string(".") : f.substr(0, slash);
}

/* The cards this machine has, by name. */
static std::vector<std::string> silvann_wheel_cards_present(void) {
    std::vector<std::string> out;
    char names[64][SILICON_FAMILIES__CARD_NAME_BYTES];
    const uint32_t n = silicon_families__cards_present(names, 64u);
    for (uint32_t i = 0; i < n; ++i) out.push_back(std::string(names[i]));
    return out;
}

/* What the chooser asks the wheel: open a family's binary from the directory found and offer its table —
 * recording what happened either way — and, once open, how many of its cards it sees. */
static bool silvann_wheel_open_family(const char* family, void* context) {
    (void)context;
    silvann_wheel_silicon_family rec;
    rec.family = family;
    const std::string file = "libsilvann_" + rec.family + SYS__OS__LIBRARY_SUFFIX;
    if (silvann_wheel_silicon_family_dir.empty()) {
        rec.path = file;
        rec.outcome = "no directory of silicon families was found";
    } else {
        rec.path = silvann_wheel_silicon_family_dir + "/" + file;
        if (!sys__os__path_exists(rec.path.c_str())) {
            /* ⛳ A FAMILY THE BUILD SKIPPED LEFT ITS REASON BESIDE WHERE ITS BINARY WOULD BE — the requests it
             *   does not answer — and that is the answer to give, rather than "not there". */
            rec.outcome = "not in " + silvann_wheel_silicon_family_dir;
            FILE* why = fopen((silvann_wheel_silicon_family_dir + "/libsilvann_" + rec.family + ".skipped").c_str(), "r");
            if (why != 0) {
                char line[512];
                if (fgets(line, sizeof line, why) != 0) {
                    rec.outcome = line;
                    while (!rec.outcome.empty() && (rec.outcome.back() == '\n' || rec.outcome.back() == '\r'))
                        rec.outcome.pop_back();
                }
                fclose(why);
            }
        } else {
            rec.silicon = silvann_wheel_load_silicon_family(rec.path, &rec.outcome);
        }
    }
    silvann_wheel_silicon_families.push_back(rec);
    return !rec.silicon.empty();
}
static uint32_t silvann_wheel_family_devices(const char* family, void* context) {
    (void)context;
    return eng_abi_devices(family);
}

static void silvann_wheel_load_silicon_families(void) {
    const char* env = getenv("SILVANN_SILICON_FAMILY_DIR");
    const std::string own = silvann_wheel_own_dir();
    const std::string candidates[3] = { env ? std::string(env) : std::string(),
                                        own.empty() ? std::string() : own + "/silicon_families",
                                        own.empty() ? std::string() : own + "/build/silicon_families" };
    if (env != 0) {
        if (silvann_wheel_is_dir(candidates[0])) silvann_wheel_silicon_family_dir = candidates[0];
    } else {
        for (const std::string& d : candidates)
            if (silvann_wheel_is_dir(d)) { silvann_wheel_silicon_family_dir = d; break; }
    }

    /* Every card to the roster: highest family first, the first whose binary opens and sees its cards.
     * ⛳ The engine keeps which families this process has opened, so a binary opens once, and a family
     *   whose binary would not open is recorded here with the loader's own reason. */
    for (const std::string& card : silvann_wheel_cards_present()) {
        silvann_wheel_card rec;
        rec.name = card;
        const char* family = eng_abi_silicon_family_choose(card.c_str(), silvann_wheel_open_family,
                                                           silvann_wheel_family_devices, 0);
        rec.family = family ? std::string(family) : std::string();
        for (unsigned int i = 0; i < eng_abi_silicon_family_attempts(); ++i) {
            const char* outcome = 0;
            const char* tried = eng_abi_silicon_family_attempt(i, &outcome);
            std::string said = outcome ? outcome : "";
            for (const silvann_wheel_silicon_family& f : silvann_wheel_silicon_families)
                if (tried != 0 && f.family == tried && f.silicon.empty()) said += ": " + f.outcome;
            rec.tried.push_back(std::make_pair(tried ? std::string(tried) : std::string(), said));
        }
        silvann_wheel_cards.push_back(rec);
    }
}

/* ══ THE MACHINE SHARED BETWEEN THREADS, AND PYTHON FREE WHILE A CALL RUNS ══════════════════════════════════
 * ⚖ 2026-11-04 the resident machine: the blocks run what the host queues, and a door that builds acts as the host
 * under the engine's own host lock (▶ `ENG__ABI__AS_THE_HOST`) — so the doors below take this lock SHARED, and
 * threads driving different blocks run their programs at once, a door that runs one waiting for its block's answer
 * holding only its share. The doors that make or unmake the machine — the boot, the shutdown, choosing its silicon
 * and its workers, loading a family — take it whole (`silvann_wheel_whole`), after every call in flight. A door
 * that runs a program — `execute`, `eval`, `run_grid`, `dispatch` — or moves memory gives the interpreter's lock
 * back first, so the server's other threads (its HTTP, the stream, a status request) run while a token's work
 * does (NN-107: /health waited 1.4-2.8 s through a 35B answer, `MEASURED` node-01,
 * measurements/2026-10-22_server_stall_node-01.md; with this lock 2-3 ms through the 35B's and GLM's answers, `MEASURED`
 * node03, measurements/2026-10-23_server_stall_node03.md). Any other door takes the engine's lock at once when it is
 * free, keeping the interpreter's, since a call of microseconds would only hand that lock to another thread
 * and wait to get it back.
 * ⛔ A THREAD NEVER WAITS FOR THE ENGINE'S LOCK WHILE IT HOLDS THE INTERPRETER'S: a program finishing in
 *   another thread needs the interpreter's lock to hand back its answer, so a door finding the engine busy
 *   gives the interpreter's lock up before it waits. The guards end in that order too: the engine's lock is
 *   let go before the interpreter's is taken again.
 * ⛳ `place`, `poll` and `take` take only the host lock inside them: they speak to running blocks whatever another
 *   thread is doing. */
static std::shared_mutex silvann_wheel_engine;

template <typename F> static auto silvann_wheel_long(F&& f) -> decltype(f()) {
    decltype(f()) got{};
    { py::gil_scoped_release step_aside;
      std::shared_lock<std::shared_mutex> share(silvann_wheel_engine);
      got = f(); }
    return got;
}

template <typename F> static auto silvann_wheel_short(F&& f) -> decltype(f()) {
    std::shared_lock<std::shared_mutex> free_now(silvann_wheel_engine, std::try_to_lock);
    if (free_now.owns_lock()) return f();
    decltype(f()) got{};
    { py::gil_scoped_release step_aside;
      std::shared_lock<std::shared_mutex> share(silvann_wheel_engine);
      got = f(); }
    return got;
}

/* The machine made or unmade: every call in flight finished first, and none begun until this one ends. */
template <typename F> static auto silvann_wheel_whole(F&& f) -> decltype(f()) {
    decltype(f()) got{};
    { py::gil_scoped_release step_aside;
      std::unique_lock<std::shared_mutex> whole(silvann_wheel_engine);
      got = f(); }
    return got;
}

/* ══ A PACKAGE'S CALLS, BOUND FROM THE TABLE IT ANSWERS ════════════════════════════════════════════════
 * Each package describes its C interface as data (`sys/contracts/abi/cpu.cuh` for the table and the
 * shapes), and this turns every row into a function under the package's name, by its shape: the C function
 * is called back through the shape's own type, a 0 from a call that can refuse is raised with the row's
 * words, and what comes back is handed to Python in the shape's form.
 * ⛳ ONE CASE PER SHAPE, AND NONE PER CALL: a package adds a call by adding a row. A row of a shape this
 *   binding does not know stops the import and names it — the package and the binding were built from
 *   different trees, and quietly leaving the call out would hide that. */
static void silvann_python_bind_surface(py::module_& m, const char* package, const sys__abi_surface* surface) {
    if (surface == 0) return;                       /* a package with no calls for a host */
    py::module_ sub = m.def_submodule(package, surface->about);
    for (uint32_t i = 0; i < surface->count; ++i) {
        const sys__abi_entry e = surface->entries[i];
        const std::string refused = e.refused ? e.refused : "refused";
        switch (e.shape) {
        case SYS__ABI_SHAPE_NUMBER_TO_NUMBER: {
            typedef unsigned long long (*call)(unsigned long long);
            const call f = (call)e.call;
            sub.def(e.name, [f](unsigned long long x) { return silvann_wheel_short([&] { return f(x); }); },
                    py::arg(e.args[0]), e.doc);
            break;
        }
        case SYS__ABI_SHAPE_ADDRESS_TO_NODE: {
            typedef int (*call)(unsigned long long, sys__heap_node*);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused](unsigned long long address) {
                sys__heap_node got;
                if (!silvann_wheel_short([&] { return f(address, &got); })) throw std::runtime_error(refused);
                std::vector<unsigned long long> args(got.args, got.args + 6);
                return py::make_tuple((unsigned int)got.dtype, got.op_code, args);
            }, py::arg(e.args[0]), e.doc);
            break;
        }
        case SYS__ABI_SHAPE_WORDS: {
            typedef int (*call)(unsigned long long*, int*);
            const call f = (call)e.call;
            std::vector<std::string> answers;
            for (uint32_t j = 0; j < SYS__ABI_MOST_WORDS && e.answers[j] != 0; ++j) answers.push_back(e.answers[j]);
            sub.def(e.name, [f, refused, answers]() {
                unsigned long long words[SYS__ABI_MOST_WORDS] = {0};
                int present[SYS__ABI_MOST_WORDS] = {0};
                if (!silvann_wheel_short([&] { return f(words, present); })) throw std::runtime_error(refused);
                py::dict out;
                for (size_t j = 0; j < answers.size(); ++j)
                    out[answers[j].c_str()] = present[j] ? py::cast(words[j]) : py::none();
                return out;
            }, e.doc);
            break;
        }
        case SYS__ABI_SHAPE_NOTHING: {
            typedef int (*call)(void);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused]() { if (!silvann_wheel_short([&] { return f(); })) throw std::runtime_error(refused); },
                    e.doc);
            break;
        }
        case SYS__ABI_SHAPE_READ_BYTES: {
            typedef int (*call)(unsigned long long, unsigned long long, void*);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused](unsigned long long address, unsigned long long bytes) {
                std::string room((size_t)bytes, '\0');
                const int ok = silvann_wheel_long([&] { return f(address, bytes, &room[0]); });
                if (!ok) throw std::runtime_error(refused);
                return py::bytes(room);
            }, py::arg(e.args[0]), py::arg(e.args[1]), e.doc);
            break;
        }
        case SYS__ABI_SHAPE_WRITE_BYTES: {
            typedef int (*call)(unsigned long long, unsigned long long, const void*);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused](unsigned long long address, py::bytes payload) {
                std::string room = payload;
                const int ok = silvann_wheel_long([&] { return f(address, room.size(), room.data()); });
                if (!ok) throw std::runtime_error(refused);
                return (unsigned long long)room.size();
            }, py::arg(e.args[0]), py::arg(e.args[1]), e.doc);
            break;
        }
        default:
            throw std::runtime_error(std::string(package) + "." + e.name + " has a shape this binding does not know ("
                                     + std::to_string(e.shape) + ")");
        }
    }
}

PYBIND11_MODULE(SILVANN_MODULE_NAME, m) {
    silvann_wheel_load_silicon_families();

    m.doc() = "the nested-frames engine: stand it up, build a program, run it";

    /* Standing up and taking down. Zero on any figure means the boot works it out, so the common call
     * takes no arguments at all — and the status is a sentence rather than a number, because a caller
     * that only learns "no" has to guess which of `eng__boot__why`'s refusals it hit. */
    m.def("boot", [](unsigned blocks, unsigned chunks, unsigned chunk_nodes, const std::string& config) {
        unsigned b = 0, c = 0, n = 0;
        const int st = silvann_wheel_whole([&] {
            const int st = eng_abi_boot_with(blocks, chunks, chunk_nodes, config.empty() ? 0 : config.c_str(),
                                             (unsigned long long)config.size());
            if (st == 0) eng_abi_geometry(&b, &c, &n);
            return st;
        });
        if (st != 0) throw std::runtime_error(std::string("boot refused: ") + eng_abi_status_text(st));
        std::map<std::string, unsigned> got;
        got["blocks"] = b; got["chunks"] = c; got["chunk_nodes"] = n;
        return got;
    }, py::arg("blocks") = 0u, py::arg("chunks") = 0u, py::arg("chunk_nodes") = 0u,
       py::arg("config") = std::string(),
       "Stand the engine up and answer the geometry it settled on. 0 means work it out.\n"
       "`config` is the whole settings file — `boot_commands{…}` and `device_params{…}` in key:value\n"
       "lines. Each package's host init reads it for its own sizing before the machine exists, and the\n"
       "device_params section is handed to the card VERBATIM for the settings register row.");

    m.def("shutdown", []() { silvann_wheel_whole([] { eng_abi_shutdown(); return 0; }); }, "Give the device memory back.");
    /* ⛳ READ WITHOUT THE WHEEL: two words, and the one question a caller asks when every other door refuses. */
    m.def("panic", []() { return eng_abi_panic(); },
          "The four-letter word the evaluator stopped with, as a number, or 0: while it is set every door refuses, and "
          "a shutdown and a boot clear it.");

    /* Which silicon. A name the binary has no card for is an error rather than a quiet fallback, because a
     * caller that asked for one card and got another would be told it had what it asked for. */
    m.def("choose_silicon", [](const std::string& name) {
        if (!silvann_wheel_whole([&] { return eng_abi_choose_silicon(name.c_str()); }))
            throw std::runtime_error("no card for silicon '" + name + "' in this binary");
    }, py::arg("name"), "Name the silicon the next boot stands up on: 'host', 'amd_rocm6_wave64', 'nvidia_cuda12'.");
    /* The workers the next boot stands up — a device of a family each, block `w` running as worker `w`. */
    m.def("choose_workers", [](const std::vector<std::pair<std::string, unsigned int>>& workers) {
        std::vector<const char*> names; std::vector<unsigned int> devices;
        for (const auto& w : workers) { names.push_back(w.first.c_str()); devices.push_back(w.second); }
        if (!silvann_wheel_whole([&] { return eng_abi_choose_workers((unsigned int)workers.size(), names.data(),
                                                                     devices.data()); }))
            throw std::runtime_error("a worker names a family this binary has no card for, or a device it does not count");
    }, py::arg("workers"), "Name the workers the next boot stands up: [(silicon, device), ...]; [] for one, as choose_silicon says.");
    m.def("act_as", [](int w) {
        if (!silvann_wheel_short([&] { return eng_abi_act_as(w); })) throw std::runtime_error("no such worker, or the machine is not free");
    }, py::arg("worker"), "Run what this thread runs next as worker `w` (its device, context, hatch, settings); -1 is back to worker 0.");
    m.def("workers", []() {
        std::vector<std::pair<std::string, unsigned int>> got;
        silvann_wheel_short([&] {
            for (unsigned int w = 0u; w < eng_abi_workers(); ++w) {
                unsigned int device = 0u;
                const char* name = eng_abi_worker_silicon(w, &device);
                got.emplace_back(std::string(name ? name : ""), device);
            }
            return 0;
        });
        py::list out;
        for (const auto& w : got) out.append(py::make_tuple(w.first, w.second));
        return out;
    }, "The workers of the machine that is up: [(silicon, device), ...].");
    m.def("load_silicon_family", [](const std::string& path) {
        std::string why;
        const std::string silicon = silvann_wheel_whole([&] { return silvann_wheel_load_silicon_family(path, &why); });
        silvann_wheel_silicon_family rec; rec.path = path; rec.silicon = silicon; rec.outcome = why;
        silvann_wheel_silicon_families.push_back(rec);
        if (silicon.empty()) throw std::runtime_error("family '" + path + "' not loaded: " + why);
        return silicon;
    }, py::arg("path"), "Load one more family by path, offer its card, and answer its silicon.");
    m.def("cards", []() {
        py::list out;
        for (const silvann_wheel_card& c : silvann_wheel_cards) {
            py::dict d;
            d["card"]           = c.name;
            d["silicon_family"] = c.family.empty() ? py::object(py::none()) : py::object(py::str(c.family));
            py::list tried;
            for (const std::pair<std::string, std::string>& t : c.tried) {
                py::dict a;
                a["silicon_family"] = t.first;
                a["outcome"]        = t.second;
                tried.append(a);
            }
            d["tried"] = tried;
            out.append(d);
        }
        return out;
    }, "Every card this machine has, by name; the silicon family that took it — None when none did; and every\n"
       "family it was offered to, highest first, with what came of it. A card with nothing tried is one no\n"
       "family includes: it is not supported.");
    m.def("silicon_families", []() {
        py::list out;
        for (const silvann_wheel_silicon_family& p : silvann_wheel_silicon_families) {
            py::dict d;
            d["path"]    = p.path;
            d["silicon"] = p.silicon.empty() ? py::object(py::none()) : py::object(py::str(p.silicon));
            d["outcome"] = p.outcome;
            out.append(d);
        }
        return out;
    }, "Every family this wheel tried to load, at import and since, and what happened to each.");
    m.def("silicon_family_dir", []() -> py::object {
        return silvann_wheel_silicon_family_dir.empty() ? py::object(py::none())
                                                  : py::object(py::str(silvann_wheel_silicon_family_dir));
    }, "The directory the families were loaded from at import, or None when none was found.");
    /* A family loaded by someone else, handed over as the address `silvann_silicon_family_get` answered. */
    m.def("offer_silicon_family", [](unsigned long long table) {
        if (!silvann_wheel_whole([&] { return eng_abi_offer_silicon_family((const void*)(uintptr_t)table); }))
            throw std::runtime_error("the family's table was refused");
    }, py::arg("table"), "Hand the engine a loaded family's table, by address.");
    m.def("devices", [](const std::string& name) { return silvann_wheel_short([&] { return eng_abi_devices(name.c_str()); }); },
          py::arg("name"), "How many devices of this silicon its family counts; 0 with no family.");
    m.def("silicon", []() -> py::object {
        const char* s = silvann_wheel_short([] { return eng_abi_silicon(); });
        return s ? py::object(py::str(s)) : py::object(py::none());
    }, "The silicon the machine stood up on, or the one a boot would use; None when there is no card.");

    /* Building a program. A cell is a kind, a verb and one value — and the three of them cover every
     * shape a program is written with, so this is the only builder there is. */
    m.def("list_create", []() { return silvann_wheel_short([] { return eng_abi_list_create(); }); },
          "An empty list. 0 means it could not be made.");
    m.def("list_append", [](unsigned long long list, unsigned kind,
                            unsigned long long verb, unsigned long long value) {
        return silvann_wheel_short([&] { return eng_abi_list_append(list, kind, verb, value); }) != 0;
    }, py::arg("list"), py::arg("kind"), py::arg("verb") = 0ull, py::arg("value") = 0ull,
       "Put one cell at the end of a list.");
    /* A whole list in one crossing. The cells go in as the triples everything here builds them as, and what
     * comes back is the cell that names the list — which is what goes into the next call's cells. */
    m.def("create_executable",
          [](const std::vector<std::tuple<unsigned int, unsigned long long, unsigned long long>>& cells) {
        std::vector<sys__heap_node> room;
        room.reserve(cells.size());
        for (const auto& c : cells)
            room.push_back(eng_abi_cell(std::get<0>(c), std::get<1>(c), std::get<2>(c)));
        sys__heap_node got;
        if (!silvann_wheel_short([&] { return eng_abi_create_executable(room.data(), (unsigned int)room.size(), &got); }))
            throw std::runtime_error("the list was refused");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("cells"),
       "Build a list from cells in one call; answers the cell that names it. It takes nothing: a caller "
       "that nested one form into another owns both and gives the inner one back.");

    /* A program frozen into something nobody may write to, and a program back from one. A caller holds
     * both after a freeze and holds the new list after a thaw — neither call gives anything back by
     * itself, the way nothing else through this door does either. */
    m.def("freeze", [](std::tuple<unsigned int, unsigned long long, unsigned long long> program) {
        sys__heap_node got;
        const sys__heap_node cell = eng_abi_cell(std::get<0>(program), std::get<1>(program), std::get<2>(program));
        if (!silvann_wheel_short([&] { return eng_abi_freeze(cell, &got); }))
            throw std::runtime_error("the program has no picture");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("program"), "Freeze a program into a picture; answers the cell that names it.");

    m.def("thaw", [](std::tuple<unsigned int, unsigned long long, unsigned long long> picture) {
        sys__heap_node got;
        const sys__heap_node cell = eng_abi_cell(std::get<0>(picture), std::get<1>(picture), std::get<2>(picture));
        if (!silvann_wheel_short([&] { return eng_abi_thaw(cell, &got); }))
            throw std::runtime_error("the picture did not come back");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("picture"), "Make a private list from a picture; answers the cell that names it.");

    /* Running one, named the way a program names anything. */
    m.def("execute", [](std::tuple<unsigned int, unsigned long long, unsigned long long> bindings,
                        std::tuple<unsigned int, unsigned long long, unsigned long long> program) {
        sys__heap_node got;
        const sys__heap_node env = eng_abi_cell(std::get<0>(bindings), std::get<1>(bindings), std::get<2>(bindings));
        const sys__heap_node pic = eng_abi_cell(std::get<0>(program), std::get<1>(program), std::get<2>(program));
        if (!silvann_wheel_long([&] { return eng_abi_execute(env, pic, &got); }))
            throw std::runtime_error("execute did not run");
        return py::make_tuple((unsigned int)got.dtype, got.args[0]);
    }, py::arg("bindings"), py::arg("program"),
       "Run a PICTURE of a program against an environment, both named by a cell.");

    m.def("release", [](unsigned long long offset) { return silvann_wheel_short([&] { return eng_abi_release(offset); }); },
          py::arg("offset"),
          "Let go of one hold; answers how many are left. Nothing this interface hands out frees itself.");
    m.def("bindings_create", [](unsigned long long symbols, unsigned long long base) {
        return silvann_wheel_short([&] { return eng_abi_bindings_create(symbols, base); });
    },    py::arg("symbols"), py::arg("base") = 0ull,
          "An environment with room for `symbols` names.");
    m.def("bindings_symbols", [](unsigned long long bindings) {
        return silvann_wheel_short([&] { return eng_abi_bindings_symbols(bindings); });
    },    py::arg("bindings"),
          "How many numbered names an environment holds; a name numbered at or past it is spelled. 0 = refused.");
    m.attr("BINDINGS_WIDTH_MAX") = eng_abi_bindings_width_max();
    /* Text. A `str` arrives as its UTF-8 bytes and a `bytes` as itself; what comes back is the cell that
     * names the string, and reading one answers its text and its hash as two `bytes`. */
    m.def("string_create", [](const std::string& text) {
        sys__heap_node got;
        if (!silvann_wheel_short([&] { return eng_abi_string_create((const unsigned char*)text.data(),
                                                                   (unsigned long long)text.size(), &got); }))
            throw std::runtime_error("the string was refused");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("text"), "A string holding this text; answers the cell that names it.");
    m.def("string_read", [](unsigned long long string) {
        std::string text((size_t)SYS__STRING__BYTES_MAX, '\0');
        std::string hash(sizeof(sys__heap_node), '\0');
        unsigned long long length = 0ull;
        if (!silvann_wheel_short([&] { return eng_abi_string_read(string, (unsigned char*)&text[0], (unsigned long long)text.size(),
                                                                 (unsigned char*)&hash[0], &length); }))
            throw std::runtime_error("not a string");
        text.resize((size_t)length);
        return py::make_tuple(py::bytes(text), py::bytes(hash));
    }, py::arg("string"), "What a string holds, as (text, hash) — both bytes.");
    /* Names. A number is the machine's and is never given back, so every composer on one machine agrees
     * on it. */
    m.def("intern", [](const std::string& name) {
        const unsigned long long symbol = silvann_wheel_short([&] {
            return eng_abi_intern((const unsigned char*)name.data(), (unsigned long long)name.size(), 0); });
        if (symbol == ~0ull) throw std::runtime_error("the name was refused");
        return symbol;
    }, py::arg("name"), "The number this name is known by on this machine, dealt the first time.");
    m.def("intern_string", [](const std::string& name) {
        unsigned long long string = 0ull;
        const unsigned long long symbol = silvann_wheel_short([&] {
            return eng_abi_intern((const unsigned char*)name.data(), (unsigned long long)name.size(), &string); });
        if (symbol == ~0ull) throw std::runtime_error("the name was refused");
        return py::make_tuple(symbol, string);
    }, py::arg("name"), "(number, string): the name's number, and the interned string a cell spells it with.");
    m.def("symbol_name", [](unsigned long long symbol) {
        std::string text((size_t)SYS__STRING__BYTES_MAX, '\0');
        unsigned long long length = 0ull;
        if (!silvann_wheel_short([&] { return eng_abi_symbol_name(symbol, (unsigned char*)&text[0],
                                                                 (unsigned long long)text.size(), &length); }))
            throw std::runtime_error("no name has that number");
        text.resize((size_t)length);
        return py::bytes(text);
    }, py::arg("symbol"), "The name a number was dealt for, as bytes.");

    /* Running one. The answer is the kind it came back as and the one value that kind carries. */
    m.def("eval", [](unsigned long long bindings, unsigned long long program) {
        unsigned kind = 0; unsigned long long value = 0ull;
        if (!silvann_wheel_long([&] { return eng_abi_eval(bindings, program, &kind, &value); }))
            throw std::runtime_error("eval did not run");
        return py::make_tuple(kind, value);
    }, py::arg("bindings"), py::arg("program"),
       "Run a PICTURE of a program; answers (kind, value). Freeze a list to get one.");

    /* Run a program on the block named, queued there with the host as the caller. The answer comes back as the
     * same pair `eval` gives; `outcome` is what happened, so a caller with no value can tell a refusal from a
     * program that answered nothing. */
    m.def("dispatch", [](unsigned long long bindings, unsigned long long program, unsigned onto) {
        unsigned kind = 0; unsigned long long value = 0ull; int outcome = 0;
        const int ok = silvann_wheel_long([&] { return eng_abi_dispatch(bindings, program, onto, &kind, &value, &outcome); });
        return py::make_tuple(kind, value, outcome, ok != 0);
    }, py::arg("bindings"), py::arg("program"), py::arg("onto") = 1u,
       "Dispatch a program picture onto a block. -> (kind, value, outcome, ran).");

    /* Run a program on the whole machine: block zero runs it, every other block resident to be given work BY it.
     * What it hands out and to whom is the program's own business. */
    /* ⛔⛔ THE LOCK COMES OFF THE LAUNCH AND GOES BACK ON BEFORE ANYTHING PYTHON IS TOUCHED, and the
     * scope is the whole of why this is written out rather than declared. A guard on the DEFINITION
     * wraps the entire body — including the tuple built at the end, which is a call into the
     * interpreter — and making one without the lock is not an error, it is a crash inside CPython
     * with this file nowhere in the trace. ⇒ ★ A GUARD NAMES WHEN IT ENDS, AND THE DEFAULT END IS
     * LATER THAN THE THING YOU MEANT. */
    m.def("run_grid", [](unsigned long long bindings, unsigned long long program) {
        unsigned kind = 0; unsigned long long value = 0ull;
        const int ok = silvann_wheel_long([&] { return eng_abi_run_grid(bindings, program, &kind, &value); });
        return py::make_tuple(kind, value, ok != 0);
    }, py::arg("bindings"), py::arg("program"),
       "Run a program with every booted block available to it. -> (kind, value, ran).");

    /* Scheduling from out here, as a program's `sys__scheduler__add`, `sys__result` and `sys__wait_for_result` do, from the host's own
     * slot (▶ `engine/abi/residency.cuh`). The blocks are resident, so they work whatever another thread is running,
     * and a block runs what they queue when it parks. */
    m.def("place", [](unsigned block, unsigned long long bindings, unsigned long long program) {
        unsigned long long answer = 0ull;
        { py::gil_scoped_release step_aside; answer = eng_abi_place(block, bindings, program); }
        return answer;
    }, py::arg("block"), py::arg("bindings"), py::arg("program"),
       "Queue a program picture on a block, into a new answer. Answers the answer, yours to `release` once read, or 0 "
       "where it was not queued: no engine, no such block, an operand missing, or the queue's refusal. The queue takes "
       "holds of its own; yours stay yours.");

    /* A process for the host's work (NN-160): a new one, this thread's requests made as one, one killed, one asked to stop
     * at the token it is on (▶ `engine/abi/residency.cuh`). */
    m.def("process_new", []() { return eng_abi_process_new(); }, "A new process: its number, the counter's next.");
    m.def("process_as", [](unsigned long long pid) { return eng_abi_process_as(pid); }, py::arg("pid"),
          "This thread's requests made as process `pid` from here on (0: each its own). Answers the one they were made as.");
    m.def("process_kill", [](unsigned long long pid) {
        unsigned long long taken = 0ull;
        { py::gil_scoped_release step_aside; taken = eng_abi_process_kill(pid); }
        return taken;
    }, py::arg("pid"), "Process `pid` killed: its requests off every queue, their answers SCKL. Answers how many it took off.");
    m.def("process_stop", [](unsigned long long pid) { return eng_abi_process_stop(pid) != 0; }, py::arg("pid"),
          "Process `pid` asked to stop at the token it is on: its program reads `(sys__scheduler__stopped)`.");

    m.def("poll", [](unsigned long long answer) {
        int ok = 0, ready = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_poll(answer, &ready); }
        if (!ok) throw std::runtime_error("no machine, or not an answer");
        return ready != 0;
    }, py::arg("answer"), "Whether that answer's value is there. Reads its state and nothing else.");

    m.def("take", [](unsigned long long answer) {
        unsigned kind = 0; unsigned long long value = 0ull; int ok = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_take(answer, &kind, &value); }
        return py::make_tuple(ok != 0, kind, value);
    }, py::arg("answer"),
       "That answer's value, once it is there. -> (read, kind, value). The answer keeps it, so a second read gets the "
       "same; a reference answered is yours to release.");

    /* The words, as the maps a caller was going to build out of them anyway. */
    m.def("verbs", []() {
        std::map<std::string, unsigned long long> out;
        for (unsigned i = 0; i < eng_abi_verb_count(); ++i) out[eng_abi_verb_name(i)] = eng_abi_verb_id(i);
        return out;
    }, "Every verb this build publishes, by name.");
    m.def("kinds", []() {
        std::map<std::string, unsigned> out;
        for (unsigned i = 0; i < eng_abi_kind_count(); ++i) out[eng_abi_kind_name(i)] = eng_abi_kind_id(i);
        return out;
    }, "Every node kind this build knows, by name.");
    /* ⛳ THE ROSTER, WHICH A CALLER NEEDS FOR A REASON THE OTHER TWO DO NOT HAVE: a package's room is one
     * allocation per package taken HOST-SIDE, and the hatch entry it lands in is indexed by package id
     * while a config file spells a name. This is what maps one to the other. */
    m.def("packages", []() {
        std::map<std::string, unsigned> out;
        for (unsigned i = 0; i < eng_abi_package_count(); ++i) out[eng_abi_package_name(i)] = eng_abi_package_id(i);
        return out;
    }, "Every package on this build's roster, by name.");

    /* ── EACH PACKAGE'S OWN C INTERFACE, UNDER ITS NAME ─────────────────────────────────────────────────
     * ⚖ *"do change it as sys.peek etc"* · *"pybind.cuh goes through the x macro and calls the same method in
     * each x macro id"*. What the engine answers sits on the module itself; what a package answers sits
     * under the package's name — asked of the package, as data, over the roster. */
#define SILVANN_PYTHON_PACKAGE(ID, NAME, ...)  silvann_python_bind_surface(m, #NAME, NAME##_abi_surface());
    PACKAGE_LIST(SILVANN_PYTHON_PACKAGE)
#undef SILVANN_PYTHON_PACKAGE
}

#endif /* SILVANN__PYBIND_CUH */
