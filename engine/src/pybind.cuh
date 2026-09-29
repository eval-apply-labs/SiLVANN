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
 * ⛳ AND IT LIVES IN THE TRANSLATION UNIT RATHER THAN IN THE ENGINE ON PURPOSE. The engine is checked
 *   for naming any binding symbol at all, because a package publishes words and the interpreter must not
 *   be written against one particular host. This file is where that rule stops applying.
 * ⛳ THE BOOT IS NOT DONE AT IMPORT. Importing a module should not claim a device, and a caller that
 *   wants a different geometry has to be able to say so before anything is taken. The families ARE
 *   loaded at import: that opens their libraries and counts nothing, so no device is claimed.
 * ══════════════════════════════════════════════════════════════════════════════════════════════════ */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "silicon_families/cards_present.cuh"   /* the cards present, before any family loads */
namespace py = pybind11;

#ifndef SILVANN_MODULE_NAME
#define SILVANN_MODULE_NAME silvann_engine_hip
#endif

/* ══ THE SILICON FAMILIES THE WHEEL LOADS ═════════════════════════════════════════════════════════
 * ⚖ *"the gpu side becomes a separate binary that is loaded at the start of the cpu session"*, and the
 * program never loads one itself — so the wheel does, here, when it is imported. A family is a
 * `libsilvann_<family>.so`; the wheel opens it, asks `silvann_silicon_family_get` for the table of the
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
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == 0) { const char* e = dlerror(); *why = e ? e : "dlopen refused"; return std::string(); }
    typedef const void* (*entry_fn)(unsigned int);
    const entry_fn open = (entry_fn)dlsym(lib, eng_abi_silicon_family_entry());
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
    struct stat st;
    return !p.empty() && stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

/* The directory the wheel's own file is in, found from the address of this function. */
static std::string silvann_wheel_own_dir(void) {
    Dl_info info;
    if (dladdr((void*)&silvann_wheel_own_dir, &info) == 0 || info.dli_fname == 0) return std::string();
    const std::string f(info.dli_fname);
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
    const std::string file = "libsilvann_" + rec.family + ".so";
    if (silvann_wheel_silicon_family_dir.empty()) {
        rec.path = file;
        rec.outcome = "no directory of silicon families was found";
    } else {
        rec.path = silvann_wheel_silicon_family_dir + "/" + file;
        struct stat st;
        if (stat(rec.path.c_str(), &st) != 0) {
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
            sub.def(e.name, [f](unsigned long long x) { return f(x); }, py::arg(e.args[0]), e.doc);
            break;
        }
        case SYS__ABI_SHAPE_ADDRESS_TO_NODE: {
            typedef int (*call)(unsigned long long, sys__heap_node*);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused](unsigned long long address) {
                sys__heap_node got;
                if (!f(address, &got)) throw std::runtime_error(refused);
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
                if (!f(words, present)) throw std::runtime_error(refused);
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
            sub.def(e.name, [f, refused]() { if (!f()) throw std::runtime_error(refused); }, e.doc);
            break;
        }
        case SYS__ABI_SHAPE_READ_BYTES: {
            typedef int (*call)(unsigned long long, unsigned long long, void*);
            const call f = (call)e.call;
            sub.def(e.name, [f, refused](unsigned long long address, unsigned long long bytes) {
                std::string room((size_t)bytes, '\0');
                int ok = 0;
                { py::gil_scoped_release step_aside; ok = f(address, bytes, &room[0]); }
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
                int ok = 0;
                { py::gil_scoped_release step_aside; ok = f(address, room.size(), room.data()); }
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
        const int st = eng_abi_boot_with(blocks, chunks, chunk_nodes,
                                         config.empty() ? 0 : config.c_str(),
                                         (unsigned long long)config.size());
        if (st != 0) throw std::runtime_error(std::string("boot refused: ") + eng_abi_status_text(st));
        unsigned b = 0, c = 0, n = 0;
        eng_abi_geometry(&b, &c, &n);
        std::map<std::string, unsigned> got;
        got["blocks"] = b; got["chunks"] = c; got["chunk_nodes"] = n;
        return got;
    }, py::arg("blocks") = 0u, py::arg("chunks") = 0u, py::arg("chunk_nodes") = 0u,
       py::arg("config") = std::string(),
       "Stand the engine up and answer the geometry it settled on. 0 means work it out.\n"
       "`config` is the whole settings file — `boot_commands{…}` and `device_params{…}` in key:value\n"
       "lines. Each package's host init reads it for its own sizing before the machine exists, and the\n"
       "device_params section is handed to the card VERBATIM for the settings register row.");

    m.def("shutdown", &eng_abi_shutdown, "Give the device memory back.");

    /* Which silicon. A name the binary has no card for is an error rather than a quiet fallback, because a
     * caller that asked for one card and got another would be told it had what it asked for. */
    m.def("choose_silicon", [](const std::string& name) {
        if (!eng_abi_choose_silicon(name.c_str()))
            throw std::runtime_error("no card for silicon '" + name + "' in this binary");
    }, py::arg("name"), "Name the silicon the next boot stands up on: 'host', 'amd_rocm6_wave64', 'nvidia_cuda12'.");
    /* The workers the next boot stands up — a device of a family each, block `w` running as worker `w`. */
    m.def("choose_workers", [](const std::vector<std::pair<std::string, unsigned int>>& workers) {
        std::vector<const char*> names; std::vector<unsigned int> devices;
        for (const auto& w : workers) { names.push_back(w.first.c_str()); devices.push_back(w.second); }
        if (!eng_abi_choose_workers((unsigned int)workers.size(), names.data(), devices.data()))
            throw std::runtime_error("a worker names a family this binary has no card for, or a device it does not count");
    }, py::arg("workers"), "Name the workers the next boot stands up: [(silicon, device), ...]; [] for one, as choose_silicon says.");
    m.def("act_as", [](int w) {
        if (!eng_abi_act_as(w)) throw std::runtime_error("no such worker, or the machine is not free");
    }, py::arg("worker"), "Run what this thread runs next as worker `w` (its device, context, hatch, settings); -1 is back to worker 0.");
    m.def("workers", []() {
        py::list out;
        for (unsigned int w = 0u; w < eng_abi_workers(); ++w) {
            unsigned int device = 0u;
            const char* name = eng_abi_worker_silicon(w, &device);
            out.append(py::make_tuple(std::string(name ? name : ""), device));
        }
        return out;
    }, "The workers of the machine that is up: [(silicon, device), ...].");
    m.def("load_silicon_family", [](const std::string& path) {
        std::string why;
        const std::string silicon = silvann_wheel_load_silicon_family(path, &why);
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
        if (!eng_abi_offer_silicon_family((const void*)(uintptr_t)table))
            throw std::runtime_error("the family's table was refused");
    }, py::arg("table"), "Hand the engine a loaded family's table, by address.");
    m.def("devices", [](const std::string& name) { return eng_abi_devices(name.c_str()); },
          py::arg("name"), "How many devices of this silicon its family counts; 0 with no family.");
    m.def("silicon", []() -> py::object {
        const char* s = eng_abi_silicon();
        return s ? py::object(py::str(s)) : py::object(py::none());
    }, "The silicon the machine stood up on, or the one a boot would use; None when there is no card.");

    /* Building a program. A cell is a kind, a verb and one value — and the three of them cover every
     * shape a program is written with, so this is the only builder there is. */
    m.def("list_create", &eng_abi_list_create, "An empty list. 0 means it could not be made.");
    m.def("list_append", [](unsigned long long list, unsigned kind,
                            unsigned long long verb, unsigned long long value) {
        return eng_abi_list_append(list, kind, verb, value) != 0;
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
        if (!eng_abi_create_executable(room.data(), (unsigned int)room.size(), &got))
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
        if (!eng_abi_freeze(eng_abi_cell(std::get<0>(program), std::get<1>(program), std::get<2>(program)),
                            &got))
            throw std::runtime_error("the program has no picture");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("program"), "Freeze a program into a picture; answers the cell that names it.");

    m.def("thaw", [](std::tuple<unsigned int, unsigned long long, unsigned long long> picture) {
        sys__heap_node got;
        if (!eng_abi_thaw(eng_abi_cell(std::get<0>(picture), std::get<1>(picture), std::get<2>(picture)),
                          &got))
            throw std::runtime_error("the picture did not come back");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("picture"), "Make a private list from a picture; answers the cell that names it.");

    /* Running one, named the way a program names anything. */
    m.def("execute", [](std::tuple<unsigned int, unsigned long long, unsigned long long> bindings,
                        std::tuple<unsigned int, unsigned long long, unsigned long long> program) {
        sys__heap_node got;
        if (!eng_abi_execute(eng_abi_cell(std::get<0>(bindings), std::get<1>(bindings), std::get<2>(bindings)),
                             eng_abi_cell(std::get<0>(program), std::get<1>(program), std::get<2>(program)),
                             &got))
            throw std::runtime_error("execute did not run");
        return py::make_tuple((unsigned int)got.dtype, got.args[0]);
    }, py::arg("bindings"), py::arg("program"),
       "Run a PICTURE of a program against an environment, both named by a cell.");

    m.def("release", &eng_abi_release, py::arg("offset"),
          "Let go of one hold; answers how many are left. Nothing this interface hands out frees itself.");
    m.def("bindings_create", &eng_abi_bindings_create,
          py::arg("symbols"), py::arg("base") = 0ull,
          "An environment with room for `symbols` names.");
    m.def("bindings_symbols", &eng_abi_bindings_symbols, py::arg("bindings"),
          "How many numbered names an environment holds; a name numbered at or past it is spelled. 0 = refused.");
    m.attr("BINDINGS_WIDTH_MAX") = eng_abi_bindings_width_max();
    /* Text. A `str` arrives as its UTF-8 bytes and a `bytes` as itself; what comes back is the cell that
     * names the string, and reading one answers its text and its hash as two `bytes`. */
    m.def("string_create", [](const std::string& text) {
        sys__heap_node got;
        if (!eng_abi_string_create((const unsigned char*)text.data(), (unsigned long long)text.size(), &got))
            throw std::runtime_error("the string was refused");
        return py::make_tuple((unsigned int)got.dtype, got.op_code, got.args[0]);
    }, py::arg("text"), "A string holding this text; answers the cell that names it.");
    m.def("string_read", [](unsigned long long string) {
        std::string text((size_t)SYS__STRING__BYTES_MAX, '\0');
        std::string hash(sizeof(sys__heap_node), '\0');
        unsigned long long length = 0ull;
        if (!eng_abi_string_read(string, (unsigned char*)&text[0], (unsigned long long)text.size(),
                                 (unsigned char*)&hash[0], &length))
            throw std::runtime_error("not a string");
        text.resize((size_t)length);
        return py::make_tuple(py::bytes(text), py::bytes(hash));
    }, py::arg("string"), "What a string holds, as (text, hash) — both bytes.");
    /* Names. A number is the machine's and is never given back, so every composer on one machine agrees
     * on it. */
    m.def("intern", [](const std::string& name) {
        const unsigned long long symbol =
            eng_abi_intern((const unsigned char*)name.data(), (unsigned long long)name.size(), 0);
        if (symbol == ~0ull) throw std::runtime_error("the name was refused");
        return symbol;
    }, py::arg("name"), "The number this name is known by on this machine, dealt the first time.");
    m.def("intern_string", [](const std::string& name) {
        unsigned long long string = 0ull;
        const unsigned long long symbol =
            eng_abi_intern((const unsigned char*)name.data(), (unsigned long long)name.size(), &string);
        if (symbol == ~0ull) throw std::runtime_error("the name was refused");
        return py::make_tuple(symbol, string);
    }, py::arg("name"), "(number, string): the name's number, and the interned string a cell spells it with.");
    m.def("symbol_name", [](unsigned long long symbol) {
        std::string text((size_t)SYS__STRING__BYTES_MAX, '\0');
        unsigned long long length = 0ull;
        if (!eng_abi_symbol_name(symbol, (unsigned char*)&text[0], (unsigned long long)text.size(), &length))
            throw std::runtime_error("no name has that number");
        text.resize((size_t)length);
        return py::bytes(text);
    }, py::arg("symbol"), "The name a number was dealt for, as bytes.");

    /* Running one. The answer is the kind it came back as and the one value that kind carries. */
    m.def("eval", [](unsigned long long bindings, unsigned long long program) {
        unsigned kind = 0; unsigned long long value = 0ull;
        if (!eng_abi_eval(bindings, program, &kind, &value))
            throw std::runtime_error("eval did not run");
        return py::make_tuple(kind, value);
    }, py::arg("bindings"), py::arg("program"),
       "Run a PICTURE of a program; answers (kind, value). Freeze a list to get one.");

    /* Run a program on a block the host never talks to, by having block zero put it there. The answer
     * comes back as the same pair `eval` gives; `outcome` is what happened, so a caller with no value
     * can tell a refusal from a program that answered nothing. */
    m.def("dispatch", [](unsigned long long bindings, unsigned long long program, unsigned onto) {
        unsigned kind = 0; unsigned long long value = 0ull; int outcome = 0;
        int ok = 0;
        { py::gil_scoped_release step_aside;
          ok = eng_abi_dispatch(bindings, program, onto, &kind, &value, &outcome); }
        return py::make_tuple(kind, value, outcome, ok != 0);
    }, py::arg("bindings"), py::arg("program"), py::arg("onto") = 1u,
       "Dispatch a program picture onto a block. -> (kind, value, outcome, ran).");

    /* Run a program on the whole machine: block zero runs it, every other block waits to be given work
     * BY it. What it dispatches and to whom is the program's own business — this only makes sure the
     * blocks it may name are there. */
    /* ⛔⛔ THE LOCK COMES OFF THE LAUNCH AND GOES BACK ON BEFORE ANYTHING PYTHON IS TOUCHED, and the
     * scope is the whole of why this is written out rather than declared. A guard on the DEFINITION
     * wraps the entire body — including the tuple built at the end, which is a call into the
     * interpreter — and making one without the lock is not an error, it is a crash inside CPython
     * with this file nowhere in the trace. ⇒ ★ A GUARD NAMES WHEN IT ENDS, AND THE DEFAULT END IS
     * LATER THAN THE THING YOU MEANT. */
    m.def("run_grid", [](unsigned long long bindings, unsigned long long program) {
        unsigned kind = 0; unsigned long long value = 0ull;
        int ok = 0;
        { py::gil_scoped_release step_aside;
          ok = eng_abi_run_grid(bindings, program, &kind, &value); }
        return py::make_tuple(kind, value, ok != 0);
    }, py::arg("bindings"), py::arg("program"),
       "Run a program with every booted block available to it. -> (kind, value, ran).");

    /* Staying up. Launch every block into its waiting loop and do not wait for the launch — from here
     * the engine is a machine you talk to rather than a call you make.
     * ⛔ EVERY SETTLING DOOR REFUSES WHILE IT IS UP. Build the programs, freeze them and take the
     * addresses FIRST; asking for a list while sixty blocks are parked would otherwise be a wait for a
     * device that has been told never to go quiet. What stays open is `place`/`poll`/`result_release`
     * and `peek`, which is the whole scheduling vocabulary. */
    m.def("reside", []() { return eng_abi_reside() != 0; },
          "Launch every block into its waiting loop and leave it running. Build everything first.");
    m.def("residing", []() { return eng_abi_residing() != 0; },
          "Whether a launch is out there that is not going to end on its own.");
    m.def("stop_residing", []() {
        int ok = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_stop_residing(); }
        return ok != 0;
    }, "Tell it to stop and wait for it. False changes nothing and may be asked again.");

    /* Scheduling from out here. The same three acts a program performs with `(sys__compute …)`,
     * `(sys__completed …)` and `(sys__result …)`, which is why the state machine did not have to grow
     * to take them. ⛔ `place` CONSUMES both operands: filling a base takes a hold of its own and the
     * host has no way to take one, so the caller's becomes the base's. Freeze again to place twice. */
    m.def("place", [](unsigned long long address, unsigned long long bindings,
                      unsigned long long program) {
        int ok = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_place(address, bindings, program); }
        return ok != 0;
    }, py::arg("address"), py::arg("bindings"), py::arg("program"),
       "Put work on a block that is already running. False means it was not taken, and a busy base "
       "is only one of the reasons: no engine, no base at that address, bindings and program "
       "disagreeing about being zero, or the send itself not going. Only a busy base is worth "
       "waiting out, so a retry loop needs a bound. TAKES OVER your hold on both operands.");

    m.def("poll", [](unsigned long long address) {
        unsigned state = 0, kind = 0; unsigned long long value = 0ull; int ok = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_poll(address, &state, &kind, &value); }
        if (!ok) throw std::runtime_error("that base did not answer");
        return py::make_tuple(state, kind, value);
    }, py::arg("address"), "How a placed computation is going. -> (state, kind, value).");

    m.def("result_release", [](unsigned long long address, unsigned block) {
        int ok = 0;
        { py::gil_scoped_release step_aside; ok = eng_abi_result_release(address, block); }
        return ok != 0;
    }, py::arg("address"), py::arg("block"),
       "Give the base back once you have taken the answer; its own block puts the computation away. "
       "Does not wait for that — `poll` until the state reads free.");

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
