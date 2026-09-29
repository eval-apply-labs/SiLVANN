#ifndef SILVANN__PACKAGES_SYS_CONTRACTS_ABI_SILICON_FAMILY_CUH
#define SILVANN__PACKAGES_SYS_CONTRACTS_ABI_SILICON_FAMILY_CUH

/* What this file needs, named where a reader — and an editor — can follow it. */
#include <stdint.h>

/* ══ ⭐⭐⭐ A SILICON FAMILY — ONE BINARY EACH, AND THE ONE THING IT EXPORTS ═══════════════════════════
 *
 * ⚖ *"the gpu side becomes a separate binary that is loaded at the start of the cpu session and works as
 * an api provider"* · *"the division between the 64 wavefront mi50 and their consumer card is not a
 * property of the runtime, it is more like the card family (which is runtime plus other caracteristics)"*.
 * A silicon family is a runtime and the traits a build was compiled for. A card's family is built by its
 * own toolchain — `hipcc`, `nvcc`, clang for OpenCL — and the host's is compiled in; a family binary
 * exports ONE symbol, `silvann_silicon_family_get`, which answers this table or nothing. The families
 * there are, and which cards each takes, are not sys's to list: they live in the tree's
 * `silicon_families/`, which answers what the packages ask.
 *
 *     id               which family this is — its number in that roster
 *     name             its name, the one a boot asks for it by
 *     package_doors    one table of doors per package, indexed by package id — each package's own shape
 *                      (`<pkg>/contracts/abi/gpu.cuh`), filled by the family from its own functions.
 *                      `sys`'s is at id 0: memory, copies, waiting, the side channel and the devices.
 *                      The program reaches a family ONLY through these: a family exports nothing else,
 *                      so a door is found by its place in a list both sides were built from, never by a
 *                      name looked up at run time.
 *
 * ⛳ THE TABLE CARRIES ITS VERSION AND ITS SIZE, AND SO DOES EVERY DOOR TABLE. A program asks for the
 * version it was written against and gets nothing from a family that does not speak it; a package reads
 * its own table only as far as the size it was built with says.
 * ⛳ `sys` NAMES NO OTHER PACKAGE'S DOORS, AND NO FAMILY. The slot is untyped here and each package reads
 * its own table as its own type; a family says its own name. So adding a package's doors, or a family,
 * never edits this file. */
#define SYS__SILICON_FAMILY__ABI_VERSION  8u   /* 7: a family says its own name · 8: numbers have gaps (host 0, 200, 300) */
#define SYS__SILICON_FAMILY__ENTRY        "silvann_silicon_family_get"

/* A family, as the program keeps it: its number, or `SYS__SILICON_FAMILY__NONE` — what asking for one
 * answers when none by that name was offered, and what a machine records before it has stood up on one. */
typedef uint32_t sys__silicon_family__id;
#define SYS__SILICON_FAMILY__NONE   0xFFFFFFFFu
/* The most workers — devices a family drives, each a block of the machine — one machine may have. */
#define SYS__SILICON__WORKERS_MAX   8u

/* The bound on a family's number. ⛳ A SLOT PER NUMBER, because the number is what a machine records
 * about the silicon it stood up on and the program's table is indexed by it — and numbers leave gaps, so
 * a family can be placed between two others. A slot is one pointer. The roster checks each number fits. */
#define SYS__SILICON_FAMILY__SLOTS  1024u

typedef struct sys__silicon_family {
    uint32_t                abi_version;     /* SYS__SILICON_FAMILY__ABI_VERSION when it was built */
    uint32_t                size;            /* sizeof this table as the family knows it           */
    sys__silicon_family__id id;              /* its number in the roster                           */
    uint32_t                package_count;   /* how many entries `package_doors` has               */
    const char*             name;            /* its name: `amd_rocm6_wave64`, `host`               */
    const void* const*      package_doors;   /* [package id] -> that package's door table          */
} sys__silicon_family;

/* The shape of the one exported symbol. It answers its table when asked for a version it speaks, and
 * null otherwise. */
typedef const sys__silicon_family* (*sys__silicon_family__get)(uint32_t abi_version);

#endif /* SILVANN__PACKAGES_SYS_CONTRACTS_ABI_SILICON_FAMILY_CUH */
