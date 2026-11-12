# arm_base — the Arm processors' source

⚖ 2026-11-09 *"it might be good for porting it to mobile phones as well"* · *"also like everything else make the arm_base
silicon platform and then add the overrides for that specific dgx sparx platform"*. As `amd_rocm_base` and the OpenCL
bases: a source a folder, not a roster family, its names its own (`arm_base__*`); a family is its card list and a
`unit.cuh` that says what the processor must have (`ARM_BASE__CAPABLE()`), its name and what it is.

| family | number | built with | takes |
|---|---|---|---|
| `arm_neon` | 55 | `-march=armv8.2-a+fp16+dotprod` | any Arm processor with Armv8.2's half precision and dot product (`HWCAP_FPHP`, `ASIMDHP`, `ASIMDDP`): a Cortex-A55/A76 on, Neoverse N1 on, Apple's M1 on, the phones of 2019 on |
| `arm_gb10` | 60 | `-march=armv9.2-a+sve2+i8mm+bf16+fp16+dotprod` | the DGX Spark's GB10 — ten Cortex-X925 (part `0xd85`) beside ten Cortex-A725 (`0xd87`), as `/proc/cpuinfo` lists them — its own overrides after arm_base's |

Both are vendor `arm`: one of them a process. `cards_present.cuh` names the machine's processors `arm_gb10` or `arm_neon`
(the GB10's parts, else the HWCAPs); each family counts no device where the kernel does not report what its flags
assume, so a build never runs an instruction the processor lacks. A processor without the dot product (a Cortex-A53, a
Raspberry Pi 3) has no Arm family: the host family runs the language and answers none of nn's doors.

## The layout

- `primitives/` — what every package asks of a family, answered on the host's threads: `x86_avx2`'s primitives with
  `__fp16` for F16C's conversions and `yield` for `pause` (`sys.cuh`, `pool.cuh`, `nn.cuh`). The pool, its sizing and
  its NUMA placement are x86_avx2's, but for the kinds of core: on a processor whose cores differ, a core under half the
  fastest one's speed (`ARM_BASE__POOL_SPEED_FLOOR`, each core's speed the system's `cpu_capacity`, else its highest clock)
  is slow, and its thread takes only the GEMVs' launches (`arm_base__zzpackage_run_every`) — every other launch is the fast
  cores' and the caller's. A machine whose cores are alike is as it was.
- `overrides/nn.cuh` — the doors that run faster than nn's generic body, every one falling back to it for what it does
  not cover:
  - the exact GEMVs (`turboquant_gemv`, `_groups`, `_groups_n`, `_groups_sum`) — at 4 bits a byte's two codes split and
    each level's half found by `tbl` in two tables of sixteen; at 8 a table of 256 floats; at 16 the code is the weight —
    eight running sums of four lanes, one order for every dot (`arm_base__zzprivate_sum8`);
  - the expert-major GEMM (`expert_rows`, `expert_rows_sum`, `expert_groups`) — a prompt's routed experts: rows decoded
    once to floats, three rows by four pairs in eight passes, each (row, pair) summed as the gemv sums it, so the gemv's bits;
  - the expert-major GEMM in integers (`expert_groups_int8`, `expert_rows_sum_int8`, with `experts_int8`) — `x` quantised
    once a call as nn's definition says, a block's products through Armv8.2's `sdot`: nn's int8 body's bytes. A family's
    own rows go to the same bodies (`arm_base__int8_rows_fn`);
  - the GEMVs in integers (`turboquant_gemv_int8`, `_groups_int8`, `_groups_sum_int8`), the same way: nn's bytes;
  - the rotations (`hadamard_rotate`, `hadamard_ladder_rows`, `hadamard_blocks`) — `x86_avx2`'s butterfly in floats where
    nn's body sums each output over its whole block;
  - the hyper-connections' logits (`hyper_logits_fp32`) summed in double, as `x86_avx2`'s;
  - and, a file a verb group as nn files them, included before the door table:
    - `attention.cuh` — the decode's weights and mix over the cache (`attention_weights`, `attention_residual_mix`, and
      their TurboQuant tiers' `_tq`) and the prompt's (`attention_causal_scores_grouped`, `attention_causal_mix`,
      `attention_mix`): the scores a block a (kv head, run of positions), each key row widened or decoded once for the
      group's queries; every mix one body, a block a (row, kv head, 16 elements), each element's sum the positions in
      nn's order — nn's bytes;
    - `deltanet.cuh` — the step, its rows and its gate named (`deltanet_step`, `_steps`, `_steps_gate`), a block a value
      head, its state 32 columns at a time and read from memory once a step; the conv (`deltanet_conv_step`, `_steps`),
      a block 64 channels — nn's arithmetic in nn's order, so nn's bytes;
    - `primitives.cuh` — the norms (`rmsnorm`, `_rows`, `_fp32`, `_fp32_rows`), the squares summed in nn's order and the
      outputs eight at a time, one body a row: nn's bytes;
    - `vector.cuh` — the zero and the add (`vector_zero`, `vector_add`) as the memory's, `x86_avx2`'s: nn's bytes.
- `../arm_gb10/overrides/nn.cuh` — the GB10's own: the int8 GEMM two rows by two pairs through i8mm's `smmla`, the same
  integer sums, so the same bytes.

## On an Arm machine

The engine module is built without `-ftls-model=initial-exec` off x86-64 (setup.py, `release/scripts/build.sh`): aarch64's
glibc has no room in its static block for the module's thread-locals. `install.sh` builds `arm_neon` and `arm_gb10` on
an aarch64 machine in `x86_avx2`'s place. The Python side asks `machine.CPU_FAMILIES` for the machine's own
(`Machine.cpu`): a model's CPU worker is that family's.

Each family gives its own logits, the same bit for bit in every process (⚖ 2026-11-10 *"i dont really mind the difference
between the setups as long as each run is reproducible"*): the two may part, as GCC fuses multiply-adds under each
`-march` as it inlines — the 35B the same through both, the 4B apart from its first generated token. The pool's launch
fences the counter before the job (`primitives/pool.cuh`): Arm, unlike x86, may show a later store before an earlier one.

`MEASURED` on the DGX Spark: measurements/2026-11-09_arm_base_gb10.md, measurements/2026-11-10_arm_pool_order_gb10.md;
on a phone (the Razer Edge, Android under Termux): measurements/2026-11-10_nn171_razer_edge.md,
measurements/2026-11-11_nn175_arm_doors_razer_edge.md.

## Not yet

The MLP's activations (`swiglu_combine`, `swiglu_pairs`), gated attention's gate (`attention_gate_rows`) and the
argmax (nn's generic body here); SVE's wider vectors (the GB10's are 128 bits, as NEON's); the GB10's two kinds of core
read and measured — its Cortex-A725 are fast by the floor where its `cpu_capacity` or clocks put them above half its
X925's (`ASSUMED`, its clocks 2.8 against 3.9 GHz; not read there); a phone's cores the system takes out of use while it
is idle — the Razer Edge's process saw CPUs 0-3, 5 and 6 (`taskset -p`: `6f`) while its cpuset held 0-7, `ASSUMED`
Qualcomm's `core_ctl` — which the pool, sized once at bind, does not get back.
