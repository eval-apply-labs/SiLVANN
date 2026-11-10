# Platforms — what SiLVANN is tested on, and what it is not yet

The README's short table, in full. A platform not listed here may well work; one that does not is an issue to open.

## Tested

**Linux with AMD Instinct MI50 (gfx906), ROCm 6** — `amd_rocm6_wave64`
- a model on one card, and the 27B's layers over two and four (tensor parallelism)
- DeepSeek-V4.1-Flash on one card with its experts on the CPU, checked against DeepSeek's own code; over both MI50s with
  an exclusive tier; and on the CPU alone from the disk

**Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595**
- every pack but GLM's checked against Hugging Face, the 35B's tier of experts included

**Linux with an AMD Instinct MI355X (gfx950), TheRock's ROCm 7.14** — `amd_rocm10` (rented)
- nn's verbs and the family's doors on the card; Qwen 3.6 35B-A3B checked against Hugging Face; GLM 5.3 Flash whole on the
  card, 66-69 tokens a second, its prompt 304-307 positions a second with the matrix cores against 127 without

**Linux with an NVIDIA H100 (sm_90) and a B200 (sm_100), CUDA 12.8, driver 595** (rented)
- the family's doors and nn's verbs on each card; the 35B, the 122B and the 27B checked against Hugging Face on the H100
  (the 35B 117-121 tokens a second after a 2,000-word prompt, the 122B 69-70, every expert on the card)
- GLM 5.3 Flash whole on the B200, every expert on the card: 40 tokens a second after a 2,000-word prompt, read at 354-374
  positions a second, its answer the one GLM gives with its experts on the CPU

**The CPU alone**
- AVX2 on a Xeon E5-2680 v4, AVX-512 on a Xeon W-2123

**Linux inside WSL2 with an AMD Radeon RX 9070 XT (gfx1201), ROCm 7.2** — `amd_rocm10`, in wave64 (rented)
- nn's verbs and the family's doors on the card, the card found inside WSL2 with nothing set; Qwen 3.5 4B `D4` checked
  against Hugging Face, 118-120 tokens a second after a 2,000-word prompt, read at 1,859-1,876 positions a second
  (through OpenCL 2.0 on the same card: 72 and 525)

**Linux inside WSL2 with an AMD Radeon RX 7900 XTX (gfx1100), ROCm 7.2** — OpenCL 2.0 (rented)
- Qwen 3.5 4B `D4` and Qwen 3.6 35B `D8E4` whole on the card checked against Hugging Face: the 35B 20 tokens a second after
  a 2,000-word prompt, read at 270 positions a second

**OpenCL 2.0 with sub-groups, on the MI50**
- the 27B and the 35B checked against Hugging Face; the 27B answering at 9-11 tokens a second and reading a prompt at 49
  positions a second
- the 122B's tier of experts: 11.7 tokens a second with 18 GB of the card (9.7 without)

**OpenCL 1.2 through NVIDIA's OpenCL, on the RTX 5060 Ti**
- the 27B checked against Hugging Face, at 1.6 tokens a second (the family is for integrated GPUs)

**Windows with an NVIDIA card, natively** (an RTX 5070 Ti, a Ryzen 7 5800X)
- built from source with CUDA 13.4, Ministral 3 14B `D4` on the card ([`users.md`](users.md#on-windows))

## Not yet tested

**On Windows**
- the prebuilt libraries on a PC (GitHub's runners, which build them, have no card)
- the CPU alone, a model's experts on the CPU or read from the disk
- AMD's and Intel's GPUs through OpenCL

**On Linux**
- booted from a USB SSD

**NVIDIA**
- tensor parallelism on NVIDIA cards
- DeepSeek-V4.1-Flash on an NVIDIA card, and GLM 5.3 Flash on one with its experts on the CPU
- other generations: the CUDA family is built for sm_75 to sm_120 (Turing to Blackwell, the B300's sm_103 among them) and
  run on sm_86 and sm_89 (an RTX 3060 and an RTX 4050 laptop's, under WSL2: nn's verbs, the doors, Qwen 3.5 4B against
  Hugging Face), sm_90, sm_100 and sm_120; an older card — Kepler, Maxwell, Pascal, Volta — goes to OpenCL 1.2, the GTX 1080 first

**AMD**
- the MI100, the MI200s and the MI300s — gfx908, gfx90a, gfx942 — which `amd_rocm10` is built for with ROCm 7 or later
  (the MI50 stays on ROCm 6, which builds it), its prompt's GEMM tiles on their matrix cores; it has run on an MI355X
  (above)
- the other Radeon `amd_rocm10` is built for — gfx1200 (the RX 9060s) — and a Radeon on Linux outside WSL2; their matrix
  instructions (WMMA) are not used yet, the prompt's GEMM taking the plain tiles there. Any other Radeon goes to OpenCL
  2.0 — RDNA3 too: built in wave64, an RX 7900 XTX answered wrongly (a shuffle reaches half its wave there); through
  OpenCL 2.0 the same card ran the 4B correctly

**OpenCL on other devices**
- the integrated GPUs `intel_opencl1_igpu` (Intel's before Gen9) and `intel_opencl2` (from Gen9) are for among them. On one, the model's memory is read where it is, not
  copied to the GPU: checked only through PoCL, a CPU's OpenCL
- which card goes to OpenCL 1.2 and which to 2 is a best guess from the vendors' published versions — OpenCL 1.2: every
  NVIDIA card CUDA does not take, Intel's iGPUs before Gen9, AMD's TeraScale and GCN 1; OpenCL 2: the rest. A card on
  the wrong one is an issue to open, with its `clinfo`

**The CPU**
- AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons)

## Not supported

- macOS
