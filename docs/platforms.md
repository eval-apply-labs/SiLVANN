# Platforms — what SiLVANN is tested on, and what it is not yet

The README's short table, in full. A platform not listed here may well work; one that does not is an issue to open.

## Tested

**Linux with AMD Instinct MI50 (gfx906), ROCm 6**
- a model on one card, and the 27B's layers over two and four (tensor parallelism)
- DeepSeek-V4.1-Flash on one card with its experts on the CPU, checked against DeepSeek's own code; over both MI50s with
  an exclusive tier; and on the CPU alone from the disk

**Linux with an NVIDIA GeForce RTX 5060 Ti (sm_120), CUDA 12.9, driver 595**
- every pack but GLM's checked against Hugging Face, the 35B's tier of experts included

**The CPU alone**
- AVX2 on a Xeon E5-2680 v4, AVX-512 on a Xeon W-2123

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
- DeepSeek-V4.1-Flash and GLM 5.3 Flash on an NVIDIA card
- other generations: the CUDA family is built for sm_75 to sm_120 (Turing to Blackwell); an older card — Kepler,
  Maxwell, Pascal, Volta — goes to OpenCL 1.2, the GTX 1080 first

**AMD**
- other generations than gfx906

**OpenCL on other devices**
- the integrated GPUs `khronos_opencl1_igpu` is for among them. On one, the model's memory is read where it is, not
  copied to the GPU: checked only through PoCL, a CPU's OpenCL
- which card goes to OpenCL 1.2 and which to 2 is a best guess from the vendors' published versions — OpenCL 1.2: every
  NVIDIA card CUDA does not take, Intel's iGPUs before Gen9, AMD's TeraScale and GCN 1; OpenCL 2: the rest. A card on
  the wrong one is an issue to open, with its `clinfo`

**The CPU**
- AVX-512 on processors other than Skylake (AMD's Zen 4 and 5, later Xeons)

## Not supported

- macOS
