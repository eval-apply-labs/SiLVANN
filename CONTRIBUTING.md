# Contributing to SiLVANN

Thank you for wanting to help. SiLVANN is a developer preview, and the most useful contributions right now are:

- **Running it on hardware we have not** — another AMD card, an NVIDIA card, an Intel or AMD iGPU through OpenCL, a
  CPU with AVX-512 — and reporting what happened, with the command, the model and the numbers.
- **Bug reports** with the smallest program or prompt that shows the problem.
- **Fixes and new silicon families** (`engine/src/silicon_families/add_a_family.md`).

## Before your first pull request

SiLVANN's engine is licensed under the AGPL-3.0 and is also offered under commercial terms. To keep both possible,
every contribution is made under the **Contributor License Agreement** in [`CLA.md`](CLA.md): you keep the
copyright in your work and grant the project a licence to distribute it under both. The **CLA Assistant** bot asks
you to accept it on your first pull request; it covers every later one.

## How we work

- Discuss larger changes first, in Discussions or an Issue, so nobody builds something that cannot be merged.
- A change comes with its evidence: a claim about speed or correctness names the command that measured it and
  the number it gave.
- The engine's build and its gates are in `engine/` (`build.sh`); keep them green.

Questions: GitHub Discussions, or `admin@evalapply.co.uk`.
