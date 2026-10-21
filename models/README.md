# models

Each model is a folder here, fetched from Hugging Face the first time it is asked for:

    python3 silvann_core.py list-models
    python3 silvann_core.py download-model silvann_tq_D8E4_qwen3.6_35b_a3b

A folder holds the packed weights, the tokenizer, the model's own licence, `lisp/` — the model's program, which the boot
reads and the evaluator runs; edit it and the next boot runs what you wrote — and `configs/default.json`, the settings
the server boots it with. A folder fetched without `lisp/` gets it from the same place the next time it is named. A model's experts run from the disk are read from its pack as they are wanted; where they do not lie
in the pack as the CPU reads them — a GLM 5.3 Flash pack downloaded before 19 October 2026, a LoRA that holds experts of
its own, a model split over two CPU sockets — they are written once to `experts/` beside its weights (about 150 GB for
GLM 5.3 Flash), and the config's `experts_dir` puts that elsewhere ([`users.md` §5](../docs/users.md#5-the-experts-on-the-disk)).
