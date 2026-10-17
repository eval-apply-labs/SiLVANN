# models

Each model is a folder here, fetched from Hugging Face the first time it is asked for:

    python3 silvann_core.py list-models
    python3 silvann_core.py download-model silvann_tq_D8E4_qwen3.6_35b_a3b

A folder holds the packed weights, the tokenizer, the model's own licence, `lisp/` — the model's program, which the boot
reads and the evaluator runs; edit it and the next boot runs what you wrote — and `configs/default.json`, the settings
the server boots it with. A folder fetched without `lisp/` gets it from the same place the next time it is named. A model with experts also writes `experts/` beside its weights the first time it runs them from the disk (the
routed experts in the layout the CPU reads them in, written once: ~150 GB for GLM 5.3 Flash, ~55 GB for the 122B, ~16 GB
for the 35B); the config's `experts_dir` puts it elsewhere.
