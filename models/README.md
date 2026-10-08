# models

Each model is a folder here, fetched from Hugging Face the first time it is asked for:

    python3 silvann_core.py list-models
    python3 silvann_core.py download-model Qwen3.6-35B-A3B_silvann_tq_D8E4

A folder holds the packed weights, the tokenizer, the model's own licence, `lisp/` — the model's program, which the boot
reads and the evaluator runs; edit it and the next boot runs what you wrote — and `configs/default.json`, the settings
the server boots it with. A folder fetched without `lisp/` gets it from the same place the next time it is named. GLM 5.3 Flash also writes `experts/` beside its weights the first time it runs from the disk (the
routed experts in the layout the CPU reads them in; ~150 GB, written once).
