;; MINISTRAL 3 — Mistral's dense models (the 14B; the 8B and 3B are the same shape), as nn's own words.
;;
;; A layer is grouped-query attention and a SwiGLU MLP, each behind an RMS norm with a plain gain: no package of its own.
;; A token is `(begin (position tok pos) (head))` — its row of the embedding, a table in RAM the card reads in place, its
;; angles a row of two tables the boot writes (▶ mistral3.py: YaRN's frequencies and Llama 4's query scale are worked out
;; there, the scale folded into the queries' table), and every layer — and an answer is `(generate tok pos n …)`
;; (▶ generate.lisp), that a token at a time until an end token, on the card. A prompt runs as rows, each layer once over a
;; chunk of them, nn's attention and MLP words: `(begin (embed_rows n) (rows first n))`.
;;
;; What the boot binds before this file is read (▶ mistral3.py `_bindings`):
;;   the shape       HIDDEN INTER EPS VOCAB RUN Q_HEADS KV_HEADS HEAD_DIM Q_WIDTH KV_WIDTH
;;                   EMBEDDING EMB_HALVES EMB_SCALES EMB_BITS — the embedding table in RAM: every token's row, a row's
;;                   halves, where the scales begin, the rows' width · HEAD_CODES HEAD_SCALES HEAD_BITS
;;                   CS_K CS_Q CS_HALVES — a position's cosines then sines, fp32, a row each position: the keys', and the
;;                   queries' with their scale in them; a row's halves
;;                   OUT STOP STOPS SEEN — `generate`'s (▶ generate.lisp) · EMBED_ON_CPU CARD CPU, and EMB_TOK where the
;;                   CPU worker holds the table: the token it decodes, a chunk's count
;;                   KV_TYPE KEYS_AT VALUES_AT KV_ROW_BYTES CACHE_BYTES — the cache, a slot a layer: its keys in the up
;;                   half, its values in the down, [position][kv head][head dim] in halves
;;                   Q_BITS K_BITS V_BITS O_BITS GATE_BITS UP_BITS DOWN_BITS — a matrix's width, the same in every layer
;;   the settings    MLP_WORDS — true where the pack keeps one of the MLP's matrices in halves
;;   a layer each    arrays indexed by the layer's place in RUN: LAYER (its number), INPUT_NORM, POST_NORM, each matrix's
;;                   two planes — Q_CODES Q_SCALES, K_…, V_…, O_…, GATE_…, UP_…, DOWN_… — MLP (its MLP's table, nn's) and,
;;                   where a prompt runs as rows, ATTENTION (its attention's table, nn's)
;;   the buffers     x h1 u hu h hr y hn hnr qq kk att attr g up gr scores logits_h res · a chunk's xs h1s and PROMPT


;; out = W·x for one matrix: its input rotated, as the pack rotated its columns; one kept in halves multiplied as it is
(defun (project codes scales bits x out rows cols)
  (if (eq? bits 16)
      (nn__expert__multiply_fp16 codes x out rows cols)
      (nn__turboquant__gemv codes scales x out rows cols bits)))

;; layer `l`'s keys or values (`half` KEYS_AT or VALUES_AT): all of them, or the row `at` bytes in
(defun (cache l half)
  (nn__expert__plane l KV_TYPE 0 half CACHE_BYTES))
(defun (cache_row l half at)
  (nn__expert__plane l KV_TYPE 0 (+ half at) KV_ROW_BYTES))


;; ── the token's row of the embedding, un-rotated (Rᵀu = S ⊙ H·u), into the residual ───────────────────────────────
;; EMBEDDING is a table in RAM the card reads a row of in place: every token's packed row (EMB_HALVES halves each), then
;; every token's scale from EMB_SCALES on — only a row crosses. With EMBED_ON_CPU the table is the CPU worker's
;; (c_embedding) and the CPU decodes the row (▶ mistral3_cpu.lisp `xe`), the card copying it across: where the card's
;; family keeps mapped RAM as a copy of its own, the table neither crosses nor takes the card's room.
(if EMBED_ON_CPU
    (defun (embed_into tok into)
      (begin (sys__node_array__set EMB_TOK 0 tok)
             (sys__compute CPU names xe) (sys__result CPU)
             (nn__buffer__copy c_emb CPU u HIDDEN_BYTES)
             (nn__hadamard__rotate u ones512 hu HIDDEN)
             (nn__vector__pointwise_mul hu signs_row into HIDDEN)
             (type x)))
    (defun (embed_into tok into)
      (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMB_HALVES) EMB_HALVES)
                                     (nn__vector__range EMBEDDING (+ EMB_SCALES tok) 1) u 1 HIDDEN EMB_BITS)
             (nn__hadamard__rotate u ones512 hu HIDDEN)
             (nn__vector__pointwise_mul hu signs_row into HIDDEN)
             (type x))))

(defun (embed tok)
  (embed_into tok x))

;; a prompt's `n` tokens, PROMPT's first `n`, a row of xs each — with the table the CPU's, its `n` rows in one hand-off
;; (▶ mistral3_cpu.lisp `xer`) and one copy into h1s, free until the layers, then each row un-rotated into xs
(if EMBED_ON_CPU
    (defun (embed_rows n)
      (let ((i 0) (at 0))
        (sys__node_array__set EMB_TOK 1 n)
        (sys__compute CPU names xer) (sys__result CPU)
        (nn__buffer__copy c_embs CPU h1s (* n HIDDEN_BYTES))
        (while '(< i n)
               '(begin (nn__hadamard__rotate (nn__vector__range h1s at HIDDEN) ones512 hu HIDDEN)
                       (nn__vector__pointwise_mul hu signs_row (nn__vector__range xs at HIDDEN) HIDDEN)
                       (set! at (+ at HIDDEN)) (set! i (+ i 1))))
        (type xs)))
    (defun (embed_rows n)
      (let ((i 0) (at 0))
        (while '(< i n)
               '(begin (embed_into (nth PROMPT i) (nn__vector__range xs at HIDDEN))
                       (set! at (+ at HIDDEN)) (set! i (+ i 1))))
        (type xs))))


;; ── a layer ─────────────────────────────────────────────────────────────────────────────────────────────────────────
;; the MLP, from the attention's sum `h1` into `x`: nn's word over the layer's table — the norm, gate and up in one
;; launch, the residual added in the down's — or, where the pack keeps a matrix of it in halves, nn's words one at a time
(if MLP_WORDS
    (defun (mlp i)
      (begin (nn__rmsnorm__apply h1 (nth POST_NORM i) h HIDDEN EPS)
             (nn__hadamard__rotate h signs hr HIDDEN)
             (project (nth GATE_CODES i) (nth GATE_SCALES i) GATE_BITS hr g INTER HIDDEN)
             (project (nth UP_CODES i) (nth UP_SCALES i) UP_BITS hr up INTER HIDDEN)
             (nn__swiglu__combine g up g INTER)
             (nn__hadamard__rotate g signs gr INTER)
             (project (nth DOWN_CODES i) (nth DOWN_SCALES i) DOWN_BITS gr y HIDDEN INTER)
             (nn__vector__add h1 y x HIDDEN)))
    (defun (mlp i)
      (nn__mlp__apply h1 (nth MLP i) x)))

(defun (layer i at len csq csk)
  (let ((l (nth LAYER i)))
    ;; attention: q k v — v straight into its cache row — RoPE on q (its angles carry the queries' scale), and on k into
    ;; its cache row, then every position
    (nn__rmsnorm__apply x (nth INPUT_NORM i) h HIDDEN EPS)
    (nn__hadamard__rotate h signs hr HIDDEN)
    (project (nth Q_CODES i) (nth Q_SCALES i) Q_BITS hr qq Q_WIDTH HIDDEN)
    (project (nth K_CODES i) (nth K_SCALES i) K_BITS hr kk KV_WIDTH HIDDEN)
    (project (nth V_CODES i) (nth V_SCALES i) V_BITS hr (cache_row l VALUES_AT at) KV_WIDTH HIDDEN)
    (nn__rope__apply qq csq qq Q_HEADS HEAD_DIM)
    (nn__rope__apply kk csk (cache_row l KEYS_AT at) KV_HEADS HEAD_DIM)
    (nn__attention__decode qq (cache l KEYS_AT) (cache l VALUES_AT) scores att Q_HEADS KV_HEADS HEAD_DIM 0 len)
    (nn__hadamard__rotate att signs attr Q_WIDTH)
    (project (nth O_CODES i) (nth O_SCALES i) O_BITS attr y HIDDEN Q_WIDTH)
    (nn__vector__add x y h1 HIDDEN)
    (mlp i)
    (type x)))

(defun (layers at len csq csk)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i at len csq csk) (set! i (+ i 1))))
    (type x)))

;; a prompt's rows `first .. first + n`, PROMPT's tokens embedded in xs: each layer once over all of them — nn's attention
;; word, its keys and values into the cache at their positions, then nn's MLP word — the rows back in xs
(defun (rows first n)
  (let ((i 0))
    (while '(< i RUN)
           '(begin (nn__attention__rows xs (nth ATTENTION i) first h1s n)
                   (nn__mlp__rows h1s (nth MLP i) xs n)
                   (set! i (+ i 1))))
    (type xs)))

;; a token's position: its embedding, its rows of the angles, every layer — its cache rows `pos` rows in
(defun (position tok pos)
  (begin (embed tok)
         (layers (* pos KV_ROW_BYTES) (+ pos 1) (nn__vector__range CS_Q (* pos CS_HALVES) CS_HALVES)
                 (nn__vector__range CS_K (* pos CS_HALVES) CS_HALVES))))


;; ── the head ────────────────────────────────────────────────────────────────────────────────────────────────────────
(defun (logits)
  (begin (nn__rmsnorm__apply x fnw hn HIDDEN EPS)
         (nn__hadamard__rotate hn signs hnr HIDDEN)
         (nn__turboquant__gemv HEAD_CODES HEAD_SCALES hnr logits_h VOCAB HIDDEN HEAD_BITS)))

(defun (head)
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)
  (begin (logits) (type x)))
