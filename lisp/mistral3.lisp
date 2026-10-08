;; MINISTRAL 3 — Mistral's dense models (the 14B; the 8B and 3B are the same shape), as nn's own words.
;;
;; A layer is grouped-query attention and a SwiGLU MLP, each behind an RMS norm with a plain gain: no package of its own.
;; A token is `(begin (position at len qs) (head))` — its byte offset in the cache rows, the positions so far, and the
;; queries' scale — with its embedding row and its rotary angles written by the host first (▶ mistral3.py: YaRN's
;; frequencies and Llama 4's query scale are worked out there).
;;
;; What the boot binds before this file is read (▶ mistral3.py `_bindings`):
;;   the shape       HIDDEN INTER EPS VOCAB RUN Q_HEADS KV_HEADS HEAD_DIM Q_WIDTH KV_WIDTH
;;                   EMB_HALVES EMB_BITS · HEAD_CODES HEAD_SCALES HEAD_BITS
;;                   KV_TYPE KEYS_AT VALUES_AT KV_ROW_BYTES CACHE_BYTES — the cache, a slot a layer: its keys in the up
;;                   half, its values in the down, [position][kv head][head dim] in halves
;;                   Q_BITS K_BITS V_BITS O_BITS GATE_BITS UP_BITS DOWN_BITS — a matrix's width, the same in every layer
;;   the settings    MLP_WORDS — true where the pack keeps one of the MLP's matrices in halves
;;   a layer each    arrays indexed by the layer's place in RUN: LAYER (its number), INPUT_NORM, POST_NORM, each matrix's
;;                   two planes — Q_CODES Q_SCALES, K_…, V_…, O_…, GATE_…, UP_…, DOWN_… — and MLP (its MLP's table, nn's)
;;   the buffers     x h1 u hu h hr y hn hnr qq kk att attr g up gr cs scores emb_codes emb_scales logits_h res


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


;; ── the token's row, un-rotated (Rᵀu = S ⊙ H·u), into the residual ─────────────────────────────────────────────────
(defun (embed codes scale)
  (begin (nn__turboquant__decode (nn__vector__range emb_codes codes EMB_HALVES) (nn__vector__range emb_scales scale 1)
                                 u 1 HIDDEN EMB_BITS)
         (nn__hadamard__rotate u ones512 hu HIDDEN)
         (nn__vector__pointwise_mul hu signs_row x HIDDEN)
         (type x)))


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

(defun (layer i at len qs)
  (let ((l (nth LAYER i)))
    ;; attention: q k v — v straight into its cache row — RoPE on q, and on k into its cache row, then every position
    (nn__rmsnorm__apply x (nth INPUT_NORM i) h HIDDEN EPS)
    (nn__hadamard__rotate h signs hr HIDDEN)
    (project (nth Q_CODES i) (nth Q_SCALES i) Q_BITS hr qq Q_WIDTH HIDDEN)
    (nn__vector__scale qq qs qq Q_WIDTH)
    (project (nth K_CODES i) (nth K_SCALES i) K_BITS hr kk KV_WIDTH HIDDEN)
    (project (nth V_CODES i) (nth V_SCALES i) V_BITS hr (cache_row l VALUES_AT at) KV_WIDTH HIDDEN)
    (nn__rope__apply qq cs qq Q_HEADS HEAD_DIM)
    (nn__rope__apply kk cs (cache_row l KEYS_AT at) KV_HEADS HEAD_DIM)
    (nn__attention__decode qq (cache l KEYS_AT) (cache l VALUES_AT) scores att Q_HEADS KV_HEADS HEAD_DIM 0 len)
    (nn__hadamard__rotate att signs attr Q_WIDTH)
    (project (nth O_CODES i) (nth O_SCALES i) O_BITS attr y HIDDEN Q_WIDTH)
    (nn__vector__add x y h1 HIDDEN)
    (mlp i)
    (type x)))

(defun (layers at len qs)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i at len qs) (set! i (+ i 1))))
    (type x)))

(defun (position at len qs)
  (begin (embed 0 0) (layers at len qs)))


;; ── the head ────────────────────────────────────────────────────────────────────────────────────────────────────────
(defun (logits)
  (begin (nn__rmsnorm__apply x fnw hn HIDDEN EPS)
         (nn__hadamard__rotate hn signs hnr HIDDEN)
         (nn__turboquant__gemv HEAD_CODES HEAD_SCALES hnr logits_h VOCAB HIDDEN HEAD_BITS)))

(defun (head)
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)
  (begin (logits) (type x)))
