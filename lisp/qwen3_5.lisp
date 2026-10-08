;; QWEN 3.5 — the DeltaNet-and-attention family: dense (the 27B) or with routed experts (the 35B-A3B, the 122B, the 397B).
;;
;; A token is `(begin (position tok pos) (head))`: the token's embedding, the rotary angles, every layer this machine runs,
;; and the head — and an answer is `(generate tok pos n …)`, that a token at a time until an end token, on the card. A
;; layer is a mixer — a DeltaNet layer's or an attention layer's, its residual included — then the MLP: dense, or the
;; routed experts. A prompt runs as rows, each layer once over a chunk of them: `(begin (embed_rows n) (rows first n))`.
;;
;; What the boot binds before this file is read (▶ qwen3_5.py `_bindings`):
;;   the shape       HIDDEN HIDDEN_BYTES VOCAB EPS THETA ROTARY RUN — RUN the layers this machine runs
;;                   EMBEDDING EMB_HALVES EMB_SCALES EMB_BITS — the embedding table in RAM, a row's halves, where the
;;                   scales begin, the rows' width
;;                   PROMPT OUT STOP STOPS SEEN — a prompt chunk's tokens, what `generate` made, the end tokens
;;                   HEAD_CODES HEAD_SCALES HEAD_BITS — the head's two planes
;;   the settings    ROTATED (the pack keeps its residual rotated) · TIERED (the attention cache in tiers) · ROUTED (the
;;                   model has routed experts) · FUSED (the MoE as one word) · EXPERTS_ON_CPU · CARD_TIER — true or false
;;   a layer each    arrays indexed by the layer's place in RUN: DELTANET (true for a DeltaNet layer), LAYER (its number),
;;                   MIXER and MIXER_ROWS (its mixer's table, a position's and a chunk's), and the MLP's tables — MLP and
;;                   MLP_ROWS (dense), MOE (the fused MoE's), PRE POST EXPERTS (the MoE in three), EXPERTS_TIER and
;;                   EXPERTS_TIER_ROWS (the card's tier). A table is a node array of planes and integers; the cells each
;;                   verb reads are named in its package's contracts/objects/ — ai_qwen_3's, and nn's for the dense MLP.
;;   the buffers     x h h1 hand picks routed xs h1s hands rrows logits_h res … (▶ qwen3_5.py `_buffers`)


;; ── the token: its row of the embedding, decoded where it is wanted ─────────────────────────────────────────────────
;; EMBEDDING is a table in RAM the card reads a row of in place: every token's packed row (EMB_HALVES halves each), then
;; every token's scale from EMB_SCALES on — the embedding stays in RAM and only a row crosses.
(if ROTATED
    (defun (embed_into tok into)                       ; the decoded row is R·e, the residual's own basis
      (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMB_HALVES) EMB_HALVES)
                                     (nn__vector__range EMBEDDING (+ EMB_SCALES tok) 1) into 1 HIDDEN EMB_BITS)
             (type x)))
    (defun (embed_into tok into)                       ; un-rotated: Rᵀu = S ⊙ H·u
      (let ((u (nn__buffer__getnew HIDDEN_BYTES)) (hu (nn__buffer__getnew HIDDEN_BYTES)))
        (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMB_HALVES) EMB_HALVES)
                                (nn__vector__range EMBEDDING (+ EMB_SCALES tok) 1) u 1 HIDDEN EMB_BITS)
        (nn__hadamard__rotate u ones512 hu HIDDEN)
        (nn__vector__pointwise_mul hu signs_row into HIDDEN)
        (type x))))

(defun (embed tok)
  (embed_into tok x))

;; a prompt's `n` tokens, PROMPT's first `n`, a row of xs each
(defun (embed_rows n)
  (let ((i 0) (at 0))
    (while '(< i n)
           '(begin (embed_into (nth PROMPT i) (nn__vector__range xs at HIDDEN))
                   (set! at (+ at HIDDEN)) (set! i (+ i 1))))
    (type xs)))


;; ── the head: the final norm, then the logits ───────────────────────────────────────────────────────────────────────
(if ROTATED
    (defun (logits)
      (begin (nn__rmsnorm__apply x fnw h HIDDEN EPS)
             (nn__turboquant__gemv HEAD_CODES HEAD_SCALES h logits_h VOCAB HIDDEN HEAD_BITS)))
    (defun (logits)
      (begin (nn__rmsnorm__apply x fnw h HIDDEN EPS)
             (nn__hadamard__rotate h signs h1 HIDDEN)
             (nn__turboquant__gemv HEAD_CODES HEAD_SCALES h1 logits_h VOCAB HIDDEN HEAD_BITS))))

(defun (head)                                          ; the greedy token
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)                                   ; the logits left for a sampler on the host
  (begin (logits) (type x)))


;; ── a layer's mixer, writing what the MLP reads into `into` ─────────────────────────────────────────────────────────
(if TIERED
    (defun (mixer i pos into)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet x (nth MIXER i) into)
          (ai_qwen_3__attention_tiered x (nth MIXER i) pos into)))
    (defun (mixer i pos into)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet x (nth MIXER i) into)
          (ai_qwen_3__attention x (nth MIXER i) pos into))))

(if TIERED
    (defun (mixer_rows i first n)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet_rows xs (nth MIXER_ROWS i) h1s n)
          (ai_qwen_3__attention_tiered_rows xs (nth MIXER_ROWS i) first h1s n)))
    (defun (mixer_rows i first n)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet_rows xs (nth MIXER_ROWS i) h1s n)
          (ai_qwen_3__attention_rows xs (nth MIXER_ROWS i) first h1s n))))


;; ── a layer at a position, and a layer over a prompt's rows: the mixer, then the MLP — one of five ──────────────────
;; ① dense · ② the MoE as one word · ③ the MoE in three on the card · ④ the routed experts on the CPU · ⑤ and a card's
;; tier of them beside the CPU's. With the experts on the CPU the card routes into `hand`, the CPU worker runs the
;; picture `xq` over the layer named in HERE (▶ qwen3_5_cpu.lisp), and the card closes the layer with the routed sum.
(if ROUTED
    (if EXPERTS_ON_CPU
        (if CARD_TIER
            (defun (layer i pos)                       ; ⑤
              (begin (mixer i pos h1)
                     (ai_qwen_3__pre_expert h1 (nth PRE i) hand picks)
                     (ai_qwen_3__experts hand picks (nth EXPERTS_TIER i) routed_k nn__expert_tier)
                     (sys__node_array__set HERE 0 i)
                     (sys__compute CPU names xq) (sys__result CPU)
                     (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
                     (nn__vector__add routed routed_k routed HIDDEN)
                     (ai_qwen_3__post_expert h1 hand routed (nth POST i) x)
                     (type x)))
            (defun (layer i pos)                       ; ④
              (begin (mixer i pos h1)
                     (ai_qwen_3__pre_expert h1 (nth PRE i) hand picks)
                     (sys__node_array__set HERE 0 i)
                     (sys__compute CPU names xq) (sys__result CPU)
                     (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
                     (ai_qwen_3__post_expert h1 hand routed (nth POST i) x)
                     (type x))))
        (if FUSED
            (defun (layer i pos)                       ; ②
              (begin (mixer i pos h1) (ai_qwen_3__moe h1 (nth MOE i) x) (type x)))
            (defun (layer i pos)                       ; ③
              (begin (mixer i pos h1)
                     (ai_qwen_3__pre_expert h1 (nth PRE i) hand picks)
                     (ai_qwen_3__experts hand picks (nth EXPERTS i) routed)
                     (ai_qwen_3__post_expert h1 hand routed (nth POST i) x)
                     (type x)))))
    (defun (layer i pos)                               ; ①
      (begin (mixer i pos h) (nn__mlp__apply h (nth MLP i) x) (type x))))

;; a prompt's: with the experts on the CPU, the card routes every row into its hand row, the CPU runs each expert once
;; over the rows that picked it, the card closes every row — waited for without a bound, as a chunk's experts read from
;; the disk can outlast `sys__result`'s wait. A card's tier takes its share of the chunk first and computes it while
;; the CPU computes the rest.
(if ROUTED
    (if EXPERTS_ON_CPU
        (if CARD_TIER
            (defun (rows_layer i first n)              ; ⑤
              (begin (mixer_rows i first n)
                     (ai_qwen_3__pre_expert_rows h1s (nth PRE i) hands n)
                     (nn__expert_tier__note hands nn__expert_tier (nth LAYER i) n HAND_ROW HAND_PICKS_AT
                                            nn__expert_major__vram_promotion_threshold_picks)
                     (sys__node_array__set HERE 0 i) (sys__node_array__set nrows 0 n)
                     (sys__compute CPU names xr)
                     (ai_qwen_3__experts_rows hands (nth EXPERTS_TIER_ROWS i) rrows_k n nn__expert_tier)
                     (while '(eq? (sys__completed CPU) (< 1 0)) '(< 0 1))
                     (sys__result CPU)
                     (nn__buffer__copy c_rrows CPU rrows ROWS_BYTES)
                     (nn__vector__add rrows rrows_k rrows ROWS_HALVES)
                     (ai_qwen_3__post_expert_rows h1s hands rrows (nth POST i) xs n)
                     (type xs)))
            (defun (rows_layer i first n)              ; ④
              (begin (mixer_rows i first n)
                     (ai_qwen_3__pre_expert_rows h1s (nth PRE i) hands n)
                     (sys__node_array__set HERE 0 i) (sys__node_array__set nrows 0 n)
                     (sys__compute CPU names xr)
                     (while '(eq? (sys__completed CPU) (< 1 0)) '(< 0 1))
                     (sys__result CPU)
                     (nn__buffer__copy c_rrows CPU rrows ROWS_BYTES)
                     (ai_qwen_3__post_expert_rows h1s hands rrows (nth POST i) xs n)
                     (type xs))))
        (defun (rows_layer i first n)                  ; ② ③ — a chunk's MoE is one word either way
          (begin (mixer_rows i first n) (ai_qwen_3__moe_rows h1s (nth MOE i) xs n) (type xs))))
    (defun (rows_layer i first n)                      ; ①
      (begin (mixer_rows i first n) (nn__mlp__rows h1s (nth MLP_ROWS i) xs n) (type xs))))


;; ── every layer this machine runs ───────────────────────────────────────────────────────────────────────────────────
(defun (layers pos)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i pos) (set! i (+ i 1))))
    (type x)))

(defun (rows first n)
  (let ((i 0))
    (while '(< i RUN) '(begin (rows_layer i first n) (set! i (+ i 1))))
    (type xs)))

;; a token's position: its embedding, the angles, the layers
(defun (position tok pos)
  (begin (embed tok) (nn__rope__angles cs pos THETA ROTARY) (layers pos)))


;; ── GENERATION: the tokens of an answer, each the next one's input, without the host between them ─────────────────
;; From `tok` at `pos`, up to `n` tokens into OUT, ending early after one of STOP's first STOPS. `draw` true draws each
;; with the card's sampler (▶ sampling.lisp: `penalty inv_t k top_p seed`), false takes the greedy one. Where the
;; penalty reads them (`reads` 1) each token goes into `seen` first, at `slot`, `count` of them there. Answers how many
;; it made.
(defun (stop? tok)
  (let ((j 0) (hit (< 1 0)))
    (while '(< j STOPS) '(begin (if (eq? (nth STOP j) tok) (set! hit (< 0 1)) 0) (set! j (+ j 1))))
    hit))

(defun (generate tok pos n draw reads count slot penalty inv_t k top_p seed)
  (let ((i 0))
    (while '(< i n)
           '(begin
              (position tok pos)
              (if (eq? reads 1)
                  (begin (sys__node_array__set seen slot tok)
                         (set! slot (+ slot 1)) (if (eq? slot SEEN) (set! slot 0) 0)
                         (if (< count SEEN) (set! count (+ count 1)) 0))
                  0)
              (set! tok (if draw (head_sample count penalty inv_t k top_p seed pos) (head)))
              (sys__node_array__set OUT i tok)
              (set! i (+ i 1)) (set! pos (+ pos 1))
              (if (stop? tok) (set! n i) 0)))
    i))
