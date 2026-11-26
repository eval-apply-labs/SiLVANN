;; INKLING 2 — `inkling2_loop_text`: 24 layers of weights, the middle twelve run three times — 48 applications a position;
;; an attention of 24 heads on 8 whose position is a bias a distance, a sink in its softmax; 128 experts of which a
;; position takes 8; and what a sublayer reads is a learned query's mix over the states its block of six began from.
;;
;; ⛳ A LAYER'S APPLICATION IS TWO VERBS OF THE MODEL'S PACKAGE AROUND ITS EXPERTS (⚖ 2026-11-24: a verb ends where the
;; evaluator has something to change). `before_experts` takes the stream to the router's choice — what the attention
;; reads, the attention, the short convolutions, the experts' own mix, the router; `after_experts` brings the experts'
;; sum back into the stream. Between them is this program's one decision in a layer: where the experts run — the card
;; that holds them, or the CPU (▶ inkling2_cpu.lisp). The arithmetic is scripts/inkling2_reference.py's, which is
;; Hugging Face's forward to 1e-6 (▶ src/docs/design/pending/inkling2_mini.md).
;;
;; ⛳ A CONVERSATION OF ANY LENGTH. A sliding layer's keys and values are a RING of the window's 1,024 positions — a
;; position's row is its place in the ring (`ring`), the new key taking the oldest's — so its cache is the same size
;; whatever the context; a full layer keeps a row a position.
;;
;; ⛳ NAMES ARE WHOLE WORDS. The language keeps every name in one place — a buffer, a procedure, a parameter of a program
;; this one runs inside (generate.lisp's, sampling.lisp's) — and a short one is taken twice before it is noticed. What
;; those two programs fix is kept as they spell it: `pos`, `tok`, `res`, `logits_h`, VOCAB.
;;
;; What the boot binds before this file is read (▶ inkling2.py `_bindings`, `_tables`):
;;   the shape       HIDDEN VOCAB EPSILON LAYERS LOOP_FROM LOOP_TO — HIDDEN_BYTES HAND_BYTES — CARD CPU ·
;;                   RING_POWERS RING_STEPS (the window times the powers of two below the context, the largest first) ·
;;                   EXPERTS_ON_CARD (true where the card holds the experts) EXPERTS_CARD (then a layer's experts table)
;;                   · CARD_TIER (true where the card keeps a tier of them, the rest the CPU's) EXPERTS_TIER (then a layer's
;;                   experts table over the tier's slots) nn__expert_tier HAND_PICKS_AT routed_tier routed_rows_tier
;;   the floats      LOGIT_CAP LOGIT_CAP_INVERSE
;;   the ends        EMBEDDING EMBEDDING_HALVES EMBEDDING_SCALES EMBEDDING_BITS — the embedding, a table in RAM a row of
;;                   which the card reads; or, EMBED_ON_CPU, the CPU worker's (c_embedding), EMB_TOK the token it decodes
;;                   · HEAD_CODES HEAD_SCALES HEAD_BITS
;;   the tables      LAYER_TABLES (a layer's weights) · APPLICATION_TABLES (an application's cache and its convolutions'
;;                   inputs) · WORK (what every application shares) — the verbs' three, ▶ the package's
;;                   contracts/objects/layer.cuh
;;   the buffers     stream · loop_start · states (the block states normed, a row each, the current one after them) ·
;;                   pass_states (what a pass leaves the next: its third block state and its last state, normed) ·
;;                   hand picks HERE routed … (▶ inkling2.py `_plan`)


;; the `index`-th run of `width` halves of a buffer: a row of several
(defun (part buffer index width)
  (nn__vector__range buffer (* index width) width))


;; ── the experts: this program's choice ──────────────────────────────────────────────────────────────────────────────
;; `before_experts` leaves the hand — the experts' input rotated, then the eight weights — and the picks. Where the card
;; holds the experts (EXPERTS_ON_CARD) it runs the model's experts verb itself; else the CPU does, and answers the
;; buffer it summed into.
;; ⭐ A CARD'S TIER OF THEM (CARD_TIER, ▶ inkling2.py `card_experts_gb`, as Qwen's): the tier's visit first — the picks the
;;   card holds marked in `picks`, so the CPU, handed the hand as it stands, computes only the rest — then the card's own
;;   into routed_tier while the CPU computes, and the two sums added. A pick the card computes is one the CPU leaves.
;;   ⛳ THE VISIT SAYS HOW MANY PICKS IT LEFT THE CPU: none, and the card computes into `routed` itself, the CPU not handed
;;   the hand at all — a lent tier holding every expert spent 3.4 ms a token on 48 empty hand-offs after 50,000 positions,
;;   50.69 ms a token against 47.27 skipped (`MEASURED` node03, measurements/2026-11-26_inkling2_long_profile_node03.md)
(if CARD_TIER
    (defun (experts layer_number)
      (if (< 0 (ai_thinkingmachines__inkling_2__experts hand picks (nth EXPERTS_TIER layer_number) routed_tier nn__expert_tier 1))
          (begin (sys__node_array__set HERE 0 layer_number)
                 (let ((expert_sum (sys__scheduler__add CPU names experts_picture)))
                   (ai_thinkingmachines__inkling_2__experts hand picks (nth EXPERTS_TIER layer_number) routed_tier nn__expert_tier)
                   (nn__buffer__copy (sys__wait_for_result expert_sum) routed HIDDEN_BYTES))
                 (nn__vector__add routed routed_tier routed HIDDEN))
          (ai_thinkingmachines__inkling_2__experts hand picks (nth EXPERTS_TIER layer_number) routed nn__expert_tier)))
    (defun (experts layer_number)
      (if EXPERTS_ON_CARD
          (ai_thinkingmachines__inkling_2__experts hand picks (nth EXPERTS_CARD layer_number) routed)
          (begin (sys__node_array__set HERE 0 layer_number)
                 (let ((expert_sum (sys__scheduler__add CPU names experts_picture)))
                   (nn__buffer__copy (sys__wait_for_result expert_sum) routed HIDDEN_BYTES))))))


;; ── a layer's application ───────────────────────────────────────────────────────────────────────────────────────────
;; `application`: which of the 48 it is — its cache and its convolutions' inputs; `with_messages` 1 in a loop layer's
;; second and third pass, where each of its inputs is first told what the pass before saw, 0 otherwise.
(defun (layer layer_number application with_messages pos ring_row)
  (begin (ai_thinkingmachines__inkling_2__before_experts (nth LAYER_TABLES layer_number) (nth APPLICATION_TABLES application)
                                                         WORK with_messages pos ring_row picks)
         (experts layer_number)
         (ai_thinkingmachines__inkling_2__after_experts (nth LAYER_TABLES layer_number) (nth APPLICATION_TABLES application)
                                                        WORK routed)))


;; ── the loop: layers 6 to 17 three times, each pass from the same start ─────────────────────────────────────────────
;; what a pass leaves the next: its third block state — `states`'s row 2, made at layer 12 — and its last state, normed
(defun (pass_end)
  (begin (nn__vector__copy (part states 2 HIDDEN) (part pass_states 0 HIDDEN) HIDDEN)
         (nn__rmsnorm__apply stream ones (part pass_states 1 HIDDEN) HIDDEN EPSILON)))

;; layers `from` up to `to`, the first's application `application`, no messages
(defun (run from to application pos ring_row)
  (let ((layer_number from))
    (while '(< layer_number to)
           '(begin (layer layer_number (+ application (- layer_number from)) 0 pos ring_row)
                   (set! layer_number (+ layer_number 1))))
    (type stream)))

;; a second or third pass: from the loop's start again, each layer told what the pass before saw
(defun (pass application pos ring_row)
  (begin (pass_end)
         (nn__vector__copy loop_start stream HIDDEN)
         (let ((layer_number LOOP_FROM))
           (while '(< layer_number LOOP_TO)
                  '(begin (layer layer_number (+ application (- layer_number LOOP_FROM)) 1 pos ring_row)
                          (set! layer_number (+ layer_number 1)))))
         (type stream)))

;; a position's place in a sliding layer's ring: `pos` less every whole turn of the window — the language divides
;; nothing, so the window times each power of two below the context is taken off where it fits, the largest first
(defun (ring pos)
  (let ((place pos) (turn 0))
    (while '(< turn RING_STEPS)
           '(begin (if (< place (nth RING_POWERS turn)) 0 (set! place (- place (nth RING_POWERS turn))))
                   (set! turn (+ turn 1))))
    place))


;; ── a position ──────────────────────────────────────────────────────────────────────────────────────────────────────
;; the token's row of the embedding, un-rotated (Rᵀu = S ⊙ H·u), normed. EMBEDDING is a table in RAM the card reads a
;; row of in place; with EMBED_ON_CPU the table is the CPU worker's (c_embedding) and the CPU decodes the row (▶
;; inkling2_cpu.lisp `embedding_picture`), the card copying it across: where the card's family keeps mapped RAM as a
;; copy of its own (OpenCL 1.2), the table neither crosses nor takes the card's room.
(if EMBED_ON_CPU
    (defun (embed tok)
      (begin (sys__node_array__set EMB_TOK 0 tok)
             (let ((row (sys__scheduler__add CPU names embedding_picture)))     ; answered as the CPU's c_emb
               (nn__buffer__copy (sys__wait_for_result row) embedding_rotated HIDDEN_BYTES))
             (nn__hadamard__rotate embedding_rotated ones512 embedding_unsigned HIDDEN)
             (nn__vector__pointwise_mul embedding_unsigned signs_row embedding HIDDEN)
             (nn__rmsnorm__apply embedding embedding_norm stream HIDDEN EPSILON)))
    (defun (embed tok)
      (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMBEDDING_HALVES) EMBEDDING_HALVES)
                                     (nn__vector__range EMBEDDING (+ EMBEDDING_SCALES tok) 1)
                                     embedding_rotated 1 HIDDEN EMBEDDING_BITS)
             (nn__hadamard__rotate embedding_rotated ones512 embedding_unsigned HIDDEN)
             (nn__vector__pointwise_mul embedding_unsigned signs_row embedding HIDDEN)
             (nn__rmsnorm__apply embedding embedding_norm stream HIDDEN EPSILON))))

(defun (position tok pos)
  (let ((ring_row (ring pos)))
    (embed tok)
    (run 0 LOOP_FROM 0 pos ring_row)
    (nn__vector__copy stream loop_start HIDDEN)
    (run LOOP_FROM LOOP_TO LOOP_FROM pos ring_row)
    (pass (+ LOOP_FROM (- LOOP_TO LOOP_FROM)) pos ring_row)
    (pass (+ LOOP_FROM (* 2 (- LOOP_TO LOOP_FROM))) pos ring_row)
    (run LOOP_TO LAYERS (+ LOOP_FROM (* 3 (- LOOP_TO LOOP_FROM))) pos ring_row)))


;; ── a prompt's positions as rows ────────────────────────────────────────────────────────────────────────────────────
;; A chunk of `row_count` positions from `from_position` — PROMPT's first `row_count` tokens — every application over all
;; of them at once: the package's verbs over the chunk's rows (▶ its layer.cuh, "A chunk"), its matrices one product
;; over the rows and its router one choice, its experts nn's rows routine; then the last row's stream and states where
;; the head reads them. The chunk's buffers hold a row a position — stream_rows, loop_start_rows, states_rows (a
;; position's STATES_ROWS), pass_states_rows (its two), hands, routed_rows — and WORK_ROWS is their table.
(defun (cpu_experts_rows layer_number row_count)
  (begin (sys__node_array__set HERE 0 layer_number)
         (sys__node_array__set HERE 1 row_count)
         ;; waited for without a bound: a chunk's experts read from a file can outlast `sys__wait_for_result`'s wait
         (let ((expert_sums (sys__scheduler__add CPU names experts_rows_picture)))
           (if CARD_TIER                           ; the card's share computed while the CPU computes its own
               (ai_thinkingmachines__inkling_2__experts_rows hands row_count (nth EXPERTS_TIER layer_number) routed_rows_tier
                                                             nn__expert_tier)
               0)
           (while '(eq? (sys__result expert_sums) (< 1 0)) '(< 0 1))
           (nn__buffer__copy (sys__wait_for_result expert_sums) routed_rows (* row_count HIDDEN_BYTES)))))
;; with a card's tier (CARD_TIER): the chunk's picks noted for it — the ones it holds marked held in the hands, which the
;; CPU then leaves, and the experts the chunk picks most promoted past the threshold the boot measured — then both shares
(defun (experts_rows layer_number row_count)
  (if EXPERTS_ON_CARD
      (ai_thinkingmachines__inkling_2__experts_rows hands row_count (nth EXPERTS_CARD layer_number) routed_rows)
      (if CARD_TIER
          (begin (nn__expert_tier__note hands nn__expert_tier layer_number row_count HAND_ROW_BYTES HAND_PICKS_AT
                                        nn__expert_major__vram_promotion_threshold_picks)
                 (cpu_experts_rows layer_number row_count)
                 (nn__vector__add routed_rows routed_rows_tier routed_rows (* row_count HIDDEN)))
          (cpu_experts_rows layer_number row_count))))

(defun (layer_rows layer_number application with_messages from_position row_count)
  (begin (ai_thinkingmachines__inkling_2__before_experts_rows (nth LAYER_TABLES layer_number) (nth APPLICATION_TABLES application)
                                                              WORK_ROWS with_messages from_position row_count)
         (experts_rows layer_number row_count)
         (ai_thinkingmachines__inkling_2__after_experts_rows (nth LAYER_TABLES layer_number) (nth APPLICATION_TABLES application)
                                                             WORK_ROWS routed_rows row_count)))

(defun (run_rows from to application from_position row_count)
  (let ((layer_number from))
    (while '(< layer_number to)
           '(begin (layer_rows layer_number (+ application (- layer_number from)) 0 from_position row_count)
                   (set! layer_number (+ layer_number 1))))
    (type stream_rows)))

;; what a pass leaves the next, each row's: its third block state and its last state, normed
(defun (pass_end_rows row_count)
  (let ((chunk_row 0))
    (while '(< chunk_row row_count)
           '(begin (nn__vector__copy (part states_rows (+ (* chunk_row STATES_ROWS) 2) HIDDEN) (part pass_states_rows (* chunk_row 2) HIDDEN)
                                     HIDDEN)
                   (nn__rmsnorm__apply (part stream_rows chunk_row HIDDEN) ones (part pass_states_rows (+ (* chunk_row 2) 1) HIDDEN)
                                       HIDDEN EPSILON)
                   (set! chunk_row (+ chunk_row 1))))
    (type pass_states_rows)))

(defun (pass_rows application from_position row_count)
  (begin (pass_end_rows row_count)
         (nn__vector__copy loop_start_rows stream_rows (* row_count HIDDEN))
         (let ((layer_number LOOP_FROM))
           (while '(< layer_number LOOP_TO)
                  '(begin (layer_rows layer_number (+ application (- layer_number LOOP_FROM)) 1 from_position row_count)
                          (set! layer_number (+ layer_number 1)))))
         (type stream_rows)))

;; the chunk's tokens, a row each, as `embed` makes a position's
(defun (embed_rows row_count)
  (let ((chunk_row 0))
    (while '(< chunk_row row_count)
           '(begin (embed (nth PROMPT chunk_row))
                   (nn__vector__copy stream (part stream_rows chunk_row HIDDEN) HIDDEN)
                   (set! chunk_row (+ chunk_row 1))))
    (type stream_rows)))

(defun (rows from_position row_count)
  (begin (embed_rows row_count)
         (run_rows 0 LOOP_FROM 0 from_position row_count)
         (nn__vector__copy stream_rows loop_start_rows (* row_count HIDDEN))
         (run_rows LOOP_FROM LOOP_TO LOOP_FROM from_position row_count)
         (pass_rows (+ LOOP_FROM (- LOOP_TO LOOP_FROM)) from_position row_count)
         (pass_rows (+ LOOP_FROM (* 2 (- LOOP_TO LOOP_FROM))) from_position row_count)
         (run_rows LOOP_TO LAYERS (+ LOOP_FROM (* 3 (- LOOP_TO LOOP_FROM))) from_position row_count)
         (nn__vector__copy (part stream_rows (- row_count 1) HIDDEN) stream HIDDEN)
         (nn__vector__copy (part states_rows (- row_count 1) STATES_HALVES) states STATES_HALVES)))


;; ── the head: the last mix over the four block states and the stream, the norm, the logits ──────────────────────────
;; a learned query's mix is an attention of one head as wide as the model: the query's scores over the four block states
;; and the stream — normed into the row after them — a softmax, the states weighed by it (nn's `decode`).
;; The logits soft-capped, LOGIT_CAP·tanh(x / LOGIT_CAP), as the model gives them: what a sampler draws from
;; (▶ sampling.lisp)
(defun (logits)
  (begin (nn__rmsnorm__apply stream ones (part states 4 HIDDEN) HIDDEN EPSILON)
         (nn__attention__decode final_query states states mix_scores final_mix 1 1 HIDDEN 0 5)
         (nn__rmsnorm__apply final_mix final_norm final_normed HIDDEN EPSILON)
         (nn__hadamard__rotate final_normed signs final_normed_rotated HIDDEN)
         (nn__turboquant__gemv HEAD_CODES HEAD_SCALES final_normed_rotated logits_h VOCAB HIDDEN HEAD_BITS)
         (nn__vector__scale logits_h LOGIT_CAP_INVERSE logits_h VOCAB)
         (nn__vector__tanh logits_h logits_h VOCAB)
         (nn__vector__scale logits_h LOGIT_CAP logits_h VOCAB)))

(defun (head)                                          ; the greedy token
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)                                   ; the logits left for a sampler on the host
  (begin (logits) (type stream)))
