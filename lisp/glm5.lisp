;; GLM 5.3 FLASH — four residual streams, Kimi Delta Attention and latent attention, routed experts on the CPUs.
;;
;; A token is `(begin (position_f codes scale pos) (head))`: its embedding into every stream, then every layer this machine
;; runs. A layer is ai_glm_5_3's sites over its tables: the attention site — a KDA layer's or an MLA layer's mixer, the
;; streams' hyper-connection around it — then the MLP site: dense, or the router, the shared expert on the card while each
;; CPU socket runs its share of every picked expert, and the shares summed into the streams. A prompt runs as rows: each
;; layer once over a chunk, `(begin (embed_rows n) (rows first n))`.
;;
;; What the boot binds before this file is read (▶ glm5.py `_bindings`):
;;   the shape       HIDDEN HIDDEN_BYTES MULT MEAN (1/MULT) VOCAB EPS RUN · EMB_HALVES EMB_BITS · HEAD_CODES HEAD_SCALES HEAD_BITS
;;                   HAND HAND_PICKS_AT HAND_BYTES — a hand row, its picks' place, a position's hand to a socket
;;                   ROWS_HANDS_BYTES ROWS_BYTES ROWS_HALVES — a chunk's hands, and its routed rows, to and from a socket
;;   the settings    CARD_TIER — true where the card holds a tier of the routed experts
;;   a layer each    arrays indexed by the layer's place in RUN: KDA (true for a KDA layer), MOE (true where it routes),
;;                   LAYER (its number), AT (the attention site's table), MO / ML (the MLP site's, routed or dense),
;;                   EXK EXKR (the card's tier's experts tables, a position's and a chunk's)
;;   a socket each   SOCKETS of them: SOCKET (its worker), ROUTED_INTO / RROWS_INTO (where its share lands on the card);
;;                   ▶ glm5_cpu.lisp for what each runs
;;   the buffers     streams xin h hr e u hu hand picks routed hm hn hnr srows hands carries rrows rwork … (▶ `_card_buffers`)


;; ── the token: its row, un-rotated (Rᵀu = S ⊙ H·u), into every stream ─────────────────────────────────────────────
(defun (embed codes scale)
  (let ((j 0))
    (nn__turboquant__decode (nn__vector__range emb_codes codes EMB_HALVES) (nn__vector__range emb_scales scale 1)
                            u 1 HIDDEN EMB_BITS)
    (nn__hadamard__rotate u ones512 hu HIDDEN)
    (nn__vector__pointwise_mul hu signs_row e HIDDEN)
    (while '(< j MULT) '(begin (nn__vector__add e zero_h (nn__vector__range streams (* j HIDDEN) HIDDEN) HIDDEN) (set! j (+ j 1))))
    (type e)))

;; a prompt's row `row`, its staged codes `codes` halves into emb_rows and its scale the `scale`-th, into its streams in srows
(defun (embed_row codes scale row)
  (let ((j 0))
    (nn__turboquant__decode (nn__vector__range emb_rows codes EMB_HALVES) (nn__vector__range emb_rscales scale 1)
                            u 1 HIDDEN EMB_BITS)
    (nn__hadamard__rotate u ones512 hu HIDDEN)
    (nn__vector__pointwise_mul hu signs_row e HIDDEN)
    (while '(< j MULT)
           '(begin (nn__vector__add e zero_h (nn__vector__range srows (* (+ (* row MULT) j) HIDDEN) HIDDEN) HIDDEN)
                   (set! j (+ j 1))))
    (type e)))

;; the chunk's `n` rows, staged one after another
(defun (embed_rows n)
  (let ((i 0) (codes 0))
    (while '(< i n) '(begin (embed_row codes i i) (set! codes (+ codes EMB_HALVES)) (set! i (+ i 1))))
    (type srows)))


;; ── the head: the streams' mean, the final norm, the logits ─────────────────────────────────────────────────────────
(defun (logits)
  (let ((j 2))
    (nn__vector__add (nn__vector__range streams 0 HIDDEN) (nn__vector__range streams HIDDEN HIDDEN) hm HIDDEN)
    (while '(< j MULT) '(begin (nn__vector__add hm (nn__vector__range streams (* j HIDDEN) HIDDEN) hm HIDDEN) (set! j (+ j 1))))
    (nn__vector__scale hm MEAN hm HIDDEN)
    (nn__rmsnorm__apply hm fnw hn HIDDEN EPS)
    (nn__hadamard__rotate hn signs hnr HIDDEN)
    (nn__turboquant__gemv HEAD_CODES HEAD_SCALES hnr logits_h VOCAB HIDDEN HEAD_BITS)))

(defun (head)                                          ; the greedy token
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)                                   ; the logits left for a sampler on the host
  (begin (logits) (type hm)))


;; ── the CPU sockets: each handed layer `i`'s share of its picks, and their shares summed back ─────────────────────────
;; HERE names the layer for the picture each runs (▶ glm5_cpu.lisp). A socket's share lands in ROUTED_INTO's — the first
;; socket's in `routed` — and the others are added onto it.
(defun (sockets_start i picture)
  (let ((s 0))
    (sys__node_array__set HERE 0 i)
    (while '(< s SOCKETS) '(begin (sys__compute (nth SOCKET s) names picture) (set! s (+ s 1))))
    (type hand)))

(defun (sockets_sum)
  (let ((s 0))
    (while '(< s SOCKETS)
           '(begin (nn__buffer__copy (nth C_ROUTED s) (nth SOCKET s) (nth ROUTED_INTO s) HIDDEN_BYTES)
                   (if (< 0 s) (nn__vector__add routed (nth ROUTED_INTO s) routed HIDDEN) 0)
                   (set! s (+ s 1))))
    (type routed)))

;; ⛳ A CHUNK'S SHARES ARE WAITED FOR WITHOUT A BOUND: `sys__result` gives up after ~15 s, and a chunk's experts read from
;;   the disk can take longer — so the card asks `sys__completed` until a socket says so, then collects
(defun (sockets_sum_rows)
  (let ((s 0))
    (while '(< s SOCKETS)
           '(begin (while '(eq? (sys__completed (nth SOCKET s)) (< 1 0)) '(< 0 1))
                   (sys__result (nth SOCKET s))
                   (set! s (+ s 1))))
    (if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0)
    (set! s 0)
    (while '(< s SOCKETS)
           '(begin (nn__buffer__copy (nth C_RROWS s) (nth SOCKET s) (nth RROWS_INTO s) ROWS_BYTES)
                   (if (< 0 s) (nn__vector__add rrows (nth RROWS_INTO s) rrows ROWS_HALVES) 0)
                   (set! s (+ s 1))))
    (type rrows)))


;; ── a layer at a position ───────────────────────────────────────────────────────────────────────────────────────────
(defun (mixer i pos)
  (if (nth KDA i)
      (ai_glm_5_3__kda streams (nth AT i))
      (ai_glm_5_3__mla streams (nth AT i) pos)))

;; the routed MLP: the router's picks into the hand, a card tier's share computed and marked, every socket handed the
;; rest while the card runs the shared expert, the shares summed — and the card tier's — and the streams closed
(defun (moe i)
  (let ((s 0))
    (ai_glm_5_3__route streams (nth MO i) hand picks)
    (if CARD_TIER (ai_glm_5_3__experts hand picks (nth EXK i) routed_k nn__expert_tier) 0)
    (sockets_start i xf)
    (ai_glm_5_3__shared hand (nth MO i))
    (while '(< s SOCKETS) '(begin (sys__result (nth SOCKET s)) (set! s (+ s 1))))
    (if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0)     ; the CPUs done: an exclusive tier's writes go now
    (sockets_sum)
    (if CARD_TIER (nn__vector__add routed routed_k routed HIDDEN) 0)
    (ai_glm_5_3__close streams (nth MO i) routed)))

(defun (layer i pos)
  (begin (mixer i pos)
         (if (nth MOE i) (moe i) (ai_glm_5_3__mlp streams (nth ML i)))
         (type streams)))

(defun (layers_f pos)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i pos) (set! i (+ i 1))))
    (type streams)))

(defun (position_f codes scale pos)
  (begin (embed codes scale) (layers_f pos)))

(defun (token_f codes scale pos)
  (begin (position_f codes scale pos) (head)))


;; ── a layer over a prompt's rows ────────────────────────────────────────────────────────────────────────────────────
(defun (moe_rows i n)
  (begin (ai_glm_5_3__route_rows srows (nth MO i) hands carries n rwork)
         ;; the card's tier told of the chunk's picks, so the prompt warms it
         (if CARD_TIER (nn__expert_tier__note hands nn__expert_tier (nth LAYER i) n HAND HAND_PICKS_AT
                                              nn__expert_major__vram_promotion_threshold_picks) 0)
         (sockets_start i xr)
         ;; the card's share computed while the CPUs compute theirs
         (if CARD_TIER (ai_glm_5_3__experts_rows hands (nth EXKR i) rrows_k n nn__expert_tier) 0)
         (sockets_sum_rows)
         (if CARD_TIER (nn__vector__add rrows rrows_k rrows ROWS_HALVES) 0)
         (ai_glm_5_3__close_rows srows (nth MO i) rrows carries n rwork)))

(defun (rows_layer i first n)
  (begin (if (nth KDA i)
             (ai_glm_5_3__kda_rows srows (nth AT i) n rwork)
             (ai_glm_5_3__mla_rows srows (nth AT i) first n rwork))
         (if (nth MOE i) (moe_rows i n) (ai_glm_5_3__mlp_rows srows (nth ML i) n rwork))
         (type srows)))

(defun (rows first n)
  (let ((i 0))
    (sys__node_array__set nrows 0 n)
    (while '(< i RUN) '(begin (rows_layer i first n) (set! i (+ i 1))))
    (type srows)))
