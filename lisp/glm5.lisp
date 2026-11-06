;; GLM 5.3 FLASH — four residual streams, Kimi Delta Attention and latent attention, routed experts on the CPUs.
;;
;; A token is `(begin (position tok pos) (head))`: its row of the embedding into every stream, then every layer this
;; machine runs. A layer is ai_z__glm_5_3's sites over its tables: the attention site — a KDA layer's or an MLA layer's mixer,
;; the streams' hyper-connection around it — then the MLP site: dense, or the router, the shared expert on the card while
;; each CPU socket runs its share of every picked expert, and the shares summed into the streams. A prompt runs as rows:
;; each layer once over a chunk, `(begin (embed_rows n) (rows first n))`. An answer is `generate` (▶ generate.lisp).
;;
;; What the boot binds before this file is read (▶ glm5.py `_bindings`):
;;   the shape       HIDDEN HIDDEN_BYTES MULT MEAN (1/MULT) VOCAB EPS RUN · HEAD_CODES HEAD_SCALES HEAD_BITS
;;                   EMBEDDING EMB_HALVES EMB_SCALES EMB_BITS — the embedding table in RAM, a row's halves, where the
;;                   scales start, its bits
;;                   HAND HAND_PICKS_AT HAND_BYTES — a hand row, its picks' place, a position's hand to a socket
;;                   ROWS_HANDS_BYTES ROWS_BYTES ROWS_HALVES — a chunk's hands, and its routed rows, to and from a socket
;;   the settings    CARD_TIER — true where the card holds a tier of the routed experts; EXPERTS_ON_CARD — true
;;                   where it holds every one of them, and no socket runs any; AHEAD — how many of the next layer's
;;                   experts each socket reads ahead from the disk, 0 as many as the disk reads while a layer computes
;;   a layer each    arrays indexed by the layer's place in RUN: KDA (true for a KDA layer), MOE (true where it routes),
;;                   LAYER (its number), AT (the attention site's table), MO / ML (the MLP site's, routed or dense),
;;                   EXK EXKR (the card's tier's experts tables, a position's and a chunk's), AHEAD_NEXT (the next MoE
;;                   layer's place in RUN where its experts are read ahead, 0 where they are not)
;;   a socket each   SOCKETS of them: SOCKET (its worker), ROUTED_INTO / RROWS_INTO (where its share lands on the card);
;;                   ▶ glm5_cpu.lisp for what each runs
;;   the buffers     streams xin h hr e u hu hand picks guess routed hm hn hnr srows hands carries rrows rwork … (▶ `_card_buffers`),
;;                   PROMPT (a chunk's tokens)


;; ── the token: its row of the embedding, un-rotated (Rᵀu = S ⊙ H·u), into every stream ──────────────────────────────
;; EMBEDDING is a table in RAM the card reads a row of in place: every token's packed row (EMB_HALVES halves each), then
;; every token's scale from EMB_SCALES on — the embedding stays in RAM and only a row crosses.
;; With EMBED_ON_CPU the table is the first socket's (c_embedding), which decodes the row (▶ glm5_cpu.lisp `xe`) for the
;; card to copy across and un-rotate: where the card's family keeps mapped RAM as a copy of its own, the table neither
;; crosses nor takes the card's room.
(if EMBED_ON_CPU
    (defun (embed_e tok)                               ; the token's row into e, decoded by the first socket
      (begin (sys__node_array__set EMB_TOK 0 tok)
             (sys__scheduler__add (nth SOCKET 0) names xe) (sys__wait (nth SOCKET 0))
             (nn__buffer__copy c_emb (nth SOCKET 0) u HIDDEN_BYTES)
             (nn__hadamard__rotate u ones512 hu HIDDEN)
             (nn__vector__pointwise_mul hu signs_row e HIDDEN)))
    (defun (embed_e tok)                               ; the token's row into e
      (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMB_HALVES) EMB_HALVES)
                                     (nn__vector__range EMBEDDING (+ EMB_SCALES tok) 1) u 1 HIDDEN EMB_BITS)
             (nn__hadamard__rotate u ones512 hu HIDDEN)
             (nn__vector__pointwise_mul hu signs_row e HIDDEN))))

(defun (embed tok)
  (let ((j 0))
    (embed_e tok)
    (while '(< j MULT) '(begin (nn__vector__add e zero_h (nn__vector__range streams (* j HIDDEN) HIDDEN) HIDDEN) (set! j (+ j 1))))
    (type e)))

;; a prompt's row `row`, its token `tok`, into its streams in srows
(defun (embed_row tok row)
  (let ((j 0))
    (embed_e tok)
    (while '(< j MULT)
           '(begin (nn__vector__add e zero_h (nn__vector__range srows (* (+ (* row MULT) j) HIDDEN) HIDDEN) HIDDEN)
                   (set! j (+ j 1))))
    (type e)))

;; the chunk's `n` rows, their tokens PROMPT's
(defun (embed_rows n)
  (let ((i 0))
    (while '(< i n) '(begin (embed_row (nth PROMPT i) i) (set! i (+ i 1))))
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
    (while '(< s SOCKETS) '(begin (sys__scheduler__add (nth SOCKET s) names picture) (set! s (+ s 1))))
    (type hand)))

(defun (sockets_wait)
  (let ((s 0))
    (while '(< s SOCKETS) '(begin (sys__wait (nth SOCKET s)) (set! s (+ s 1))))))

(defun (sockets_sum)
  (let ((s 0))
    (while '(< s SOCKETS)
           '(begin (nn__buffer__copy (nth C_ROUTED s) (nth SOCKET s) (nth ROUTED_INTO s) HIDDEN_BYTES)
                   (if (< 0 s) (nn__vector__add routed (nth ROUTED_INTO s) routed HIDDEN) 0)
                   (set! s (+ s 1))))
    (type routed)))

;; ⛔ A TOP-LEVEL `if` ANSWERS TRUE ON EITHER ARM: the boot defines it as it defines a `defun` (▶ machine.py `defun`), and
;;   refuses any other answer — an arm of `0` stopped GLM's boot wherever SOCKETS is not 2, a one-socket machine or the
;;   experts on the card (the host engine on node03, 2026-10-30: the form answered an integer, kind 8)
;; ⛳ TWO SOCKETS WRITTEN OUT — a two-socket machine's layer starts, waits for and sums its sockets without the loops'
;;   own steps (a counter, its test, its increment, each an apply on the card's path), in the loops' order. MEASURED
;;   2026-10-30, debian01 (the DL380), GLM on one card, no tier: the card's applies 5,072 -> 2,930 a token, its
;;   evaluator's own 7.65 -> 5.22 ms, 12.02 and 11.85 tokens/s against 12.30 and 12.24, the same tokens
;;   (measurements/2026-10-30_nn146_threads_floaters_dl380.md)
(if (eq? SOCKETS 2)
    (begin
      (defun (sockets_start i picture)
        (begin (sys__node_array__set HERE 0 i)
               (sys__scheduler__add (nth SOCKET 0) names picture)
               (sys__scheduler__add (nth SOCKET 1) names picture)
               (type hand)))
      (defun (sockets_wait)
        (begin (sys__wait (nth SOCKET 0)) (sys__wait (nth SOCKET 1))))
      (defun (sockets_sum)
        (begin (nn__buffer__copy (nth C_ROUTED 0) (nth SOCKET 0) (nth ROUTED_INTO 0) HIDDEN_BYTES)
               (nn__buffer__copy (nth C_ROUTED 1) (nth SOCKET 1) (nth ROUTED_INTO 1) HIDDEN_BYTES)
               (nn__vector__add routed (nth ROUTED_INTO 1) routed HIDDEN)
               (type routed))))
    (< 0 1))

;; ⛳ A CHUNK'S SHARES ARE WAITED FOR WITHOUT A BOUND: `sys__wait` gives up after ~15 s, and a chunk's experts read from
;;   the disk can take longer — so the card asks `sys__result` until a socket says so, then collects
(defun (sockets_sum_rows)
  (let ((s 0))
    (while '(< s SOCKETS)
           '(begin (while '(eq? (sys__result (nth SOCKET s)) (< 1 0)) '(< 0 1))
                   (sys__wait (nth SOCKET s))
                   (set! s (+ s 1))))
    (if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0)
    (set! s 0)
    ;; ⛳ THE CHUNK'S ROWS BACK, NOT A WHOLE CHUNK'S: a draft's check is two rows, and a chunk is 256 — MEASURED 2026-10-19,
    ;;   node-01, the two rows' hand-off was 5.5 ms a layer copying 256 rows' hands out and sums back (▶ src_glm5_rows2_profile)
    (while '(< s SOCKETS)
           '(begin (nn__buffer__copy (nth C_RROWS s) (nth SOCKET s) (nth RROWS_INTO s) (* (nth nrows 0) HIDDEN_BYTES))
                   (if (< 0 s) (nn__vector__add rrows (nth RROWS_INTO s) rrows (* (nth nrows 0) HIDDEN)) 0)
                   (set! s (+ s 1))))
    (type rrows)))


;; ── a layer at a position ───────────────────────────────────────────────────────────────────────────────────────────
(defun (mixer i pos)
  (if (nth KDA i)
      (ai_z__glm_5_3__kda streams (nth AT i))
      (ai_z__glm_5_3__mla streams (nth AT i) pos)))

;; the routed MLP: the router's picks into the hand, a card tier's share computed and marked, every socket handed the
;; rest while the card runs the shared expert, the shares summed — and the card tier's — and the streams closed.
;; ⛳ THE TIER'S EXPERTS BEFORE THE SOCKETS START, NOT BESIDE THEM (NN-99 ④, measured): the visit-first order Qwen takes
;;   (▶ qwen3_5.lisp) drains the card's queue a layer to start the sockets early — GLM's card experts are short, and the
;;   drain costs more than they hide: a token 118-123 against 109-116 ms on one card (18 GB), 106.5 against 100.7 on two
;;   (18 GB a card), 85.6 against 80.9 on four (24 GB a card); four cards' runs vary ±5 ms (▶ glm5.py's group `moe`, the same)
;; the tier's experts and the sockets started, in the order VISIT_FIRST names (glm5.py's `card_visit_first`, off by default)
(defun (tier_and_sockets i)
  (if CARD_TIER
      (if VISIT_FIRST
          (begin (ai_z__glm_5_3__experts hand picks (nth EXK i) routed_k nn__expert_tier 1)
                 (sockets_start i xf_ready)
                 (ai_z__glm_5_3__experts hand picks (nth EXK i) routed_k nn__expert_tier))
          (begin (ai_z__glm_5_3__experts hand picks (nth EXK i) routed_k nn__expert_tier) (sockets_start i xf)))
      (sockets_start i xf)))
;; the router — and, where the experts are read ahead (AHEAD_NEXT, NN-138), the next MoE layer's router on the same input,
;; its likeliest picks into `guess` for the sockets to read the first of while this layer computes (▶ glm5_cpu.lisp `xf`)
(defun (route i)
  (if (< 0 (nth AHEAD_NEXT i))
      (ai_z__glm_5_3__route streams (nth MO i) hand picks (nth MO (nth AHEAD_NEXT i)) guess)
      (ai_z__glm_5_3__route streams (nth MO i) hand picks)))
;; with EXPERTS_ON_CARD every routed expert of these layers is in a slot on the card: the card computes every pick, and
;; no socket is started or waited for
(if EXPERTS_ON_CARD
    (defun (moe i)
      (begin (ai_z__glm_5_3__route streams (nth MO i) hand picks)
             (ai_z__glm_5_3__experts hand picks (nth EXK i) routed)
             (ai_z__glm_5_3__shared hand (nth MO i))
             (ai_z__glm_5_3__close streams (nth MO i) routed)))
    (defun (moe i)
      (begin
        (route i)
        (tier_and_sockets i)
        (ai_z__glm_5_3__shared hand (nth MO i))
        (sockets_wait)
        (if CARD_TIER (nn__expert_tier__written nn__expert_tier) 0)     ; the CPUs done: an exclusive tier's writes go now
        (sockets_sum)
        (if CARD_TIER (nn__vector__add routed routed_k routed HIDDEN) 0)
        (ai_z__glm_5_3__close streams (nth MO i) routed))))

(defun (layer i pos)
  (begin (mixer i pos)
         (if (nth MOE i) (moe i) (ai_z__glm_5_3__mlp streams (nth ML i)))
         (type streams)))

(defun (layers_f pos)
  (let ((i 0))
    (while '(< i RUN) '(begin (layer i pos) (set! i (+ i 1))))
    (type streams)))

;; a token's position: its row of the embedding into the streams, then the layers
(defun (position tok pos)
  (begin (embed tok) (layers_f pos)))


;; ── two seats: two conversations on one card, the card on one while the sockets compute the other's experts ─────────
;; A seat is a conversation's cartridge on the card: its streams, its hand and picks, its attention tables over its own
;; state and its MLP tables over its own scratch (the MoE site's carry lives there from the router to the close). The
;; weights are the same for both. A section runs on seat `k` inside a `let` binding that seat's names — STREAMS_OF HAND_OF
;; PICKS_OF AT_OF MO_OF, each seat's by seat — so the sites above run unchanged, and the names are unbound when it ends.
;; The sockets read the names as they stood when the programs were defined: they are told the seat in HERE's second cell
;; (▶ glm5_cpu.lisp `xf2`), each seat its own hand and routed share on each of them. A seat stepped a position at a time
;; — `position` inside its `let`, which binds SEAT as well — hands its sockets the same way.
;; The pair's step: seat 1 half a layer behind seat 0. A MoE layer runs seat 0's attention and router while the sockets
;; finish seat 1's previous experts, then seat 0's experts while the card runs seat 1's attention and router, then seat
;; 1's while the card runs seat 0's next layer — the card and the sockets each on one seat while the other works. A seat's
;; experts collected, the other's start before the first's are summed and closed: each seat's share lands in its own
;; buffers on the sockets, so the sockets wait only for the hand-off.
(if (< 1 SEATS)
    (begin
      ;; a seat's attention and router: its picks read back
      (defun (seat_front k i pos)
        (let ((streams (nth STREAMS_OF k)) (hand (nth HAND_OF k)) (picks (nth PICKS_OF k)) (AT (nth AT_OF k)) (MO (nth MO_OF k)))
          (mixer i pos)
          (route i)
          (type streams)))
      ;; the sockets handed a seat's picks
      (defun (seat_start k i)
        (let ((s 0))
          (sys__node_array__set HERE 0 i)
          (sys__node_array__set HERE 1 k)
          (while '(< s SOCKETS) '(begin (sys__scheduler__add (nth SOCKET s) names xf2) (set! s (+ s 1))))
          (type hand)))
      ;; the card's shared expert on a seat's hand
      (defun (seat_shared k i)
        (let ((hand (nth HAND_OF k)) (MO (nth MO_OF k)))
          (ai_z__glm_5_3__shared hand (nth MO i))))
      ;; a seat's shares copied off the sockets and summed into routed
      (defun (seat_sum k)
        (let ((s 0))
          (while '(< s SOCKETS)
                 '(begin (nn__buffer__copy (nth (nth C_ROUTED_OF k) s) (nth SOCKET s) (nth ROUTED_INTO s) HIDDEN_BYTES)
                         (if (< 0 s) (nn__vector__add routed (nth ROUTED_INTO s) routed HIDDEN) 0)
                         (set! s (+ s 1))))
          (type routed)))
      ;; a seat's experts, collected, summed and closed into its streams
      (defun (seat_finish k i)
        (let ((streams (nth STREAMS_OF k)) (MO (nth MO_OF k)))
          (seat_sum k)
          (ai_z__glm_5_3__close streams (nth MO i) routed)
          (type streams)))
      (defun (seat_back k i)
        (begin (sockets_wait) (seat_finish k i)))
      ;; a position at a time on seat SEAT: `moe`'s sockets told the seat and summed from its buffers (no tier with seats)
      (defun (tier_and_sockets i)
        (let ((s 0))
          (sys__node_array__set HERE 0 i)
          (sys__node_array__set HERE 1 SEAT)
          (while '(< s SOCKETS) '(begin (sys__scheduler__add (nth SOCKET s) names xf2) (set! s (+ s 1))))
          (type hand)))
      (defun (sockets_sum)
        (seat_sum SEAT))
      ;; a dense layer on a seat
      (defun (seat_dense k i pos)
        (let ((streams (nth STREAMS_OF k)) (AT (nth AT_OF k)))
          (mixer i pos)
          (ai_z__glm_5_3__mlp streams (nth ML i))
          (type streams)))
      ;; every layer for two seats, seat `ka` at `pa` and seat `kb` at `pb`, `kb` the one behind; `out` while `kb`'s
      ;; experts of layer `behind` are on the sockets
      (defun (pair_layers ka pa kb pb)
        (let ((i 0) (behind 0) (out (< 1 0)))
          (while '(< i RUN)
                 '(begin
                    (if (nth MOE i)
                        (begin (seat_front ka i pa)
                               (if out
                                   (begin (sockets_wait) (seat_start ka i) (seat_finish kb behind))
                                   (seat_start ka i))
                               (seat_shared ka i)
                               (seat_front kb i pb)
                               (sockets_wait)
                               (seat_start kb i)
                               (seat_finish ka i)
                               (seat_shared kb i)
                               (set! behind i)
                               (set! out (< 0 1)))
                        (begin (if out (begin (seat_back kb behind) (set! out (< 1 0))) 0)
                               (seat_dense ka i pa)
                               (seat_dense kb i pb)))
                    (set! i (+ i 1))))
          (if out (seat_back kb behind) 0)
          (type streams)))
      ;; two seats' tokens in and every layer; each seat's head is the runtime's, inside the seat's `let`
      ;; (▶ glm5.py `pair_step`)
      (defun (seat_embed k tok)
        (let ((streams (nth STREAMS_OF k))) (embed tok)))
      (defun (pair_position ka ta pa kb tb pb)
        (begin (seat_embed ka ta) (seat_embed kb tb) (pair_layers ka pa kb pb))))
    (< 0 1))


;; ── a layer over a prompt's rows ────────────────────────────────────────────────────────────────────────────────────
(if EXPERTS_ON_CARD
    (defun (moe_rows i n)
      (begin (ai_z__glm_5_3__route_rows srows (nth MO i) hands carries n rwork)
             (ai_z__glm_5_3__experts_rows hands (nth EXKR i) rrows n)
             (ai_z__glm_5_3__close_rows srows (nth MO i) rrows carries n rwork)))
    (defun (moe_rows i n)
      (begin (ai_z__glm_5_3__route_rows srows (nth MO i) hands carries n rwork)
             ;; the card's tier told of the chunk's picks, so the prompt warms it
             (if CARD_TIER (nn__expert_tier__note hands nn__expert_tier (nth LAYER i) n HAND HAND_PICKS_AT
                                                  nn__expert_major__vram_promotion_threshold_picks) 0)
             (sockets_start i xr)
             ;; the card's share computed while the CPUs compute theirs
             (if CARD_TIER (ai_z__glm_5_3__experts_rows hands (nth EXKR i) rrows_k n nn__expert_tier) 0)
             (sockets_sum_rows)
             (if CARD_TIER (nn__vector__add rrows rrows_k rrows (* n HIDDEN)) 0)
             (ai_z__glm_5_3__close_rows srows (nth MO i) rrows carries n rwork))))

(defun (rows_layer i first n)
  (begin (if (nth KDA i)
             (ai_z__glm_5_3__kda_rows srows (nth AT i) n rwork)
             (ai_z__glm_5_3__mla_rows srows (nth AT i) first n rwork))
         (if (nth MOE i) (moe_rows i n) (ai_z__glm_5_3__mlp_rows srows (nth ML i) n rwork))
         (type srows)))

(defun (rows first n)
  (let ((i 0))
    (sys__node_array__set nrows 0 n)
    (while '(< i RUN) '(begin (rows_layer i first n) (set! i (+ i 1))))
    (type srows)))


;; ── a prompt read layer-major: a slice of chunks, every layer once over all of it (NN-115 ③) ─────────────────────────
;; With STREAM (▶ glm5.py `_prefill_streamed`) a slice of chunks is held on the card — SROWS_S its rows' streams, HANDS_S
;; and CARRIES_S its MoE sites' hands and carries, RROWS_S its routed rows — and each layer runs once over it: every chunk's
;; attention and router in order, then the layer's experts once over all the slice's rows, streamed onto the card through
;; the tier's ring (a note with NOTE_STREAM: every pick the card's, the sockets computing none, the ones picked the line's
;; times or more into the tier first), then every chunk's close. `w` rows a chunk, the slice's last chunk `last_n`.
(if STREAM
    (begin
      ;; a chunk's `n` tokens, PROMPT's first `n`, into the slice's streams from its row `row`
      (defun (embed_slice_row tok row)
        (let ((j 0))
          (embed_e tok)
          (while '(< j MULT)
                 '(begin (nn__vector__add e zero_h (nn__vector__range SROWS_S (* (+ (* row MULT) j) HIDDEN) HIDDEN) HIDDEN)
                         (set! j (+ j 1))))
          (type e)))
      (defun (embed_slice row n)
        (let ((i 0))
          (while '(< i n) '(begin (embed_slice_row (nth PROMPT i) (+ row i)) (set! i (+ i 1))))
          (type SROWS_S)))
      ;; chunk `c`'s `n` rows of a part of the slice, `halves` a row
      (defun (part_rows v c n halves w)
        (nn__vector__range v (* c (* w halves)) (* n halves)))
      (defun (slice_layer i first k last_n w)
        (let ((c 0) (n w) (all (+ (* (- k 1) w) last_n)))
          (while '(< c k)
                 '(begin (set! n (if (eq? c (- k 1)) last_n w))
                         (if (nth KDA i)
                             (ai_z__glm_5_3__kda_rows (part_rows SROWS_S c n SROW_HALVES w) (nth AT i) n rwork)
                             (ai_z__glm_5_3__mla_rows (part_rows SROWS_S c n SROW_HALVES w) (nth AT i) (+ first (* c w)) n rwork))
                         (if (nth MOE i)
                             (ai_z__glm_5_3__route_rows (part_rows SROWS_S c n SROW_HALVES w) (nth MO i)
                                                        (part_rows HANDS_S c n HAND_HALVES w)
                                                        (part_rows CARRIES_S c n CARRY_HALVES w) n rwork)
                             (ai_z__glm_5_3__mlp_rows (part_rows SROWS_S c n SROW_HALVES w) (nth ML i) n rwork))
                         (set! c (+ c 1))))
          (if (nth MOE i)
              (begin
                (nn__expert_tier__note HANDS_S nn__expert_tier (nth LAYER i) all HAND HAND_PICKS_AT
                                       (+ nn__expert_major__vram_promotion_threshold_picks NOTE_STREAM))
                (ai_z__glm_5_3__experts_rows HANDS_S (nth EXKS i) RROWS_S all nn__expert_tier)
                (set! c 0)
                (while '(< c k)
                       '(begin (set! n (if (eq? c (- k 1)) last_n w))
                               (ai_z__glm_5_3__close_rows (part_rows SROWS_S c n SROW_HALVES w) (nth MO i)
                                                          (part_rows RROWS_S c n HIDDEN w)
                                                          (part_rows CARRIES_S c n CARRY_HALVES w) n rwork)
                               (set! c (+ c 1))))
                (type SROWS_S))
              (type SROWS_S))
          (type SROWS_S)))
      (defun (rows_slice first k last_n w)
        (let ((i 0))
          (sys__node_array__set nrows 0 w)
          (while '(< i RUN) '(begin (slice_layer i first k last_n w) (set! i (+ i 1))))
          (type SROWS_S))))
    (defun (rows_slice first k last_n w)               ; no slice: the prompt is read a chunk at a time
      (type srows)))
