;; QWEN3.8-FLASH-NEXT — `qwen4_exp`: Gated DeltaNet and Qwen Sparse Attention, four residual streams through a gated
;; residual at every site, 512 experts of 640 and a shared one, an n-gram embedding at its second layer.
;;
;; ⛳ THE FIRST PROGRAM, IN nn's WORDS ALONE (▶ src/docs/design/pending/qwen4_exp.md ⑤): no verb of a model package yet —
;; each site is a sequence of nn's words over its layer's table, checked against test/src_qwen4_exp_oracle.py. A token is
;; `(begin (position tok pos) (head))`, a position at a time; the routed experts are the CPU's (▶ qwen4_exp_cpu.lisp),
;; the shared expert the card's while they compute. The four streams are fp32 (`sf`), as DeepSeek's are.
;;
;; What the boot binds before this file is read (▶ qwen4_exp.py `_bindings`):
;;   the shape       HIDDEN STREAMS SW (the four streams' width) RANK (the gated residual's, padded to the rotation's
;;                   ladder) KH VH LD KEY VAL CONV_CH QH KVH AD QW KVW ROT THETA
;;                   NE TOPK INTER VOCAB EPS RUN CONV_K DIL — HIDDEN_BYTES HAND_BYTES — CARD CPU
;;                   QUARTER TWO NEG — the program's float constants · GATE_SIGMOID the DeltaNet's output gate ·
;;                   GUESS AHEAD — the next layer's picks guessed, and how many of them the CPU reads ahead (0 as many as
;;                   the disk starts)
;;                   EMBEDDING EMB_HALVES EMB_SCALES EMB_BITS — the embedding, a table in RAM a row of which the card reads
;;                   HEAD_CODES HEAD_SCALES HEAD_BITS · HEADT the last mixer's table
;;   a layer each    arrays indexed by the layer: LT (its table: a matrix three cells — codes, scales, bits — at C_*, a norm
;;                   or a vector one), ATTN (true for an attention layer), NGRAM_AT (true for the n-gram layer), AHEAD_NEXT
;;                   (the next layer, where its experts are read ahead; 0 where none), S32S CWINS
;;                   (a DeltaNet layer's state and conv window), KCS VCS (an attention layer's keys and values), GVNS (the
;;                   n-gram layer's normed values, a position each)
;;   the buffers     sf (the streams, fp32) mix mix1 mixd sf2 gvf ngr (a position's n-gram rows, ▶ `position`) prev2 (the
;;                   two tokens before it) NGT (the n-gram table, ▶ nn's `nn__ngram__rows`) hand
;;                   picks HERE routed … — and a prompt chunk's, a row a position: sfR ngrR csR lgR hands m2R mr2R inj2R
;;                   sdowR routedR, its tokens `toks` (▶ qwen4_exp.py `_buffers`); HAND_ROW HAND_HALVES HAND_PICKS_AT
;;                   HAND_W_AT the hand row's layout (ai_qwen_3's AI_QWEN_3__HAND__ROW)
;;   the verbs       VERBS (true: the sites ai_qwen_4's verbs, false: nn's words) · Q4 the shape table they read · q4s
;;                   their scratch
;;   the sockets     SOCKETS (1, or 2: the experts' parts over two CPU workers, CPU and CPU + 1) · ACT_BYTES a socket's
;;                   activations · with two: routed1 (the second half of the routed sum), each socket's c_routed0
;;                   c_routed1, and the CPU's arrays EXS C_HAND C_ACT C_OTHER C_ROUTED (▶ qwen4_exp_cpu.lisp `xu` `xd`)
;;   the seats       CARDS, SEATS (a card's) · CARD this card's worker · each seat's names its own view's, the
;;                   program's names bound to that seat's buffers and state, `names` the view its CPU experts run in,
;;                   HERE what its steps carry (▶ the chain, qwen4_exp.py `load`); every seat's `close` hands the CPU its
;;                   experts with `sys__scheduler__add`, which queues them, so seats on their own chains share the CPU in
;;                   the order their requests arrive (▶ sys's scheduler__header.cuh)


;; ── a record times a vector: the rotated vector where the record is quantised, the plain one where it is exact (d = 16) ──
(defun (mv t c plain rot out rows cols)
  (nn__turboquant__gemv (nth t c) (nth t (+ c 1)) (if (eq? (nth t (+ c 2)) 16) plain rot) out rows cols (nth t (+ c 2))))

;; the rank's rotation as the pack's ladder makes it — a 256 then a 128 — which `nn__hadamard__rotate` does not place
;; below 512 (▶ the design note: the rotation's rule to bring to the format)
(defun (rotate_rank x out)
  (begin (nn__hadamard__blocks (nn__vector__range x 0 256) signs (nn__vector__range out 0 256) 256 256 0)
         (nn__hadamard__blocks (nn__vector__range x 256 128) signs (nn__vector__range out 256 128) 128 128 0)))


;; ── the gated residual ───────────────────────────────────────────────────────────────────────────────────────────────
;; its before-step over the fp32 streams: each stream normed by its slice of the gain, the mix's two maps (the rank
;; padded at the front, so its first values are 0), the weighted mean over the streams into `into`, and the four
;; injection weights into `inj` — none for the last mixer, its `c_inj` 0
(defun (gated_pre t c_norm c_down c_up c_inj into inj)
  (let ((j 0))
    (while '(< j STREAMS)
           '(begin (nn__rmsnorm__apply_fp32 (nn__vector__range sf (* j (* 2 HIDDEN)) (* 2 HIDDEN))
                                            (nn__vector__range (nth t c_norm) (* j HIDDEN) HIDDEN)
                                            (nn__vector__range an (* j HIDDEN) HIDDEN) HIDDEN EPS)
                   (set! j (+ j 1))))
    (nn__hadamard__rotate an signs anr SW)
    (mv t c_down an anr a1 RANK SW)
    (nn__vector__scale a1 QUARTER a2 RANK)
    (nn__swiglu__combine a2 ones a3 RANK)
    (rotate_rank a3 a3r)
    (mv t c_up a3 a3r gt SW RANK)
    (nn__sigmoid__apply gt gs SW)
    (nn__vector__pointwise_mul gs an gn SW)
    (nn__vector__add (nn__vector__range gn 0 HIDDEN) (nn__vector__range gn HIDDEN HIDDEN) m01 HIDDEN)
    (nn__vector__add m01 (nn__vector__range gn (* 2 HIDDEN) HIDDEN) m012 HIDDEN)
    (nn__vector__add m012 (nn__vector__range gn (* 3 HIDDEN) HIDDEN) m4 HIDDEN)
    (nn__vector__scale m4 QUARTER into HIDDEN)
    (if (< 0 c_inj)
        (begin (mv t c_inj an anr ij STREAMS SW)
               (nn__vector__scale ij QUARTER ij2 STREAMS)
               (nn__sigmoid__apply ij2 ij3 STREAMS)
               (nn__vector__scale ij3 TWO inj STREAMS))
        (type into))))

;; its after-step: stream j += inj_j · y — `mix` holds no pre, the weights as post, the identity as comb
(defun (inject y inj)
  (begin (nn__vector__widen inj (nn__vector__range mix 8 8) STREAMS)
         (nn__hyper__post_fp32 sf y mix sf HIDDEN STREAMS)))


;; ── the DeltaNet: its gates, the conv, then every head's step in one launch, its output gate a sigmoid ──────────────
(defun (deltanet t s32 cwin)
  (begin
    (nn__hadamard__rotate m signs mr HIDDEN)
    (mv t C_QKV m mr mixed CONV_CH HIDDEN)
    (mv t C_Z m mr z VAL HIDDEN)
    (mv t C_A m mr av VH HIDDEN)
    (mv t C_B m mr bv VH HIDDEN)
    (nn__sigmoid__apply bv beta VH)
    (nn__vector__add av (nth t C_DTB) sumb VH)
    (nn__vector__softplus sumb spb VH)
    (nn__vector__exp (nth t C_ALOG) eab VH)
    (nn__vector__pointwise_mul spb eab prodb VH)
    (nn__vector__scale prodb NEG gb VH)
    (nn__deltanet__conv_step (nth t C_CONV) cwin mixed conved CONV_CH)
    (nn__deltanet__step s32 conved z beta gb (nth t C_GNORM) core KH VH LD GATE_SIGMOID)
    (nn__hadamard__rotate core signs crot VAL)
    (mv t C_OUT core crot y HIDDEN VAL)))


;; ── the attention: per-head q and k norms, rope on each head's first ROT, the cache, the sigmoid gate ────────────────
;; ⛳ QSA's indexer is not here: up to 2,051 positions it chooses every block, and the boot holds the context below that
(defun (attention t kc vc pos)
  (begin
    (nn__hadamard__rotate m signs mr HIDDEN)
    (mv t C_Q m mr qg (* 2 QW) HIDDEN)
    (mv t C_K m mr kk KVW HIDDEN)
    (mv t C_V m mr vv KVW HIDDEN)
    (let ((h 0))
      (while '(< h QH)
             '(begin (nn__rmsnorm__apply (nn__vector__range qg (* h (* 2 AD)) AD) (nth t C_QN)
                                         (nn__vector__range qb (* h AD) AD) AD EPS)
                     (nn__rope__apply (nn__vector__range qb (* h AD) ROT) cs (nn__vector__range qb (* h AD) ROT) 1 ROT)
                     (nn__sigmoid__apply (nn__vector__range qg (+ (* h (* 2 AD)) AD) AD) (nn__vector__range gsig (* h AD) AD) AD)
                     (set! h (+ h 1)))))
    (let ((kh 0))
      (while '(< kh KVH)
             '(let ((at (* (+ (* pos KVH) kh) AD)))
                (nn__rmsnorm__apply (nn__vector__range kk (* kh AD) AD) (nth t C_KN) (nn__vector__range kc at AD) AD EPS)
                (nn__rope__apply (nn__vector__range kc at ROT) cs (nn__vector__range kc at ROT) 1 ROT)
                (set! kh (+ kh 1)))))
    (nn__vector__copy vv (nn__vector__range vc (* pos KVW) KVW) KVW)
    (nn__attention__decode qb kc vc scores ao QH KVH AD 0 (+ pos 1))
    (nn__vector__pointwise_mul ao gsig aog QW)
    (nn__hadamard__rotate aog signs aogr QW)
    (mv t C_O aog aogr y HIDDEN QW)))


;; ── the n-gram embedding, before the attention site of its layer ────────────────────────────────────────────────────
;; the key and the value from the rows the host wrote into `ngr`; DeepSeek's gate (σ(sign·√|k̂·q̂/√H|), the norms' gains
;; folded into C_WENG) into the fp32 streams; the gated values exactly — the streams after the gate less before it —
;; normed into the history; their dilated conv over it, silu, added
(defun (ngram t gvn pos)
  (begin
    (nn__hadamard__rotate ngr signs ngrr HIDDEN)
    (mv t C_KEY ngr ngrr (nn__vector__range kv 0 SW) SW HIDDEN)
    (mv t C_VAL ngr ngrr (nn__vector__range kv SW HIDDEN) HIDDEN HIDDEN)
    (nn__vector__copy sf (nn__vector__range sf2 (* 2 SW) (* 2 SW)) (* 2 SW))
    (nn__engram__gate_fp32 sf kv (nth t C_WENG) HIDDEN STREAMS EPS)
    (nn__vector__copy sf (nn__vector__range sf2 0 (* 2 SW)) (* 2 SW))
    (nn__hyper__post_fp32 sf2 zb mixd gvf SW 2)
    (let ((j 0))
      (while '(< j STREAMS)
             '(begin (nn__rmsnorm__apply_fp32 (nn__vector__range gvf (* j (* 2 HIDDEN)) (* 2 HIDDEN))
                                              (nn__vector__range (nth t C_NC) (* j HIDDEN) HIDDEN)
                                              (nn__vector__range gvn (+ (* pos SW) (* j HIDDEN)) HIDDEN) HIDDEN EPS)
                     (set! j (+ j 1)))))
    (nn__vector__zero cv SW)
    (let ((k 0))
      (while '(< k CONV_K)
             '(let ((q (- pos (* (- (- CONV_K 1) k) DIL))))
                (if (< q 0)
                    (type cv)
                    (begin (nn__vector__pointwise_mul (nn__vector__range gvn (* q SW) SW) (nth t (+ C_TAP k)) cvt SW)
                           (nn__vector__add cv cvt cv SW)))
                (set! k (+ k 1)))))
    (nn__swiglu__combine cv ones cvs SW)
    (nn__hyper__post_fp32 sf cvs mix1 sf SW 1)))


;; ── the MoE: the router's ten into the hand, the CPU's routed sum, the card's shared expert beside it ──────────────
(defun (route t i)
  (begin (gated_pre t C_MN C_MD C_MU C_MI m2 inj2)
         (nn__hadamard__rotate m2 signs mr2 HIDDEN)
         (mv t C_GATE m2 mr2 lg NE HIDDEN)
         (nn__vector__top_k lg NE TOPK picks wts)
         (nn__vector__copy mr2 (nn__vector__range hand 0 HIDDEN) HIDDEN)
         (nn__vector__copy wts (nn__vector__range hand HIDDEN TOPK) TOPK)))

;; where the experts are read ahead from the disk (AHEAD_NEXT, NN-138): the next layer's router on the same input, its
;; GUESS likeliest, the likeliest first, into `guess` for the CPU to read the first of (▶ qwen4_exp_cpu.lisp)
(defun (guess_next i)
  (begin (mv (nth LT (nth AHEAD_NEXT i)) C_GATE m2 mr2 lgn NE HIDDEN)
         (nn__vector__top_k lgn NE GUESS guess gwts)))

;; the shared expert over `x` (and `xr`, it rotated) into `out`, gated by its own sigmoid
(defun (shared t x xr out)
  (begin (mv t C_SG x xr sgo INTER HIDDEN)
         (mv t C_SU x xr suo INTER HIDDEN)
         (nn__swiglu__combine sgo suo sact INTER)
         (nn__hadamard__rotate sact signs sactr INTER)
         (mv t C_SD sact sactr sdo HIDDEN INTER)
         (mv t C_SEG x xr segv 1 HIDDEN)
         (nn__sigmoid__apply segv segs 1)
         (nn__vector__scale_at sdo segs 0 out HIDDEN)))

(defun (close t i)
  (begin (sys__node_array__set HERE 0 i)
         (sys__scheduler__add CPU names xq)
         (shared t m2 mr2 sdow)
         (sys__wait CPU)
         (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
         (nn__vector__add routed sdow y2 HIDDEN)
         (inject y2 inj2)))

;; ⭐ THE EXPERTS OVER TWO SOCKETS (SOCKETS 2, ▶ qwen4_exp_cpu.lisp `xu` `xd`): both started on their activations while
;;   the card runs the shared expert, then both on their rows of the routed sum, and the two halves added here.
;; ⛔ A TOP-LEVEL `if` ANSWERS TRUE ON EITHER ARM: the boot defines it as it defines a `defun` (▶ machine.py `defun`)
(if (eq? SOCKETS 2)
    (defun (close t i)
      (begin (sys__node_array__set HERE 0 i)
             (sys__scheduler__add CPU names xu)
             (sys__scheduler__add (+ CPU 1) names xu)
             (shared t m2 mr2 sdow)
             (sys__wait CPU)
             (sys__wait (+ CPU 1))
             (sys__scheduler__add CPU names xd)
             (sys__scheduler__add (+ CPU 1) names xd)
             (sys__wait CPU)
             (sys__wait (+ CPU 1))
             (nn__buffer__copy c_routed0 CPU routed HIDDEN_BYTES)
             (nn__buffer__copy c_routed1 (+ CPU 1) routed1 HIDDEN_BYTES)
             (nn__vector__add routed routed1 routed HIDDEN)
             (nn__vector__add routed sdow y2 HIDDEN)
             (inject y2 inj2)))
    (< 0 1))

;; ⭐ A CARD'S TIER OF THE ROUTED EXPERTS (CARD_TIER, ▶ qwen4_exp.py `card_experts_gb`, as Qwen 3.5's ⑤): the tier's visit
;;   first — the picks the card holds marked, so the CPU, handed the hand as it stands, computes only the rest beside the
;;   card's (▶ qwen4_exp_cpu.lisp `xq_ready`) — then the card's own picks into routed_k and the shared expert, and the two
;;   routed sums added before the shared one
(if CARD_TIER
    (defun (close t i)
      (begin (ai_qwen_3__experts hand picks (nth EXPERTS_TIER i) routed_k nn__expert_tier 1)
             (sys__node_array__set HERE 0 i)
             (sys__scheduler__add CPU names xq_ready)
             (ai_qwen_3__experts hand picks (nth EXPERTS_TIER i) routed_k nn__expert_tier)
             (shared t m2 mr2 sdow)
             (sys__wait CPU)
             (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
             (nn__vector__add routed routed_k routed HIDDEN)
             (nn__vector__add routed sdow y2 HIDDEN)
             (inject y2 inj2)))
    (< 0 1))

;; ── a layer's front: its n-gram site, its gated residual, its mixer, its router into the hand ────────────────────────
;; One procedure for each kind of layer — its mixer, the n-gram site or none, the next layer's experts read ahead or not —
;; and each layer's chosen when the file loads, into FRONTS (▶ the end of the file): a token's layer calls its own front,
;; `((nth FRONTS i) t i pos)`, and tests nothing about the layer.
(defun (front_deltanet t i pos)
  (begin (gated_pre t C_AN C_AD C_AU C_AI m inj)
         (deltanet t (nth S32S i) (nth CWINS i))
         (inject y inj)
         (route t i)))

(defun (front_attention t i pos)
  (begin (gated_pre t C_AN C_AD C_AU C_AI m inj)
         (attention t (nth KCS i) (nth VCS i) pos)
         (inject y inj)
         (route t i)))

(defun (front_ngram_deltanet t i pos) (begin (ngram t (nth GVNS i) pos) (front_deltanet t i pos)))
(defun (front_ngram_attention t i pos) (begin (ngram t (nth GVNS i) pos) (front_attention t i pos)))

;; a layer whose next layer's experts are read ahead: its own front (PLAIN_FRONTS), then the guess
(defun (front_ahead t i pos) (begin ((nth PLAIN_FRONTS i) t i pos) (guess_next i)))

;; ⭐ THE SITES AS ai_qwen_4's VERBS (VERBS, ▶ qwen4_exp.py `verbs`): each the doors its words above launch, in their
;;   order, from one apply — the same bits, without the ranges, the loops and the table reads between them. Q4 is the
;;   shape they read (▶ ai_qwen_4's contracts/objects/layer.cuh); the words stay the reference. The router's verb queues
;;   the hand's copy of mr2 before the top-k's wait and writes the weights into the hand itself, not `wts` — the same
;;   values in the hand; the inject is the rows verb at one row. The two fronts name the verbs themselves, so a token's
;;   front is one procedure of five verbs.
(if VERBS
    (begin
      (defun (gated_pre t c_norm c_down c_up c_inj into inj)
        (ai_qwen_4__gated_pre sf t c_norm into (if (< 0 c_inj) inj 0) Q4))
      (defun (inject y inj) (ai_qwen_4__inject_rows sf y inj 1 Q4))
      (defun (route t i)
        (begin (ai_qwen_4__gated_pre sf t C_MN m2 inj2 Q4)
               (ai_qwen_4__route m2 mr2 t lg picks hand TOPK Q4)))
      (defun (deltanet t s32 cwin) (ai_qwen_4__deltanet m t s32 cwin y Q4))
      (defun (attention t kc vc pos) (ai_qwen_4__attention m t kc vc pos y Q4))
      (defun (shared t x xr out) (ai_qwen_4__shared x xr t out Q4))
      (defun (front_deltanet t i pos)
        (begin (ai_qwen_4__gated_pre sf t C_AN m inj Q4)
               (ai_qwen_4__deltanet m t (nth S32S i) (nth CWINS i) y Q4)
               (ai_qwen_4__inject_rows sf y inj 1 Q4)
               (ai_qwen_4__gated_pre sf t C_MN m2 inj2 Q4)
               (ai_qwen_4__route m2 mr2 t lg picks hand TOPK Q4)))
      (defun (front_attention t i pos)
        (begin (ai_qwen_4__gated_pre sf t C_AN m inj Q4)
               (ai_qwen_4__attention m t (nth KCS i) (nth VCS i) pos y Q4)
               (ai_qwen_4__inject_rows sf y inj 1 Q4)
               (ai_qwen_4__gated_pre sf t C_MN m2 inj2 Q4)
               (ai_qwen_4__route m2 mr2 t lg picks hand TOPK Q4))))
    (< 0 1))


;; ── every layer, a position: each layer's front, then its close ─────────────────────────────────────────────────────
(defun (layers pos)
  (let ((i 0))
    (while '(< i RUN) '(begin ((nth FRONTS i) (nth LT i) i pos) (close (nth LT i) i) (set! i (+ i 1))))
    (type sf)))

;; ⭐ A POSITION READS ITS OWN N-GRAM ROWS (▶ nn's `nn__ngram__rows`, qwen4_exp.py `_ngram_table`): the token's n-grams
;;   after the two in `prev2` hashed, their rows read from the table's file into `ngr` for the n-gram layer, `prev2` moved
;;   on — so a position, and a run of them (`generate`), needs nothing of the host's
;; the token's row of the embedding, un-rotated (Rᵀu = S ⊙ H·u), into every stream
(defun (embed tok)
  (begin (nn__turboquant__decode (nn__vector__range EMBEDDING (* tok EMB_HALVES) EMB_HALVES)
                                 (nn__vector__range EMBEDDING (+ EMB_SCALES tok) 1) eu 1 HIDDEN EMB_BITS)
         (nn__hadamard__rotate eu ones512 ehu HIDDEN)
         (nn__vector__pointwise_mul ehu signs_row e HIDDEN)
         (let ((j 0))
           (while '(< j STREAMS)
                  '(begin (nn__vector__widen e (nn__vector__range sf (* j (* 2 HIDDEN)) (* 2 HIDDEN)) HIDDEN)
                          (set! j (+ j 1)))))
         (type sf)))

(defun (position tok pos)
  (begin (nn__ngram__rows tok prev2 NGT ngr) (embed tok) (nn__rope__angles cs pos THETA ROT) (layers pos)))


;; ── a prompt, a chunk of rows at a time: positions `first .. first + n` ──────────────────────────────────────────────
;; Each layer takes the whole chunk before the next: every row's mixer and its MoE's before-step a row at a time, in
;; order, as a position runs them (the DeltaNet's state, the cache and the n-gram history filled row after row); then the
;; chunk's router picks into its hand rows at once, the CPU's routed experts over them once, expert-major (▶
;; qwen4_exp_cpu.lisp `xr`), while the card runs each row's shared expert; then each row's sum into its streams. A row's
;; streams live in `sfR` between its sites and are run in `sf`.
(defun (row_in r) (nn__vector__copy (nn__vector__range sfR (* r (* 2 SW)) (* 2 SW)) sf (* 2 SW)))
(defun (row_out r) (nn__vector__copy sf (nn__vector__range sfR (* r (* 2 SW)) (* 2 SW)) (* 2 SW)))
(defun (in_row buf r width) (nn__vector__range buf (* r width) width))

(defun (rows_layer i first n)
  (let ((t (nth LT i)) (r 0))
    (while '(< r n)
           '(let ((pos (+ first r)))
              (row_in r)
              (nn__vector__copy (in_row csR r (* 2 ROT)) cs (* 2 ROT))
              (if (nth NGRAM_AT i)
                  (begin (nn__vector__copy (in_row ngrR r HIDDEN) ngr HIDDEN) (ngram t (nth GVNS i) pos))
                  (type sf))
              (gated_pre t C_AN C_AD C_AU C_AI m inj)
              (if (nth ATTN i) (attention t (nth KCS i) (nth VCS i) pos) (deltanet t (nth S32S i) (nth CWINS i)))
              (inject y inj)
              (gated_pre t C_MN C_MD C_MU C_MI m2 inj2)
              (nn__hadamard__rotate m2 signs mr2 HIDDEN)
              (mv t C_GATE m2 mr2 (in_row lgR r NE) NE HIDDEN)
              (nn__vector__copy mr2 (nn__vector__range hands (* r HAND_HALVES) HIDDEN) HIDDEN)
              (nn__vector__copy m2 (in_row m2R r HIDDEN) HIDDEN)
              (nn__vector__copy mr2 (in_row mr2R r HIDDEN) HIDDEN)
              (nn__vector__copy inj2 (in_row inj2R r 8) STREAMS)
              (row_out r)
              (set! r (+ r 1))))
    (nn__vector__top_k_rows lgR NE TOPK n hands HAND_PICKS_AT HAND_W_AT HAND_ROW)
    (sys__node_array__set HERE 0 i)
    (sys__node_array__set HERE 1 n)
    (sys__scheduler__add CPU names xr)
    (set! r 0)
    (while '(< r n)
           '(begin (shared t (in_row m2R r HIDDEN) (in_row mr2R r HIDDEN) (in_row sdowR r HIDDEN))
                   (set! r (+ r 1))))
    (sys__wait CPU)
    (nn__buffer__copy c_rrows CPU routedR (* n HIDDEN_BYTES))
    (set! r 0)
    (while '(< r n)
           '(begin (row_in r)
                   (nn__vector__add (in_row routedR r HIDDEN) (in_row sdowR r HIDDEN) y2 HIDDEN)
                   (inject y2 (in_row inj2R r 8))
                   (row_out r)
                   (set! r (+ r 1))))))

;; the chunk's tokens are in `toks`, its n-gram rows in `ngrR` (▶ qwen4_exp.py `prefill`); its last row's streams are
;; left in `sf` for the head. `card` 1 has each layer's routed experts computed on the card (`rows_layer_card`, PROMPT_CARD),
;; 0 on the CPU (`rows_layer`). Each layer collects what it left: in the manual mode a chunk's 48 layers are one
;; evaluation, and what they let go waits for a collect (▶ sys's `garbage_collection__header.cuh`); in the ordinary
;; mode the collect finds nothing waiting
(defun (prompt_rows first n card)
  (let ((r 0) (i 0) (on_card (eq? card 1)))
    (while '(< r n)
           '(begin (embed (nth toks r))
                   (row_out r)
                   (nn__rope__angles (in_row csR r (* 2 ROT)) (+ first r) THETA ROT)
                   (set! r (+ r 1))))
    (while '(< i RUN) '(begin (if on_card (rows_layer_card i first n) (rows_layer i first n))
                              (sys__garbage_collection__collect)
                              (set! i (+ i 1))))
    (row_in (- n 1))
    (type sf)))

;; ⭐ A CHUNK'S ROWS AT ONCE (VERBS, ▶ ai_qwen_4's `*_rows`): each site once over the chunk — its matrices one GEMM over every
;;   row, the DeltaNet's recurrence and the attention walking the rows inside their launches — where the words above run a
;;   row at a time. Q4R is the rows' shape (their scratch, a row's angles in csR, the chunk's scores, a mix a row). The
;;   n-gram layer's site stays a row at a time: its convolution reads the history the rows before it write. A rows form
;;   sums in its own order: it is held to a position at a time's by a bound, not bits (▶ test/src_qwen4_exp_runtime.py).
(if VERBS
    (defun (rows_layer i first n)
      (let ((t (nth LT i)) (r 0))
        (if (nth NGRAM_AT i)
            (while '(< r n)
                   '(begin (row_in r)
                           (nn__vector__copy (in_row ngrR r HIDDEN) ngr HIDDEN)
                           (ngram t (nth GVNS i) (+ first r))
                           (row_out r)
                           (set! r (+ r 1))))
            (type sf))
        (ai_qwen_4__gated_pre_rows sfR t C_AN mR injR n Q4R)
        (if (nth ATTN i)
            (ai_qwen_4__attention_rows mR t (nth KCS i) (nth VCS i) first n yR Q4R)
            (ai_qwen_4__deltanet_rows mR t (nth S32S i) (nth CWINS i) yR n Q4R))
        (ai_qwen_4__inject_rows sfR yR injR n Q4R)
        (ai_qwen_4__gated_pre_rows sfR t C_MN m2R inj2R n Q4R)
        (ai_qwen_4__route_rows m2R mr2R t lgR hands n Q4R)
        (nn__vector__top_k_rows lgR NE TOPK n hands HAND_PICKS_AT HAND_W_AT HAND_ROW)
        (sys__node_array__set HERE 0 i)
        (sys__node_array__set HERE 1 n)
        (sys__scheduler__add CPU names xr)
        (ai_qwen_4__shared_rows m2R mr2R t sdowR n Q4R)
        (sys__wait CPU)
        (nn__buffer__copy c_rrows CPU routedR (* n HIDDEN_BYTES))
        (nn__vector__add routedR sdowR yR (* n HIDDEN))
        (ai_qwen_4__inject_rows sfR yR inj2R n Q4R)))
    (< 0 1))

;; ⭐ A PROMPT'S EXPERTS ON ITS OWN CARD (PROMPT_CARD, ▶ qwen4_exp.py `prompt_on`): the chunk's picks noted for the card's
;;   tier — every one the card's: the ones its tier holds (CARD_TIER) computed from it, the rest streamed from the CPU's
;;   memory through the tier's ring a batch ahead of the GEMM (NOTE_STREAM; the threshold past any chunk's picks of one
;;   expert, so none is promoted) — over the rows, into the routed rows, the CPU left to the other cards. A chunk chooses
;;   it or the CPU's `rows_layer` by its length (▶ `prompt_rows`). A card past the first reads its seat's prompt in its
;;   seat's view: its tier, its ring.
(if PROMPT_CARD
    (defun (rows_layer_card i first n)
      (let ((t (nth LT i)) (r 0))
        (if (nth NGRAM_AT i)
            (while '(< r n)
                   '(begin (row_in r)
                           (nn__vector__copy (in_row ngrR r HIDDEN) ngr HIDDEN)
                           (ngram t (nth GVNS i) (+ first r))
                           (row_out r)
                           (set! r (+ r 1))))
            (type sf))
        (ai_qwen_4__gated_pre_rows sfR t C_AN mR injR n Q4R)
        (if (nth ATTN i)
            (ai_qwen_4__attention_rows mR t (nth KCS i) (nth VCS i) first n yR Q4R)
            (ai_qwen_4__deltanet_rows mR t (nth S32S i) (nth CWINS i) yR n Q4R))
        (ai_qwen_4__inject_rows sfR yR injR n Q4R)
        (ai_qwen_4__gated_pre_rows sfR t C_MN m2R inj2R n Q4R)
        (ai_qwen_4__route_rows m2R mr2R t lgR hands n Q4R)
        (nn__vector__top_k_rows lgR NE TOPK n hands HAND_PICKS_AT HAND_W_AT HAND_ROW)
        (nn__expert_tier__note hands nn__expert_tier i n HAND_ROW HAND_PICKS_AT
                               (+ nn__expert_major__vram_promotion_threshold_picks NOTE_STREAM))
        (ai_qwen_3__experts_rows hands (nth EXPERTS_CARD_ROWS i) routedR n nn__expert_tier)
        (ai_qwen_4__shared_rows m2R mr2R t sdowR n Q4R)
        (nn__vector__add routedR sdowR yR (* n HIDDEN))
        (ai_qwen_4__inject_rows sfR yR inj2R n Q4R)))
    (< 0 1))


;; ── the head: the last mixer folds the streams, then the logits ─────────────────────────────────────────────────────
(defun (logits)
  (begin (gated_pre HEADT C_AN C_AD C_AU 0 mh inj)
         (nn__hadamard__rotate mh signs mhr HIDDEN)
         (nn__turboquant__gemv HEAD_CODES HEAD_SCALES mhr logits_h VOCAB HIDDEN HEAD_BITS)))

(defun (head)                                          ; the greedy token
  (begin (logits) (nn__argmax__find logits_h res VOCAB) (nn__buffer__read res)))

(defun (head_logits)                                   ; the logits left for a sampler on the host
  (begin (logits) (type sf)))


;; ── a position as a chain of steps, a layer's experts between two ─────────────────────────────────────────────────
;; ⚖ 2026-11-04 *"the program becomes ((attention) (router) (compute_local_gemv) (ask_gemv)) ((get_gemv)
;; (compute_final_layer_part)) so it is simply a self-queuing exercise, removing all custom work"*. A step runs a layer's
;; front, hands the CPU its experts and runs the card's shared expert beside them, and ends continued on the CPU's answer
;; (▶ sys's `sys__scheduler__continue`); the next step, once that answer is filled, closes the layer and runs the next
;; one's front. So the card's block is free while the CPU computes, and whatever else is queued on it — another seat's
;; step — runs meanwhile. Each seat runs in a view of its own (▶ qwen4_exp.py `load`): the program's names bound to that
;; seat's buffers and state, `names` the view the CPU computes its experts in. Its HERE is what its steps carry between
;; them: 0 the layer, 1 the position, 2 how the position ends (▶ `ends`). The last step answers the position's token.
;; One CPU worker: the experts over two sockets run a position whole (`position`).
;; ⛔ A PARAMETER IS A NAME LIKE ANY OTHER, bound while its procedure runs: these are `ending`, as `e` is the embedding's
;;   buffer that `embed` reads
(defun (ends ending) (if (eq? ending 1) (head) (begin (if (eq? ending 2) (head_logits) (type sf)) 0)))

(defun (chain_ask t i)
  (begin (sys__node_array__set HERE 0 i)
         (let ((r (sys__scheduler__add CPU names xq)))
           (shared t m2 mr2 sdow)
           (sys__scheduler__continue CARD 0 c_close '(r)))))

(defun (chain_close)
  (let ((i (+ (nth HERE 0) 1)))
    (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
    (nn__vector__add routed sdow y2 HIDDEN)
    (inject y2 inj2)
    (if (< i RUN)
        (begin ((nth FRONTS i) (nth LT i) i (nth HERE 1)) (chain_ask (nth LT i) i))
        (ends (nth HERE 2)))))

;; with a card's tier (CARD_TIER, ▶ the tiered `close`): the visit before the CPU is asked, the card's own picks beside it
(if CARD_TIER
    (begin
      (defun (chain_ask t i)
        (begin (ai_qwen_3__experts hand picks (nth EXPERTS_TIER i) routed_k nn__expert_tier 1)
               (sys__node_array__set HERE 0 i)
               (let ((r (sys__scheduler__add CPU names xq_ready)))
                 (ai_qwen_3__experts hand picks (nth EXPERTS_TIER i) routed_k nn__expert_tier)
                 (shared t m2 mr2 sdow)
                 (sys__scheduler__continue CARD 0 c_close '(r)))))
      (defun (chain_close)
        (let ((i (+ (nth HERE 0) 1)))
          (nn__buffer__copy c_routed CPU routed HIDDEN_BYTES)
          (nn__vector__add routed routed_k routed HIDDEN)
          (nn__vector__add routed sdow y2 HIDDEN)
          (inject y2 inj2)
          (if (< i RUN)
              (begin ((nth FRONTS i) (nth LT i) i (nth HERE 1)) (chain_ask (nth LT i) i))
              (ends (nth HERE 2))))))
    (< 0 1))

;; the position's first step: its token in, its angles, the first layer's front and its experts asked
(defun (chain_position tok pos ending)
  (begin (sys__node_array__set HERE 1 pos)
         (sys__node_array__set HERE 2 ending)
         (nn__ngram__rows tok prev2 NGT ngr)
         (embed tok)
         (nn__rope__angles cs pos THETA ROT)
         ((nth FRONTS 0) (nth LT 0) 0 pos)
         (chain_ask (nth LT 0) 0)))

;; what a step continues as: the layer's close and the next one's front — inline, as every step of a seat's (▶
;; qwen4_exp.py `_inline`): it keeps the card's scratch between its verbs (NN-159)
(picture c_close (sys__scheduler__compute_inline '(chain_close)))


;; ── each layer's front, chosen when the file loads ──────────────────────────────────────────────────────────────────
;; By the layer's mixer (ATTN), the n-gram site (NGRAM_AT) and the read-ahead (AHEAD_NEXT): the procedure itself into
;; PLAIN_FRONTS, and into FRONTS the one a token's layer calls — `front_ahead` where the next layer's experts are read
;; ahead, else the plain one. ⛔ IT COMES LAST: an array holds the procedure a name meant when the array was filled, so
;; a `defun` after this would not reach it.
(if (< 0 RUN)
    (let ((i 0))
      (while '(< i RUN)
             '(begin (sys__node_array__set PLAIN_FRONTS i
                                           (if (nth ATTN i)
                                               (if (nth NGRAM_AT i) front_ngram_attention front_attention)
                                               (if (nth NGRAM_AT i) front_ngram_deltanet front_deltanet)))
                     (sys__node_array__set FRONTS i (if (< 0 (nth AHEAD_NEXT i)) front_ahead (nth PLAIN_FRONTS i)))
                     (set! i (+ i 1))))
      (< 0 1))
    (< 0 1))
