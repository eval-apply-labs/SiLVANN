;; QWEN3.8-FLASH-NEXT's ROUTED EXPERTS ON THE CPU — what the card hands the CPU worker, a layer at a time (▶ qwen4_exp.lisp
;; `close`): the hand copied across — the router's input rotated, then the ten weights, in ai_qwen_3's hand-off layout —
;; and the layer HERE names run over it with that layer's experts table, the CPU worker's own; where the card guessed the
;; next layer's picks (AHEAD_NEXT, NN-138), the first of the guess the CPU does not hold read ahead from the disk — AHEAD
;; experts, or as many as the disk starts while a layer computes. Each picture ends with the buffer it computed into, so
;; that buffer is its answer: the card, waiting on the answer, is left holding it.

(worker 1
  (picture xq (begin (nn__buffer__copy hand c_hand HAND_BYTES)
                     (if (< 0 (nth AHEAD_NEXT (nth HERE 0)))
                         (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed
                                             guess (nth EXPERTS_CPU (nth AHEAD_NEXT (nth HERE 0))) AHEAD 1)
                         (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed))
                     c_routed))
  ;; the card's tier visited and the card waited for (CARD_TIER, ▶ qwen4_exp.lisp's tiered `close`): the hand read beside
  ;; the card's own experts, and only the picks the tier did not mark
  (picture xq_ready (begin (nn__buffer__copy hand c_hand HAND_BYTES 1)
                           (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed)
                           c_routed))
  ;; a prompt's chunk (▶ qwen4_exp.lisp `rows_layer`): its HERE 1 hand rows copied across in one, the layer's routed
  ;; experts once over them, expert-major, into c_rrows
  (picture xr (begin (nn__buffer__copy hands c_hands (* (nth HERE 1) HAND_ROW))
                     (ai_qwen_3__experts_rows c_hands (nth EXPERTS_CPU_ROWS (nth HERE 0)) c_rrows (nth HERE 1))
                     c_rrows))
  ;; ⭐ THE EXPERTS OVER TWO SOCKETS (SOCKETS 2, ▶ qwen4_exp.py `sockets`): socket `s` is worker CPU + s and holds its
  ;; part of every expert in its own memory, its tables EXS's and its buffers C_HAND C_ACT C_OTHER C_ROUTED's. First
  ;; the hand copied across and its activations — every pick's I halves, its own half in their places, zeros elsewhere
  (picture xu (let ((s (- (sys__worker) CPU)))
                (nn__buffer__copy hand (nth C_HAND s) HAND_BYTES)
                (ai_qwen_3__experts_up (nth C_HAND s) picks (nth (nth EXS s) (nth HERE 0)) (nth C_ACT s))))
  ;; then, both sockets' activations made, the other's copied across and this socket's rows of the routed sum — its
  ;; answer, C_ROUTED's
  (picture xd (let ((s (- (sys__worker) CPU)))
                (nn__buffer__copy (nth C_ACT (- 1 s)) (nth C_OTHER s) ACT_BYTES)
                (ai_qwen_3__experts_down (nth C_HAND s) picks (nth (nth EXS s) (nth HERE 0)) (nth C_ACT s)
                                         (nth C_OTHER s) (nth C_ROUTED s))
                (nth C_ROUTED s))))

;; what a CPU block sees: the picture, the tables, the card's hand and picks — the bindings as they stand
(view names)
