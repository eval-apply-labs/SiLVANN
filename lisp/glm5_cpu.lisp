;; GLM 5.3's ROUTED EXPERTS ON THE CPU SOCKETS — what the card hands each socket, a layer at a time (▶ glm5.lisp).
;; A socket is a worker after the cards' — socket `s` is worker `s + FIRST_SOCKET` — and holds its part of every expert in its own
;; memory. Each picture copies the card's hand across and runs the layer HERE names over it, with that socket's experts
;; table: EX a position's, EXR a chunk's, each an array a socket of arrays a layer. C_HAND C_ROUTED C_HANDS C_RROWS are
;; each socket's own buffers.

(worker 1
  ;;   and where the card guessed the next MoE layer's picks (AHEAD_NEXT, NN-138), this socket's part of the first of the
  ;;   guess it does not hold read ahead from the disk — AHEAD experts, or as many as the disk reads while a layer computes,
  ;;   an expert a part on each of the SOCKETS
  (picture xf (let ((s (- (sys__worker) FIRST_SOCKET)))
                (nn__buffer__copy hand CARD (nth C_HAND s) HAND_BYTES)
                (if (< 0 (nth AHEAD_NEXT (nth HERE 0)))
                    (ai_z__glm_5_3__experts (nth C_HAND s) picks (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED s)
                                            guess (nth (nth EX s) (nth AHEAD_NEXT (nth HERE 0))) AHEAD SOCKETS)
                    (ai_z__glm_5_3__experts (nth C_HAND s) picks (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED s)))))
  ;; the card's tier visited and the card waited for (▶ glm5.lisp `tier_and_sockets`, VISIT_FIRST): the hand read beside
  ;;   the card's experts
  (picture xf_ready (let ((s (- (sys__worker) FIRST_SOCKET)))
                      (nn__buffer__copy hand CARD (nth C_HAND s) HAND_BYTES 1)
                      (ai_z__glm_5_3__experts (nth C_HAND s) picks (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED s))))
  (picture xr (let ((s (- (sys__worker) FIRST_SOCKET)))
                (nn__buffer__copy hands CARD (nth C_HANDS s) (* (nth nrows 0) HAND))   ; the chunk's rows, not a whole chunk's
                (ai_z__glm_5_3__experts_rows (nth C_HANDS s) (nth (nth EXR s) (nth HERE 0)) (nth C_RROWS s) (nth nrows 0))))
  ;; the embedding the first socket's (EMBED_ON_CPU, ▶ glm5.lisp `embed_e`): a token's row, decoded from c_embedding into c_emb
  (picture xe (let ((tok (nth EMB_TOK 0)))
                (nn__turboquant__decode (nn__vector__range c_embedding (* tok EMB_HALVES) EMB_HALVES)
                                        (nn__vector__range c_embedding (+ EMB_SCALES tok) 1) c_emb 1 HIDDEN EMB_BITS))))

;; what a socket's block sees: the pictures, the tables, the card's hand, picks and guess — the bindings as they stand
(view names)
