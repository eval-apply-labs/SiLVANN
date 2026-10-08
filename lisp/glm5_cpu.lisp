;; GLM 5.3's ROUTED EXPERTS ON THE CPU SOCKETS — what the card hands each socket, a layer at a time (▶ glm5.lisp).
;; A socket is the worker after the card's — socket `s` is worker `s + 1` — and holds its part of every expert in its own
;; memory. Each picture copies the card's hand across and runs the layer HERE names over it, with that socket's experts
;; table: EX a position's, EXR a chunk's, each an array a socket of arrays a layer. C_HAND C_ROUTED C_HANDS C_RROWS are
;; each socket's own buffers.

(worker 1
  (picture xf (let ((s (- (sys__worker) 1)))
                (nn__buffer__copy hand CARD (nth C_HAND s) HAND_BYTES)
                (ai_glm_5_3__experts (nth C_HAND s) picks (nth (nth EX s) (nth HERE 0)) (nth C_ROUTED s))))
  (picture xr (let ((s (- (sys__worker) 1)))
                (nn__buffer__copy hands CARD (nth C_HANDS s) ROWS_HANDS_BYTES)
                (ai_glm_5_3__experts_rows (nth C_HANDS s) (nth (nth EXR s) (nth HERE 0)) (nth C_RROWS s) (nth nrows 0)))))

;; what a socket's block sees: the pictures, the tables, the card's hand and picks — the bindings as they stand
(view names)
