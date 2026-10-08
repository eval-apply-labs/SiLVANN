;; QWEN 3.5's ROUTED EXPERTS ON THE CPU — what the card hands the CPU worker, a layer at a time (▶ qwen3_5.lisp ④ ⑤).
;; Each picture copies the card's hand across and runs the layer HERE names over it, with that layer's experts table,
;; which is the CPU worker's own: EXPERTS_CPU a position's, EXPERTS_CPU_ROWS a chunk's.

(worker 1
  (picture xq (begin (nn__buffer__copy hand CARD c_hand HAND_BYTES)
                     (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed)))
  (picture xr (begin (nn__buffer__copy hands CARD c_hands HANDS_BYTES)
                     (ai_qwen_3__experts_rows c_hands (nth EXPERTS_CPU_ROWS (nth HERE 0)) c_rrows (nth nrows 0)))))

;; what a CPU block sees: the pictures, the tables, the card's hand and picks — the bindings as they stand
(view names)
