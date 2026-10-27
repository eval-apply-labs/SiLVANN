;; QWEN 3.5's ROUTED EXPERTS ON THE CPU — what the card hands the CPU worker, a layer at a time (▶ qwen3_5.lisp ④ ⑤).
;; Each picture copies the card's hand across and runs the layer HERE names over it, with that layer's experts table,
;; which is the CPU worker's own: EXPERTS_CPU a position's, EXPERTS_CPU_ROWS a chunk's.

(worker 1                                     ; the CPU: a tensor-parallel group's boot moves it after its cards
  (picture xq (begin (nn__buffer__copy hand CARD c_hand HAND_BYTES)
                     (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed)))
  ;; the card's tier visited and the card waited for (▶ qwen3_5.lisp ⑤): the hand read beside the card's experts
  (picture xq_ready (begin (nn__buffer__copy hand CARD c_hand HAND_BYTES 1)
                           (ai_qwen_3__experts c_hand picks (nth EXPERTS_CPU (nth HERE 0)) c_routed)))
  (picture xr (begin (nn__buffer__copy hands CARD c_hands HANDS_BYTES)
                     (ai_qwen_3__experts_rows c_hands (nth EXPERTS_CPU_ROWS (nth HERE 0)) c_rrows (nth nrows 0))))
  ;; the embedding the CPU's (EMBED_ON_CPU, ▶ qwen3_5.lisp `embed_into`): a token's row decoded from c_embedding into
  ;; c_emb, and a prompt chunk's — EMB_TOK's count of PROMPT's tokens — into c_embs, a row each
  (picture xe (let ((tok (nth EMB_TOK 0)))
                (nn__turboquant__decode (nn__vector__range c_embedding (* tok EMB_HALVES) EMB_HALVES)
                                        (nn__vector__range c_embedding (+ EMB_SCALES tok) 1) c_emb 1 HIDDEN EMB_BITS)))
  (picture xer (let ((i 0) (at 0) (n (nth EMB_TOK 1)))
                 (while '(< i n)
                        '(let ((tok (nth PROMPT i)))
                           (nn__turboquant__decode (nn__vector__range c_embedding (* tok EMB_HALVES) EMB_HALVES)
                                                   (nn__vector__range c_embedding (+ EMB_SCALES tok) 1)
                                                   (nn__vector__range c_embs at HIDDEN) 1 HIDDEN EMB_BITS)
                           (set! at (+ at HIDDEN)) (set! i (+ i 1)))))))

;; what a CPU block sees: the pictures, the tables, the card's hand and picks — the bindings as they stand
(view names)
