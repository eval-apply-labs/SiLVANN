;; INKLING 2 ON THE CPU — what the card hands the CPU worker: the routed experts, a layer's application at a time
;; (▶ inkling2.lisp `experts`): the hand copied across — the router's input rotated, then the eight weights — and the
;; layer HERE names run over it by the model's own experts verb (ai_thinkingmachines__inkling_2's: nn's routed experts,
;; the activation soft-capped by the table's two caps) with that layer's experts table, the CPU worker's own. The
;; picture ends with the buffer it summed into, so that buffer is its answer: the card, waiting on the answer, is left
;; holding it.

(worker 1
  (picture experts_picture
           (begin (nn__buffer__copy hand cpu_hand HAND_BYTES)
                  (ai_thinkingmachines__inkling_2__experts cpu_hand picks (nth EXPERTS_CPU (nth HERE 0)) cpu_routed)
                  cpu_routed))
  ;; the embedding the CPU's (EMBED_ON_CPU, ▶ inkling2.lisp `embed`): a token's row — EMB_TOK's first — decoded from
  ;; c_embedding into c_emb, its answer: the card, waiting on it, copies the row across and un-rotates it
  (picture embedding_picture (let ((tok (nth EMB_TOK 0)))
                               (nn__turboquant__decode (nn__vector__range c_embedding (* tok EMBEDDING_HALVES) EMBEDDING_HALVES)
                                                       (nn__vector__range c_embedding (+ EMBEDDING_SCALES tok) 1)
                                                       c_emb 1 HIDDEN EMBEDDING_BITS)
                               c_emb)))

;; what a CPU block sees: the picture, the tables, the card's hand and picks — the bindings as they stand
(view names)
