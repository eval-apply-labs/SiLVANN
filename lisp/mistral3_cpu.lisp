;; MINISTRAL 3's EMBEDDING ON THE CPU — what the card hands the CPU worker (▶ mistral3.lisp `embed_into`, EMBED_ON_CPU).
;; The table is the CPU worker's (c_embedding): a token's row — EMB_TOK's first — decoded into c_emb, and a prompt chunk's,
;; EMB_TOK's second the count of PROMPT's tokens, into c_embs, a row each. The card copies them across and un-rotates them.

(worker 1
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

;; what a CPU block sees: the pictures, the tables, the prompt's tokens — the bindings as they stand
(view names)
