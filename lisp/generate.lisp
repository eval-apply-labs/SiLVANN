;; AN ANSWER ON THE CARD — the tokens of an answer, each the next one's input, without the host between them.
;;
;; Any family's program that has a token's position and a head answers with it: `(position tok pos)` — the token's
;; embedding, its angles and every layer this machine runs — `(head)` the greedy token, and `head_sample` the drawn one
;; (▶ sampling.lisp). A model's folder carries this file beside its family's own (▶ `ModelFolder.program`).
;;
;; What the boot binds before this file is read (▶ silvann_runtime/generate.py `buffers`):
;;   OUT STOP STOPS — what `generate` made, the end tokens and how many of them · seen SEEN — the sampler's recent tokens

;; From `tok` at `pos`, up to `n` tokens into OUT, ending early after one of STOP's first STOPS. `draw` true draws each
;; with the card's sampler (▶ sampling.lisp: `penalty inv_t k top_p seed`), false takes the greedy one. Where the
;; penalty reads them (`reads` 1) each token goes into `seen` first, at `slot`, `count` of them there. Answers how many
;; it made.
(defun (stop? tok)
  (let ((j 0) (hit (< 1 0)))
    (while '(< j STOPS) '(begin (if (eq? (nth STOP j) tok) (set! hit (< 0 1)) 0) (set! j (+ j 1))))
    hit))

(defun (generate tok pos n draw reads count slot penalty inv_t k top_p seed)
  (let ((i 0))
    (while '(< i n)
           '(begin
              (position tok pos)
              (if (eq? reads 1)
                  (begin (sys__node_array__set seen slot tok)
                         (set! slot (+ slot 1)) (if (eq? slot SEEN) (set! slot 0) 0)
                         (if (< count SEEN) (set! count (+ count 1)) 0))
                  0)
              (set! tok (if draw (head_sample count penalty inv_t k top_p seed pos) (head)))
              (sys__node_array__set OUT i tok)
              (set! i (+ i 1)) (set! pos (+ pos 1))
              (if (stop? tok) (set! n i) 0)))
    i))
