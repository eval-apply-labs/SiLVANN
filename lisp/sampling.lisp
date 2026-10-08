;; THE SAMPLER ON THE CARD — the next token drawn where the logits are; only the token comes back. ▶ sampling.py
;;
;; Every model's program defines `(logits)`, which leaves the head's logits in `logits_h`, and binds VOCAB. The draw
;; is the host sampler's arithmetic: the repetition penalty over the recent tokens in `seen` (the first `count` of
;; them), the temperature, the `k` largest and their softmax, the top-p cut, one draw. The random number is `seed` and
;; `pos` hashed, so a seed replays a conversation on any card.

(defun (head_sample count penalty inv_t k top_p seed pos)
  (begin (logits)
         (nn__vector__penalize logits_h VOCAB seen count penalty)
         (nn__vector__scale logits_h inv_t logits_h VOCAB)
         (nn__vector__top_k logits_h VOCAB k sample_picks sample_weights)
         (nn__vector__draw sample_weights k top_p seed pos res)
         (nth sample_picks (nn__buffer__read res))))
