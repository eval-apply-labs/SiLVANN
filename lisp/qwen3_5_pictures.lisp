;; QWEN 3.5'S PICTURES — what a prompt with a picture needs beyond qwen3_5.lisp, read after it (▶ qwen3_5.py
;; `_procedures`) by a runtime that reads pictures (v0.7.0 on). A runtime before it never reads this file, so the
;; qwen3_5.lisp a pack carries stays one every runtime runs.
;;
;; A picture's tokens have a row and a column of their own (M-RoPE: a token's three positions, time, row and column), and
;; after a picture a token's rotary position is no longer its place in the cache. What the boot binds before this file is
;; read (▶ qwen3_5.py `_bindings`):
;;   LAG     how far a token's rotary position is behind its place in the cache (0 until a picture: its tokens take
;;           fewer positions than places)
;;   ROWPOS  true while a prompt chunk's rotary positions are the host's, three a row in `rowpos`
;;   MROPE   their sections, as nn's `rope_angles_mrope_rows` takes them
;; It defines again, over qwen3_5.lisp's, a chunk's mixer and a token's position.

;; a chunk's rows are the cache's `first ..`; their rotary positions are those, or the host's in `rowpos` (ROWPOS)
(if TIERED
    (defun (mixer_rows i first n)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet_rows xs (nth MIXER_ROWS i) h1s n)
          (if ROWPOS
              (ai_qwen_3__attention_tiered_rows xs (nth MIXER_ROWS i) first h1s n rowpos MROPE)
              (ai_qwen_3__attention_tiered_rows xs (nth MIXER_ROWS i) first h1s n))))
    (defun (mixer_rows i first n)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet_rows xs (nth MIXER_ROWS i) h1s n)
          (if ROWPOS
              (ai_qwen_3__attention_rows xs (nth MIXER_ROWS i) first h1s n rowpos MROPE)
              (ai_qwen_3__attention_rows xs (nth MIXER_ROWS i) first h1s n)))))

;; a token's position: its embedding, the angles (at its rotary position, LAG behind its place in the cache), the layers
(defun (position tok pos)
  (begin (embed tok) (nn__rope__angles cs (- pos LAG) THETA ROTARY) (layers pos)))
