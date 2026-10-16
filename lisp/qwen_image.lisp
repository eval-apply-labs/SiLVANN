;; QWEN-IMAGE 2.1 — a picture from a prompt and reference pictures: its diffusion transformer, a step at a time.
;;
;; A picture's latents are `(denoise steps)`: the prompt's modulation (t = 0) once, then each step — the timestep's
;; modulation, the sequence's rows, every block, and the step's end, which moves the latents along the flow by the step's
;; dsigma. The first step runs every row of the sequence — the prompt's text runs, its references' latent rows, the
;; picture's — and each block keeps the prompt's keys and values; a later step runs the picture's rows alone over them:
;; the prompt is modulated from t = 0 and reads no later row, so it is the same at every step.
;;
;; What the boot binds before this file is read (▶ qwen_image.py `_bindings`):
;;   the transformer  QG, its own table · BLOCK, a block's table each · BLOCKS, how many — the cells each verb reads are
;;                    named in ai_qwen_image's contracts/objects/ (step.cuh, block.cuh)
;;   the rows         x modulation scale rotary limits sinusoids encoder conditions latents prediction
;; and before each picture (▶ qwen_image.py `bind_picture`):
;;   TOTAL PREFIX TARGET  the sequence's rows, the prompt's, the picture's
;;   RUNS             the prompt's runs; RUN_TEXT (true for a text run, false for a reference's rows), RUN_FROM (its first
;;                    row in `encoder`, or in `conditions`), RUN_FIRST (its first row in the sequence), RUN_ROWS
;;   DSIGMA           a step's sigma after it less its own; the step's sinusoid is `sinusoids`' row one past its own,
;;                    row 0 being t = 0's

;; ── the sequence's rows: the prompt's runs, then the picture's ──────────────────────────────────────────────────────
(defun (prompt_rows)
  (let ((i 0))
    (while '(< i RUNS)
           '(begin (if (nth RUN_TEXT i)
                       (ai_qwen_image__text encoder x QG (nth RUN_ROWS i) (nth RUN_FROM i) (nth RUN_FIRST i))
                       (ai_qwen_image__latents conditions x QG (nth RUN_ROWS i) (nth RUN_FROM i) (nth RUN_FIRST i)))
                   (set! i (+ i 1))))
    (type x)))

;; ── every block over `rows` rows of the sequence from `first` ───────────────────────────────────────────────────────
(defun (blocks rows first)
  (let ((i 0))
    (while '(< i BLOCKS)
           '(begin (ai_qwen_image__block x modulation rotary limits (nth BLOCK i) rows first PREFIX)
                   (set! i (+ i 1))))
    (type x)))

;; ── a step: the first over every row, a later one over the picture's ────────────────────────────────────────────────
(defun (step s)
  (begin (ai_qwen_image__time sinusoids (+ s 1) modulation 1 scale QG)
         (if (< s 1)
             (begin (prompt_rows)
                    (ai_qwen_image__latents latents x QG TARGET 0 PREFIX)
                    (blocks TOTAL 0)
                    (ai_qwen_image__out x scale prediction latents QG TARGET PREFIX (nth DSIGMA s)))
             (begin (ai_qwen_image__latents latents x QG TARGET 0 0)
                    (blocks TARGET PREFIX)
                    (ai_qwen_image__out x scale prediction latents QG TARGET 0 (nth DSIGMA s))))))

(defun (denoise steps)
  (let ((s 0))
    (ai_qwen_image__time sinusoids 0 modulation 0 scale QG)
    (while '(< s steps) '(begin (step s) (set! s (+ s 1))))
    (type latents)))
