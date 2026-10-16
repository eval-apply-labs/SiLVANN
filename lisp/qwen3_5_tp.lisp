;; QWEN 3.5 OVER SEVERAL CARDS — TENSOR PARALLEL (NN-87). Read after qwen3_5.lisp when the boot asks for it, and it
;; redefines `layers` and `rows`: a dense model's every layer is split over CARDS cards — each holds its share of every
;; layer's heads and of its MLP's width, and the same columns of the projection back — so each card's mixer and MLP is a
;; partial sum, and the verbs, given the group as their last argument, add the cards' partials over the cards' own memory
;; and hand every card the whole (▶ nn's contracts/objects/partials.cuh). The embedding and the head stay card 0's: card 0
;; embeds, and the group's reduce of its rows with every other card's zeros is every card's residual.
;;
;; What the boot binds besides qwen3_5.lisp's (▶ qwen3_5.py `_bindings`):
;;   CARDS GROUP          how many cards, and their group's number
;;   TP_POS               a node array: the position, set by card 0 before it hands the others their token — a
;;                        prompt chunk's first position, and TP_N its rows
;;   TP_I                 a node array, a cell a card: the layer a card's picture is at — a picture writes no binding
;;   X_OF H_OF ZERO_OF CS_OF MIXER_OF MLP_OF XS_OF H1S_OF MIXER_ROWS_OF MLP_ROWS_OF
;;                        a card's own buffers and tables, an array indexed by its worker: a picture names them through
;;                        (sys__worker), as it cannot call a procedure


;; ── card 0: its layers, the others handed theirs first and collected last ───────────────────────────────────────────
(if TIERED
    (defun (mixer_tp i pos into)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet x (nth MIXER i) into x GROUP)
          (ai_qwen_3__attention_tiered x (nth MIXER i) pos into x GROUP)))
    (defun (mixer_tp i pos into)
      (if (nth DELTANET i)
          (ai_qwen_3__deltanet x (nth MIXER i) into x GROUP)
          (ai_qwen_3__attention x (nth MIXER i) pos into x GROUP))))

(defun (layers pos)
  (let ((i 0) (s 1))
    (sys__node_array__set TP_POS 0 pos)
    (while '(< s CARDS) '(begin (sys__compute s tp_names tp_card) (set! s (+ s 1))))
    (nn__partials__reduce GROUP x x HIDDEN)
    (while '(< i RUN) '(begin (mixer_tp i pos h) (nn__mlp__apply h (nth MLP i) x h GROUP) (set! i (+ i 1))))
    (set! s 1)
    (while '(< s CARDS) '(begin (sys__result s) (set! s (+ s 1))))
    (type x)))


;; ── every other card: its angles, its residual from card 0, its layers — one picture, the same for each card ─────────
(picture tp_card
  (begin
    (nn__rope__angles (nth CS_OF (sys__worker)) (nth TP_POS 0) THETA ROTARY)
    (nn__partials__reduce GROUP (nth ZERO_OF (sys__worker)) (nth X_OF (sys__worker)) HIDDEN)
    (sys__node_array__set TP_I (sys__worker) 0)
    (while '(< (nth TP_I (sys__worker)) RUN)
           '(begin
              (if (nth DELTANET (nth TP_I (sys__worker)))
                  (ai_qwen_3__deltanet (nth X_OF (sys__worker)) (nth (nth MIXER_OF (sys__worker)) (nth TP_I (sys__worker)))
                                       (nth H_OF (sys__worker)) (nth X_OF (sys__worker)) GROUP)
                  (if TIERED
                      (ai_qwen_3__attention_tiered (nth X_OF (sys__worker)) (nth (nth MIXER_OF (sys__worker)) (nth TP_I (sys__worker)))
                                                   (nth TP_POS 0) (nth H_OF (sys__worker)) (nth X_OF (sys__worker)) GROUP)
                      (ai_qwen_3__attention (nth X_OF (sys__worker)) (nth (nth MIXER_OF (sys__worker)) (nth TP_I (sys__worker)))
                                            (nth TP_POS 0) (nth H_OF (sys__worker)) (nth X_OF (sys__worker)) GROUP)))
              (nn__mlp__apply (nth H_OF (sys__worker)) (nth (nth MLP_OF (sys__worker)) (nth TP_I (sys__worker)))
                              (nth X_OF (sys__worker)) (nth H_OF (sys__worker)) GROUP)
              (sys__node_array__set TP_I (sys__worker) (+ (nth TP_I (sys__worker)) 1))))
    (type x)))

;; ── a prompt's chunk as rows: card 0 embeds them, hands every other card its rows, and runs its own ────────────────────
;; ⛳ a chunk's rows are at most the hot ring of a tiered cache (▶ qwen3_5.py `prefill`), as on one card
(defun (mixer_rows_tp i first n)
  (if (nth DELTANET i)
      (ai_qwen_3__deltanet_rows xs (nth MIXER_ROWS i) h1s n GROUP)
      (if TIERED
          (ai_qwen_3__attention_tiered_rows xs (nth MIXER_ROWS i) first h1s n GROUP)
          (ai_qwen_3__attention_rows xs (nth MIXER_ROWS i) first h1s n GROUP))))

(defun (rows first n)
  (let ((i 0) (s 1))
    (sys__node_array__set TP_POS 0 first)
    (sys__node_array__set TP_N 0 n)
    (while '(< s CARDS) '(begin (sys__compute s tp_names tp_rows_card) (set! s (+ s 1))))
    (nn__partials__reduce GROUP xs xs (* n HIDDEN))
    (while '(< i RUN) '(begin (mixer_rows_tp i first n) (nn__mlp__rows h1s (nth MLP_ROWS i) xs n GROUP) (set! i (+ i 1))))
    (set! s 1)
    (while '(< s CARDS) '(begin (sys__result s) (set! s (+ s 1))))
    (type xs)))

;; every other card: zeros where card 0 has the chunk's embeddings, the group's reduce of them its rows, then its layers
(picture tp_rows_card
  (begin
    (nn__vector__zero (nth XS_OF (sys__worker)) (* (nth TP_N 0) HIDDEN))
    (nn__partials__reduce GROUP (nth XS_OF (sys__worker)) (nth XS_OF (sys__worker)) (* (nth TP_N 0) HIDDEN))
    (sys__node_array__set TP_I (sys__worker) 0)
    (while '(< (nth TP_I (sys__worker)) RUN)
           '(begin
              (if (nth DELTANET (nth TP_I (sys__worker)))
                  (ai_qwen_3__deltanet_rows (nth XS_OF (sys__worker)) (nth (nth MIXER_ROWS_OF (sys__worker)) (nth TP_I (sys__worker)))
                                            (nth H1S_OF (sys__worker)) (nth TP_N 0) GROUP)
                  (if TIERED
                      (ai_qwen_3__attention_tiered_rows (nth XS_OF (sys__worker)) (nth (nth MIXER_ROWS_OF (sys__worker)) (nth TP_I (sys__worker)))
                                                        (nth TP_POS 0) (nth H1S_OF (sys__worker)) (nth TP_N 0) GROUP)
                      (ai_qwen_3__attention_rows (nth XS_OF (sys__worker)) (nth (nth MIXER_ROWS_OF (sys__worker)) (nth TP_I (sys__worker)))
                                                 (nth TP_POS 0) (nth H1S_OF (sys__worker)) (nth TP_N 0) GROUP)))
              (nn__mlp__rows (nth H1S_OF (sys__worker)) (nth (nth MLP_ROWS_OF (sys__worker)) (nth TP_I (sys__worker)))
                             (nth XS_OF (sys__worker)) (nth TP_N 0) GROUP)
              (sys__node_array__set TP_I (sys__worker) (+ (nth TP_I (sys__worker)) 1))))
    (type x)))

;; what a card's picture sees: the bindings as they stand
(view tp_names)
