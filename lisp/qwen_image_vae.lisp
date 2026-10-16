;; QWEN-IMAGE 2.1's PICTURE DECODER (its VAE) — a picture's latents to its pixels, on the card.
;;
;; `(vae_decode h w)` over latents of h x w (a picture's sixteenth a side) already in `z`: the latents' convolution and
;; the decoder's first, the middle — a resnet, the attention, a resnet — then each level's resnets, every level but the
;; last widened to the next resolution, and the last convolution into `T`, the picture's pixels (four channels, RGBA).
;; Each convolution is `ai_qwen_image__conv` over its table; the middle's attention is `ai_qwen_image__vae_attention`.
;;
;; Four picture-sized planes carry it: X the layer's input, Y and T a resnet's, D the next resolution's. A resnet that
;; keeps its width runs in place (its second convolution adds onto its input); one that changes it writes its shortcut
;; into the other plane and adds onto that, and the two planes change places. A level's widening is made first, into D,
;; so its resnets run in place and its upsampling convolution adds onto it; then D is the level's output.
;;
;; What the boot binds before this file is read (▶ qwen_image_vae.py `CardVae._tables`):
;;   the planes       z X Y T D
;;   the tables       VAE_post_quant VAE_conv_in VAE_conv_out VAE_attn; the middle's resnets VAE_mid_r0_c1 … _c2,
;;                    VAE_mid_r1_c1 … _c2 — the cells each verb reads are named in ai_qwen_image's contracts/objects/vae.cuh
;;   the levels       LEVELS, RESNETS (each level's); level i's resnet r is entry i·RESNETS + r of UP_C1, UP_C2 and
;;                    UP_SHORT (its shortcut's table, where UP_HAS_SHORT says it has one); UP_DUP and UP_UP a level's
;;                    widening and upsampling (every level but the last)

;; ── a resnet over `a`: in place, its output in `a`; with a shortcut, into `b` ──────────────────────────────────────
(defun (vae_resnet a tmp c1 c2 h w)
  (begin (ai_qwen_image__conv a tmp c1 h w)
         (ai_qwen_image__conv tmp a c2 h w)))

(defun (vae_resnet_short a b tmp short c1 c2 h w)
  (begin (ai_qwen_image__conv a b short h w)
         (ai_qwen_image__conv a tmp c1 h w)
         (ai_qwen_image__conv tmp b c2 h w)))

;; ── the decode ──────────────────────────────────────────────────────────────────────────────────────────────────
(defun (vae_decode h w)
  (let ((x X) (y Y) (d D) (s X) (hh h) (ww w) (i 0) (r 0) (k 0))
    (ai_qwen_image__conv z y VAE_post_quant h w)
    (ai_qwen_image__conv y x VAE_conv_in h w)
    (vae_resnet x T VAE_mid_r0_c1 VAE_mid_r0_c2 h w)
    (ai_qwen_image__vae_attention x VAE_attn h w)
    (vae_resnet x T VAE_mid_r1_c1 VAE_mid_r1_c2 h w)
    (while '(< i LEVELS)
           '(begin (if (< i (- LEVELS 1)) (ai_qwen_image__conv x d (nth UP_DUP i) hh ww) (type x))
                   (set! r 0)
                   (while '(< r RESNETS)
                          '(begin (set! k (+ (* i RESNETS) r))
                                  (if (nth UP_HAS_SHORT k)
                                      (begin (vae_resnet_short x y T (nth UP_SHORT k) (nth UP_C1 k) (nth UP_C2 k) hh ww)
                                             (set! s x) (set! x y) (set! y s))
                                      (vae_resnet x T (nth UP_C1 k) (nth UP_C2 k) hh ww))
                                  (set! r (+ r 1))))
                   (if (< i (- LEVELS 1))
                       (begin (ai_qwen_image__conv x d (nth UP_UP i) hh ww)
                              (set! s x) (set! x d) (set! d s)
                              (set! hh (* 2 hh)) (set! ww (* 2 ww)))
                       (type x))
                   (set! i (+ i 1))))
    (ai_qwen_image__conv x T VAE_conv_out hh ww)
    (type T)))
