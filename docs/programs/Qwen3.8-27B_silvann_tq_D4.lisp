;; qwen38_27b_tq4 (qwen3_5_text) — the procedures silvann_runtime/qwen3_5.py wrote at this boot
;; for max_context 100000, options {} — other settings write other tables and procedures
;; 134 programs, in the order they were defined

(defun (embed codes scale) (let
    ((u
        (nn__buffer__getnew 10240))
      (hu
        (nn__buffer__getnew 10240)))
    (nn__turboquant__decode
      (nn__vector__range emb_codes codes 2560)
      (nn__vector__range emb_scales scale 1) u 1 5120 8)
    (nn__hadamard__rotate u ones512 hu 5120)
    (nn__vector__pointwise_mul hu signs_row x 5120)
    (type x)))

(defun (embed_row codes scale at) (let
    ((u
        (nn__buffer__getnew 10240))
      (hu
        (nn__buffer__getnew 10240)))
    (nn__turboquant__decode
      (nn__vector__range emb_codes codes 2560)
      (nn__vector__range emb_scales scale 1) u 1 5120 8)
    (nn__hadamard__rotate u ones512 hu 5120)
    (nn__vector__pointwise_mul hu signs_row
      (nn__vector__range xs at 5120) 5120)
    (type xs)))

(defun (rows0 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr0 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr0 xs n)
    (type xs)))

(defun (rows1 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr1 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr1 xs n)
    (type xs)))

(defun (rows2 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr2 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr2 xs n)
    (type xs)))

(defun (rows3 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr3 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr3 xs n)
    (type xs)))

(defun (rows4 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr4 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr4 xs n)
    (type xs)))

(defun (rows5 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr5 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr5 xs n)
    (type xs)))

(defun (rows6 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr6 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr6 xs n)
    (type xs)))

(defun (rows7 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr7 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr7 xs n)
    (type xs)))

(defun (rows8 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr8 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr8 xs n)
    (type xs)))

(defun (rows9 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr9 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr9 xs n)
    (type xs)))

(defun (rows10 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr10 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr10 xs n)
    (type xs)))

(defun (rows11 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr11 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr11 xs n)
    (type xs)))

(defun (rows12 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr12 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr12 xs n)
    (type xs)))

(defun (rows13 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr13 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr13 xs n)
    (type xs)))

(defun (rows14 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr14 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr14 xs n)
    (type xs)))

(defun (rows15 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr15 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr15 xs n)
    (type xs)))

(defun (rows16 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr16 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr16 xs n)
    (type xs)))

(defun (rows17 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr17 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr17 xs n)
    (type xs)))

(defun (rows18 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr18 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr18 xs n)
    (type xs)))

(defun (rows19 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr19 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr19 xs n)
    (type xs)))

(defun (rows20 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr20 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr20 xs n)
    (type xs)))

(defun (rows21 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr21 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr21 xs n)
    (type xs)))

(defun (rows22 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr22 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr22 xs n)
    (type xs)))

(defun (rows23 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr23 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr23 xs n)
    (type xs)))

(defun (rows24 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr24 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr24 xs n)
    (type xs)))

(defun (rows25 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr25 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr25 xs n)
    (type xs)))

(defun (rows26 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr26 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr26 xs n)
    (type xs)))

(defun (rows27 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr27 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr27 xs n)
    (type xs)))

(defun (rows28 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr28 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr28 xs n)
    (type xs)))

(defun (rows29 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr29 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr29 xs n)
    (type xs)))

(defun (rows30 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr30 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr30 xs n)
    (type xs)))

(defun (rows31 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr31 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr31 xs n)
    (type xs)))

(defun (rows32 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr32 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr32 xs n)
    (type xs)))

(defun (rows33 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr33 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr33 xs n)
    (type xs)))

(defun (rows34 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr34 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr34 xs n)
    (type xs)))

(defun (rows35 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr35 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr35 xs n)
    (type xs)))

(defun (rows36 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr36 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr36 xs n)
    (type xs)))

(defun (rows37 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr37 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr37 xs n)
    (type xs)))

(defun (rows38 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr38 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr38 xs n)
    (type xs)))

(defun (rows39 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr39 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr39 xs n)
    (type xs)))

(defun (rows40 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr40 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr40 xs n)
    (type xs)))

(defun (rows41 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr41 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr41 xs n)
    (type xs)))

(defun (rows42 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr42 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr42 xs n)
    (type xs)))

(defun (rows43 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr43 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr43 xs n)
    (type xs)))

(defun (rows44 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr44 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr44 xs n)
    (type xs)))

(defun (rows45 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr45 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr45 xs n)
    (type xs)))

(defun (rows46 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr46 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr46 xs n)
    (type xs)))

(defun (rows47 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr47 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr47 xs n)
    (type xs)))

(defun (rows48 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr48 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr48 xs n)
    (type xs)))

(defun (rows49 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr49 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr49 xs n)
    (type xs)))

(defun (rows50 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr50 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr50 xs n)
    (type xs)))

(defun (rows51 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr51 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr51 xs n)
    (type xs)))

(defun (rows52 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr52 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr52 xs n)
    (type xs)))

(defun (rows53 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr53 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr53 xs n)
    (type xs)))

(defun (rows54 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr54 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr54 xs n)
    (type xs)))

(defun (rows55 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr55 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr55 xs n)
    (type xs)))

(defun (rows56 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr56 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr56 xs n)
    (type xs)))

(defun (rows57 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr57 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr57 xs n)
    (type xs)))

(defun (rows58 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr58 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr58 xs n)
    (type xs)))

(defun (rows59 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr59 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr59 xs n)
    (type xs)))

(defun (rows60 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr60 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr60 xs n)
    (type xs)))

(defun (rows61 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr61 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr61 xs n)
    (type xs)))

(defun (rows62 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr62 h1s n)
    (ai_qwen_3__mlp_rows h1s mlr62 xs n)
    (type xs)))

(defun (rows63 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr63 first h1s n)
    (ai_qwen_3__mlp_rows h1s mlr63 xs n)
    (type xs)))

(defun (rows first n) (begin
    (rows0 first n)
    (rows1 first n)
    (rows2 first n)
    (rows3 first n)
    (rows4 first n)
    (rows5 first n)
    (rows6 first n)
    (rows7 first n)
    (rows8 first n)
    (rows9 first n)
    (rows10 first n)
    (rows11 first n)
    (rows12 first n)
    (rows13 first n)
    (rows14 first n)
    (rows15 first n)
    (rows16 first n)
    (rows17 first n)
    (rows18 first n)
    (rows19 first n)
    (rows20 first n)
    (rows21 first n)
    (rows22 first n)
    (rows23 first n)
    (rows24 first n)
    (rows25 first n)
    (rows26 first n)
    (rows27 first n)
    (rows28 first n)
    (rows29 first n)
    (rows30 first n)
    (rows31 first n)
    (rows32 first n)
    (rows33 first n)
    (rows34 first n)
    (rows35 first n)
    (rows36 first n)
    (rows37 first n)
    (rows38 first n)
    (rows39 first n)
    (rows40 first n)
    (rows41 first n)
    (rows42 first n)
    (rows43 first n)
    (rows44 first n)
    (rows45 first n)
    (rows46 first n)
    (rows47 first n)
    (rows48 first n)
    (rows49 first n)
    (rows50 first n)
    (rows51 first n)
    (rows52 first n)
    (rows53 first n)
    (rows54 first n)
    (rows55 first n)
    (rows56 first n)
    (rows57 first n)
    (rows58 first n)
    (rows59 first n)
    (rows60 first n)
    (rows61 first n)
    (rows62 first n)
    (rows63 first n)
    (type xs)))

(defun (head) (begin
    (nn__rmsnorm__apply x fnw h 5120)
    (nn__hadamard__rotate h signs h1 5120)
    (nn__turboquant__gemv
      (nn__expert__plane 0 20 0 0 1271398400)
      (nn__expert__plane 0 20 0 1271398400 496640) h1 logits_h 248320 5120 8)
    (nn__argmax__find logits_h res 248320)
    (nn__buffer__read res)))

(defun (head_logits) (begin
    (nn__rmsnorm__apply x fnw h 5120)
    (nn__hadamard__rotate h signs h1 5120)
    (nn__turboquant__gemv
      (nn__expert__plane 0 20 0 0 1271398400)
      (nn__expert__plane 0 20 0 1271398400 496640) h1 logits_h 248320 5120 8)
    (type x)))

(defun (layer0 pos) (begin
    (ai_qwen_3__deltanet x mx0 h)
    (ai_qwen_3__mlp h ml0 x)
    (type x)))

(defun (layer1 pos) (begin
    (ai_qwen_3__deltanet x mx1 h)
    (ai_qwen_3__mlp h ml1 x)
    (type x)))

(defun (layer2 pos) (begin
    (ai_qwen_3__deltanet x mx2 h)
    (ai_qwen_3__mlp h ml2 x)
    (type x)))

(defun (layer3 pos) (begin
    (ai_qwen_3__attention_tiered x mx3 pos h)
    (ai_qwen_3__mlp h ml3 x)
    (type x)))

(defun (layer4 pos) (begin
    (ai_qwen_3__deltanet x mx4 h)
    (ai_qwen_3__mlp h ml4 x)
    (type x)))

(defun (layer5 pos) (begin
    (ai_qwen_3__deltanet x mx5 h)
    (ai_qwen_3__mlp h ml5 x)
    (type x)))

(defun (layer6 pos) (begin
    (ai_qwen_3__deltanet x mx6 h)
    (ai_qwen_3__mlp h ml6 x)
    (type x)))

(defun (layer7 pos) (begin
    (ai_qwen_3__attention_tiered x mx7 pos h)
    (ai_qwen_3__mlp h ml7 x)
    (type x)))

(defun (layer8 pos) (begin
    (ai_qwen_3__deltanet x mx8 h)
    (ai_qwen_3__mlp h ml8 x)
    (type x)))

(defun (layer9 pos) (begin
    (ai_qwen_3__deltanet x mx9 h)
    (ai_qwen_3__mlp h ml9 x)
    (type x)))

(defun (layer10 pos) (begin
    (ai_qwen_3__deltanet x mx10 h)
    (ai_qwen_3__mlp h ml10 x)
    (type x)))

(defun (layer11 pos) (begin
    (ai_qwen_3__attention_tiered x mx11 pos h)
    (ai_qwen_3__mlp h ml11 x)
    (type x)))

(defun (layer12 pos) (begin
    (ai_qwen_3__deltanet x mx12 h)
    (ai_qwen_3__mlp h ml12 x)
    (type x)))

(defun (layer13 pos) (begin
    (ai_qwen_3__deltanet x mx13 h)
    (ai_qwen_3__mlp h ml13 x)
    (type x)))

(defun (layer14 pos) (begin
    (ai_qwen_3__deltanet x mx14 h)
    (ai_qwen_3__mlp h ml14 x)
    (type x)))

(defun (layer15 pos) (begin
    (ai_qwen_3__attention_tiered x mx15 pos h)
    (ai_qwen_3__mlp h ml15 x)
    (type x)))

(defun (layer16 pos) (begin
    (ai_qwen_3__deltanet x mx16 h)
    (ai_qwen_3__mlp h ml16 x)
    (type x)))

(defun (layer17 pos) (begin
    (ai_qwen_3__deltanet x mx17 h)
    (ai_qwen_3__mlp h ml17 x)
    (type x)))

(defun (layer18 pos) (begin
    (ai_qwen_3__deltanet x mx18 h)
    (ai_qwen_3__mlp h ml18 x)
    (type x)))

(defun (layer19 pos) (begin
    (ai_qwen_3__attention_tiered x mx19 pos h)
    (ai_qwen_3__mlp h ml19 x)
    (type x)))

(defun (layer20 pos) (begin
    (ai_qwen_3__deltanet x mx20 h)
    (ai_qwen_3__mlp h ml20 x)
    (type x)))

(defun (layer21 pos) (begin
    (ai_qwen_3__deltanet x mx21 h)
    (ai_qwen_3__mlp h ml21 x)
    (type x)))

(defun (layer22 pos) (begin
    (ai_qwen_3__deltanet x mx22 h)
    (ai_qwen_3__mlp h ml22 x)
    (type x)))

(defun (layer23 pos) (begin
    (ai_qwen_3__attention_tiered x mx23 pos h)
    (ai_qwen_3__mlp h ml23 x)
    (type x)))

(defun (layer24 pos) (begin
    (ai_qwen_3__deltanet x mx24 h)
    (ai_qwen_3__mlp h ml24 x)
    (type x)))

(defun (layer25 pos) (begin
    (ai_qwen_3__deltanet x mx25 h)
    (ai_qwen_3__mlp h ml25 x)
    (type x)))

(defun (layer26 pos) (begin
    (ai_qwen_3__deltanet x mx26 h)
    (ai_qwen_3__mlp h ml26 x)
    (type x)))

(defun (layer27 pos) (begin
    (ai_qwen_3__attention_tiered x mx27 pos h)
    (ai_qwen_3__mlp h ml27 x)
    (type x)))

(defun (layer28 pos) (begin
    (ai_qwen_3__deltanet x mx28 h)
    (ai_qwen_3__mlp h ml28 x)
    (type x)))

(defun (layer29 pos) (begin
    (ai_qwen_3__deltanet x mx29 h)
    (ai_qwen_3__mlp h ml29 x)
    (type x)))

(defun (layer30 pos) (begin
    (ai_qwen_3__deltanet x mx30 h)
    (ai_qwen_3__mlp h ml30 x)
    (type x)))

(defun (layer31 pos) (begin
    (ai_qwen_3__attention_tiered x mx31 pos h)
    (ai_qwen_3__mlp h ml31 x)
    (type x)))

(defun (layer32 pos) (begin
    (ai_qwen_3__deltanet x mx32 h)
    (ai_qwen_3__mlp h ml32 x)
    (type x)))

(defun (layer33 pos) (begin
    (ai_qwen_3__deltanet x mx33 h)
    (ai_qwen_3__mlp h ml33 x)
    (type x)))

(defun (layer34 pos) (begin
    (ai_qwen_3__deltanet x mx34 h)
    (ai_qwen_3__mlp h ml34 x)
    (type x)))

(defun (layer35 pos) (begin
    (ai_qwen_3__attention_tiered x mx35 pos h)
    (ai_qwen_3__mlp h ml35 x)
    (type x)))

(defun (layer36 pos) (begin
    (ai_qwen_3__deltanet x mx36 h)
    (ai_qwen_3__mlp h ml36 x)
    (type x)))

(defun (layer37 pos) (begin
    (ai_qwen_3__deltanet x mx37 h)
    (ai_qwen_3__mlp h ml37 x)
    (type x)))

(defun (layer38 pos) (begin
    (ai_qwen_3__deltanet x mx38 h)
    (ai_qwen_3__mlp h ml38 x)
    (type x)))

(defun (layer39 pos) (begin
    (ai_qwen_3__attention_tiered x mx39 pos h)
    (ai_qwen_3__mlp h ml39 x)
    (type x)))

(defun (layer40 pos) (begin
    (ai_qwen_3__deltanet x mx40 h)
    (ai_qwen_3__mlp h ml40 x)
    (type x)))

(defun (layer41 pos) (begin
    (ai_qwen_3__deltanet x mx41 h)
    (ai_qwen_3__mlp h ml41 x)
    (type x)))

(defun (layer42 pos) (begin
    (ai_qwen_3__deltanet x mx42 h)
    (ai_qwen_3__mlp h ml42 x)
    (type x)))

(defun (layer43 pos) (begin
    (ai_qwen_3__attention_tiered x mx43 pos h)
    (ai_qwen_3__mlp h ml43 x)
    (type x)))

(defun (layer44 pos) (begin
    (ai_qwen_3__deltanet x mx44 h)
    (ai_qwen_3__mlp h ml44 x)
    (type x)))

(defun (layer45 pos) (begin
    (ai_qwen_3__deltanet x mx45 h)
    (ai_qwen_3__mlp h ml45 x)
    (type x)))

(defun (layer46 pos) (begin
    (ai_qwen_3__deltanet x mx46 h)
    (ai_qwen_3__mlp h ml46 x)
    (type x)))

(defun (layer47 pos) (begin
    (ai_qwen_3__attention_tiered x mx47 pos h)
    (ai_qwen_3__mlp h ml47 x)
    (type x)))

(defun (layer48 pos) (begin
    (ai_qwen_3__deltanet x mx48 h)
    (ai_qwen_3__mlp h ml48 x)
    (type x)))

(defun (layer49 pos) (begin
    (ai_qwen_3__deltanet x mx49 h)
    (ai_qwen_3__mlp h ml49 x)
    (type x)))

(defun (layer50 pos) (begin
    (ai_qwen_3__deltanet x mx50 h)
    (ai_qwen_3__mlp h ml50 x)
    (type x)))

(defun (layer51 pos) (begin
    (ai_qwen_3__attention_tiered x mx51 pos h)
    (ai_qwen_3__mlp h ml51 x)
    (type x)))

(defun (layer52 pos) (begin
    (ai_qwen_3__deltanet x mx52 h)
    (ai_qwen_3__mlp h ml52 x)
    (type x)))

(defun (layer53 pos) (begin
    (ai_qwen_3__deltanet x mx53 h)
    (ai_qwen_3__mlp h ml53 x)
    (type x)))

(defun (layer54 pos) (begin
    (ai_qwen_3__deltanet x mx54 h)
    (ai_qwen_3__mlp h ml54 x)
    (type x)))

(defun (layer55 pos) (begin
    (ai_qwen_3__attention_tiered x mx55 pos h)
    (ai_qwen_3__mlp h ml55 x)
    (type x)))

(defun (layer56 pos) (begin
    (ai_qwen_3__deltanet x mx56 h)
    (ai_qwen_3__mlp h ml56 x)
    (type x)))

(defun (layer57 pos) (begin
    (ai_qwen_3__deltanet x mx57 h)
    (ai_qwen_3__mlp h ml57 x)
    (type x)))

(defun (layer58 pos) (begin
    (ai_qwen_3__deltanet x mx58 h)
    (ai_qwen_3__mlp h ml58 x)
    (type x)))

(defun (layer59 pos) (begin
    (ai_qwen_3__attention_tiered x mx59 pos h)
    (ai_qwen_3__mlp h ml59 x)
    (type x)))

(defun (layer60 pos) (begin
    (ai_qwen_3__deltanet x mx60 h)
    (ai_qwen_3__mlp h ml60 x)
    (type x)))

(defun (layer61 pos) (begin
    (ai_qwen_3__deltanet x mx61 h)
    (ai_qwen_3__mlp h ml61 x)
    (type x)))

(defun (layer62 pos) (begin
    (ai_qwen_3__deltanet x mx62 h)
    (ai_qwen_3__mlp h ml62 x)
    (type x)))

(defun (layer63 pos) (begin
    (ai_qwen_3__attention_tiered x mx63 pos h)
    (ai_qwen_3__mlp h ml63 x)
    (type x)))

(defun (layers pos) (begin
    (layer0 pos)
    (layer1 pos)
    (layer2 pos)
    (layer3 pos)
    (layer4 pos)
    (layer5 pos)
    (layer6 pos)
    (layer7 pos)
    (layer8 pos)
    (layer9 pos)
    (layer10 pos)
    (layer11 pos)
    (layer12 pos)
    (layer13 pos)
    (layer14 pos)
    (layer15 pos)
    (layer16 pos)
    (layer17 pos)
    (layer18 pos)
    (layer19 pos)
    (layer20 pos)
    (layer21 pos)
    (layer22 pos)
    (layer23 pos)
    (layer24 pos)
    (layer25 pos)
    (layer26 pos)
    (layer27 pos)
    (layer28 pos)
    (layer29 pos)
    (layer30 pos)
    (layer31 pos)
    (layer32 pos)
    (layer33 pos)
    (layer34 pos)
    (layer35 pos)
    (layer36 pos)
    (layer37 pos)
    (layer38 pos)
    (layer39 pos)
    (layer40 pos)
    (layer41 pos)
    (layer42 pos)
    (layer43 pos)
    (layer44 pos)
    (layer45 pos)
    (layer46 pos)
    (layer47 pos)
    (layer48 pos)
    (layer49 pos)
    (layer50 pos)
    (layer51 pos)
    (layer52 pos)
    (layer53 pos)
    (layer54 pos)
    (layer55 pos)
    (layer56 pos)
    (layer57 pos)
    (layer58 pos)
    (layer59 pos)
    (layer60 pos)
    (layer61 pos)
    (layer62 pos)
    (layer63 pos)
    (type x)))

