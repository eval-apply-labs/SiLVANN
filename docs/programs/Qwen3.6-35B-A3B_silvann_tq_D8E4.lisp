;; qwen36_35b_a3b_tq4 (qwen3_5_moe_text) — the procedures silvann_runtime/qwen3_5.py wrote at this boot
;; for max_context 100000, options {} — other settings write other tables and procedures
;; 86 programs, in the order they were defined

(defun (embed codes scale) (let
    ((u
        (nn__buffer__getnew 4096))
      (hu
        (nn__buffer__getnew 4096)))
    (nn__turboquant__decode
      (nn__vector__range emb_codes codes 1024)
      (nn__vector__range emb_scales scale 1) u 1 2048 8)
    (nn__hadamard__rotate u ones512 hu 2048)
    (nn__vector__pointwise_mul hu signs_row x 2048)
    (type x)))

(defun (embed_row codes scale at) (let
    ((u
        (nn__buffer__getnew 4096))
      (hu
        (nn__buffer__getnew 4096)))
    (nn__turboquant__decode
      (nn__vector__range emb_codes codes 1024)
      (nn__vector__range emb_scales scale 1) u 1 2048 8)
    (nn__hadamard__rotate u ones512 hu 2048)
    (nn__vector__pointwise_mul hu signs_row
      (nn__vector__range xs at 2048) 2048)
    (type xs)))

(defun (rows0 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr0 h1s n)
    (ai_qwen_3__moe_rows h1s mr0 xs n)
    (type xs)))

(defun (rows1 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr1 h1s n)
    (ai_qwen_3__moe_rows h1s mr1 xs n)
    (type xs)))

(defun (rows2 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr2 h1s n)
    (ai_qwen_3__moe_rows h1s mr2 xs n)
    (type xs)))

(defun (rows3 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr3 first h1s n)
    (ai_qwen_3__moe_rows h1s mr3 xs n)
    (type xs)))

(defun (rows4 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr4 h1s n)
    (ai_qwen_3__moe_rows h1s mr4 xs n)
    (type xs)))

(defun (rows5 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr5 h1s n)
    (ai_qwen_3__moe_rows h1s mr5 xs n)
    (type xs)))

(defun (rows6 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr6 h1s n)
    (ai_qwen_3__moe_rows h1s mr6 xs n)
    (type xs)))

(defun (rows7 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr7 first h1s n)
    (ai_qwen_3__moe_rows h1s mr7 xs n)
    (type xs)))

(defun (rows8 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr8 h1s n)
    (ai_qwen_3__moe_rows h1s mr8 xs n)
    (type xs)))

(defun (rows9 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr9 h1s n)
    (ai_qwen_3__moe_rows h1s mr9 xs n)
    (type xs)))

(defun (rows10 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr10 h1s n)
    (ai_qwen_3__moe_rows h1s mr10 xs n)
    (type xs)))

(defun (rows11 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr11 first h1s n)
    (ai_qwen_3__moe_rows h1s mr11 xs n)
    (type xs)))

(defun (rows12 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr12 h1s n)
    (ai_qwen_3__moe_rows h1s mr12 xs n)
    (type xs)))

(defun (rows13 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr13 h1s n)
    (ai_qwen_3__moe_rows h1s mr13 xs n)
    (type xs)))

(defun (rows14 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr14 h1s n)
    (ai_qwen_3__moe_rows h1s mr14 xs n)
    (type xs)))

(defun (rows15 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr15 first h1s n)
    (ai_qwen_3__moe_rows h1s mr15 xs n)
    (type xs)))

(defun (rows16 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr16 h1s n)
    (ai_qwen_3__moe_rows h1s mr16 xs n)
    (type xs)))

(defun (rows17 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr17 h1s n)
    (ai_qwen_3__moe_rows h1s mr17 xs n)
    (type xs)))

(defun (rows18 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr18 h1s n)
    (ai_qwen_3__moe_rows h1s mr18 xs n)
    (type xs)))

(defun (rows19 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr19 first h1s n)
    (ai_qwen_3__moe_rows h1s mr19 xs n)
    (type xs)))

(defun (rows20 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr20 h1s n)
    (ai_qwen_3__moe_rows h1s mr20 xs n)
    (type xs)))

(defun (rows21 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr21 h1s n)
    (ai_qwen_3__moe_rows h1s mr21 xs n)
    (type xs)))

(defun (rows22 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr22 h1s n)
    (ai_qwen_3__moe_rows h1s mr22 xs n)
    (type xs)))

(defun (rows23 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr23 first h1s n)
    (ai_qwen_3__moe_rows h1s mr23 xs n)
    (type xs)))

(defun (rows24 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr24 h1s n)
    (ai_qwen_3__moe_rows h1s mr24 xs n)
    (type xs)))

(defun (rows25 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr25 h1s n)
    (ai_qwen_3__moe_rows h1s mr25 xs n)
    (type xs)))

(defun (rows26 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr26 h1s n)
    (ai_qwen_3__moe_rows h1s mr26 xs n)
    (type xs)))

(defun (rows27 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr27 first h1s n)
    (ai_qwen_3__moe_rows h1s mr27 xs n)
    (type xs)))

(defun (rows28 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr28 h1s n)
    (ai_qwen_3__moe_rows h1s mr28 xs n)
    (type xs)))

(defun (rows29 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr29 h1s n)
    (ai_qwen_3__moe_rows h1s mr29 xs n)
    (type xs)))

(defun (rows30 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr30 h1s n)
    (ai_qwen_3__moe_rows h1s mr30 xs n)
    (type xs)))

(defun (rows31 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr31 first h1s n)
    (ai_qwen_3__moe_rows h1s mr31 xs n)
    (type xs)))

(defun (rows32 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr32 h1s n)
    (ai_qwen_3__moe_rows h1s mr32 xs n)
    (type xs)))

(defun (rows33 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr33 h1s n)
    (ai_qwen_3__moe_rows h1s mr33 xs n)
    (type xs)))

(defun (rows34 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr34 h1s n)
    (ai_qwen_3__moe_rows h1s mr34 xs n)
    (type xs)))

(defun (rows35 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr35 first h1s n)
    (ai_qwen_3__moe_rows h1s mr35 xs n)
    (type xs)))

(defun (rows36 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr36 h1s n)
    (ai_qwen_3__moe_rows h1s mr36 xs n)
    (type xs)))

(defun (rows37 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr37 h1s n)
    (ai_qwen_3__moe_rows h1s mr37 xs n)
    (type xs)))

(defun (rows38 first n) (begin
    (ai_qwen_3__deltanet_rows xs dr38 h1s n)
    (ai_qwen_3__moe_rows h1s mr38 xs n)
    (type xs)))

(defun (rows39 first n) (begin
    (ai_qwen_3__attention_tiered_rows xs atr39 first h1s n)
    (ai_qwen_3__moe_rows h1s mr39 xs n)
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
    (type xs)))

(defun (head) (begin
    (nn__rmsnorm__apply x fnw h 2048)
    (nn__hadamard__rotate h signs h1 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 0 23 0 0 508559360)
      (nn__expert__plane 0 23 0 508559360 496640) h1 logits_h 248320 2048 8)
    (nn__argmax__find logits_h res 248320)
    (nn__buffer__read res)))

(defun (head_logits) (begin
    (nn__rmsnorm__apply x fnw h 2048)
    (nn__hadamard__rotate h signs h1 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 0 23 0 0 508559360)
      (nn__expert__plane 0 23 0 508559360 496640) h1 logits_h 248320 2048 8)
    (type x)))

(defun (layer0 pos) (begin
    (ai_qwen_3__deltanet x mx0 h1)
    (ai_qwen_3__moe h1 mr0 x)
    (type x)))

(defun (layer1 pos) (begin
    (ai_qwen_3__deltanet x mx1 h1)
    (ai_qwen_3__moe h1 mr1 x)
    (type x)))

(defun (layer2 pos) (begin
    (ai_qwen_3__deltanet x mx2 h1)
    (ai_qwen_3__moe h1 mr2 x)
    (type x)))

(defun (layer3 pos) (begin
    (ai_qwen_3__attention_tiered x mx3 pos h1)
    (ai_qwen_3__moe h1 mr3 x)
    (type x)))

(defun (layer4 pos) (begin
    (ai_qwen_3__deltanet x mx4 h1)
    (ai_qwen_3__moe h1 mr4 x)
    (type x)))

(defun (layer5 pos) (begin
    (ai_qwen_3__deltanet x mx5 h1)
    (ai_qwen_3__moe h1 mr5 x)
    (type x)))

(defun (layer6 pos) (begin
    (ai_qwen_3__deltanet x mx6 h1)
    (ai_qwen_3__moe h1 mr6 x)
    (type x)))

(defun (layer7 pos) (begin
    (ai_qwen_3__attention_tiered x mx7 pos h1)
    (ai_qwen_3__moe h1 mr7 x)
    (type x)))

(defun (layer8 pos) (begin
    (ai_qwen_3__deltanet x mx8 h1)
    (ai_qwen_3__moe h1 mr8 x)
    (type x)))

(defun (layer9 pos) (begin
    (ai_qwen_3__deltanet x mx9 h1)
    (ai_qwen_3__moe h1 mr9 x)
    (type x)))

(defun (layer10 pos) (begin
    (ai_qwen_3__deltanet x mx10 h1)
    (ai_qwen_3__moe h1 mr10 x)
    (type x)))

(defun (layer11 pos) (begin
    (ai_qwen_3__attention_tiered x mx11 pos h1)
    (ai_qwen_3__moe h1 mr11 x)
    (type x)))

(defun (layer12 pos) (begin
    (ai_qwen_3__deltanet x mx12 h1)
    (ai_qwen_3__moe h1 mr12 x)
    (type x)))

(defun (layer13 pos) (begin
    (ai_qwen_3__deltanet x mx13 h1)
    (ai_qwen_3__moe h1 mr13 x)
    (type x)))

(defun (layer14 pos) (begin
    (ai_qwen_3__deltanet x mx14 h1)
    (ai_qwen_3__moe h1 mr14 x)
    (type x)))

(defun (layer15 pos) (begin
    (ai_qwen_3__attention_tiered x mx15 pos h1)
    (ai_qwen_3__moe h1 mr15 x)
    (type x)))

(defun (layer16 pos) (begin
    (ai_qwen_3__deltanet x mx16 h1)
    (ai_qwen_3__moe h1 mr16 x)
    (type x)))

(defun (layer17 pos) (begin
    (ai_qwen_3__deltanet x mx17 h1)
    (ai_qwen_3__moe h1 mr17 x)
    (type x)))

(defun (layer18 pos) (begin
    (ai_qwen_3__deltanet x mx18 h1)
    (ai_qwen_3__moe h1 mr18 x)
    (type x)))

(defun (layer19 pos) (begin
    (ai_qwen_3__attention_tiered x mx19 pos h1)
    (ai_qwen_3__moe h1 mr19 x)
    (type x)))

(defun (layer20 pos) (begin
    (ai_qwen_3__deltanet x mx20 h1)
    (ai_qwen_3__moe h1 mr20 x)
    (type x)))

(defun (layer21 pos) (begin
    (ai_qwen_3__deltanet x mx21 h1)
    (ai_qwen_3__moe h1 mr21 x)
    (type x)))

(defun (layer22 pos) (begin
    (ai_qwen_3__deltanet x mx22 h1)
    (ai_qwen_3__moe h1 mr22 x)
    (type x)))

(defun (layer23 pos) (begin
    (ai_qwen_3__attention_tiered x mx23 pos h1)
    (ai_qwen_3__moe h1 mr23 x)
    (type x)))

(defun (layer24 pos) (begin
    (ai_qwen_3__deltanet x mx24 h1)
    (ai_qwen_3__moe h1 mr24 x)
    (type x)))

(defun (layer25 pos) (begin
    (ai_qwen_3__deltanet x mx25 h1)
    (ai_qwen_3__moe h1 mr25 x)
    (type x)))

(defun (layer26 pos) (begin
    (ai_qwen_3__deltanet x mx26 h1)
    (ai_qwen_3__moe h1 mr26 x)
    (type x)))

(defun (layer27 pos) (begin
    (ai_qwen_3__attention_tiered x mx27 pos h1)
    (ai_qwen_3__moe h1 mr27 x)
    (type x)))

(defun (layer28 pos) (begin
    (ai_qwen_3__deltanet x mx28 h1)
    (ai_qwen_3__moe h1 mr28 x)
    (type x)))

(defun (layer29 pos) (begin
    (ai_qwen_3__deltanet x mx29 h1)
    (ai_qwen_3__moe h1 mr29 x)
    (type x)))

(defun (layer30 pos) (begin
    (ai_qwen_3__deltanet x mx30 h1)
    (ai_qwen_3__moe h1 mr30 x)
    (type x)))

(defun (layer31 pos) (begin
    (ai_qwen_3__attention_tiered x mx31 pos h1)
    (ai_qwen_3__moe h1 mr31 x)
    (type x)))

(defun (layer32 pos) (begin
    (ai_qwen_3__deltanet x mx32 h1)
    (ai_qwen_3__moe h1 mr32 x)
    (type x)))

(defun (layer33 pos) (begin
    (ai_qwen_3__deltanet x mx33 h1)
    (ai_qwen_3__moe h1 mr33 x)
    (type x)))

(defun (layer34 pos) (begin
    (ai_qwen_3__deltanet x mx34 h1)
    (ai_qwen_3__moe h1 mr34 x)
    (type x)))

(defun (layer35 pos) (begin
    (ai_qwen_3__attention_tiered x mx35 pos h1)
    (ai_qwen_3__moe h1 mr35 x)
    (type x)))

(defun (layer36 pos) (begin
    (ai_qwen_3__deltanet x mx36 h1)
    (ai_qwen_3__moe h1 mr36 x)
    (type x)))

(defun (layer37 pos) (begin
    (ai_qwen_3__deltanet x mx37 h1)
    (ai_qwen_3__moe h1 mr37 x)
    (type x)))

(defun (layer38 pos) (begin
    (ai_qwen_3__deltanet x mx38 h1)
    (ai_qwen_3__moe h1 mr38 x)
    (type x)))

(defun (layer39 pos) (begin
    (ai_qwen_3__attention_tiered x mx39 pos h1)
    (ai_qwen_3__moe h1 mr39 x)
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
    (type x)))

