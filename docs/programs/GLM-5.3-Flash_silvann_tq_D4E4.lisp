;; glm53_flash_tq4_v3 (glm5_next_text) — the procedures silvann_runtime/glm5.py wrote at this boot
;; for max_context 100000, options {"experts_from": "disk"} — other settings write other tables and procedures
;; 434 programs, in the order they were defined

(defun (embed codes scale) (begin
    (nn__turboquant__decode
      (nn__vector__range emb_codes codes 2048)
      (nn__vector__range emb_scales scale 1) u 1 4096 8)
    (nn__hadamard__rotate u ones512 hu 4096)
    (nn__vector__pointwise_mul hu signs_row e 4096)
    (nn__vector__add e zero_h
      (nn__vector__range streams 0 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range streams 4096 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range streams 8192 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range streams 12288 4096) 4096)
    (type e)))

(defun (head) (begin
    (nn__vector__add
      (nn__vector__range streams 0 4096)
      (nn__vector__range streams 4096 4096) hm 4096)
    (nn__vector__add hm
      (nn__vector__range streams 8192 4096) hm 4096)
    (nn__vector__add hm
      (nn__vector__range streams 12288 4096) hm 4096)
    (nn__vector__scale hm 0.25 hm 4096)
    (nn__rmsnorm__apply hm fnw hn 4096)
    (nn__hadamard__rotate hn signs hnr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 0 29 0 0 634388480)
      (nn__expert__plane 0 29 0 634388480 309760) hnr logits_h 154880 4096 8)
    (nn__argmax__find logits_h res 154880)
    (nn__buffer__read res)))

(defun (head_logits) (begin
    (nn__vector__add
      (nn__vector__range streams 0 4096)
      (nn__vector__range streams 4096 4096) hm 4096)
    (nn__vector__add hm
      (nn__vector__range streams 8192 4096) hm 4096)
    (nn__vector__add hm
      (nn__vector__range streams 12288 4096) hm 4096)
    (nn__vector__scale hm 0.25 hm 4096)
    (nn__rmsnorm__apply hm fnw hn 4096)
    (nn__hadamard__rotate hn signs hnr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 0 29 0 0 634388480)
      (nn__expert__plane 0 29 0 634388480 309760) hnr logits_h 154880 4096 8)
    (type hm)))

(defun (a0 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 0 1 0 0 786432)
      (nn__expert__plane 0 0 0 0 48)
      (nn__expert__plane 0 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 0 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 0 16 1 0 16777216)
      (nn__expert__plane 0 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 0 16 0 0 16777216)
      (nn__expert__plane 0 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 0 16 2 0 16777216)
      (nn__expert__plane 0 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 0 15 1 0 65536)
      (nn__expert__plane 0 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 0 15 0 0 65536)
      (nn__expert__plane 0 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 0 15 2 0 65536)
      (nn__expert__plane 0 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 0 13 0 0 262144)
      (nn__expert__plane 0 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 0 14 0 0 524288)
      (nn__expert__plane 0 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 0 11 0 0 131072)
      (nn__expert__plane 0 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 0 13 1 0 262144)
      (nn__expert__plane 0 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 0 14 1 0 524288)
      (nn__expert__plane 0 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 0 30 0 0 4194304) conved fb
      (nn__expert__plane 0 12 0 0 16384)
      (nn__expert__plane 0 10 0 0 128) gate
      (nn__expert__plane 0 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 0 18 0 0 16777216)
      (nn__expert__plane 0 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 0 1 1 0 786432)
      (nn__expert__plane 0 0 1 0 48)
      (nn__expert__plane 0 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 0 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 0 7 0 0 25165824)
      (nn__expert__plane 0 7 0 25165824 24576) hr
      (nn__vector__range gu 0 12288) 12288 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 0 7 1 0 25165824)
      (nn__expert__plane 0 7 1 25165824 24576) hr
      (nn__vector__range gu 12288 12288) 12288 4096 4)
    (nn__swiglu__clamped gu act 12288 10.0)
    (nn__hadamard__rotate act signs actr 12288)
    (nn__turboquant__gemv
      (nn__expert__plane 0 4 0 0 25165824)
      (nn__expert__plane 0 4 0 25165824 8192) actr y 4096 12288 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a1 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 1 1 0 0 786432)
      (nn__expert__plane 1 0 0 0 48)
      (nn__expert__plane 1 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 1 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 1 16 1 0 16777216)
      (nn__expert__plane 1 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 1 16 0 0 16777216)
      (nn__expert__plane 1 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 1 16 2 0 16777216)
      (nn__expert__plane 1 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 1 15 1 0 65536)
      (nn__expert__plane 1 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 1 15 0 0 65536)
      (nn__expert__plane 1 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 1 15 2 0 65536)
      (nn__expert__plane 1 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 1 13 0 0 262144)
      (nn__expert__plane 1 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 1 14 0 0 524288)
      (nn__expert__plane 1 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 1 11 0 0 131072)
      (nn__expert__plane 1 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 1 13 1 0 262144)
      (nn__expert__plane 1 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 1 14 1 0 524288)
      (nn__expert__plane 1 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 1 30 0 0 4194304) conved fb
      (nn__expert__plane 1 12 0 0 16384)
      (nn__expert__plane 1 10 0 0 128) gate
      (nn__expert__plane 1 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 1 18 0 0 16777216)
      (nn__expert__plane 1 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 1 1 1 0 786432)
      (nn__expert__plane 1 0 1 0 48)
      (nn__expert__plane 1 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 1 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 1 7 0 0 25165824)
      (nn__expert__plane 1 7 0 25165824 24576) hr
      (nn__vector__range gu 0 12288) 12288 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 1 7 1 0 25165824)
      (nn__expert__plane 1 7 1 25165824 24576) hr
      (nn__vector__range gu 12288 12288) 12288 4096 4)
    (nn__swiglu__clamped gu act 12288 10.0)
    (nn__hadamard__rotate act signs actr 12288)
    (nn__turboquant__gemv
      (nn__expert__plane 1 4 0 0 25165824)
      (nn__expert__plane 1 4 0 25165824 8192) actr y 4096 12288 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a2 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 2 1 0 0 786432)
      (nn__expert__plane 2 0 0 0 48)
      (nn__expert__plane 2 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 2 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 2 16 1 0 16777216)
      (nn__expert__plane 2 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 2 16 0 0 16777216)
      (nn__expert__plane 2 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 2 16 2 0 16777216)
      (nn__expert__plane 2 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 2 15 1 0 65536)
      (nn__expert__plane 2 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 2 15 0 0 65536)
      (nn__expert__plane 2 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 2 15 2 0 65536)
      (nn__expert__plane 2 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 2 13 0 0 262144)
      (nn__expert__plane 2 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 2 14 0 0 524288)
      (nn__expert__plane 2 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 2 11 0 0 131072)
      (nn__expert__plane 2 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 2 13 1 0 262144)
      (nn__expert__plane 2 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 2 14 1 0 524288)
      (nn__expert__plane 2 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 2 30 0 0 4194304) conved fb
      (nn__expert__plane 2 12 0 0 16384)
      (nn__expert__plane 2 10 0 0 128) gate
      (nn__expert__plane 2 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 2 18 0 0 16777216)
      (nn__expert__plane 2 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 2 1 1 0 786432)
      (nn__expert__plane 2 0 1 0 48)
      (nn__expert__plane 2 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 2 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 2 7 0 0 25165824)
      (nn__expert__plane 2 7 0 25165824 24576) hr
      (nn__vector__range gu 0 12288) 12288 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 2 7 1 0 25165824)
      (nn__expert__plane 2 7 1 25165824 24576) hr
      (nn__vector__range gu 12288 12288) 12288 4096 4)
    (nn__swiglu__clamped gu act 12288 10.0)
    (nn__hadamard__rotate act signs actr 12288)
    (nn__turboquant__gemv
      (nn__expert__plane 2 4 0 0 25165824)
      (nn__expert__plane 2 4 0 25165824 8192) actr y 4096 12288 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a3 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 3 1 2 0 786432)
      (nn__expert__plane 3 0 2 0 48)
      (nn__expert__plane 3 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 3 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 3 27 0 0 3145728)
      (nn__expert__plane 3 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 3 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 3 28 0 0 12582912)
      (nn__expert__plane 3 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 3 23 0 0 1048576)
      (nn__expert__plane 3 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 3 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 3 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 3 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 3 31 0 0 102400000)
      (nn__expert__plane 3 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 3 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 3 25 0 0 33554432)
      (nn__expert__plane 3 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 3 1 3 0 786432)
      (nn__expert__plane 3 0 3 0 48)
      (nn__expert__plane 3 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 3 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 3 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 3 5 1 0 576) 2.5) 
    (type streams)))

(defun (s3) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 3 9 2 0 4194304)
      (nn__expert__plane 3 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 3 9 3 0 4194304)
      (nn__expert__plane 3 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 3 8 1 0 4194304)
      (nn__expert__plane 3 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c3) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a4 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 4 1 0 0 786432)
      (nn__expert__plane 4 0 0 0 48)
      (nn__expert__plane 4 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 4 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 4 16 1 0 16777216)
      (nn__expert__plane 4 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 4 16 0 0 16777216)
      (nn__expert__plane 4 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 4 16 2 0 16777216)
      (nn__expert__plane 4 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 4 15 1 0 65536)
      (nn__expert__plane 4 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 4 15 0 0 65536)
      (nn__expert__plane 4 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 4 15 2 0 65536)
      (nn__expert__plane 4 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 4 13 0 0 262144)
      (nn__expert__plane 4 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 4 14 0 0 524288)
      (nn__expert__plane 4 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 4 11 0 0 131072)
      (nn__expert__plane 4 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 4 13 1 0 262144)
      (nn__expert__plane 4 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 4 14 1 0 524288)
      (nn__expert__plane 4 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 4 30 0 0 4194304) conved fb
      (nn__expert__plane 4 12 0 0 16384)
      (nn__expert__plane 4 10 0 0 128) gate
      (nn__expert__plane 4 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 4 18 0 0 16777216)
      (nn__expert__plane 4 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 4 1 1 0 786432)
      (nn__expert__plane 4 0 1 0 48)
      (nn__expert__plane 4 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 4 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 4 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 4 5 0 0 576) 2.5) 
    (type streams)))

(defun (s4) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 4 9 0 0 4194304)
      (nn__expert__plane 4 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 4 9 1 0 4194304)
      (nn__expert__plane 4 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 4 8 0 0 4194304)
      (nn__expert__plane 4 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c4) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a5 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 5 1 0 0 786432)
      (nn__expert__plane 5 0 0 0 48)
      (nn__expert__plane 5 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 5 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 5 16 1 0 16777216)
      (nn__expert__plane 5 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 5 16 0 0 16777216)
      (nn__expert__plane 5 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 5 16 2 0 16777216)
      (nn__expert__plane 5 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 5 15 1 0 65536)
      (nn__expert__plane 5 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 5 15 0 0 65536)
      (nn__expert__plane 5 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 5 15 2 0 65536)
      (nn__expert__plane 5 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 5 13 0 0 262144)
      (nn__expert__plane 5 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 5 14 0 0 524288)
      (nn__expert__plane 5 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 5 11 0 0 131072)
      (nn__expert__plane 5 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 5 13 1 0 262144)
      (nn__expert__plane 5 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 5 14 1 0 524288)
      (nn__expert__plane 5 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 5 30 0 0 4194304) conved fb
      (nn__expert__plane 5 12 0 0 16384)
      (nn__expert__plane 5 10 0 0 128) gate
      (nn__expert__plane 5 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 5 18 0 0 16777216)
      (nn__expert__plane 5 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 5 1 1 0 786432)
      (nn__expert__plane 5 0 1 0 48)
      (nn__expert__plane 5 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 5 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 5 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 5 5 0 0 576) 2.5) 
    (type streams)))

(defun (s5) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 5 9 0 0 4194304)
      (nn__expert__plane 5 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 5 9 1 0 4194304)
      (nn__expert__plane 5 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 5 8 0 0 4194304)
      (nn__expert__plane 5 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c5) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a6 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 6 1 0 0 786432)
      (nn__expert__plane 6 0 0 0 48)
      (nn__expert__plane 6 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 6 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 6 16 1 0 16777216)
      (nn__expert__plane 6 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 6 16 0 0 16777216)
      (nn__expert__plane 6 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 6 16 2 0 16777216)
      (nn__expert__plane 6 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 6 15 1 0 65536)
      (nn__expert__plane 6 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 6 15 0 0 65536)
      (nn__expert__plane 6 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 6 15 2 0 65536)
      (nn__expert__plane 6 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 6 13 0 0 262144)
      (nn__expert__plane 6 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 6 14 0 0 524288)
      (nn__expert__plane 6 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 6 11 0 0 131072)
      (nn__expert__plane 6 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 6 13 1 0 262144)
      (nn__expert__plane 6 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 6 14 1 0 524288)
      (nn__expert__plane 6 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 6 30 0 0 4194304) conved fb
      (nn__expert__plane 6 12 0 0 16384)
      (nn__expert__plane 6 10 0 0 128) gate
      (nn__expert__plane 6 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 6 18 0 0 16777216)
      (nn__expert__plane 6 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 6 1 1 0 786432)
      (nn__expert__plane 6 0 1 0 48)
      (nn__expert__plane 6 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 6 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 6 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 6 5 0 0 576) 2.5) 
    (type streams)))

(defun (s6) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 6 9 0 0 4194304)
      (nn__expert__plane 6 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 6 9 1 0 4194304)
      (nn__expert__plane 6 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 6 8 0 0 4194304)
      (nn__expert__plane 6 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c6) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a7 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 7 1 2 0 786432)
      (nn__expert__plane 7 0 2 0 48)
      (nn__expert__plane 7 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 7 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 7 27 0 0 3145728)
      (nn__expert__plane 7 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 7 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 7 28 0 0 12582912)
      (nn__expert__plane 7 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 7 23 0 0 1048576)
      (nn__expert__plane 7 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 7 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 7 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 7 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 7 31 0 0 102400000)
      (nn__expert__plane 7 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 7 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 7 25 0 0 33554432)
      (nn__expert__plane 7 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 7 1 3 0 786432)
      (nn__expert__plane 7 0 3 0 48)
      (nn__expert__plane 7 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 7 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 7 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 7 5 1 0 576) 2.5) 
    (type streams)))

(defun (s7) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 7 9 2 0 4194304)
      (nn__expert__plane 7 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 7 9 3 0 4194304)
      (nn__expert__plane 7 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 7 8 1 0 4194304)
      (nn__expert__plane 7 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c7) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a8 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 8 1 0 0 786432)
      (nn__expert__plane 8 0 0 0 48)
      (nn__expert__plane 8 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 8 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 8 16 1 0 16777216)
      (nn__expert__plane 8 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 8 16 0 0 16777216)
      (nn__expert__plane 8 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 8 16 2 0 16777216)
      (nn__expert__plane 8 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 8 15 1 0 65536)
      (nn__expert__plane 8 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 8 15 0 0 65536)
      (nn__expert__plane 8 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 8 15 2 0 65536)
      (nn__expert__plane 8 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 8 13 0 0 262144)
      (nn__expert__plane 8 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 8 14 0 0 524288)
      (nn__expert__plane 8 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 8 11 0 0 131072)
      (nn__expert__plane 8 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 8 13 1 0 262144)
      (nn__expert__plane 8 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 8 14 1 0 524288)
      (nn__expert__plane 8 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 8 30 0 0 4194304) conved fb
      (nn__expert__plane 8 12 0 0 16384)
      (nn__expert__plane 8 10 0 0 128) gate
      (nn__expert__plane 8 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 8 18 0 0 16777216)
      (nn__expert__plane 8 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 8 1 1 0 786432)
      (nn__expert__plane 8 0 1 0 48)
      (nn__expert__plane 8 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 8 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 8 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 8 5 0 0 576) 2.5) 
    (type streams)))

(defun (s8) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 8 9 0 0 4194304)
      (nn__expert__plane 8 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 8 9 1 0 4194304)
      (nn__expert__plane 8 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 8 8 0 0 4194304)
      (nn__expert__plane 8 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c8) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a9 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 9 1 0 0 786432)
      (nn__expert__plane 9 0 0 0 48)
      (nn__expert__plane 9 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 9 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 9 16 1 0 16777216)
      (nn__expert__plane 9 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 9 16 0 0 16777216)
      (nn__expert__plane 9 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 9 16 2 0 16777216)
      (nn__expert__plane 9 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 9 15 1 0 65536)
      (nn__expert__plane 9 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 9 15 0 0 65536)
      (nn__expert__plane 9 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 9 15 2 0 65536)
      (nn__expert__plane 9 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 9 13 0 0 262144)
      (nn__expert__plane 9 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 9 14 0 0 524288)
      (nn__expert__plane 9 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 9 11 0 0 131072)
      (nn__expert__plane 9 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 9 13 1 0 262144)
      (nn__expert__plane 9 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 9 14 1 0 524288)
      (nn__expert__plane 9 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 9 30 0 0 4194304) conved fb
      (nn__expert__plane 9 12 0 0 16384)
      (nn__expert__plane 9 10 0 0 128) gate
      (nn__expert__plane 9 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 9 18 0 0 16777216)
      (nn__expert__plane 9 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 9 1 1 0 786432)
      (nn__expert__plane 9 0 1 0 48)
      (nn__expert__plane 9 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 9 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 9 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 9 5 0 0 576) 2.5) 
    (type streams)))

(defun (s9) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 9 9 0 0 4194304)
      (nn__expert__plane 9 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 9 9 1 0 4194304)
      (nn__expert__plane 9 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 9 8 0 0 4194304)
      (nn__expert__plane 9 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c9) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a10 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 10 1 0 0 786432)
      (nn__expert__plane 10 0 0 0 48)
      (nn__expert__plane 10 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 10 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 10 16 1 0 16777216)
      (nn__expert__plane 10 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 10 16 0 0 16777216)
      (nn__expert__plane 10 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 10 16 2 0 16777216)
      (nn__expert__plane 10 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 10 15 1 0 65536)
      (nn__expert__plane 10 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 10 15 0 0 65536)
      (nn__expert__plane 10 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 10 15 2 0 65536)
      (nn__expert__plane 10 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 10 13 0 0 262144)
      (nn__expert__plane 10 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 10 14 0 0 524288)
      (nn__expert__plane 10 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 10 11 0 0 131072)
      (nn__expert__plane 10 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 10 13 1 0 262144)
      (nn__expert__plane 10 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 10 14 1 0 524288)
      (nn__expert__plane 10 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 10 30 0 0 4194304) conved fb
      (nn__expert__plane 10 12 0 0 16384)
      (nn__expert__plane 10 10 0 0 128) gate
      (nn__expert__plane 10 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 10 18 0 0 16777216)
      (nn__expert__plane 10 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 10 1 1 0 786432)
      (nn__expert__plane 10 0 1 0 48)
      (nn__expert__plane 10 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 10 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 10 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 10 5 0 0 576) 2.5) 
    (type streams)))

(defun (s10) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 10 9 0 0 4194304)
      (nn__expert__plane 10 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 10 9 1 0 4194304)
      (nn__expert__plane 10 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 10 8 0 0 4194304)
      (nn__expert__plane 10 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c10) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a11 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 11 1 2 0 786432)
      (nn__expert__plane 11 0 2 0 48)
      (nn__expert__plane 11 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 11 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 11 27 0 0 3145728)
      (nn__expert__plane 11 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 11 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 11 28 0 0 12582912)
      (nn__expert__plane 11 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 11 23 0 0 1048576)
      (nn__expert__plane 11 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 11 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 11 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 11 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 11 31 0 0 102400000)
      (nn__expert__plane 11 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 11 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 11 25 0 0 33554432)
      (nn__expert__plane 11 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 11 1 3 0 786432)
      (nn__expert__plane 11 0 3 0 48)
      (nn__expert__plane 11 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 11 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 11 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 11 5 1 0 576) 2.5) 
    (type streams)))

(defun (s11) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 11 9 2 0 4194304)
      (nn__expert__plane 11 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 11 9 3 0 4194304)
      (nn__expert__plane 11 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 11 8 1 0 4194304)
      (nn__expert__plane 11 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c11) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a12 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 12 1 0 0 786432)
      (nn__expert__plane 12 0 0 0 48)
      (nn__expert__plane 12 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 12 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 12 16 1 0 16777216)
      (nn__expert__plane 12 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 12 16 0 0 16777216)
      (nn__expert__plane 12 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 12 16 2 0 16777216)
      (nn__expert__plane 12 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 12 15 1 0 65536)
      (nn__expert__plane 12 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 12 15 0 0 65536)
      (nn__expert__plane 12 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 12 15 2 0 65536)
      (nn__expert__plane 12 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 12 13 0 0 262144)
      (nn__expert__plane 12 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 12 14 0 0 524288)
      (nn__expert__plane 12 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 12 11 0 0 131072)
      (nn__expert__plane 12 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 12 13 1 0 262144)
      (nn__expert__plane 12 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 12 14 1 0 524288)
      (nn__expert__plane 12 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 12 30 0 0 4194304) conved fb
      (nn__expert__plane 12 12 0 0 16384)
      (nn__expert__plane 12 10 0 0 128) gate
      (nn__expert__plane 12 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 12 18 0 0 16777216)
      (nn__expert__plane 12 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 12 1 1 0 786432)
      (nn__expert__plane 12 0 1 0 48)
      (nn__expert__plane 12 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 12 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 12 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 12 5 0 0 576) 2.5) 
    (type streams)))

(defun (s12) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 12 9 0 0 4194304)
      (nn__expert__plane 12 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 12 9 1 0 4194304)
      (nn__expert__plane 12 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 12 8 0 0 4194304)
      (nn__expert__plane 12 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c12) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a13 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 13 1 0 0 786432)
      (nn__expert__plane 13 0 0 0 48)
      (nn__expert__plane 13 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 13 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 13 16 1 0 16777216)
      (nn__expert__plane 13 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 13 16 0 0 16777216)
      (nn__expert__plane 13 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 13 16 2 0 16777216)
      (nn__expert__plane 13 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 13 15 1 0 65536)
      (nn__expert__plane 13 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 13 15 0 0 65536)
      (nn__expert__plane 13 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 13 15 2 0 65536)
      (nn__expert__plane 13 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 13 13 0 0 262144)
      (nn__expert__plane 13 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 13 14 0 0 524288)
      (nn__expert__plane 13 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 13 11 0 0 131072)
      (nn__expert__plane 13 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 13 13 1 0 262144)
      (nn__expert__plane 13 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 13 14 1 0 524288)
      (nn__expert__plane 13 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 13 30 0 0 4194304) conved fb
      (nn__expert__plane 13 12 0 0 16384)
      (nn__expert__plane 13 10 0 0 128) gate
      (nn__expert__plane 13 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 13 18 0 0 16777216)
      (nn__expert__plane 13 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 13 1 1 0 786432)
      (nn__expert__plane 13 0 1 0 48)
      (nn__expert__plane 13 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 13 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 13 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 13 5 0 0 576) 2.5) 
    (type streams)))

(defun (s13) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 13 9 0 0 4194304)
      (nn__expert__plane 13 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 13 9 1 0 4194304)
      (nn__expert__plane 13 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 13 8 0 0 4194304)
      (nn__expert__plane 13 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c13) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a14 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 14 1 0 0 786432)
      (nn__expert__plane 14 0 0 0 48)
      (nn__expert__plane 14 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 14 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 14 16 1 0 16777216)
      (nn__expert__plane 14 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 14 16 0 0 16777216)
      (nn__expert__plane 14 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 14 16 2 0 16777216)
      (nn__expert__plane 14 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 14 15 1 0 65536)
      (nn__expert__plane 14 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 14 15 0 0 65536)
      (nn__expert__plane 14 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 14 15 2 0 65536)
      (nn__expert__plane 14 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 14 13 0 0 262144)
      (nn__expert__plane 14 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 14 14 0 0 524288)
      (nn__expert__plane 14 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 14 11 0 0 131072)
      (nn__expert__plane 14 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 14 13 1 0 262144)
      (nn__expert__plane 14 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 14 14 1 0 524288)
      (nn__expert__plane 14 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 14 30 0 0 4194304) conved fb
      (nn__expert__plane 14 12 0 0 16384)
      (nn__expert__plane 14 10 0 0 128) gate
      (nn__expert__plane 14 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 14 18 0 0 16777216)
      (nn__expert__plane 14 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 14 1 1 0 786432)
      (nn__expert__plane 14 0 1 0 48)
      (nn__expert__plane 14 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 14 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 14 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 14 5 0 0 576) 2.5) 
    (type streams)))

(defun (s14) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 14 9 0 0 4194304)
      (nn__expert__plane 14 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 14 9 1 0 4194304)
      (nn__expert__plane 14 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 14 8 0 0 4194304)
      (nn__expert__plane 14 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c14) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a15 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 15 1 2 0 786432)
      (nn__expert__plane 15 0 2 0 48)
      (nn__expert__plane 15 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 15 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 15 27 0 0 3145728)
      (nn__expert__plane 15 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 15 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 15 28 0 0 12582912)
      (nn__expert__plane 15 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 15 23 0 0 1048576)
      (nn__expert__plane 15 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 15 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 15 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 15 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 15 31 0 0 102400000)
      (nn__expert__plane 15 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 15 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 15 25 0 0 33554432)
      (nn__expert__plane 15 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 15 1 3 0 786432)
      (nn__expert__plane 15 0 3 0 48)
      (nn__expert__plane 15 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 15 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 15 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 15 5 1 0 576) 2.5) 
    (type streams)))

(defun (s15) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 15 9 2 0 4194304)
      (nn__expert__plane 15 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 15 9 3 0 4194304)
      (nn__expert__plane 15 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 15 8 1 0 4194304)
      (nn__expert__plane 15 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c15) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a16 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 16 1 0 0 786432)
      (nn__expert__plane 16 0 0 0 48)
      (nn__expert__plane 16 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 16 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 16 16 1 0 16777216)
      (nn__expert__plane 16 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 16 16 0 0 16777216)
      (nn__expert__plane 16 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 16 16 2 0 16777216)
      (nn__expert__plane 16 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 16 15 1 0 65536)
      (nn__expert__plane 16 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 16 15 0 0 65536)
      (nn__expert__plane 16 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 16 15 2 0 65536)
      (nn__expert__plane 16 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 16 13 0 0 262144)
      (nn__expert__plane 16 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 16 14 0 0 524288)
      (nn__expert__plane 16 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 16 11 0 0 131072)
      (nn__expert__plane 16 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 16 13 1 0 262144)
      (nn__expert__plane 16 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 16 14 1 0 524288)
      (nn__expert__plane 16 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 16 30 0 0 4194304) conved fb
      (nn__expert__plane 16 12 0 0 16384)
      (nn__expert__plane 16 10 0 0 128) gate
      (nn__expert__plane 16 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 16 18 0 0 16777216)
      (nn__expert__plane 16 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 16 1 1 0 786432)
      (nn__expert__plane 16 0 1 0 48)
      (nn__expert__plane 16 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 16 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 16 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 16 5 0 0 576) 2.5) 
    (type streams)))

(defun (s16) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 16 9 0 0 4194304)
      (nn__expert__plane 16 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 16 9 1 0 4194304)
      (nn__expert__plane 16 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 16 8 0 0 4194304)
      (nn__expert__plane 16 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c16) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a17 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 17 1 0 0 786432)
      (nn__expert__plane 17 0 0 0 48)
      (nn__expert__plane 17 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 17 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 17 16 1 0 16777216)
      (nn__expert__plane 17 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 17 16 0 0 16777216)
      (nn__expert__plane 17 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 17 16 2 0 16777216)
      (nn__expert__plane 17 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 17 15 1 0 65536)
      (nn__expert__plane 17 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 17 15 0 0 65536)
      (nn__expert__plane 17 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 17 15 2 0 65536)
      (nn__expert__plane 17 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 17 13 0 0 262144)
      (nn__expert__plane 17 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 17 14 0 0 524288)
      (nn__expert__plane 17 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 17 11 0 0 131072)
      (nn__expert__plane 17 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 17 13 1 0 262144)
      (nn__expert__plane 17 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 17 14 1 0 524288)
      (nn__expert__plane 17 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 17 30 0 0 4194304) conved fb
      (nn__expert__plane 17 12 0 0 16384)
      (nn__expert__plane 17 10 0 0 128) gate
      (nn__expert__plane 17 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 17 18 0 0 16777216)
      (nn__expert__plane 17 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 17 1 1 0 786432)
      (nn__expert__plane 17 0 1 0 48)
      (nn__expert__plane 17 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 17 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 17 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 17 5 0 0 576) 2.5) 
    (type streams)))

(defun (s17) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 17 9 0 0 4194304)
      (nn__expert__plane 17 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 17 9 1 0 4194304)
      (nn__expert__plane 17 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 17 8 0 0 4194304)
      (nn__expert__plane 17 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c17) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a18 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 18 1 0 0 786432)
      (nn__expert__plane 18 0 0 0 48)
      (nn__expert__plane 18 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 18 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 18 16 1 0 16777216)
      (nn__expert__plane 18 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 18 16 0 0 16777216)
      (nn__expert__plane 18 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 18 16 2 0 16777216)
      (nn__expert__plane 18 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 18 15 1 0 65536)
      (nn__expert__plane 18 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 18 15 0 0 65536)
      (nn__expert__plane 18 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 18 15 2 0 65536)
      (nn__expert__plane 18 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 18 13 0 0 262144)
      (nn__expert__plane 18 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 18 14 0 0 524288)
      (nn__expert__plane 18 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 18 11 0 0 131072)
      (nn__expert__plane 18 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 18 13 1 0 262144)
      (nn__expert__plane 18 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 18 14 1 0 524288)
      (nn__expert__plane 18 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 18 30 0 0 4194304) conved fb
      (nn__expert__plane 18 12 0 0 16384)
      (nn__expert__plane 18 10 0 0 128) gate
      (nn__expert__plane 18 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 18 18 0 0 16777216)
      (nn__expert__plane 18 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 18 1 1 0 786432)
      (nn__expert__plane 18 0 1 0 48)
      (nn__expert__plane 18 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 18 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 18 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 18 5 0 0 576) 2.5) 
    (type streams)))

(defun (s18) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 18 9 0 0 4194304)
      (nn__expert__plane 18 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 18 9 1 0 4194304)
      (nn__expert__plane 18 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 18 8 0 0 4194304)
      (nn__expert__plane 18 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c18) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a19 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 19 1 2 0 786432)
      (nn__expert__plane 19 0 2 0 48)
      (nn__expert__plane 19 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 19 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 19 27 0 0 3145728)
      (nn__expert__plane 19 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 19 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 19 28 0 0 12582912)
      (nn__expert__plane 19 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 19 23 0 0 1048576)
      (nn__expert__plane 19 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 19 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 19 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 19 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 19 31 0 0 102400000)
      (nn__expert__plane 19 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 19 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 19 25 0 0 33554432)
      (nn__expert__plane 19 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 19 1 3 0 786432)
      (nn__expert__plane 19 0 3 0 48)
      (nn__expert__plane 19 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 19 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 19 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 19 5 1 0 576) 2.5) 
    (type streams)))

(defun (s19) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 19 9 2 0 4194304)
      (nn__expert__plane 19 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 19 9 3 0 4194304)
      (nn__expert__plane 19 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 19 8 1 0 4194304)
      (nn__expert__plane 19 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c19) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a20 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 20 1 0 0 786432)
      (nn__expert__plane 20 0 0 0 48)
      (nn__expert__plane 20 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 20 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 20 16 1 0 16777216)
      (nn__expert__plane 20 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 20 16 0 0 16777216)
      (nn__expert__plane 20 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 20 16 2 0 16777216)
      (nn__expert__plane 20 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 20 15 1 0 65536)
      (nn__expert__plane 20 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 20 15 0 0 65536)
      (nn__expert__plane 20 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 20 15 2 0 65536)
      (nn__expert__plane 20 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 20 13 0 0 262144)
      (nn__expert__plane 20 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 20 14 0 0 524288)
      (nn__expert__plane 20 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 20 11 0 0 131072)
      (nn__expert__plane 20 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 20 13 1 0 262144)
      (nn__expert__plane 20 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 20 14 1 0 524288)
      (nn__expert__plane 20 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 20 30 0 0 4194304) conved fb
      (nn__expert__plane 20 12 0 0 16384)
      (nn__expert__plane 20 10 0 0 128) gate
      (nn__expert__plane 20 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 20 18 0 0 16777216)
      (nn__expert__plane 20 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 20 1 1 0 786432)
      (nn__expert__plane 20 0 1 0 48)
      (nn__expert__plane 20 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 20 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 20 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 20 5 0 0 576) 2.5) 
    (type streams)))

(defun (s20) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 20 9 0 0 4194304)
      (nn__expert__plane 20 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 20 9 1 0 4194304)
      (nn__expert__plane 20 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 20 8 0 0 4194304)
      (nn__expert__plane 20 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c20) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a21 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 21 1 0 0 786432)
      (nn__expert__plane 21 0 0 0 48)
      (nn__expert__plane 21 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 21 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 21 16 1 0 16777216)
      (nn__expert__plane 21 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 21 16 0 0 16777216)
      (nn__expert__plane 21 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 21 16 2 0 16777216)
      (nn__expert__plane 21 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 21 15 1 0 65536)
      (nn__expert__plane 21 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 21 15 0 0 65536)
      (nn__expert__plane 21 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 21 15 2 0 65536)
      (nn__expert__plane 21 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 21 13 0 0 262144)
      (nn__expert__plane 21 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 21 14 0 0 524288)
      (nn__expert__plane 21 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 21 11 0 0 131072)
      (nn__expert__plane 21 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 21 13 1 0 262144)
      (nn__expert__plane 21 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 21 14 1 0 524288)
      (nn__expert__plane 21 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 21 30 0 0 4194304) conved fb
      (nn__expert__plane 21 12 0 0 16384)
      (nn__expert__plane 21 10 0 0 128) gate
      (nn__expert__plane 21 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 21 18 0 0 16777216)
      (nn__expert__plane 21 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 21 1 1 0 786432)
      (nn__expert__plane 21 0 1 0 48)
      (nn__expert__plane 21 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 21 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 21 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 21 5 0 0 576) 2.5) 
    (type streams)))

(defun (s21) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 21 9 0 0 4194304)
      (nn__expert__plane 21 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 21 9 1 0 4194304)
      (nn__expert__plane 21 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 21 8 0 0 4194304)
      (nn__expert__plane 21 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c21) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a22 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 22 1 0 0 786432)
      (nn__expert__plane 22 0 0 0 48)
      (nn__expert__plane 22 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 22 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 22 16 1 0 16777216)
      (nn__expert__plane 22 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 22 16 0 0 16777216)
      (nn__expert__plane 22 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 22 16 2 0 16777216)
      (nn__expert__plane 22 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 22 15 1 0 65536)
      (nn__expert__plane 22 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 22 15 0 0 65536)
      (nn__expert__plane 22 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 22 15 2 0 65536)
      (nn__expert__plane 22 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 22 13 0 0 262144)
      (nn__expert__plane 22 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 22 14 0 0 524288)
      (nn__expert__plane 22 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 22 11 0 0 131072)
      (nn__expert__plane 22 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 22 13 1 0 262144)
      (nn__expert__plane 22 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 22 14 1 0 524288)
      (nn__expert__plane 22 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 22 30 0 0 4194304) conved fb
      (nn__expert__plane 22 12 0 0 16384)
      (nn__expert__plane 22 10 0 0 128) gate
      (nn__expert__plane 22 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 22 18 0 0 16777216)
      (nn__expert__plane 22 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 22 1 1 0 786432)
      (nn__expert__plane 22 0 1 0 48)
      (nn__expert__plane 22 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 22 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 22 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 22 5 0 0 576) 2.5) 
    (type streams)))

(defun (s22) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 22 9 0 0 4194304)
      (nn__expert__plane 22 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 22 9 1 0 4194304)
      (nn__expert__plane 22 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 22 8 0 0 4194304)
      (nn__expert__plane 22 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c22) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a23 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 23 1 2 0 786432)
      (nn__expert__plane 23 0 2 0 48)
      (nn__expert__plane 23 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 23 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 23 27 0 0 3145728)
      (nn__expert__plane 23 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 23 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 23 28 0 0 12582912)
      (nn__expert__plane 23 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 23 23 0 0 1048576)
      (nn__expert__plane 23 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 23 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 23 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 23 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 23 31 0 0 102400000)
      (nn__expert__plane 23 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 23 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 23 25 0 0 33554432)
      (nn__expert__plane 23 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 23 1 3 0 786432)
      (nn__expert__plane 23 0 3 0 48)
      (nn__expert__plane 23 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 23 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 23 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 23 5 1 0 576) 2.5) 
    (type streams)))

(defun (s23) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 23 9 2 0 4194304)
      (nn__expert__plane 23 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 23 9 3 0 4194304)
      (nn__expert__plane 23 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 23 8 1 0 4194304)
      (nn__expert__plane 23 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c23) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a24 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 24 1 0 0 786432)
      (nn__expert__plane 24 0 0 0 48)
      (nn__expert__plane 24 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 24 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 24 16 1 0 16777216)
      (nn__expert__plane 24 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 24 16 0 0 16777216)
      (nn__expert__plane 24 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 24 16 2 0 16777216)
      (nn__expert__plane 24 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 24 15 1 0 65536)
      (nn__expert__plane 24 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 24 15 0 0 65536)
      (nn__expert__plane 24 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 24 15 2 0 65536)
      (nn__expert__plane 24 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 24 13 0 0 262144)
      (nn__expert__plane 24 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 24 14 0 0 524288)
      (nn__expert__plane 24 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 24 11 0 0 131072)
      (nn__expert__plane 24 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 24 13 1 0 262144)
      (nn__expert__plane 24 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 24 14 1 0 524288)
      (nn__expert__plane 24 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 24 30 0 0 4194304) conved fb
      (nn__expert__plane 24 12 0 0 16384)
      (nn__expert__plane 24 10 0 0 128) gate
      (nn__expert__plane 24 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 24 18 0 0 16777216)
      (nn__expert__plane 24 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 24 1 1 0 786432)
      (nn__expert__plane 24 0 1 0 48)
      (nn__expert__plane 24 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 24 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 24 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 24 5 0 0 576) 2.5) 
    (type streams)))

(defun (s24) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 24 9 0 0 4194304)
      (nn__expert__plane 24 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 24 9 1 0 4194304)
      (nn__expert__plane 24 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 24 8 0 0 4194304)
      (nn__expert__plane 24 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c24) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a25 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 25 1 0 0 786432)
      (nn__expert__plane 25 0 0 0 48)
      (nn__expert__plane 25 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 25 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 25 16 1 0 16777216)
      (nn__expert__plane 25 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 25 16 0 0 16777216)
      (nn__expert__plane 25 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 25 16 2 0 16777216)
      (nn__expert__plane 25 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 25 15 1 0 65536)
      (nn__expert__plane 25 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 25 15 0 0 65536)
      (nn__expert__plane 25 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 25 15 2 0 65536)
      (nn__expert__plane 25 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 25 13 0 0 262144)
      (nn__expert__plane 25 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 25 14 0 0 524288)
      (nn__expert__plane 25 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 25 11 0 0 131072)
      (nn__expert__plane 25 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 25 13 1 0 262144)
      (nn__expert__plane 25 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 25 14 1 0 524288)
      (nn__expert__plane 25 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 25 30 0 0 4194304) conved fb
      (nn__expert__plane 25 12 0 0 16384)
      (nn__expert__plane 25 10 0 0 128) gate
      (nn__expert__plane 25 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 25 18 0 0 16777216)
      (nn__expert__plane 25 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 25 1 1 0 786432)
      (nn__expert__plane 25 0 1 0 48)
      (nn__expert__plane 25 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 25 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 25 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 25 5 0 0 576) 2.5) 
    (type streams)))

(defun (s25) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 25 9 0 0 4194304)
      (nn__expert__plane 25 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 25 9 1 0 4194304)
      (nn__expert__plane 25 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 25 8 0 0 4194304)
      (nn__expert__plane 25 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c25) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a26 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 26 1 0 0 786432)
      (nn__expert__plane 26 0 0 0 48)
      (nn__expert__plane 26 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 26 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 26 16 1 0 16777216)
      (nn__expert__plane 26 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 26 16 0 0 16777216)
      (nn__expert__plane 26 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 26 16 2 0 16777216)
      (nn__expert__plane 26 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 26 15 1 0 65536)
      (nn__expert__plane 26 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 26 15 0 0 65536)
      (nn__expert__plane 26 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 26 15 2 0 65536)
      (nn__expert__plane 26 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 26 13 0 0 262144)
      (nn__expert__plane 26 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 26 14 0 0 524288)
      (nn__expert__plane 26 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 26 11 0 0 131072)
      (nn__expert__plane 26 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 26 13 1 0 262144)
      (nn__expert__plane 26 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 26 14 1 0 524288)
      (nn__expert__plane 26 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 26 30 0 0 4194304) conved fb
      (nn__expert__plane 26 12 0 0 16384)
      (nn__expert__plane 26 10 0 0 128) gate
      (nn__expert__plane 26 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 26 18 0 0 16777216)
      (nn__expert__plane 26 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 26 1 1 0 786432)
      (nn__expert__plane 26 0 1 0 48)
      (nn__expert__plane 26 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 26 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 26 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 26 5 0 0 576) 2.5) 
    (type streams)))

(defun (s26) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 26 9 0 0 4194304)
      (nn__expert__plane 26 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 26 9 1 0 4194304)
      (nn__expert__plane 26 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 26 8 0 0 4194304)
      (nn__expert__plane 26 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c26) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a27 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 27 1 2 0 786432)
      (nn__expert__plane 27 0 2 0 48)
      (nn__expert__plane 27 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 27 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 27 27 0 0 3145728)
      (nn__expert__plane 27 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 27 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 27 28 0 0 12582912)
      (nn__expert__plane 27 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 27 23 0 0 1048576)
      (nn__expert__plane 27 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 27 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 27 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 27 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 27 31 0 0 102400000)
      (nn__expert__plane 27 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 27 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 27 25 0 0 33554432)
      (nn__expert__plane 27 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 27 1 3 0 786432)
      (nn__expert__plane 27 0 3 0 48)
      (nn__expert__plane 27 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 27 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 27 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 27 5 1 0 576) 2.5) 
    (type streams)))

(defun (s27) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 27 9 2 0 4194304)
      (nn__expert__plane 27 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 27 9 3 0 4194304)
      (nn__expert__plane 27 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 27 8 1 0 4194304)
      (nn__expert__plane 27 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c27) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a28 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 28 1 0 0 786432)
      (nn__expert__plane 28 0 0 0 48)
      (nn__expert__plane 28 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 28 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 28 16 1 0 16777216)
      (nn__expert__plane 28 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 28 16 0 0 16777216)
      (nn__expert__plane 28 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 28 16 2 0 16777216)
      (nn__expert__plane 28 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 28 15 1 0 65536)
      (nn__expert__plane 28 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 28 15 0 0 65536)
      (nn__expert__plane 28 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 28 15 2 0 65536)
      (nn__expert__plane 28 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 28 13 0 0 262144)
      (nn__expert__plane 28 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 28 14 0 0 524288)
      (nn__expert__plane 28 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 28 11 0 0 131072)
      (nn__expert__plane 28 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 28 13 1 0 262144)
      (nn__expert__plane 28 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 28 14 1 0 524288)
      (nn__expert__plane 28 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 28 30 0 0 4194304) conved fb
      (nn__expert__plane 28 12 0 0 16384)
      (nn__expert__plane 28 10 0 0 128) gate
      (nn__expert__plane 28 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 28 18 0 0 16777216)
      (nn__expert__plane 28 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 28 1 1 0 786432)
      (nn__expert__plane 28 0 1 0 48)
      (nn__expert__plane 28 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 28 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 28 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 28 5 0 0 576) 2.5) 
    (type streams)))

(defun (s28) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 28 9 0 0 4194304)
      (nn__expert__plane 28 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 28 9 1 0 4194304)
      (nn__expert__plane 28 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 28 8 0 0 4194304)
      (nn__expert__plane 28 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c28) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a29 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 29 1 0 0 786432)
      (nn__expert__plane 29 0 0 0 48)
      (nn__expert__plane 29 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 29 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 29 16 1 0 16777216)
      (nn__expert__plane 29 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 29 16 0 0 16777216)
      (nn__expert__plane 29 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 29 16 2 0 16777216)
      (nn__expert__plane 29 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 29 15 1 0 65536)
      (nn__expert__plane 29 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 29 15 0 0 65536)
      (nn__expert__plane 29 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 29 15 2 0 65536)
      (nn__expert__plane 29 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 29 13 0 0 262144)
      (nn__expert__plane 29 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 29 14 0 0 524288)
      (nn__expert__plane 29 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 29 11 0 0 131072)
      (nn__expert__plane 29 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 29 13 1 0 262144)
      (nn__expert__plane 29 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 29 14 1 0 524288)
      (nn__expert__plane 29 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 29 30 0 0 4194304) conved fb
      (nn__expert__plane 29 12 0 0 16384)
      (nn__expert__plane 29 10 0 0 128) gate
      (nn__expert__plane 29 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 29 18 0 0 16777216)
      (nn__expert__plane 29 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 29 1 1 0 786432)
      (nn__expert__plane 29 0 1 0 48)
      (nn__expert__plane 29 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 29 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 29 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 29 5 0 0 576) 2.5) 
    (type streams)))

(defun (s29) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 29 9 0 0 4194304)
      (nn__expert__plane 29 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 29 9 1 0 4194304)
      (nn__expert__plane 29 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 29 8 0 0 4194304)
      (nn__expert__plane 29 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c29) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a30 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 30 1 0 0 786432)
      (nn__expert__plane 30 0 0 0 48)
      (nn__expert__plane 30 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 30 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 30 16 1 0 16777216)
      (nn__expert__plane 30 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 30 16 0 0 16777216)
      (nn__expert__plane 30 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 30 16 2 0 16777216)
      (nn__expert__plane 30 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 30 15 1 0 65536)
      (nn__expert__plane 30 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 30 15 0 0 65536)
      (nn__expert__plane 30 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 30 15 2 0 65536)
      (nn__expert__plane 30 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 30 13 0 0 262144)
      (nn__expert__plane 30 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 30 14 0 0 524288)
      (nn__expert__plane 30 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 30 11 0 0 131072)
      (nn__expert__plane 30 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 30 13 1 0 262144)
      (nn__expert__plane 30 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 30 14 1 0 524288)
      (nn__expert__plane 30 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 30 30 0 0 4194304) conved fb
      (nn__expert__plane 30 12 0 0 16384)
      (nn__expert__plane 30 10 0 0 128) gate
      (nn__expert__plane 30 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 30 18 0 0 16777216)
      (nn__expert__plane 30 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 30 1 1 0 786432)
      (nn__expert__plane 30 0 1 0 48)
      (nn__expert__plane 30 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 30 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 30 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 30 5 0 0 576) 2.5) 
    (type streams)))

(defun (s30) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 30 9 0 0 4194304)
      (nn__expert__plane 30 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 30 9 1 0 4194304)
      (nn__expert__plane 30 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 30 8 0 0 4194304)
      (nn__expert__plane 30 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c30) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a31 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 31 1 2 0 786432)
      (nn__expert__plane 31 0 2 0 48)
      (nn__expert__plane 31 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 31 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 31 27 0 0 3145728)
      (nn__expert__plane 31 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 31 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 31 28 0 0 12582912)
      (nn__expert__plane 31 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 31 23 0 0 1048576)
      (nn__expert__plane 31 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 31 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 31 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 31 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 31 31 0 0 102400000)
      (nn__expert__plane 31 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 31 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 31 25 0 0 33554432)
      (nn__expert__plane 31 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 31 1 3 0 786432)
      (nn__expert__plane 31 0 3 0 48)
      (nn__expert__plane 31 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 31 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 31 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 31 5 1 0 576) 2.5) 
    (type streams)))

(defun (s31) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 31 9 2 0 4194304)
      (nn__expert__plane 31 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 31 9 3 0 4194304)
      (nn__expert__plane 31 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 31 8 1 0 4194304)
      (nn__expert__plane 31 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c31) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a32 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 32 1 0 0 786432)
      (nn__expert__plane 32 0 0 0 48)
      (nn__expert__plane 32 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 32 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 32 16 1 0 16777216)
      (nn__expert__plane 32 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 32 16 0 0 16777216)
      (nn__expert__plane 32 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 32 16 2 0 16777216)
      (nn__expert__plane 32 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 32 15 1 0 65536)
      (nn__expert__plane 32 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 32 15 0 0 65536)
      (nn__expert__plane 32 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 32 15 2 0 65536)
      (nn__expert__plane 32 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 32 13 0 0 262144)
      (nn__expert__plane 32 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 32 14 0 0 524288)
      (nn__expert__plane 32 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 32 11 0 0 131072)
      (nn__expert__plane 32 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 32 13 1 0 262144)
      (nn__expert__plane 32 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 32 14 1 0 524288)
      (nn__expert__plane 32 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 32 30 0 0 4194304) conved fb
      (nn__expert__plane 32 12 0 0 16384)
      (nn__expert__plane 32 10 0 0 128) gate
      (nn__expert__plane 32 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 32 18 0 0 16777216)
      (nn__expert__plane 32 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 32 1 1 0 786432)
      (nn__expert__plane 32 0 1 0 48)
      (nn__expert__plane 32 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 32 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 32 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 32 5 0 0 576) 2.5) 
    (type streams)))

(defun (s32) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 32 9 0 0 4194304)
      (nn__expert__plane 32 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 32 9 1 0 4194304)
      (nn__expert__plane 32 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 32 8 0 0 4194304)
      (nn__expert__plane 32 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c32) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a33 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 33 1 0 0 786432)
      (nn__expert__plane 33 0 0 0 48)
      (nn__expert__plane 33 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 33 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 33 16 1 0 16777216)
      (nn__expert__plane 33 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 33 16 0 0 16777216)
      (nn__expert__plane 33 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 33 16 2 0 16777216)
      (nn__expert__plane 33 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 33 15 1 0 65536)
      (nn__expert__plane 33 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 33 15 0 0 65536)
      (nn__expert__plane 33 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 33 15 2 0 65536)
      (nn__expert__plane 33 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 33 13 0 0 262144)
      (nn__expert__plane 33 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 33 14 0 0 524288)
      (nn__expert__plane 33 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 33 11 0 0 131072)
      (nn__expert__plane 33 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 33 13 1 0 262144)
      (nn__expert__plane 33 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 33 14 1 0 524288)
      (nn__expert__plane 33 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 33 30 0 0 4194304) conved fb
      (nn__expert__plane 33 12 0 0 16384)
      (nn__expert__plane 33 10 0 0 128) gate
      (nn__expert__plane 33 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 33 18 0 0 16777216)
      (nn__expert__plane 33 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 33 1 1 0 786432)
      (nn__expert__plane 33 0 1 0 48)
      (nn__expert__plane 33 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 33 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 33 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 33 5 0 0 576) 2.5) 
    (type streams)))

(defun (s33) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 33 9 0 0 4194304)
      (nn__expert__plane 33 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 33 9 1 0 4194304)
      (nn__expert__plane 33 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 33 8 0 0 4194304)
      (nn__expert__plane 33 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c33) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a34 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 34 1 0 0 786432)
      (nn__expert__plane 34 0 0 0 48)
      (nn__expert__plane 34 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 34 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 34 16 1 0 16777216)
      (nn__expert__plane 34 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 34 16 0 0 16777216)
      (nn__expert__plane 34 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 34 16 2 0 16777216)
      (nn__expert__plane 34 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 34 15 1 0 65536)
      (nn__expert__plane 34 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 34 15 0 0 65536)
      (nn__expert__plane 34 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 34 15 2 0 65536)
      (nn__expert__plane 34 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 34 13 0 0 262144)
      (nn__expert__plane 34 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 34 14 0 0 524288)
      (nn__expert__plane 34 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 34 11 0 0 131072)
      (nn__expert__plane 34 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 34 13 1 0 262144)
      (nn__expert__plane 34 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 34 14 1 0 524288)
      (nn__expert__plane 34 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 34 30 0 0 4194304) conved fb
      (nn__expert__plane 34 12 0 0 16384)
      (nn__expert__plane 34 10 0 0 128) gate
      (nn__expert__plane 34 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 34 18 0 0 16777216)
      (nn__expert__plane 34 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 34 1 1 0 786432)
      (nn__expert__plane 34 0 1 0 48)
      (nn__expert__plane 34 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 34 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 34 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 34 5 0 0 576) 2.5) 
    (type streams)))

(defun (s34) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 34 9 0 0 4194304)
      (nn__expert__plane 34 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 34 9 1 0 4194304)
      (nn__expert__plane 34 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 34 8 0 0 4194304)
      (nn__expert__plane 34 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c34) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a35 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 35 1 2 0 786432)
      (nn__expert__plane 35 0 2 0 48)
      (nn__expert__plane 35 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 35 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 35 27 0 0 3145728)
      (nn__expert__plane 35 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 35 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 35 28 0 0 12582912)
      (nn__expert__plane 35 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 35 23 0 0 1048576)
      (nn__expert__plane 35 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 35 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 35 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 35 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 35 31 0 0 102400000)
      (nn__expert__plane 35 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 35 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 35 25 0 0 33554432)
      (nn__expert__plane 35 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 35 1 3 0 786432)
      (nn__expert__plane 35 0 3 0 48)
      (nn__expert__plane 35 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 35 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 35 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 35 5 1 0 576) 2.5) 
    (type streams)))

(defun (s35) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 35 9 2 0 4194304)
      (nn__expert__plane 35 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 35 9 3 0 4194304)
      (nn__expert__plane 35 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 35 8 1 0 4194304)
      (nn__expert__plane 35 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c35) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a36 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 36 1 0 0 786432)
      (nn__expert__plane 36 0 0 0 48)
      (nn__expert__plane 36 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 36 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 36 16 1 0 16777216)
      (nn__expert__plane 36 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 36 16 0 0 16777216)
      (nn__expert__plane 36 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 36 16 2 0 16777216)
      (nn__expert__plane 36 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 36 15 1 0 65536)
      (nn__expert__plane 36 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 36 15 0 0 65536)
      (nn__expert__plane 36 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 36 15 2 0 65536)
      (nn__expert__plane 36 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 36 13 0 0 262144)
      (nn__expert__plane 36 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 36 14 0 0 524288)
      (nn__expert__plane 36 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 36 11 0 0 131072)
      (nn__expert__plane 36 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 36 13 1 0 262144)
      (nn__expert__plane 36 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 36 14 1 0 524288)
      (nn__expert__plane 36 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 36 30 0 0 4194304) conved fb
      (nn__expert__plane 36 12 0 0 16384)
      (nn__expert__plane 36 10 0 0 128) gate
      (nn__expert__plane 36 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 36 18 0 0 16777216)
      (nn__expert__plane 36 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 36 1 1 0 786432)
      (nn__expert__plane 36 0 1 0 48)
      (nn__expert__plane 36 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 36 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 36 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 36 5 0 0 576) 2.5) 
    (type streams)))

(defun (s36) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 36 9 0 0 4194304)
      (nn__expert__plane 36 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 36 9 1 0 4194304)
      (nn__expert__plane 36 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 36 8 0 0 4194304)
      (nn__expert__plane 36 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c36) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a37 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 37 1 0 0 786432)
      (nn__expert__plane 37 0 0 0 48)
      (nn__expert__plane 37 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 37 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 37 16 1 0 16777216)
      (nn__expert__plane 37 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 37 16 0 0 16777216)
      (nn__expert__plane 37 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 37 16 2 0 16777216)
      (nn__expert__plane 37 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 37 15 1 0 65536)
      (nn__expert__plane 37 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 37 15 0 0 65536)
      (nn__expert__plane 37 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 37 15 2 0 65536)
      (nn__expert__plane 37 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 37 13 0 0 262144)
      (nn__expert__plane 37 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 37 14 0 0 524288)
      (nn__expert__plane 37 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 37 11 0 0 131072)
      (nn__expert__plane 37 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 37 13 1 0 262144)
      (nn__expert__plane 37 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 37 14 1 0 524288)
      (nn__expert__plane 37 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 37 30 0 0 4194304) conved fb
      (nn__expert__plane 37 12 0 0 16384)
      (nn__expert__plane 37 10 0 0 128) gate
      (nn__expert__plane 37 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 37 18 0 0 16777216)
      (nn__expert__plane 37 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 37 1 1 0 786432)
      (nn__expert__plane 37 0 1 0 48)
      (nn__expert__plane 37 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 37 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 37 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 37 5 0 0 576) 2.5) 
    (type streams)))

(defun (s37) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 37 9 0 0 4194304)
      (nn__expert__plane 37 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 37 9 1 0 4194304)
      (nn__expert__plane 37 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 37 8 0 0 4194304)
      (nn__expert__plane 37 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c37) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a38 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 38 1 0 0 786432)
      (nn__expert__plane 38 0 0 0 48)
      (nn__expert__plane 38 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 38 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 38 16 1 0 16777216)
      (nn__expert__plane 38 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 38 16 0 0 16777216)
      (nn__expert__plane 38 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 38 16 2 0 16777216)
      (nn__expert__plane 38 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 38 15 1 0 65536)
      (nn__expert__plane 38 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 38 15 0 0 65536)
      (nn__expert__plane 38 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 38 15 2 0 65536)
      (nn__expert__plane 38 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 38 13 0 0 262144)
      (nn__expert__plane 38 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 38 14 0 0 524288)
      (nn__expert__plane 38 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 38 11 0 0 131072)
      (nn__expert__plane 38 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 38 13 1 0 262144)
      (nn__expert__plane 38 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 38 14 1 0 524288)
      (nn__expert__plane 38 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 38 30 0 0 4194304) conved fb
      (nn__expert__plane 38 12 0 0 16384)
      (nn__expert__plane 38 10 0 0 128) gate
      (nn__expert__plane 38 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 38 18 0 0 16777216)
      (nn__expert__plane 38 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 38 1 1 0 786432)
      (nn__expert__plane 38 0 1 0 48)
      (nn__expert__plane 38 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 38 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 38 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 38 5 0 0 576) 2.5) 
    (type streams)))

(defun (s38) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 38 9 0 0 4194304)
      (nn__expert__plane 38 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 38 9 1 0 4194304)
      (nn__expert__plane 38 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 38 8 0 0 4194304)
      (nn__expert__plane 38 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c38) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a39 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 39 1 2 0 786432)
      (nn__expert__plane 39 0 2 0 48)
      (nn__expert__plane 39 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 39 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 39 27 0 0 3145728)
      (nn__expert__plane 39 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 39 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 39 28 0 0 12582912)
      (nn__expert__plane 39 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 39 23 0 0 1048576)
      (nn__expert__plane 39 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 39 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 39 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 39 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 39 31 0 0 102400000)
      (nn__expert__plane 39 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 39 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 39 25 0 0 33554432)
      (nn__expert__plane 39 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 39 1 3 0 786432)
      (nn__expert__plane 39 0 3 0 48)
      (nn__expert__plane 39 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 39 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 39 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 39 5 1 0 576) 2.5) 
    (type streams)))

(defun (s39) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 39 9 2 0 4194304)
      (nn__expert__plane 39 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 39 9 3 0 4194304)
      (nn__expert__plane 39 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 39 8 1 0 4194304)
      (nn__expert__plane 39 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c39) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a40 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 40 1 0 0 786432)
      (nn__expert__plane 40 0 0 0 48)
      (nn__expert__plane 40 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 40 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 40 16 1 0 16777216)
      (nn__expert__plane 40 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 40 16 0 0 16777216)
      (nn__expert__plane 40 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 40 16 2 0 16777216)
      (nn__expert__plane 40 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 40 15 1 0 65536)
      (nn__expert__plane 40 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 40 15 0 0 65536)
      (nn__expert__plane 40 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 40 15 2 0 65536)
      (nn__expert__plane 40 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 40 13 0 0 262144)
      (nn__expert__plane 40 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 40 14 0 0 524288)
      (nn__expert__plane 40 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 40 11 0 0 131072)
      (nn__expert__plane 40 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 40 13 1 0 262144)
      (nn__expert__plane 40 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 40 14 1 0 524288)
      (nn__expert__plane 40 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 40 30 0 0 4194304) conved fb
      (nn__expert__plane 40 12 0 0 16384)
      (nn__expert__plane 40 10 0 0 128) gate
      (nn__expert__plane 40 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 40 18 0 0 16777216)
      (nn__expert__plane 40 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 40 1 1 0 786432)
      (nn__expert__plane 40 0 1 0 48)
      (nn__expert__plane 40 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 40 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 40 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 40 5 0 0 576) 2.5) 
    (type streams)))

(defun (s40) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 40 9 0 0 4194304)
      (nn__expert__plane 40 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 40 9 1 0 4194304)
      (nn__expert__plane 40 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 40 8 0 0 4194304)
      (nn__expert__plane 40 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c40) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a41 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 41 1 0 0 786432)
      (nn__expert__plane 41 0 0 0 48)
      (nn__expert__plane 41 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 41 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 41 16 1 0 16777216)
      (nn__expert__plane 41 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 41 16 0 0 16777216)
      (nn__expert__plane 41 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 41 16 2 0 16777216)
      (nn__expert__plane 41 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 41 15 1 0 65536)
      (nn__expert__plane 41 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 41 15 0 0 65536)
      (nn__expert__plane 41 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 41 15 2 0 65536)
      (nn__expert__plane 41 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 41 13 0 0 262144)
      (nn__expert__plane 41 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 41 14 0 0 524288)
      (nn__expert__plane 41 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 41 11 0 0 131072)
      (nn__expert__plane 41 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 41 13 1 0 262144)
      (nn__expert__plane 41 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 41 14 1 0 524288)
      (nn__expert__plane 41 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 41 30 0 0 4194304) conved fb
      (nn__expert__plane 41 12 0 0 16384)
      (nn__expert__plane 41 10 0 0 128) gate
      (nn__expert__plane 41 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 41 18 0 0 16777216)
      (nn__expert__plane 41 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 41 1 1 0 786432)
      (nn__expert__plane 41 0 1 0 48)
      (nn__expert__plane 41 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 41 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 41 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 41 5 0 0 576) 2.5) 
    (type streams)))

(defun (s41) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 41 9 0 0 4194304)
      (nn__expert__plane 41 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 41 9 1 0 4194304)
      (nn__expert__plane 41 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 41 8 0 0 4194304)
      (nn__expert__plane 41 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c41) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a42 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 42 1 0 0 786432)
      (nn__expert__plane 42 0 0 0 48)
      (nn__expert__plane 42 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 42 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 42 16 1 0 16777216)
      (nn__expert__plane 42 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 42 16 0 0 16777216)
      (nn__expert__plane 42 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 42 16 2 0 16777216)
      (nn__expert__plane 42 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 42 15 1 0 65536)
      (nn__expert__plane 42 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 42 15 0 0 65536)
      (nn__expert__plane 42 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 42 15 2 0 65536)
      (nn__expert__plane 42 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 42 13 0 0 262144)
      (nn__expert__plane 42 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 42 14 0 0 524288)
      (nn__expert__plane 42 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 42 11 0 0 131072)
      (nn__expert__plane 42 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 42 13 1 0 262144)
      (nn__expert__plane 42 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 42 14 1 0 524288)
      (nn__expert__plane 42 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 42 30 0 0 4194304) conved fb
      (nn__expert__plane 42 12 0 0 16384)
      (nn__expert__plane 42 10 0 0 128) gate
      (nn__expert__plane 42 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 42 18 0 0 16777216)
      (nn__expert__plane 42 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 42 1 1 0 786432)
      (nn__expert__plane 42 0 1 0 48)
      (nn__expert__plane 42 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 42 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 42 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 42 5 0 0 576) 2.5) 
    (type streams)))

(defun (s42) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 42 9 0 0 4194304)
      (nn__expert__plane 42 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 42 9 1 0 4194304)
      (nn__expert__plane 42 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 42 8 0 0 4194304)
      (nn__expert__plane 42 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c42) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a43 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 43 1 2 0 786432)
      (nn__expert__plane 43 0 2 0 48)
      (nn__expert__plane 43 2 2 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 43 3 2 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 43 27 0 0 3145728)
      (nn__expert__plane 43 27 0 3145728 3072) hr qa 1536 4096 4)
    (nn__rmsnorm__apply qa
      (nn__expert__plane 43 26 0 0 3072) qn 1536)
    (nn__hadamard__rotate qn signs qnr 1536)
    (nn__turboquant__gemv
      (nn__expert__plane 43 28 0 0 12582912)
      (nn__expert__plane 43 28 0 12582912 32768) qnr q 16384 1536 4)
    (nn__turboquant__gemv
      (nn__expert__plane 43 23 0 0 1048576)
      (nn__expert__plane 43 23 0 1048576 1024) hr ckv 512 4096 4)
    (nn__rmsnorm__apply ckv
      (nn__expert__plane 43 22 0 0 1024) ckn 512)
    (nn__hadamard__rotate ckn signs
      (nn__expert__plane 43 31 0
        (sys__add 0 at) 1024) 512)
    (nn__attention__absorb q
      (nn__expert__plane 43 31 0 108806144 33554432) qabs 64 256 512 512 1.4142135623730951)
    (nn__attention__decode qabs
      (nn__expert__plane 43 31 0 0 102400000)
      (nn__expert__plane 43 31 0 0 102400000) scores olat 64 1 512 0 len)
    (nn__attention__expand olat
      (nn__expert__plane 43 31 0 108806144 33554432) vh 64 256 512 512 256)
    (nn__hadamard__rotate vh signs vhr 16384)
    (nn__turboquant__gemv
      (nn__expert__plane 43 25 0 0 33554432)
      (nn__expert__plane 43 25 0 33554432 8192) vhr y 4096 16384 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 43 1 3 0 786432)
      (nn__expert__plane 43 0 3 0 48)
      (nn__expert__plane 43 2 3 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 43 3 3 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 43 6 1 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 43 5 1 0 576) 2.5) 
    (type streams)))

(defun (s43) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 43 9 2 0 4194304)
      (nn__expert__plane 43 9 2 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 43 9 3 0 4194304)
      (nn__expert__plane 43 9 3 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 43 8 1 0 4194304)
      (nn__expert__plane 43 8 1 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c43) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (a44 at len) (begin
    (nn__hyper__pre streams
      (nn__expert__plane 44 1 0 0 786432)
      (nn__expert__plane 44 0 0 0 48)
      (nn__expert__plane 44 2 0 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 44 3 0 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__turboquant__gemv
      (nn__expert__plane 44 16 1 0 16777216)
      (nn__expert__plane 44 16 1 16777216 16384) hr
      (nn__vector__range qkv 0 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 44 16 0 0 16777216)
      (nn__expert__plane 44 16 0 16777216 16384) hr
      (nn__vector__range qkv 8192 8192) 8192 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 44 16 2 0 16777216)
      (nn__expert__plane 44 16 2 16777216 16384) hr
      (nn__vector__range qkv 16384 8192) 8192 4096 4)
    (nn__deltanet__conv_step
      (nn__expert__plane 44 15 1 0 65536)
      (nn__expert__plane 44 30 0 4194304 49152)
      (nn__vector__range qkv 0 8192)
      (nn__vector__range conved 0 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 44 15 0 0 65536)
      (nn__expert__plane 44 30 0 4243456 49152)
      (nn__vector__range qkv 8192 8192)
      (nn__vector__range conved 8192 8192) 8192)
    (nn__deltanet__conv_step
      (nn__expert__plane 44 15 2 0 65536)
      (nn__expert__plane 44 30 0 4292608 49152)
      (nn__vector__range qkv 16384 8192)
      (nn__vector__range conved 16384 8192) 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 44 13 0 0 262144)
      (nn__expert__plane 44 13 0 262144 256) hr fa 128 4096 4)
    (nn__hadamard__rotate fa signs far 128)
    (nn__turboquant__gemv
      (nn__expert__plane 44 14 0 0 524288)
      (nn__expert__plane 44 14 0 524288 16384) far
      (nn__vector__range fb 0 8192) 8192 128 4)
    (nn__turboquant__gemv
      (nn__expert__plane 44 11 0 0 131072)
      (nn__expert__plane 44 11 0 131072 128) hr
      (nn__vector__range fb 8192 64) 64 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 44 13 1 0 262144)
      (nn__expert__plane 44 13 1 262144 256) hr ga 128 4096 4)
    (nn__hadamard__rotate ga signs gar 128)
    (nn__turboquant__gemv
      (nn__expert__plane 44 14 1 0 524288)
      (nn__expert__plane 44 14 1 524288 16384) gar gate 8192 128 4)
    (nn__kda__step
      (nn__expert__plane 44 30 0 0 4194304) conved fb
      (nn__expert__plane 44 12 0 0 16384)
      (nn__expert__plane 44 10 0 0 128) gate
      (nn__expert__plane 44 17 0 0 256) ko 64 128 -5.0 1e-05)
    (nn__hadamard__rotate ko signs kor 8192)
    (nn__turboquant__gemv
      (nn__expert__plane 44 18 0 0 16777216)
      (nn__expert__plane 44 18 0 16777216 8192) kor y 4096 8192 4)
    (nn__hyper__post streams y mix streams 4096 4)
    (nn__hyper__pre streams
      (nn__expert__plane 44 1 1 0 786432)
      (nn__expert__plane 44 0 1 0 48)
      (nn__expert__plane 44 2 1 0 6) xin mix 4096 4 20 1e-05 1e-06)
    (nn__rmsnorm__apply xin
      (nn__expert__plane 44 3 1 0 8192) h 4096)
    (nn__hadamard__rotate h signs hr 4096)
    (nn__expert__multiply_fp16
      (nn__expert__plane 44 6 0 0 2359296) h logits 288 4096)
    (nn__vector__top_k_biased logits 288 8 picks weights
      (nn__expert__plane 44 5 0 0 576) 2.5) 
    (type streams)))

(defun (s44) (begin
    (nn__turboquant__gemv
      (nn__expert__plane 44 9 0 0 4194304)
      (nn__expert__plane 44 9 0 4194304 4096) hr
      (nn__vector__range sgu 0 2048) 2048 4096 4)
    (nn__turboquant__gemv
      (nn__expert__plane 44 9 1 0 4194304)
      (nn__expert__plane 44 9 1 4194304 4096) hr
      (nn__vector__range sgu 2048 2048) 2048 4096 4)
    (nn__swiglu__clamped sgu sact 2048 10.0)
    (nn__hadamard__rotate sact signs sactr 2048)
    (nn__turboquant__gemv
      (nn__expert__plane 44 8 0 0 4194304)
      (nn__expert__plane 44 8 0 4194304 8192) sactr sh 4096 2048 4)
    (type sh)))

(defun (c44) (begin
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__vector__add sh routed y 4096)
    (nn__hyper__post streams y mix streams 4096 4)
    (type streams)))

(defun (x3) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 0 4194304)
        (nn__expert__plane 3 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 3 0 x 4198400 2097152)
        (nn__expert__plane 3 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw3 — run by another worker when a program hands it over
(x3)

(defun (x4) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 0 4194304)
        (nn__expert__plane 4 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 4 0 x 4198400 2097152)
        (nn__expert__plane 4 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw4 — run by another worker when a program hands it over
(x4)

(defun (x5) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 0 4194304)
        (nn__expert__plane 5 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 5 0 x 4198400 2097152)
        (nn__expert__plane 5 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw5 — run by another worker when a program hands it over
(x5)

(defun (x6) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 0 4194304)
        (nn__expert__plane 6 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 6 0 x 4198400 2097152)
        (nn__expert__plane 6 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw6 — run by another worker when a program hands it over
(x6)

(defun (x7) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 0 4194304)
        (nn__expert__plane 7 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 7 0 x 4198400 2097152)
        (nn__expert__plane 7 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw7 — run by another worker when a program hands it over
(x7)

(defun (x8) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 0 4194304)
        (nn__expert__plane 8 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 8 0 x 4198400 2097152)
        (nn__expert__plane 8 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw8 — run by another worker when a program hands it over
(x8)

(defun (x9) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 0 4194304)
        (nn__expert__plane 9 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 9 0 x 4198400 2097152)
        (nn__expert__plane 9 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw9 — run by another worker when a program hands it over
(x9)

(defun (x10) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 0 4194304)
        (nn__expert__plane 10 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 10 0 x 4198400 2097152)
        (nn__expert__plane 10 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw10 — run by another worker when a program hands it over
(x10)

(defun (x11) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 0 4194304)
        (nn__expert__plane 11 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 11 0 x 4198400 2097152)
        (nn__expert__plane 11 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw11 — run by another worker when a program hands it over
(x11)

(defun (x12) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 0 4194304)
        (nn__expert__plane 12 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 12 0 x 4198400 2097152)
        (nn__expert__plane 12 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw12 — run by another worker when a program hands it over
(x12)

(defun (x13) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 0 4194304)
        (nn__expert__plane 13 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 13 0 x 4198400 2097152)
        (nn__expert__plane 13 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw13 — run by another worker when a program hands it over
(x13)

(defun (x14) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 0 4194304)
        (nn__expert__plane 14 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 14 0 x 4198400 2097152)
        (nn__expert__plane 14 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw14 — run by another worker when a program hands it over
(x14)

(defun (x15) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 0 4194304)
        (nn__expert__plane 15 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 15 0 x 4198400 2097152)
        (nn__expert__plane 15 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw15 — run by another worker when a program hands it over
(x15)

(defun (x16) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 0 4194304)
        (nn__expert__plane 16 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 16 0 x 4198400 2097152)
        (nn__expert__plane 16 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw16 — run by another worker when a program hands it over
(x16)

(defun (x17) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 0 4194304)
        (nn__expert__plane 17 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 17 0 x 4198400 2097152)
        (nn__expert__plane 17 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw17 — run by another worker when a program hands it over
(x17)

(defun (x18) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 0 4194304)
        (nn__expert__plane 18 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 18 0 x 4198400 2097152)
        (nn__expert__plane 18 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw18 — run by another worker when a program hands it over
(x18)

(defun (x19) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 0 4194304)
        (nn__expert__plane 19 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 19 0 x 4198400 2097152)
        (nn__expert__plane 19 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw19 — run by another worker when a program hands it over
(x19)

(defun (x20) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 0 4194304)
        (nn__expert__plane 20 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 20 0 x 4198400 2097152)
        (nn__expert__plane 20 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw20 — run by another worker when a program hands it over
(x20)

(defun (x21) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 0 4194304)
        (nn__expert__plane 21 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 21 0 x 4198400 2097152)
        (nn__expert__plane 21 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw21 — run by another worker when a program hands it over
(x21)

(defun (x22) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 0 4194304)
        (nn__expert__plane 22 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 22 0 x 4198400 2097152)
        (nn__expert__plane 22 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw22 — run by another worker when a program hands it over
(x22)

(defun (x23) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 0 4194304)
        (nn__expert__plane 23 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 23 0 x 4198400 2097152)
        (nn__expert__plane 23 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw23 — run by another worker when a program hands it over
(x23)

(defun (x24) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 0 4194304)
        (nn__expert__plane 24 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 24 0 x 4198400 2097152)
        (nn__expert__plane 24 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw24 — run by another worker when a program hands it over
(x24)

(defun (x25) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 0 4194304)
        (nn__expert__plane 25 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 25 0 x 4198400 2097152)
        (nn__expert__plane 25 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw25 — run by another worker when a program hands it over
(x25)

(defun (x26) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 0 4194304)
        (nn__expert__plane 26 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 26 0 x 4198400 2097152)
        (nn__expert__plane 26 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw26 — run by another worker when a program hands it over
(x26)

(defun (x27) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 0 4194304)
        (nn__expert__plane 27 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 27 0 x 4198400 2097152)
        (nn__expert__plane 27 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw27 — run by another worker when a program hands it over
(x27)

(defun (x28) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 0 4194304)
        (nn__expert__plane 28 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 28 0 x 4198400 2097152)
        (nn__expert__plane 28 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw28 — run by another worker when a program hands it over
(x28)

(defun (x29) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 0 4194304)
        (nn__expert__plane 29 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 29 0 x 4198400 2097152)
        (nn__expert__plane 29 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw29 — run by another worker when a program hands it over
(x29)

(defun (x30) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 0 4194304)
        (nn__expert__plane 30 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 30 0 x 4198400 2097152)
        (nn__expert__plane 30 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw30 — run by another worker when a program hands it over
(x30)

(defun (x31) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 0 4194304)
        (nn__expert__plane 31 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 31 0 x 4198400 2097152)
        (nn__expert__plane 31 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw31 — run by another worker when a program hands it over
(x31)

(defun (x32) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 0 4194304)
        (nn__expert__plane 32 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 32 0 x 4198400 2097152)
        (nn__expert__plane 32 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw32 — run by another worker when a program hands it over
(x32)

(defun (x33) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 0 4194304)
        (nn__expert__plane 33 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 33 0 x 4198400 2097152)
        (nn__expert__plane 33 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw33 — run by another worker when a program hands it over
(x33)

(defun (x34) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 0 4194304)
        (nn__expert__plane 34 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 34 0 x 4198400 2097152)
        (nn__expert__plane 34 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw34 — run by another worker when a program hands it over
(x34)

(defun (x35) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 0 4194304)
        (nn__expert__plane 35 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 35 0 x 4198400 2097152)
        (nn__expert__plane 35 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw35 — run by another worker when a program hands it over
(x35)

(defun (x36) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 0 4194304)
        (nn__expert__plane 36 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 36 0 x 4198400 2097152)
        (nn__expert__plane 36 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw36 — run by another worker when a program hands it over
(x36)

(defun (x37) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 0 4194304)
        (nn__expert__plane 37 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 37 0 x 4198400 2097152)
        (nn__expert__plane 37 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw37 — run by another worker when a program hands it over
(x37)

(defun (x38) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 0 4194304)
        (nn__expert__plane 38 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 38 0 x 4198400 2097152)
        (nn__expert__plane 38 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw38 — run by another worker when a program hands it over
(x38)

(defun (x39) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 0 4194304)
        (nn__expert__plane 39 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 39 0 x 4198400 2097152)
        (nn__expert__plane 39 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw39 — run by another worker when a program hands it over
(x39)

(defun (x40) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 0 4194304)
        (nn__expert__plane 40 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 40 0 x 4198400 2097152)
        (nn__expert__plane 40 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw40 — run by another worker when a program hands it over
(x40)

(defun (x41) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 0 4194304)
        (nn__expert__plane 41 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 41 0 x 4198400 2097152)
        (nn__expert__plane 41 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw41 — run by another worker when a program hands it over
(x41)

(defun (x42) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 0 4194304)
        (nn__expert__plane 42 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 42 0 x 4198400 2097152)
        (nn__expert__plane 42 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw42 — run by another worker when a program hands it over
(x42)

(defun (x43) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 0 4194304)
        (nn__expert__plane 43 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 43 0 x 4198400 2097152)
        (nn__expert__plane 43 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw43 — run by another worker when a program hands it over
(x43)

(defun (x44) (begin
    (nn__buffer__copy hr 0 c_hr 8192)
    (nn__buffer__copy weights 0 c_w 16)
    (nn__vector__zero c_routed 4096)
    (let
      ((x
          (sys__node_array__get picks 0)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 0 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 1)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 1 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 2)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 2 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 3)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 3 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 4)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 4 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 5)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 5 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 6)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 6 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (let
      ((x
          (sys__node_array__get picks 7)))
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 0 4194304)
        (nn__expert__plane 44 0 x 4194304 4096) c_hr c_gu 4096 4096 4)
      (nn__swiglu__clamped c_gu c_act 2048 10.0)
      (nn__hadamard__rotate c_act c_signs c_actr 2048)
      (nn__turboquant__gemv
        (nn__expert__plane 44 0 x 4198400 2097152)
        (nn__expert__plane 44 0 x 6295552 8192) c_actr c_e 4096 2048 4)
      (nn__vector__scale_at c_e c_w 7 c_e 4096)
      (nn__vector__add c_routed c_e c_routed 4096))
    (type c_routed)))

;; picture xw44 — run by another worker when a program hands it over
(x44)

;; picture xf3_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex3_0 c_routed))

;; picture xf4_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex4_0 c_routed))

;; picture xf5_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex5_0 c_routed))

;; picture xf6_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex6_0 c_routed))

;; picture xf7_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex7_0 c_routed))

;; picture xf8_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex8_0 c_routed))

;; picture xf9_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex9_0 c_routed))

;; picture xf10_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex10_0 c_routed))

;; picture xf11_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex11_0 c_routed))

;; picture xf12_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex12_0 c_routed))

;; picture xf13_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex13_0 c_routed))

;; picture xf14_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex14_0 c_routed))

;; picture xf15_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex15_0 c_routed))

;; picture xf16_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex16_0 c_routed))

;; picture xf17_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex17_0 c_routed))

;; picture xf18_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex18_0 c_routed))

;; picture xf19_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex19_0 c_routed))

;; picture xf20_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex20_0 c_routed))

;; picture xf21_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex21_0 c_routed))

;; picture xf22_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex22_0 c_routed))

;; picture xf23_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex23_0 c_routed))

;; picture xf24_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex24_0 c_routed))

;; picture xf25_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex25_0 c_routed))

;; picture xf26_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex26_0 c_routed))

;; picture xf27_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex27_0 c_routed))

;; picture xf28_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex28_0 c_routed))

;; picture xf29_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex29_0 c_routed))

;; picture xf30_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex30_0 c_routed))

;; picture xf31_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex31_0 c_routed))

;; picture xf32_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex32_0 c_routed))

;; picture xf33_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex33_0 c_routed))

;; picture xf34_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex34_0 c_routed))

;; picture xf35_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex35_0 c_routed))

;; picture xf36_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex36_0 c_routed))

;; picture xf37_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex37_0 c_routed))

;; picture xf38_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex38_0 c_routed))

;; picture xf39_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex39_0 c_routed))

;; picture xf40_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex40_0 c_routed))

;; picture xf41_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex41_0 c_routed))

;; picture xf42_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex42_0 c_routed))

;; picture xf43_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex43_0 c_routed))

;; picture xf44_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c_hand 8208) (ai_glm_5_3__experts c_hand picks ex44_0 c_routed))

;; picture xf3_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex3_1 c1_routed))

;; picture xf4_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex4_1 c1_routed))

;; picture xf5_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex5_1 c1_routed))

;; picture xf6_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex6_1 c1_routed))

;; picture xf7_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex7_1 c1_routed))

;; picture xf8_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex8_1 c1_routed))

;; picture xf9_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex9_1 c1_routed))

;; picture xf10_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex10_1 c1_routed))

;; picture xf11_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex11_1 c1_routed))

;; picture xf12_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex12_1 c1_routed))

;; picture xf13_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex13_1 c1_routed))

;; picture xf14_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex14_1 c1_routed))

;; picture xf15_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex15_1 c1_routed))

;; picture xf16_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex16_1 c1_routed))

;; picture xf17_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex17_1 c1_routed))

;; picture xf18_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex18_1 c1_routed))

;; picture xf19_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex19_1 c1_routed))

;; picture xf20_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex20_1 c1_routed))

;; picture xf21_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex21_1 c1_routed))

;; picture xf22_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex22_1 c1_routed))

;; picture xf23_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex23_1 c1_routed))

;; picture xf24_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex24_1 c1_routed))

;; picture xf25_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex25_1 c1_routed))

;; picture xf26_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex26_1 c1_routed))

;; picture xf27_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex27_1 c1_routed))

;; picture xf28_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex28_1 c1_routed))

;; picture xf29_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex29_1 c1_routed))

;; picture xf30_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex30_1 c1_routed))

;; picture xf31_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex31_1 c1_routed))

;; picture xf32_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex32_1 c1_routed))

;; picture xf33_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex33_1 c1_routed))

;; picture xf34_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex34_1 c1_routed))

;; picture xf35_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex35_1 c1_routed))

;; picture xf36_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex36_1 c1_routed))

;; picture xf37_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex37_1 c1_routed))

;; picture xf38_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex38_1 c1_routed))

;; picture xf39_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex39_1 c1_routed))

;; picture xf40_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex40_1 c1_routed))

;; picture xf41_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex41_1 c1_routed))

;; picture xf42_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex42_1 c1_routed))

;; picture xf43_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex43_1 c1_routed))

;; picture xf44_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hand 0 c1_hand 8208) (ai_glm_5_3__experts c1_hand picks ex44_1 c1_routed))

;; picture xr3_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr3_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr4_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr4_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr5_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr5_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr6_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr6_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr7_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr7_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr8_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr8_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr9_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr9_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr10_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr10_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr11_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr11_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr12_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr12_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr13_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr13_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr14_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr14_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr15_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr15_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr16_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr16_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr17_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr17_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr18_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr18_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr19_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr19_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr20_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr20_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr21_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr21_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr22_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr22_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr23_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr23_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr24_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr24_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr25_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr25_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr26_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr26_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr27_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr27_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr28_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr28_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr29_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr29_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr30_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr30_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr31_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr31_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr32_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr32_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr33_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr33_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr34_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr34_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr35_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr35_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr36_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr36_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr37_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr37_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr38_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr38_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr39_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr39_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr40_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr40_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr41_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr41_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr42_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr42_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr43_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr43_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr44_0 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c_hands 2117632) (ai_glm_5_3__experts_rows c_hands exr44_0 c_rrows
    (sys__node_array__get nrows 0)))

;; picture xr3_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr3_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr4_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr4_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr5_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr5_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr6_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr6_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr7_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr7_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr8_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr8_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr9_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr9_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr10_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr10_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr11_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr11_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr12_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr12_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr13_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr13_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr14_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr14_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr15_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr15_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr16_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr16_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr17_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr17_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr18_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr18_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr19_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr19_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr20_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr20_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr21_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr21_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr22_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr22_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr23_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr23_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr24_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr24_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr25_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr25_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr26_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr26_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr27_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr27_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr28_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr28_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr29_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr29_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr30_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr30_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr31_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr31_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr32_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr32_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr33_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr33_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr34_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr34_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr35_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr35_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr36_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr36_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr37_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr37_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr38_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr38_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr39_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr39_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr40_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr40_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr41_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr41_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr42_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr42_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr43_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr43_1 c1_rrows
    (sys__node_array__get nrows 0)))

;; picture xr44_1 — run by another worker when a program hands it over
(begin (nn__buffer__copy hands 0 c1_hands 2117632) (ai_glm_5_3__experts_rows c1_hands exr44_1 c1_rrows
    (sys__node_array__get nrows 0)))

(defun (r0 first n) (begin
    (ai_glm_5_3__kda_rows srows at0 n)
    (ai_glm_5_3__mlp_rows srows ml0 n)
    (type srows)))

(defun (r1 first n) (begin
    (ai_glm_5_3__kda_rows srows at1 n)
    (ai_glm_5_3__mlp_rows srows ml1 n)
    (type srows)))

(defun (r2 first n) (begin
    (ai_glm_5_3__kda_rows srows at2 n)
    (ai_glm_5_3__mlp_rows srows ml2 n)
    (type srows)))

(defun (r3 first n) (begin
    (ai_glm_5_3__mla_rows srows at3 first n)
    (ai_glm_5_3__route_rows srows mo3 hands carries n)
    (sys__compute 1 names xr3_0)
    (sys__compute 2 names xr3_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo3 rrows carries n)
    (type srows)))

(defun (r4 first n) (begin
    (ai_glm_5_3__kda_rows srows at4 n)
    (ai_glm_5_3__route_rows srows mo4 hands carries n)
    (sys__compute 1 names xr4_0)
    (sys__compute 2 names xr4_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo4 rrows carries n)
    (type srows)))

(defun (r5 first n) (begin
    (ai_glm_5_3__kda_rows srows at5 n)
    (ai_glm_5_3__route_rows srows mo5 hands carries n)
    (sys__compute 1 names xr5_0)
    (sys__compute 2 names xr5_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo5 rrows carries n)
    (type srows)))

(defun (r6 first n) (begin
    (ai_glm_5_3__kda_rows srows at6 n)
    (ai_glm_5_3__route_rows srows mo6 hands carries n)
    (sys__compute 1 names xr6_0)
    (sys__compute 2 names xr6_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo6 rrows carries n)
    (type srows)))

(defun (r7 first n) (begin
    (ai_glm_5_3__mla_rows srows at7 first n)
    (ai_glm_5_3__route_rows srows mo7 hands carries n)
    (sys__compute 1 names xr7_0)
    (sys__compute 2 names xr7_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo7 rrows carries n)
    (type srows)))

(defun (r8 first n) (begin
    (ai_glm_5_3__kda_rows srows at8 n)
    (ai_glm_5_3__route_rows srows mo8 hands carries n)
    (sys__compute 1 names xr8_0)
    (sys__compute 2 names xr8_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo8 rrows carries n)
    (type srows)))

(defun (r9 first n) (begin
    (ai_glm_5_3__kda_rows srows at9 n)
    (ai_glm_5_3__route_rows srows mo9 hands carries n)
    (sys__compute 1 names xr9_0)
    (sys__compute 2 names xr9_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo9 rrows carries n)
    (type srows)))

(defun (r10 first n) (begin
    (ai_glm_5_3__kda_rows srows at10 n)
    (ai_glm_5_3__route_rows srows mo10 hands carries n)
    (sys__compute 1 names xr10_0)
    (sys__compute 2 names xr10_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo10 rrows carries n)
    (type srows)))

(defun (r11 first n) (begin
    (ai_glm_5_3__mla_rows srows at11 first n)
    (ai_glm_5_3__route_rows srows mo11 hands carries n)
    (sys__compute 1 names xr11_0)
    (sys__compute 2 names xr11_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo11 rrows carries n)
    (type srows)))

(defun (r12 first n) (begin
    (ai_glm_5_3__kda_rows srows at12 n)
    (ai_glm_5_3__route_rows srows mo12 hands carries n)
    (sys__compute 1 names xr12_0)
    (sys__compute 2 names xr12_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo12 rrows carries n)
    (type srows)))

(defun (r13 first n) (begin
    (ai_glm_5_3__kda_rows srows at13 n)
    (ai_glm_5_3__route_rows srows mo13 hands carries n)
    (sys__compute 1 names xr13_0)
    (sys__compute 2 names xr13_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo13 rrows carries n)
    (type srows)))

(defun (r14 first n) (begin
    (ai_glm_5_3__kda_rows srows at14 n)
    (ai_glm_5_3__route_rows srows mo14 hands carries n)
    (sys__compute 1 names xr14_0)
    (sys__compute 2 names xr14_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo14 rrows carries n)
    (type srows)))

(defun (r15 first n) (begin
    (ai_glm_5_3__mla_rows srows at15 first n)
    (ai_glm_5_3__route_rows srows mo15 hands carries n)
    (sys__compute 1 names xr15_0)
    (sys__compute 2 names xr15_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo15 rrows carries n)
    (type srows)))

(defun (r16 first n) (begin
    (ai_glm_5_3__kda_rows srows at16 n)
    (ai_glm_5_3__route_rows srows mo16 hands carries n)
    (sys__compute 1 names xr16_0)
    (sys__compute 2 names xr16_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo16 rrows carries n)
    (type srows)))

(defun (r17 first n) (begin
    (ai_glm_5_3__kda_rows srows at17 n)
    (ai_glm_5_3__route_rows srows mo17 hands carries n)
    (sys__compute 1 names xr17_0)
    (sys__compute 2 names xr17_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo17 rrows carries n)
    (type srows)))

(defun (r18 first n) (begin
    (ai_glm_5_3__kda_rows srows at18 n)
    (ai_glm_5_3__route_rows srows mo18 hands carries n)
    (sys__compute 1 names xr18_0)
    (sys__compute 2 names xr18_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo18 rrows carries n)
    (type srows)))

(defun (r19 first n) (begin
    (ai_glm_5_3__mla_rows srows at19 first n)
    (ai_glm_5_3__route_rows srows mo19 hands carries n)
    (sys__compute 1 names xr19_0)
    (sys__compute 2 names xr19_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo19 rrows carries n)
    (type srows)))

(defun (r20 first n) (begin
    (ai_glm_5_3__kda_rows srows at20 n)
    (ai_glm_5_3__route_rows srows mo20 hands carries n)
    (sys__compute 1 names xr20_0)
    (sys__compute 2 names xr20_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo20 rrows carries n)
    (type srows)))

(defun (r21 first n) (begin
    (ai_glm_5_3__kda_rows srows at21 n)
    (ai_glm_5_3__route_rows srows mo21 hands carries n)
    (sys__compute 1 names xr21_0)
    (sys__compute 2 names xr21_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo21 rrows carries n)
    (type srows)))

(defun (r22 first n) (begin
    (ai_glm_5_3__kda_rows srows at22 n)
    (ai_glm_5_3__route_rows srows mo22 hands carries n)
    (sys__compute 1 names xr22_0)
    (sys__compute 2 names xr22_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo22 rrows carries n)
    (type srows)))

(defun (r23 first n) (begin
    (ai_glm_5_3__mla_rows srows at23 first n)
    (ai_glm_5_3__route_rows srows mo23 hands carries n)
    (sys__compute 1 names xr23_0)
    (sys__compute 2 names xr23_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo23 rrows carries n)
    (type srows)))

(defun (r24 first n) (begin
    (ai_glm_5_3__kda_rows srows at24 n)
    (ai_glm_5_3__route_rows srows mo24 hands carries n)
    (sys__compute 1 names xr24_0)
    (sys__compute 2 names xr24_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo24 rrows carries n)
    (type srows)))

(defun (r25 first n) (begin
    (ai_glm_5_3__kda_rows srows at25 n)
    (ai_glm_5_3__route_rows srows mo25 hands carries n)
    (sys__compute 1 names xr25_0)
    (sys__compute 2 names xr25_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo25 rrows carries n)
    (type srows)))

(defun (r26 first n) (begin
    (ai_glm_5_3__kda_rows srows at26 n)
    (ai_glm_5_3__route_rows srows mo26 hands carries n)
    (sys__compute 1 names xr26_0)
    (sys__compute 2 names xr26_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo26 rrows carries n)
    (type srows)))

(defun (r27 first n) (begin
    (ai_glm_5_3__mla_rows srows at27 first n)
    (ai_glm_5_3__route_rows srows mo27 hands carries n)
    (sys__compute 1 names xr27_0)
    (sys__compute 2 names xr27_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo27 rrows carries n)
    (type srows)))

(defun (r28 first n) (begin
    (ai_glm_5_3__kda_rows srows at28 n)
    (ai_glm_5_3__route_rows srows mo28 hands carries n)
    (sys__compute 1 names xr28_0)
    (sys__compute 2 names xr28_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo28 rrows carries n)
    (type srows)))

(defun (r29 first n) (begin
    (ai_glm_5_3__kda_rows srows at29 n)
    (ai_glm_5_3__route_rows srows mo29 hands carries n)
    (sys__compute 1 names xr29_0)
    (sys__compute 2 names xr29_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo29 rrows carries n)
    (type srows)))

(defun (r30 first n) (begin
    (ai_glm_5_3__kda_rows srows at30 n)
    (ai_glm_5_3__route_rows srows mo30 hands carries n)
    (sys__compute 1 names xr30_0)
    (sys__compute 2 names xr30_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo30 rrows carries n)
    (type srows)))

(defun (r31 first n) (begin
    (ai_glm_5_3__mla_rows srows at31 first n)
    (ai_glm_5_3__route_rows srows mo31 hands carries n)
    (sys__compute 1 names xr31_0)
    (sys__compute 2 names xr31_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo31 rrows carries n)
    (type srows)))

(defun (r32 first n) (begin
    (ai_glm_5_3__kda_rows srows at32 n)
    (ai_glm_5_3__route_rows srows mo32 hands carries n)
    (sys__compute 1 names xr32_0)
    (sys__compute 2 names xr32_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo32 rrows carries n)
    (type srows)))

(defun (r33 first n) (begin
    (ai_glm_5_3__kda_rows srows at33 n)
    (ai_glm_5_3__route_rows srows mo33 hands carries n)
    (sys__compute 1 names xr33_0)
    (sys__compute 2 names xr33_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo33 rrows carries n)
    (type srows)))

(defun (r34 first n) (begin
    (ai_glm_5_3__kda_rows srows at34 n)
    (ai_glm_5_3__route_rows srows mo34 hands carries n)
    (sys__compute 1 names xr34_0)
    (sys__compute 2 names xr34_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo34 rrows carries n)
    (type srows)))

(defun (r35 first n) (begin
    (ai_glm_5_3__mla_rows srows at35 first n)
    (ai_glm_5_3__route_rows srows mo35 hands carries n)
    (sys__compute 1 names xr35_0)
    (sys__compute 2 names xr35_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo35 rrows carries n)
    (type srows)))

(defun (r36 first n) (begin
    (ai_glm_5_3__kda_rows srows at36 n)
    (ai_glm_5_3__route_rows srows mo36 hands carries n)
    (sys__compute 1 names xr36_0)
    (sys__compute 2 names xr36_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo36 rrows carries n)
    (type srows)))

(defun (r37 first n) (begin
    (ai_glm_5_3__kda_rows srows at37 n)
    (ai_glm_5_3__route_rows srows mo37 hands carries n)
    (sys__compute 1 names xr37_0)
    (sys__compute 2 names xr37_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo37 rrows carries n)
    (type srows)))

(defun (r38 first n) (begin
    (ai_glm_5_3__kda_rows srows at38 n)
    (ai_glm_5_3__route_rows srows mo38 hands carries n)
    (sys__compute 1 names xr38_0)
    (sys__compute 2 names xr38_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo38 rrows carries n)
    (type srows)))

(defun (r39 first n) (begin
    (ai_glm_5_3__mla_rows srows at39 first n)
    (ai_glm_5_3__route_rows srows mo39 hands carries n)
    (sys__compute 1 names xr39_0)
    (sys__compute 2 names xr39_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo39 rrows carries n)
    (type srows)))

(defun (r40 first n) (begin
    (ai_glm_5_3__kda_rows srows at40 n)
    (ai_glm_5_3__route_rows srows mo40 hands carries n)
    (sys__compute 1 names xr40_0)
    (sys__compute 2 names xr40_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo40 rrows carries n)
    (type srows)))

(defun (r41 first n) (begin
    (ai_glm_5_3__kda_rows srows at41 n)
    (ai_glm_5_3__route_rows srows mo41 hands carries n)
    (sys__compute 1 names xr41_0)
    (sys__compute 2 names xr41_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo41 rrows carries n)
    (type srows)))

(defun (r42 first n) (begin
    (ai_glm_5_3__kda_rows srows at42 n)
    (ai_glm_5_3__route_rows srows mo42 hands carries n)
    (sys__compute 1 names xr42_0)
    (sys__compute 2 names xr42_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo42 rrows carries n)
    (type srows)))

(defun (r43 first n) (begin
    (ai_glm_5_3__mla_rows srows at43 first n)
    (ai_glm_5_3__route_rows srows mo43 hands carries n)
    (sys__compute 1 names xr43_0)
    (sys__compute 2 names xr43_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo43 rrows carries n)
    (type srows)))

(defun (r44 first n) (begin
    (ai_glm_5_3__kda_rows srows at44 n)
    (ai_glm_5_3__route_rows srows mo44 hands carries n)
    (sys__compute 1 names xr44_0)
    (sys__compute 2 names xr44_1)
    (while '(sys__eq
        (sys__completed 1)
        (< 1 0)) '(< 0 1))
    (sys__result 1)
    (while '(sys__eq
        (sys__completed 2)
        (< 1 0)) '(< 0 1))
    (sys__result 2)
    (nn__buffer__copy c_rrows 1 rrows 2097152)
    (nn__buffer__copy c1_rrows 2 rrows1 2097152)
    (nn__vector__add rrows rrows1 rrows 1048576)
    (ai_glm_5_3__close_rows srows mo44 rrows carries n)
    (type srows)))

(defun (embed_row codes scale at0 at1 at2 at3) (begin
    (nn__turboquant__decode
      (nn__vector__range emb_rows codes 2048)
      (nn__vector__range emb_rscales scale 1) u 1 4096 8)
    (nn__hadamard__rotate u ones512 hu 4096)
    (nn__vector__pointwise_mul hu signs_row e 4096)
    (nn__vector__add e zero_h
      (nn__vector__range srows at0 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range srows at1 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range srows at2 4096) 4096)
    (nn__vector__add e zero_h
      (nn__vector__range srows at3 4096) 4096)
    (type e)))

(defun (token_f codes scale pos) (begin
    (embed codes scale)
    (ai_glm_5_3__kda streams at0)
    (ai_glm_5_3__mlp streams ml0)
    (ai_glm_5_3__kda streams at1)
    (ai_glm_5_3__mlp streams ml1)
    (ai_glm_5_3__kda streams at2)
    (ai_glm_5_3__mlp streams ml2)
    (ai_glm_5_3__mla streams at3 pos)
    (ai_glm_5_3__route streams mo3 hand picks)
    (sys__compute 1 names xf3_0)
    (sys__compute 2 names xf3_1)
    (ai_glm_5_3__shared hand mo3)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo3 routed)
    (ai_glm_5_3__kda streams at4)
    (ai_glm_5_3__route streams mo4 hand picks)
    (sys__compute 1 names xf4_0)
    (sys__compute 2 names xf4_1)
    (ai_glm_5_3__shared hand mo4)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo4 routed)
    (ai_glm_5_3__kda streams at5)
    (ai_glm_5_3__route streams mo5 hand picks)
    (sys__compute 1 names xf5_0)
    (sys__compute 2 names xf5_1)
    (ai_glm_5_3__shared hand mo5)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo5 routed)
    (ai_glm_5_3__kda streams at6)
    (ai_glm_5_3__route streams mo6 hand picks)
    (sys__compute 1 names xf6_0)
    (sys__compute 2 names xf6_1)
    (ai_glm_5_3__shared hand mo6)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo6 routed)
    (ai_glm_5_3__mla streams at7 pos)
    (ai_glm_5_3__route streams mo7 hand picks)
    (sys__compute 1 names xf7_0)
    (sys__compute 2 names xf7_1)
    (ai_glm_5_3__shared hand mo7)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo7 routed)
    (ai_glm_5_3__kda streams at8)
    (ai_glm_5_3__route streams mo8 hand picks)
    (sys__compute 1 names xf8_0)
    (sys__compute 2 names xf8_1)
    (ai_glm_5_3__shared hand mo8)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo8 routed)
    (ai_glm_5_3__kda streams at9)
    (ai_glm_5_3__route streams mo9 hand picks)
    (sys__compute 1 names xf9_0)
    (sys__compute 2 names xf9_1)
    (ai_glm_5_3__shared hand mo9)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo9 routed)
    (ai_glm_5_3__kda streams at10)
    (ai_glm_5_3__route streams mo10 hand picks)
    (sys__compute 1 names xf10_0)
    (sys__compute 2 names xf10_1)
    (ai_glm_5_3__shared hand mo10)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo10 routed)
    (ai_glm_5_3__mla streams at11 pos)
    (ai_glm_5_3__route streams mo11 hand picks)
    (sys__compute 1 names xf11_0)
    (sys__compute 2 names xf11_1)
    (ai_glm_5_3__shared hand mo11)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo11 routed)
    (ai_glm_5_3__kda streams at12)
    (ai_glm_5_3__route streams mo12 hand picks)
    (sys__compute 1 names xf12_0)
    (sys__compute 2 names xf12_1)
    (ai_glm_5_3__shared hand mo12)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo12 routed)
    (ai_glm_5_3__kda streams at13)
    (ai_glm_5_3__route streams mo13 hand picks)
    (sys__compute 1 names xf13_0)
    (sys__compute 2 names xf13_1)
    (ai_glm_5_3__shared hand mo13)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo13 routed)
    (ai_glm_5_3__kda streams at14)
    (ai_glm_5_3__route streams mo14 hand picks)
    (sys__compute 1 names xf14_0)
    (sys__compute 2 names xf14_1)
    (ai_glm_5_3__shared hand mo14)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo14 routed)
    (ai_glm_5_3__mla streams at15 pos)
    (ai_glm_5_3__route streams mo15 hand picks)
    (sys__compute 1 names xf15_0)
    (sys__compute 2 names xf15_1)
    (ai_glm_5_3__shared hand mo15)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo15 routed)
    (ai_glm_5_3__kda streams at16)
    (ai_glm_5_3__route streams mo16 hand picks)
    (sys__compute 1 names xf16_0)
    (sys__compute 2 names xf16_1)
    (ai_glm_5_3__shared hand mo16)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo16 routed)
    (ai_glm_5_3__kda streams at17)
    (ai_glm_5_3__route streams mo17 hand picks)
    (sys__compute 1 names xf17_0)
    (sys__compute 2 names xf17_1)
    (ai_glm_5_3__shared hand mo17)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo17 routed)
    (ai_glm_5_3__kda streams at18)
    (ai_glm_5_3__route streams mo18 hand picks)
    (sys__compute 1 names xf18_0)
    (sys__compute 2 names xf18_1)
    (ai_glm_5_3__shared hand mo18)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo18 routed)
    (ai_glm_5_3__mla streams at19 pos)
    (ai_glm_5_3__route streams mo19 hand picks)
    (sys__compute 1 names xf19_0)
    (sys__compute 2 names xf19_1)
    (ai_glm_5_3__shared hand mo19)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo19 routed)
    (ai_glm_5_3__kda streams at20)
    (ai_glm_5_3__route streams mo20 hand picks)
    (sys__compute 1 names xf20_0)
    (sys__compute 2 names xf20_1)
    (ai_glm_5_3__shared hand mo20)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo20 routed)
    (ai_glm_5_3__kda streams at21)
    (ai_glm_5_3__route streams mo21 hand picks)
    (sys__compute 1 names xf21_0)
    (sys__compute 2 names xf21_1)
    (ai_glm_5_3__shared hand mo21)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo21 routed)
    (ai_glm_5_3__kda streams at22)
    (ai_glm_5_3__route streams mo22 hand picks)
    (sys__compute 1 names xf22_0)
    (sys__compute 2 names xf22_1)
    (ai_glm_5_3__shared hand mo22)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo22 routed)
    (ai_glm_5_3__mla streams at23 pos)
    (ai_glm_5_3__route streams mo23 hand picks)
    (sys__compute 1 names xf23_0)
    (sys__compute 2 names xf23_1)
    (ai_glm_5_3__shared hand mo23)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo23 routed)
    (ai_glm_5_3__kda streams at24)
    (ai_glm_5_3__route streams mo24 hand picks)
    (sys__compute 1 names xf24_0)
    (sys__compute 2 names xf24_1)
    (ai_glm_5_3__shared hand mo24)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo24 routed)
    (ai_glm_5_3__kda streams at25)
    (ai_glm_5_3__route streams mo25 hand picks)
    (sys__compute 1 names xf25_0)
    (sys__compute 2 names xf25_1)
    (ai_glm_5_3__shared hand mo25)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo25 routed)
    (ai_glm_5_3__kda streams at26)
    (ai_glm_5_3__route streams mo26 hand picks)
    (sys__compute 1 names xf26_0)
    (sys__compute 2 names xf26_1)
    (ai_glm_5_3__shared hand mo26)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo26 routed)
    (ai_glm_5_3__mla streams at27 pos)
    (ai_glm_5_3__route streams mo27 hand picks)
    (sys__compute 1 names xf27_0)
    (sys__compute 2 names xf27_1)
    (ai_glm_5_3__shared hand mo27)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo27 routed)
    (ai_glm_5_3__kda streams at28)
    (ai_glm_5_3__route streams mo28 hand picks)
    (sys__compute 1 names xf28_0)
    (sys__compute 2 names xf28_1)
    (ai_glm_5_3__shared hand mo28)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo28 routed)
    (ai_glm_5_3__kda streams at29)
    (ai_glm_5_3__route streams mo29 hand picks)
    (sys__compute 1 names xf29_0)
    (sys__compute 2 names xf29_1)
    (ai_glm_5_3__shared hand mo29)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo29 routed)
    (ai_glm_5_3__kda streams at30)
    (ai_glm_5_3__route streams mo30 hand picks)
    (sys__compute 1 names xf30_0)
    (sys__compute 2 names xf30_1)
    (ai_glm_5_3__shared hand mo30)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo30 routed)
    (ai_glm_5_3__mla streams at31 pos)
    (ai_glm_5_3__route streams mo31 hand picks)
    (sys__compute 1 names xf31_0)
    (sys__compute 2 names xf31_1)
    (ai_glm_5_3__shared hand mo31)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo31 routed)
    (ai_glm_5_3__kda streams at32)
    (ai_glm_5_3__route streams mo32 hand picks)
    (sys__compute 1 names xf32_0)
    (sys__compute 2 names xf32_1)
    (ai_glm_5_3__shared hand mo32)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo32 routed)
    (ai_glm_5_3__kda streams at33)
    (ai_glm_5_3__route streams mo33 hand picks)
    (sys__compute 1 names xf33_0)
    (sys__compute 2 names xf33_1)
    (ai_glm_5_3__shared hand mo33)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo33 routed)
    (ai_glm_5_3__kda streams at34)
    (ai_glm_5_3__route streams mo34 hand picks)
    (sys__compute 1 names xf34_0)
    (sys__compute 2 names xf34_1)
    (ai_glm_5_3__shared hand mo34)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo34 routed)
    (ai_glm_5_3__mla streams at35 pos)
    (ai_glm_5_3__route streams mo35 hand picks)
    (sys__compute 1 names xf35_0)
    (sys__compute 2 names xf35_1)
    (ai_glm_5_3__shared hand mo35)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo35 routed)
    (ai_glm_5_3__kda streams at36)
    (ai_glm_5_3__route streams mo36 hand picks)
    (sys__compute 1 names xf36_0)
    (sys__compute 2 names xf36_1)
    (ai_glm_5_3__shared hand mo36)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo36 routed)
    (ai_glm_5_3__kda streams at37)
    (ai_glm_5_3__route streams mo37 hand picks)
    (sys__compute 1 names xf37_0)
    (sys__compute 2 names xf37_1)
    (ai_glm_5_3__shared hand mo37)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo37 routed)
    (ai_glm_5_3__kda streams at38)
    (ai_glm_5_3__route streams mo38 hand picks)
    (sys__compute 1 names xf38_0)
    (sys__compute 2 names xf38_1)
    (ai_glm_5_3__shared hand mo38)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo38 routed)
    (ai_glm_5_3__mla streams at39 pos)
    (ai_glm_5_3__route streams mo39 hand picks)
    (sys__compute 1 names xf39_0)
    (sys__compute 2 names xf39_1)
    (ai_glm_5_3__shared hand mo39)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo39 routed)
    (ai_glm_5_3__kda streams at40)
    (ai_glm_5_3__route streams mo40 hand picks)
    (sys__compute 1 names xf40_0)
    (sys__compute 2 names xf40_1)
    (ai_glm_5_3__shared hand mo40)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo40 routed)
    (ai_glm_5_3__kda streams at41)
    (ai_glm_5_3__route streams mo41 hand picks)
    (sys__compute 1 names xf41_0)
    (sys__compute 2 names xf41_1)
    (ai_glm_5_3__shared hand mo41)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo41 routed)
    (ai_glm_5_3__kda streams at42)
    (ai_glm_5_3__route streams mo42 hand picks)
    (sys__compute 1 names xf42_0)
    (sys__compute 2 names xf42_1)
    (ai_glm_5_3__shared hand mo42)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo42 routed)
    (ai_glm_5_3__mla streams at43 pos)
    (ai_glm_5_3__route streams mo43 hand picks)
    (sys__compute 1 names xf43_0)
    (sys__compute 2 names xf43_1)
    (ai_glm_5_3__shared hand mo43)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo43 routed)
    (ai_glm_5_3__kda streams at44)
    (ai_glm_5_3__route streams mo44 hand picks)
    (sys__compute 1 names xf44_0)
    (sys__compute 2 names xf44_1)
    (ai_glm_5_3__shared hand mo44)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo44 routed)
    (head)))

(defun (token_f_logits codes scale pos) (begin
    (embed codes scale)
    (ai_glm_5_3__kda streams at0)
    (ai_glm_5_3__mlp streams ml0)
    (ai_glm_5_3__kda streams at1)
    (ai_glm_5_3__mlp streams ml1)
    (ai_glm_5_3__kda streams at2)
    (ai_glm_5_3__mlp streams ml2)
    (ai_glm_5_3__mla streams at3 pos)
    (ai_glm_5_3__route streams mo3 hand picks)
    (sys__compute 1 names xf3_0)
    (sys__compute 2 names xf3_1)
    (ai_glm_5_3__shared hand mo3)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo3 routed)
    (ai_glm_5_3__kda streams at4)
    (ai_glm_5_3__route streams mo4 hand picks)
    (sys__compute 1 names xf4_0)
    (sys__compute 2 names xf4_1)
    (ai_glm_5_3__shared hand mo4)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo4 routed)
    (ai_glm_5_3__kda streams at5)
    (ai_glm_5_3__route streams mo5 hand picks)
    (sys__compute 1 names xf5_0)
    (sys__compute 2 names xf5_1)
    (ai_glm_5_3__shared hand mo5)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo5 routed)
    (ai_glm_5_3__kda streams at6)
    (ai_glm_5_3__route streams mo6 hand picks)
    (sys__compute 1 names xf6_0)
    (sys__compute 2 names xf6_1)
    (ai_glm_5_3__shared hand mo6)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo6 routed)
    (ai_glm_5_3__mla streams at7 pos)
    (ai_glm_5_3__route streams mo7 hand picks)
    (sys__compute 1 names xf7_0)
    (sys__compute 2 names xf7_1)
    (ai_glm_5_3__shared hand mo7)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo7 routed)
    (ai_glm_5_3__kda streams at8)
    (ai_glm_5_3__route streams mo8 hand picks)
    (sys__compute 1 names xf8_0)
    (sys__compute 2 names xf8_1)
    (ai_glm_5_3__shared hand mo8)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo8 routed)
    (ai_glm_5_3__kda streams at9)
    (ai_glm_5_3__route streams mo9 hand picks)
    (sys__compute 1 names xf9_0)
    (sys__compute 2 names xf9_1)
    (ai_glm_5_3__shared hand mo9)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo9 routed)
    (ai_glm_5_3__kda streams at10)
    (ai_glm_5_3__route streams mo10 hand picks)
    (sys__compute 1 names xf10_0)
    (sys__compute 2 names xf10_1)
    (ai_glm_5_3__shared hand mo10)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo10 routed)
    (ai_glm_5_3__mla streams at11 pos)
    (ai_glm_5_3__route streams mo11 hand picks)
    (sys__compute 1 names xf11_0)
    (sys__compute 2 names xf11_1)
    (ai_glm_5_3__shared hand mo11)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo11 routed)
    (ai_glm_5_3__kda streams at12)
    (ai_glm_5_3__route streams mo12 hand picks)
    (sys__compute 1 names xf12_0)
    (sys__compute 2 names xf12_1)
    (ai_glm_5_3__shared hand mo12)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo12 routed)
    (ai_glm_5_3__kda streams at13)
    (ai_glm_5_3__route streams mo13 hand picks)
    (sys__compute 1 names xf13_0)
    (sys__compute 2 names xf13_1)
    (ai_glm_5_3__shared hand mo13)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo13 routed)
    (ai_glm_5_3__kda streams at14)
    (ai_glm_5_3__route streams mo14 hand picks)
    (sys__compute 1 names xf14_0)
    (sys__compute 2 names xf14_1)
    (ai_glm_5_3__shared hand mo14)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo14 routed)
    (ai_glm_5_3__mla streams at15 pos)
    (ai_glm_5_3__route streams mo15 hand picks)
    (sys__compute 1 names xf15_0)
    (sys__compute 2 names xf15_1)
    (ai_glm_5_3__shared hand mo15)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo15 routed)
    (ai_glm_5_3__kda streams at16)
    (ai_glm_5_3__route streams mo16 hand picks)
    (sys__compute 1 names xf16_0)
    (sys__compute 2 names xf16_1)
    (ai_glm_5_3__shared hand mo16)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo16 routed)
    (ai_glm_5_3__kda streams at17)
    (ai_glm_5_3__route streams mo17 hand picks)
    (sys__compute 1 names xf17_0)
    (sys__compute 2 names xf17_1)
    (ai_glm_5_3__shared hand mo17)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo17 routed)
    (ai_glm_5_3__kda streams at18)
    (ai_glm_5_3__route streams mo18 hand picks)
    (sys__compute 1 names xf18_0)
    (sys__compute 2 names xf18_1)
    (ai_glm_5_3__shared hand mo18)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo18 routed)
    (ai_glm_5_3__mla streams at19 pos)
    (ai_glm_5_3__route streams mo19 hand picks)
    (sys__compute 1 names xf19_0)
    (sys__compute 2 names xf19_1)
    (ai_glm_5_3__shared hand mo19)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo19 routed)
    (ai_glm_5_3__kda streams at20)
    (ai_glm_5_3__route streams mo20 hand picks)
    (sys__compute 1 names xf20_0)
    (sys__compute 2 names xf20_1)
    (ai_glm_5_3__shared hand mo20)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo20 routed)
    (ai_glm_5_3__kda streams at21)
    (ai_glm_5_3__route streams mo21 hand picks)
    (sys__compute 1 names xf21_0)
    (sys__compute 2 names xf21_1)
    (ai_glm_5_3__shared hand mo21)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo21 routed)
    (ai_glm_5_3__kda streams at22)
    (ai_glm_5_3__route streams mo22 hand picks)
    (sys__compute 1 names xf22_0)
    (sys__compute 2 names xf22_1)
    (ai_glm_5_3__shared hand mo22)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo22 routed)
    (ai_glm_5_3__mla streams at23 pos)
    (ai_glm_5_3__route streams mo23 hand picks)
    (sys__compute 1 names xf23_0)
    (sys__compute 2 names xf23_1)
    (ai_glm_5_3__shared hand mo23)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo23 routed)
    (ai_glm_5_3__kda streams at24)
    (ai_glm_5_3__route streams mo24 hand picks)
    (sys__compute 1 names xf24_0)
    (sys__compute 2 names xf24_1)
    (ai_glm_5_3__shared hand mo24)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo24 routed)
    (ai_glm_5_3__kda streams at25)
    (ai_glm_5_3__route streams mo25 hand picks)
    (sys__compute 1 names xf25_0)
    (sys__compute 2 names xf25_1)
    (ai_glm_5_3__shared hand mo25)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo25 routed)
    (ai_glm_5_3__kda streams at26)
    (ai_glm_5_3__route streams mo26 hand picks)
    (sys__compute 1 names xf26_0)
    (sys__compute 2 names xf26_1)
    (ai_glm_5_3__shared hand mo26)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo26 routed)
    (ai_glm_5_3__mla streams at27 pos)
    (ai_glm_5_3__route streams mo27 hand picks)
    (sys__compute 1 names xf27_0)
    (sys__compute 2 names xf27_1)
    (ai_glm_5_3__shared hand mo27)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo27 routed)
    (ai_glm_5_3__kda streams at28)
    (ai_glm_5_3__route streams mo28 hand picks)
    (sys__compute 1 names xf28_0)
    (sys__compute 2 names xf28_1)
    (ai_glm_5_3__shared hand mo28)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo28 routed)
    (ai_glm_5_3__kda streams at29)
    (ai_glm_5_3__route streams mo29 hand picks)
    (sys__compute 1 names xf29_0)
    (sys__compute 2 names xf29_1)
    (ai_glm_5_3__shared hand mo29)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo29 routed)
    (ai_glm_5_3__kda streams at30)
    (ai_glm_5_3__route streams mo30 hand picks)
    (sys__compute 1 names xf30_0)
    (sys__compute 2 names xf30_1)
    (ai_glm_5_3__shared hand mo30)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo30 routed)
    (ai_glm_5_3__mla streams at31 pos)
    (ai_glm_5_3__route streams mo31 hand picks)
    (sys__compute 1 names xf31_0)
    (sys__compute 2 names xf31_1)
    (ai_glm_5_3__shared hand mo31)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo31 routed)
    (ai_glm_5_3__kda streams at32)
    (ai_glm_5_3__route streams mo32 hand picks)
    (sys__compute 1 names xf32_0)
    (sys__compute 2 names xf32_1)
    (ai_glm_5_3__shared hand mo32)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo32 routed)
    (ai_glm_5_3__kda streams at33)
    (ai_glm_5_3__route streams mo33 hand picks)
    (sys__compute 1 names xf33_0)
    (sys__compute 2 names xf33_1)
    (ai_glm_5_3__shared hand mo33)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo33 routed)
    (ai_glm_5_3__kda streams at34)
    (ai_glm_5_3__route streams mo34 hand picks)
    (sys__compute 1 names xf34_0)
    (sys__compute 2 names xf34_1)
    (ai_glm_5_3__shared hand mo34)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo34 routed)
    (ai_glm_5_3__mla streams at35 pos)
    (ai_glm_5_3__route streams mo35 hand picks)
    (sys__compute 1 names xf35_0)
    (sys__compute 2 names xf35_1)
    (ai_glm_5_3__shared hand mo35)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo35 routed)
    (ai_glm_5_3__kda streams at36)
    (ai_glm_5_3__route streams mo36 hand picks)
    (sys__compute 1 names xf36_0)
    (sys__compute 2 names xf36_1)
    (ai_glm_5_3__shared hand mo36)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo36 routed)
    (ai_glm_5_3__kda streams at37)
    (ai_glm_5_3__route streams mo37 hand picks)
    (sys__compute 1 names xf37_0)
    (sys__compute 2 names xf37_1)
    (ai_glm_5_3__shared hand mo37)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo37 routed)
    (ai_glm_5_3__kda streams at38)
    (ai_glm_5_3__route streams mo38 hand picks)
    (sys__compute 1 names xf38_0)
    (sys__compute 2 names xf38_1)
    (ai_glm_5_3__shared hand mo38)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo38 routed)
    (ai_glm_5_3__mla streams at39 pos)
    (ai_glm_5_3__route streams mo39 hand picks)
    (sys__compute 1 names xf39_0)
    (sys__compute 2 names xf39_1)
    (ai_glm_5_3__shared hand mo39)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo39 routed)
    (ai_glm_5_3__kda streams at40)
    (ai_glm_5_3__route streams mo40 hand picks)
    (sys__compute 1 names xf40_0)
    (sys__compute 2 names xf40_1)
    (ai_glm_5_3__shared hand mo40)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo40 routed)
    (ai_glm_5_3__kda streams at41)
    (ai_glm_5_3__route streams mo41 hand picks)
    (sys__compute 1 names xf41_0)
    (sys__compute 2 names xf41_1)
    (ai_glm_5_3__shared hand mo41)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo41 routed)
    (ai_glm_5_3__kda streams at42)
    (ai_glm_5_3__route streams mo42 hand picks)
    (sys__compute 1 names xf42_0)
    (sys__compute 2 names xf42_1)
    (ai_glm_5_3__shared hand mo42)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo42 routed)
    (ai_glm_5_3__mla streams at43 pos)
    (ai_glm_5_3__route streams mo43 hand picks)
    (sys__compute 1 names xf43_0)
    (sys__compute 2 names xf43_1)
    (ai_glm_5_3__shared hand mo43)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo43 routed)
    (ai_glm_5_3__kda streams at44)
    (ai_glm_5_3__route streams mo44 hand picks)
    (sys__compute 1 names xf44_0)
    (sys__compute 2 names xf44_1)
    (ai_glm_5_3__shared hand mo44)
    (sys__result 1)
    (sys__result 2)
    (nn__buffer__copy c_routed 1 routed 8192)
    (nn__buffer__copy c1_routed 2 routed1 8192)
    (nn__vector__add routed routed1 routed 4096)
    (ai_glm_5_3__close streams mo44 routed)
    (head_logits)))

(defun (token codes scale at len) (begin
    (embed codes scale)
    (a0 at len)
    (a1 at len)
    (a2 at len)
    (a3 at len)
    (sys__compute 1 names xw3)
    (s3)
    (sys__result 1)
    (c3)
    (a4 at len)
    (sys__compute 1 names xw4)
    (s4)
    (sys__result 1)
    (c4)
    (a5 at len)
    (sys__compute 1 names xw5)
    (s5)
    (sys__result 1)
    (c5)
    (a6 at len)
    (sys__compute 1 names xw6)
    (s6)
    (sys__result 1)
    (c6)
    (a7 at len)
    (sys__compute 1 names xw7)
    (s7)
    (sys__result 1)
    (c7)
    (a8 at len)
    (sys__compute 1 names xw8)
    (s8)
    (sys__result 1)
    (c8)
    (a9 at len)
    (sys__compute 1 names xw9)
    (s9)
    (sys__result 1)
    (c9)
    (a10 at len)
    (sys__compute 1 names xw10)
    (s10)
    (sys__result 1)
    (c10)
    (a11 at len)
    (sys__compute 1 names xw11)
    (s11)
    (sys__result 1)
    (c11)
    (a12 at len)
    (sys__compute 1 names xw12)
    (s12)
    (sys__result 1)
    (c12)
    (a13 at len)
    (sys__compute 1 names xw13)
    (s13)
    (sys__result 1)
    (c13)
    (a14 at len)
    (sys__compute 1 names xw14)
    (s14)
    (sys__result 1)
    (c14)
    (a15 at len)
    (sys__compute 1 names xw15)
    (s15)
    (sys__result 1)
    (c15)
    (a16 at len)
    (sys__compute 1 names xw16)
    (s16)
    (sys__result 1)
    (c16)
    (a17 at len)
    (sys__compute 1 names xw17)
    (s17)
    (sys__result 1)
    (c17)
    (a18 at len)
    (sys__compute 1 names xw18)
    (s18)
    (sys__result 1)
    (c18)
    (a19 at len)
    (sys__compute 1 names xw19)
    (s19)
    (sys__result 1)
    (c19)
    (a20 at len)
    (sys__compute 1 names xw20)
    (s20)
    (sys__result 1)
    (c20)
    (a21 at len)
    (sys__compute 1 names xw21)
    (s21)
    (sys__result 1)
    (c21)
    (a22 at len)
    (sys__compute 1 names xw22)
    (s22)
    (sys__result 1)
    (c22)
    (a23 at len)
    (sys__compute 1 names xw23)
    (s23)
    (sys__result 1)
    (c23)
    (a24 at len)
    (sys__compute 1 names xw24)
    (s24)
    (sys__result 1)
    (c24)
    (a25 at len)
    (sys__compute 1 names xw25)
    (s25)
    (sys__result 1)
    (c25)
    (a26 at len)
    (sys__compute 1 names xw26)
    (s26)
    (sys__result 1)
    (c26)
    (a27 at len)
    (sys__compute 1 names xw27)
    (s27)
    (sys__result 1)
    (c27)
    (a28 at len)
    (sys__compute 1 names xw28)
    (s28)
    (sys__result 1)
    (c28)
    (a29 at len)
    (sys__compute 1 names xw29)
    (s29)
    (sys__result 1)
    (c29)
    (a30 at len)
    (sys__compute 1 names xw30)
    (s30)
    (sys__result 1)
    (c30)
    (a31 at len)
    (sys__compute 1 names xw31)
    (s31)
    (sys__result 1)
    (c31)
    (a32 at len)
    (sys__compute 1 names xw32)
    (s32)
    (sys__result 1)
    (c32)
    (a33 at len)
    (sys__compute 1 names xw33)
    (s33)
    (sys__result 1)
    (c33)
    (a34 at len)
    (sys__compute 1 names xw34)
    (s34)
    (sys__result 1)
    (c34)
    (a35 at len)
    (sys__compute 1 names xw35)
    (s35)
    (sys__result 1)
    (c35)
    (a36 at len)
    (sys__compute 1 names xw36)
    (s36)
    (sys__result 1)
    (c36)
    (a37 at len)
    (sys__compute 1 names xw37)
    (s37)
    (sys__result 1)
    (c37)
    (a38 at len)
    (sys__compute 1 names xw38)
    (s38)
    (sys__result 1)
    (c38)
    (a39 at len)
    (sys__compute 1 names xw39)
    (s39)
    (sys__result 1)
    (c39)
    (a40 at len)
    (sys__compute 1 names xw40)
    (s40)
    (sys__result 1)
    (c40)
    (a41 at len)
    (sys__compute 1 names xw41)
    (s41)
    (sys__result 1)
    (c41)
    (a42 at len)
    (sys__compute 1 names xw42)
    (s42)
    (sys__result 1)
    (c42)
    (a43 at len)
    (sys__compute 1 names xw43)
    (s43)
    (sys__result 1)
    (c43)
    (a44 at len)
    (sys__compute 1 names xw44)
    (s44)
    (sys__result 1)
    (c44)
    (head)))

(defun (token_logits codes scale at len) (begin
    (embed codes scale)
    (a0 at len)
    (a1 at len)
    (a2 at len)
    (a3 at len)
    (sys__compute 1 names xw3)
    (s3)
    (sys__result 1)
    (c3)
    (a4 at len)
    (sys__compute 1 names xw4)
    (s4)
    (sys__result 1)
    (c4)
    (a5 at len)
    (sys__compute 1 names xw5)
    (s5)
    (sys__result 1)
    (c5)
    (a6 at len)
    (sys__compute 1 names xw6)
    (s6)
    (sys__result 1)
    (c6)
    (a7 at len)
    (sys__compute 1 names xw7)
    (s7)
    (sys__result 1)
    (c7)
    (a8 at len)
    (sys__compute 1 names xw8)
    (s8)
    (sys__result 1)
    (c8)
    (a9 at len)
    (sys__compute 1 names xw9)
    (s9)
    (sys__result 1)
    (c9)
    (a10 at len)
    (sys__compute 1 names xw10)
    (s10)
    (sys__result 1)
    (c10)
    (a11 at len)
    (sys__compute 1 names xw11)
    (s11)
    (sys__result 1)
    (c11)
    (a12 at len)
    (sys__compute 1 names xw12)
    (s12)
    (sys__result 1)
    (c12)
    (a13 at len)
    (sys__compute 1 names xw13)
    (s13)
    (sys__result 1)
    (c13)
    (a14 at len)
    (sys__compute 1 names xw14)
    (s14)
    (sys__result 1)
    (c14)
    (a15 at len)
    (sys__compute 1 names xw15)
    (s15)
    (sys__result 1)
    (c15)
    (a16 at len)
    (sys__compute 1 names xw16)
    (s16)
    (sys__result 1)
    (c16)
    (a17 at len)
    (sys__compute 1 names xw17)
    (s17)
    (sys__result 1)
    (c17)
    (a18 at len)
    (sys__compute 1 names xw18)
    (s18)
    (sys__result 1)
    (c18)
    (a19 at len)
    (sys__compute 1 names xw19)
    (s19)
    (sys__result 1)
    (c19)
    (a20 at len)
    (sys__compute 1 names xw20)
    (s20)
    (sys__result 1)
    (c20)
    (a21 at len)
    (sys__compute 1 names xw21)
    (s21)
    (sys__result 1)
    (c21)
    (a22 at len)
    (sys__compute 1 names xw22)
    (s22)
    (sys__result 1)
    (c22)
    (a23 at len)
    (sys__compute 1 names xw23)
    (s23)
    (sys__result 1)
    (c23)
    (a24 at len)
    (sys__compute 1 names xw24)
    (s24)
    (sys__result 1)
    (c24)
    (a25 at len)
    (sys__compute 1 names xw25)
    (s25)
    (sys__result 1)
    (c25)
    (a26 at len)
    (sys__compute 1 names xw26)
    (s26)
    (sys__result 1)
    (c26)
    (a27 at len)
    (sys__compute 1 names xw27)
    (s27)
    (sys__result 1)
    (c27)
    (a28 at len)
    (sys__compute 1 names xw28)
    (s28)
    (sys__result 1)
    (c28)
    (a29 at len)
    (sys__compute 1 names xw29)
    (s29)
    (sys__result 1)
    (c29)
    (a30 at len)
    (sys__compute 1 names xw30)
    (s30)
    (sys__result 1)
    (c30)
    (a31 at len)
    (sys__compute 1 names xw31)
    (s31)
    (sys__result 1)
    (c31)
    (a32 at len)
    (sys__compute 1 names xw32)
    (s32)
    (sys__result 1)
    (c32)
    (a33 at len)
    (sys__compute 1 names xw33)
    (s33)
    (sys__result 1)
    (c33)
    (a34 at len)
    (sys__compute 1 names xw34)
    (s34)
    (sys__result 1)
    (c34)
    (a35 at len)
    (sys__compute 1 names xw35)
    (s35)
    (sys__result 1)
    (c35)
    (a36 at len)
    (sys__compute 1 names xw36)
    (s36)
    (sys__result 1)
    (c36)
    (a37 at len)
    (sys__compute 1 names xw37)
    (s37)
    (sys__result 1)
    (c37)
    (a38 at len)
    (sys__compute 1 names xw38)
    (s38)
    (sys__result 1)
    (c38)
    (a39 at len)
    (sys__compute 1 names xw39)
    (s39)
    (sys__result 1)
    (c39)
    (a40 at len)
    (sys__compute 1 names xw40)
    (s40)
    (sys__result 1)
    (c40)
    (a41 at len)
    (sys__compute 1 names xw41)
    (s41)
    (sys__result 1)
    (c41)
    (a42 at len)
    (sys__compute 1 names xw42)
    (s42)
    (sys__result 1)
    (c42)
    (a43 at len)
    (sys__compute 1 names xw43)
    (s43)
    (sys__result 1)
    (c43)
    (a44 at len)
    (sys__compute 1 names xw44)
    (s44)
    (sys__result 1)
    (c44)
    (head_logits)))

