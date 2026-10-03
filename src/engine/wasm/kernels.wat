;; NeuroViz conv kernels: WebAssembly with 128-bit SIMD (two float64 lanes).
;;
;; Rebuild src/engine/wasm/kernels.ts after editing:  npm run build:wasm
;;
;; Every function works on byte offsets into the imported memory. Arrays are
;; row-major float32 except `acc`, a float64 scratch area.
;;
;; Exactness: the matmuls compute each output element as the float64 sum, in
;; ascending p, of float32 × float32 products (exact in float64), rounded to
;; float32 once: the same arithmetic as the JS kernels in tensor.ts, so the
;; results are bitwise identical. SIMD runs across output elements (j), never
;; across the summation, and there is no fused multiply-add. A zero a-value's
;; products are skipped only when `skip` is set (the caller sets it when B is
;; all finite, as tensor.ts does, so 0 · Inf still gives NaN).
(module
  (import "env" "memory" (memory 1))

  ;; acc[0 .. n) += av · B[0 .. n)   (float64 acc, float32 B)
  ;; (Unrolling to two vectors per step was tried: no measurable gain.)
  (func $axpy (param $acc i32) (param $B i32) (param $n i32) (param $av f64)
    (local $j i32) (local $n2 i32) (local $avv v128) (local $pa i32)
    (local.set $avv (f64x2.splat (local.get $av)))
    (local.set $n2 (i32.and (local.get $n) (i32.const -2)))
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $j) (local.get $n2)))
        (local.set $pa (i32.add (local.get $acc) (i32.shl (local.get $j) (i32.const 3))))
        (v128.store
          (local.get $pa)
          (f64x2.add
            (v128.load (local.get $pa))
            (f64x2.mul
              (local.get $avv)
              (f64x2.promote_low_f32x4
                (v128.load64_zero
                  (i32.add (local.get $B) (i32.shl (local.get $j) (i32.const 2))))))))
        (local.set $j (i32.add (local.get $j) (i32.const 2)))
        (br $next)))
    (if (i32.lt_u (local.get $j) (local.get $n))
      (then
        (local.set $pa (i32.add (local.get $acc) (i32.shl (local.get $j) (i32.const 3))))
        (f64.store
          (local.get $pa)
          (f64.add
            (f64.load (local.get $pa))
            (f64.mul
              (local.get $av)
              (f64.promote_f32
                (f32.load (i32.add (local.get $B) (i32.shl (local.get $j) (i32.const 2)))))))))))

  ;; C[0 .. n) = float32(acc[0 .. n))
  (func $store (param $C i32) (param $acc i32) (param $n i32)
    (local $j i32)
    (block $done
      (loop $next
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (f32.store
          (i32.add (local.get $C) (i32.shl (local.get $j) (i32.const 2)))
          (f32.demote_f64
            (f64.load (i32.add (local.get $acc) (i32.shl (local.get $j) (i32.const 3))))))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $next))))

  ;; C[m, n] = A[m, k] · B[k, n]. acc: n float64s.
  (func (export "nn")
    (param $A i32) (param $B i32) (param $C i32) (param $acc i32)
    (param $m i32) (param $k i32) (param $n i32) (param $skip i32)
    (local $i i32) (local $p i32) (local $av f64) (local $arow i32)
    (block $done_i
      (loop $next_i
        (br_if $done_i (i32.ge_u (local.get $i) (local.get $m)))
        (memory.fill (local.get $acc) (i32.const 0) (i32.shl (local.get $n) (i32.const 3)))
        (local.set $arow
          (i32.add (local.get $A) (i32.shl (i32.mul (local.get $i) (local.get $k)) (i32.const 2))))
        (local.set $p (i32.const 0))
        (block $done_p
          (loop $next_p
            (br_if $done_p (i32.ge_u (local.get $p) (local.get $k)))
            (local.set $av
              (f64.promote_f32
                (f32.load (i32.add (local.get $arow) (i32.shl (local.get $p) (i32.const 2))))))
            (if (i32.eqz (i32.and (local.get $skip) (f64.eq (local.get $av) (f64.const 0))))
              (then
                (call $axpy
                  (local.get $acc)
                  (i32.add
                    (local.get $B)
                    (i32.shl (i32.mul (local.get $p) (local.get $n)) (i32.const 2)))
                  (local.get $n)
                  (local.get $av))))
            (local.set $p (i32.add (local.get $p) (i32.const 1)))
            (br $next_p)))
        (call $store
          (i32.add (local.get $C) (i32.shl (i32.mul (local.get $i) (local.get $n)) (i32.const 2)))
          (local.get $acc)
          (local.get $n))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $next_i))))

  ;; C[m, n] = Aᵀ · B with A [k, m], B [k, n]. acc: m·n float64s.
  (func (export "tn")
    (param $A i32) (param $B i32) (param $C i32) (param $acc i32)
    (param $m i32) (param $k i32) (param $n i32) (param $skip i32)
    (local $i i32) (local $p i32) (local $av f64) (local $arow i32) (local $brow i32)
    (memory.fill
      (local.get $acc)
      (i32.const 0)
      (i32.shl (i32.mul (local.get $m) (local.get $n)) (i32.const 3)))
    (block $done_p
      (loop $next_p
        (br_if $done_p (i32.ge_u (local.get $p) (local.get $k)))
        (local.set $arow
          (i32.add (local.get $A) (i32.shl (i32.mul (local.get $p) (local.get $m)) (i32.const 2))))
        (local.set $brow
          (i32.add (local.get $B) (i32.shl (i32.mul (local.get $p) (local.get $n)) (i32.const 2))))
        (local.set $i (i32.const 0))
        (block $done_i
          (loop $next_i
            (br_if $done_i (i32.ge_u (local.get $i) (local.get $m)))
            (local.set $av
              (f64.promote_f32
                (f32.load (i32.add (local.get $arow) (i32.shl (local.get $i) (i32.const 2))))))
            (if (i32.eqz (i32.and (local.get $skip) (f64.eq (local.get $av) (f64.const 0))))
              (then
                (call $axpy
                  (i32.add
                    (local.get $acc)
                    (i32.shl (i32.mul (local.get $i) (local.get $n)) (i32.const 3)))
                  (local.get $brow)
                  (local.get $n)
                  (local.get $av))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $next_i)))
        (local.set $p (i32.add (local.get $p) (i32.const 1)))
        (br $next_p)))
    (call $store (local.get $C) (local.get $acc) (i32.mul (local.get $m) (local.get $n))))

  ;; im2col (add = 0) or col2im (add = 1) for images [b0, b0 + m) of x [N, c, h, w]:
  ;; col [m·oh·ow, c·k·k], one row per output position, zeros over the padded edge.
  ;; col2im adds each entry of col back into x (float32 adds, in the same order as the JS loop).
  (func (export "patches")
    (param $x i32) (param $col i32) (param $b0 i32) (param $m i32)
    (param $c i32) (param $h i32) (param $w i32)
    (param $k i32) (param $s i32) (param $pad i32) (param $oh i32) (param $ow i32) (param $add i32)
    (local $b i32) (local $oy i32) (local $ox i32) (local $ci i32) (local $ky i32) (local $kx i32)
    (local $iy i32) (local $ix i32) (local $o i32) (local $plane i32) (local $ro i32)
    (local $px i32) (local $inside i32) (local $bend i32)
    (local.set $o (local.get $col))
    (local.set $b (local.get $b0))
    (local.set $bend (i32.add (local.get $b0) (local.get $m)))
    (block $done_b (loop $next_b
      (br_if $done_b (i32.ge_u (local.get $b) (local.get $bend)))
      (local.set $oy (i32.const 0))
      (block $done_oy (loop $next_oy
        (br_if $done_oy (i32.ge_u (local.get $oy) (local.get $oh)))
        (local.set $ox (i32.const 0))
        (block $done_ox (loop $next_ox
          (br_if $done_ox (i32.ge_u (local.get $ox) (local.get $ow)))
          (local.set $ci (i32.const 0))
          (block $done_ci (loop $next_ci
            (br_if $done_ci (i32.ge_u (local.get $ci) (local.get $c)))
            (local.set $plane
              (i32.mul (i32.add (i32.mul (local.get $b) (local.get $c)) (local.get $ci)) (local.get $h)))
            (local.set $ky (i32.const 0))
            (block $done_ky (loop $next_ky
              (br_if $done_ky (i32.ge_u (local.get $ky) (local.get $k)))
              (local.set $iy
                (i32.add (i32.sub (i32.mul (local.get $oy) (local.get $s)) (local.get $pad)) (local.get $ky)))
              (if (i32.or (i32.lt_s (local.get $iy) (i32.const 0)) (i32.ge_s (local.get $iy) (local.get $h)))
                (then
                  (if (i32.eqz (local.get $add))
                    (then (memory.fill (local.get $o) (i32.const 0) (i32.shl (local.get $k) (i32.const 2)))))
                  (local.set $o (i32.add (local.get $o) (i32.shl (local.get $k) (i32.const 2)))))
                (else
                  (local.set $ro (i32.mul (i32.add (local.get $plane) (local.get $iy)) (local.get $w)))
                  (local.set $kx (i32.const 0))
                  (block $done_kx (loop $next_kx
                    (br_if $done_kx (i32.ge_u (local.get $kx) (local.get $k)))
                    (local.set $ix
                      (i32.add (i32.sub (i32.mul (local.get $ox) (local.get $s)) (local.get $pad)) (local.get $kx)))
                    (local.set $inside
                      (i32.and (i32.ge_s (local.get $ix) (i32.const 0)) (i32.lt_s (local.get $ix) (local.get $w))))
                    (local.set $px
                      (i32.add (local.get $x) (i32.shl (i32.add (local.get $ro) (local.get $ix)) (i32.const 2))))
                    (if (local.get $add)
                      (then
                        (if (local.get $inside)
                          (then (f32.store (local.get $px)
                            (f32.add (f32.load (local.get $px)) (f32.load (local.get $o)))))))
                      (else
                        (if (local.get $inside)
                          (then (f32.store (local.get $o) (f32.load (local.get $px))))
                          (else (f32.store (local.get $o) (f32.const 0))))))
                    (local.set $o (i32.add (local.get $o) (i32.const 4)))
                    (local.set $kx (i32.add (local.get $kx) (i32.const 1)))
                    (br $next_kx)))))
              (local.set $ky (i32.add (local.get $ky) (i32.const 1)))
              (br $next_ky)))
            (local.set $ci (i32.add (local.get $ci) (i32.const 1)))
            (br $next_ci)))
          (local.set $ox (i32.add (local.get $ox) (i32.const 1)))
          (br $next_ox)))
        (local.set $oy (i32.add (local.get $oy) (i32.const 1)))
        (br $next_oy)))
      (local.set $b (i32.add (local.get $b) (i32.const 1)))
      (br $next_b))))
)
