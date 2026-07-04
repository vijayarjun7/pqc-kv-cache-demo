# H2 Experiment — Full Pipeline Benchmark

## H2 Hypothesis
Net latency improvement is measurable and consistent at d=256 on commodity hardware after accounting for rotation overhead — compared to vanilla Kyber-512 baseline (~45,200 AVX2 cycles per encapsulation).

**Note:** this run tests d=64 (real Llama-3.2-1B head dimension from experiments/03_real_kv), not d=256 as in the original H2 statement. Cycle counts are not reported — only wall-clock μs, measured via `time.perf_counter_ns()` — since no hardware cycle counter (e.g. rdtsc) is used here, μs and the cited AVX2 cycle figure are not directly comparable.

## Important Caveat
liboqs's public `encap_secret()` is a black box: it generates its own internal randomness and does not accept the rotated KV vector as input. This benchmark cannot make Kyber's actual NTT run *on* the rotated data — that requires an instrumented liboqs build (flagged as separate future work under H1-efficiency). What is measured here is the **practical end-to-end pipeline cost**: rotation + quantization + encoding overhead, stacked on top of an unchanged Kyber-512 call, compared against calling Kyber-512 alone. A negative result here does not disconfirm H1-efficiency's internal reduction-pressure claim — it only shows that, via the public Python bindings, added preprocessing cannot make a fixed-cost black-box operation faster.

## H2 Verdict (practical pipeline version)
**NOT SUPPORTED ❌ in Python implementation**

- Vanilla Kyber-512 mean:        6.25 μs
- Rotated pipeline mean (total): 53.22 μs
  - Preprocessing (rotate+quantize+encode): 46.41 μs
  - Kyber-512 encapsulate component:        6.81 μs
- Net delta: -751.8% (regression)

**Reason:** Python preprocessing overhead dominates at d=64 versus compiled C
Kyber-512.

**What this means:** QuantRot-PQC preprocessing cost is much greater than
Kyber's own cost at d=64, in this pure-Python implementation — not
necessarily a property of the algorithm itself.

**What this does NOT mean:**
- HD³ rotation is inherently expensive.
- A C/CUDA implementation would show the same results.
- The approach is infeasible in production.

## Vectorization Comparison (d=64)

| Implementation | Preprocessing | Total pipeline | Net delta |
|---|---|---|---|
| Pure Python loop | 110.89 μs | 117.94 μs | -1954% |
| Matmul (BLAS) | 32.01 μs | 40.46 μs | -560% |
| Reshape FWHT | 46.41 μs | 53.22 μs | -752% |
| Vanilla Kyber-512 | — | 6.25 μs | baseline |

At d=64, Kyber-512's compiled C implementation is faster than any of the
Python preprocessing variants tried here. The matmul approach (32.01 μs) beat
the asymptotically-better reshape FWHT (~46 μs) at this scale — a single
BLAS matmul call outperforms 18 separate numpy reshape/slice/copy operations
(6 levels × 3 rotation applications) once per-call Python/numpy overhead is
accounted for. A production path — a compiled HD³ rotation kernel (C
extension or CUDA) — would be expected to bring preprocessing to well under
1 μs, comparable to Kyber's own cost, but that is not something this
Python benchmark can demonstrate directly.

## Full Latency Stats (μs)

| Stage | Mean | Median | Std | Min | Max |
|-------|------|--------|-----|-----|-----|
| Vanilla Kyber-512 | 6.25 | 5.88 | 1.17 | 5.71 | 27.17 |
| Preprocessing (rotate+quantize+encode) | 46.41 | 45.54 | 4.17 | 42.83 | 98.00 |
| Kyber-512 component (pipeline) | 6.81 | 6.62 | 0.67 | 6.29 | 15.62 |
| Total pipeline | 53.22 | 52.31 | 4.68 | 49.17 | 113.62 |

## Experimental Setup
- Trials: 500 (one real KV vector per trial)
- KV vectors: experiments/03_real_kv/real_kv_vectors.npy
- Dimension d: 64
- HD³ applications: 3
- Warmup trials (uncounted): 5
- Timer: time.perf_counter_ns()

## Interpretation
The rotated pipeline costs more end-to-end than vanilla Kyber-512 alone, as expected: rotation/quantization/encoding add real Python-side overhead on top of an unchanged, fixed-cost Kyber call. This does not test — and cannot disconfirm — H1-efficiency's claim about reduced internal NTT reduction pressure, which requires an instrumented liboqs build to observe directly.

## Next Step
To test H1-efficiency and the cycle-level version of H2 properly, instrument liboqs's Kyber NTT implementation directly (e.g. count conditional Barrett/Montgomery reductions per call) and compare rotated vs unrotated coefficient inputs at that level.