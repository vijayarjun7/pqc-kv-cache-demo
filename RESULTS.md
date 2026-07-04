# Results

Benchmark output from the PQC KV-cache overhead demo. These numbers come from
running the simulation locally — see [Reproducibility](#reproducibility) to
regenerate them yourself.

## Hardware

<!-- Fill in with your machine's details before publishing results -->

- **Device:**
- **CPU:**
- **RAM:**
- **OS:**
- **Python version:**
- **NumPy version:**

## Results Table

| Mode          | Latency | Memory | Delta                                     |
|---------------|---------|--------|--------------------------------------------|
| Baseline      | ~0.06s  | ~196 MB | —                                          |
| PQC simulated | ~10.0s  | ~452 MB | +17,000% latency, +130% memory             |
| Optimized PQC | ~2.9s   | ~192 MB | -71% vs PQC, -57% vs PQC memory            |

The optimized path recovers most of the latency lost to the simulated PQC
overhead while landing at roughly the same memory footprint as the baseline.

## Reproducibility

- All runs use `np.random.seed(42)`, so the random matrices, key polynomials,
  and rotation matrices are identical across runs on the same hardware.
- Results are deterministic given the seed, but **absolute timings depend on
  hardware** (CPU architecture, clock speed, memory bandwidth, and BLAS/FFT
  backend). Relative deltas (percent overhead, percent speedup) should be
  fairly stable across machines; absolute seconds will not be.

## Expected Ranges by Hardware

| Hardware        | Baseline    | PQC simulated | Optimized PQC |
|-----------------|-------------|----------------|----------------|
| MacBook M1/M2    | 0.04–0.08s  | 6–12s          | 1.5–4s         |
| Intel i7/i9      | 0.05–0.1s   | 8–15s          | 2–5s           |
| Cloud GPU        | 0.02–0.05s  | 4–8s           | 1–3s           |

Note that this benchmark is CPU-bound (FFT and matmul on NumPy arrays), so
"Cloud GPU" here reflects the host CPU of a typical GPU instance, not GPU
acceleration — nothing in this demo runs on the GPU.

## What Each Mode Measures

### Baseline

- Flatten KV tensor to `(16384, 1024)`.
- Matmul with a random `(1024, 1024)` weight matrix `W`.
- Layer-norm surrogate + tanh activation.
- Measures raw LLM projection overhead with no cryptographic operations —
  this is the reference cost of touching the KV cache at all.

### PQC simulated

- Reshape to `(-1, 64)` → 131,072 blocks.
- Each block: forward FFT → multiply by a random complex key polynomial →
  inverse FFT → mod 3329.
- Allocate a 2x ciphertext expansion array.
- Measures the overhead shape of Kyber's NTT-based polynomial arithmetic
  applied per KV block, plus the memory cost of ciphertext expansion.

### Optimized PQC

- QR decomposition on a random `(64, 64)` matrix → orthogonal `Q`.
- Rotate all head vectors: `kv.reshape(-1, 64) @ Q.T`.
- int8-quantize with a global scale (4x byte reduction).
- Dequantize → run the same NTT loop on the smaller representation.
- Measures the overhead of a QuantRot-inspired optimization: rotate into a
  basis that concentrates values, quantize to int8, then pay the NTT cost on
  a representation that's a quarter the size.

## Simulation Methodology

This demo approximates cryptographic and optimization costs; it is **not** a
production PQC implementation. Specifically:

- The NTT (number-theoretic transform) used in real Kyber is over the ring
  `Z_q` with `q = 3329`. This demo uses NumPy's FFT as a stand-in — it has a
  comparable computational shape (O(n log n) butterfly structure) but is not
  a modular NTT and provides no actual security guarantees.
- The int8 optimization is genuinely faster: it produces a real speedup from
  reduced cache footprint, not a simulated one. Smaller data fits more
  comfortably in cache, which is why the optimized path's speedup is
  reproducible and not just a scaling artifact.
- The int8 → float32 dequantization step byte-reinterprets data so that 4
  int8 values pack into a single float32 slot, matching the 4x reduction
  claimed above.
- This is **not liboqs**, not constant-time, and not suitable for security
  claims of any kind. If you need real Kyber benchmarks, use
  [liboqs](https://github.com/open-quantum-safe/liboqs) or a vetted PQC
  library instead. This repo exists to make the *shape* of the overhead
  intuitive, not to certify performance of an actual PQC scheme.
