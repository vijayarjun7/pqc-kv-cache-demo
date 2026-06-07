# Benchmark Results

## demo.py -- Single-Run Benchmark

### KV-Cache Configuration

| Parameter | Value |
|-----------|-------|
| Layers | 32 |
| Tokens | 512 |
| Heads | 16 |
| Head dim | 64 |
| Total float32 values | 16,777,216 |
| Raw tensor size | ~64 MB |
| Random seed | 42 (deterministic) |

### Results

| Mode | Latency (s) | Peak Memory (MB) | vs Baseline |
|------|-------------|------------------|-------------|
| Baseline | ~0.058 | ~196 | -- |
| PQC (simulated Kyber) | ~9.982 | ~452 | +17,072% latency, +130% memory |
| Optimized PQC | ~2.889 | ~192 | -71.1% latency vs PQC, -57.5% memory vs PQC |

*Numbers are representative -- vary by CPU, memory bandwidth, and Python version.*

### What drives each number

**Baseline** -- single matrix projection (16384x1024 @ 1024x1024), layer-norm, tanh activation. Dominated by one BLAS matmul.

**PQC** -- 131,072 NTT blocks x (rfft + polynomial multiply + irfft + mod 3329). No vectorisation across blocks in demo.py; Python loop is the bottleneck, which amplifies the overhead relative to production.

**Optimized PQC** -- QR orthogonal rotation + int8 quantization reduces the float32 tensor to int8 (4x byte reduction), then byte-reinterpretation as float32 halves the block count again. Net: ~4x fewer NTT iterations -> ~71% latency recovery vs naive PQC.

---

## app.py -- API Load Simulator

The Streamlit dashboard (streamlit run app.py) sweeps 6 checkpoints from 1% to 100% of a configurable max request count and reports **per-request** average latency, throughput (req/s), and peak batch memory.

### Simulation constants (simulation.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| HEADS | 8 | attention heads per request |
| HEAD_DIM | 32 | per-head dimension |
| NTT_BLOCK | 32 | polynomial block size |
| KYBER_Q | 3329 | Kyber modulus |
| EXPANSION | 2 | ciphertext expansion factor |

### Typical output at payload_scale=4, max_requests=100

| Mode | Avg Latency | Throughput | Peak Memory |
|------|-------------|------------|-------------|
| Baseline | ~0.02 ms | ~800 req/s | ~0.5 MB |
| Secure | ~0.15 ms | ~120 req/s | ~1.2 MB |
| Optimized Secure | ~0.05 ms | ~380 req/s | ~0.6 MB |

*Exact numbers vary by hardware. Run `streamlit run app.py` to generate your own.*

---

## Reproducibility Notes

- `demo.py` sets `np.random.seed(42)` globally -- results are deterministic on the same hardware.
- `simulation.py` uses `np.random.default_rng(seed + i)` per request -- also deterministic per payload_scale/max_requests combination.
- Both tools measure wall-clock time with `time.perf_counter()` and memory with `tracemalloc` (Python heap only).
- Results scale approximately linearly with tensor size. Halving TOKENS or LAYERS roughly halves PQC latency.

---

## Simulation Limitations

These results are from a **NumPy FFT simulation**, not a production Kyber implementation.

| Aspect | This demo | Production Kyber |
|--------|-----------|-----------------|
| NTT arithmetic | numpy.fft (float64) | Integer NTT over Z_3329 |
| Key generation | Random complex array | Actual lattice key gen |
| Security | None (simulation) | 128-bit post-quantum |
| Constant factors | Higher (Python overhead) | Lower (C/SIMD) |
| Overhead ratio | Structurally plausible | Hardware-dependent |

The overhead *ratio* (PQC vs Baseline) is structurally sound. The absolute numbers are not comparable to production.
