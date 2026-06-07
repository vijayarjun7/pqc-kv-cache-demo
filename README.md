# PQC KV-Cache Demo

> Simulates post-quantum cryptography overhead on LLM KV-cache inference — with a QuantRot-inspired optimization and an interactive Streamlit dashboard.

---

## What This Is

Post-quantum cryptographic schemes like **CRYSTALS-Kyber** protect LLM inference APIs from "harvest now, decrypt later" attacks — but their NTT polynomial arithmetic and ~2x ciphertext expansion add serious latency and memory overhead at inference time.

This repo contains two complementary tools:

1. **`demo.py`** — Single-run benchmark: measures Baseline vs PQC vs Optimized PQC on a full KV-cache tensor (32 layers x 512 tokens x 16 heads x 64 dim).
2. **`app.py` + `simulation.py`** — Interactive Streamlit dashboard: sweeps request volume (1-500 requests) and shows latency, memory, and throughput trends across all three modes.

The **QuantRot-inspired optimization** (orthogonal rotation + int8 quantization before encryption) reduces PQC overhead by ~60-75% by shrinking the NTT input size 4x before the expensive polynomial arithmetic runs.

> WARNING: **Simulation disclaimer:** All cryptography is *simulated*. NTT transforms use NumPy FFT as a structural stand-in for true Number Theoretic Transforms over Z_q. This is a performance engineering prototype, not a production Kyber implementation. Results demonstrate overhead *patterns*, not exact production numbers.

---

## Key Results (`demo.py`)

| Mode | Latency | Peak Memory | vs Baseline |
|------|---------|-------------|-------------|
| Baseline | ~0.06s | ~196 MB | - |
| PQC (simulated) | ~10.0s | ~452 MB | +17,000% latency, +130% memory |
| Optimized PQC | ~2.9s | ~192 MB | -71% vs PQC latency, -57% vs PQC memory |

*Results are deterministic (`np.random.seed(42)`) but vary by hardware.*

---

## How to Run

```bash
pip install -r requirements.txt

# Single benchmark run
python demo.py

# Interactive Streamlit dashboard
streamlit run app.py
```

**Requirements:** Python 3.9+, numpy, matplotlib, streamlit, plotly, pandas (all in `requirements.txt`).

---

## Repo Structure

```
demo.py                Single benchmark: Baseline / PQC / Optimized PQC
simulation.py          Simulation engine for the Streamlit dashboard (no UI imports)
app.py                 Streamlit dashboard -- interactive API load simulator
requirements.txt       Python dependencies
benchmark_results.png  Chart output from demo.py
```

---

## What the Optimization Does

Orthogonal rotation (QR decomp) decorrelates dimensions, then int8 quantization clips dynamic range for 4x byte reduction, then 4x mean-pool compression reduces NTT block count by 4x, so NTT polynomial arithmetic runs on 4x smaller input before ciphertext expansion (2x, Kyber-768 structure).

Fewer NTT blocks = fewer FFT calls = lower real compute time.

---

## Research Context

This demo is a precursor to ongoing **QuantRot-PQC** research exploring whether HD3 t-design rotation formally reduces CRYSTALS-Kyber NTT input coefficient dynamic range on real KV vectors from Llama-3 models.

See [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) for the full picture and [RESULTS.md](RESULTS.md) for benchmark details.

---

## License

MIT
