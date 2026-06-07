# Research Context

This demo is a precursor to the **QuantRot-PQC** research project.

---

## The Problem

Post-quantum cryptographic schemes -- specifically **CRYSTALS-Kyber** (now standardized as NIST ML-KEM) -- protect LLM inference APIs from "harvest now, decrypt later" attacks. But their NTT polynomial arithmetic introduces serious inference overhead:

- ~170x latency increase over unencrypted baseline (simulated; varies in production)
- ~2x memory expansion from Kyber ciphertext structure (u + v components)
- Overhead scales with every token in the KV-cache -- the primary memory bottleneck in transformer inference

This overhead makes naive PQC deployment impractical at LLM serving scale.

---

## The QuantRot Hypothesis

**QuantRot** is an orthogonal-rotation preprocessing step for transformer KV-cache vectors, originally proposed to improve quantization accuracy. The hypothesis being tested in the research:

> Applying HD3 t-design rotation to KV vectors before CRYSTALS-Kyber NTT encryption formally reduces the dynamic range of NTT input coefficients -- lowering effective ciphertext expansion and polynomial arithmetic cost -- making quantum-safe LLM inference practically viable.

The intuition: orthogonal rotation distributes energy more uniformly across dimensions. Uniformly distributed coefficients compress more efficiently before modular arithmetic, reducing the effective bit-width required for accurate NTT computation.

---

## What This Demo Shows

| Component | What it demonstrates |
|-----------|---------------------|
| `demo.py` | PQC overhead magnitude on a realistic KV-cache tensor; QuantRot-inspired rotation + int8 quantization recovers ~71% of latency |
| `simulation.py` | Overhead scales with request volume; optimization maintains recovery ratio |
| `app.py` | Interactive exploration of the overhead/recovery tradeoff across workload sizes |

The demo uses **QR decomposition** as a stand-in for the HD3 t-design rotation. The full research uses structured Hadamard-based rotations with theoretical guarantees on coefficient distribution.

---

## Research Status

| Phase | Status |
|-------|--------|
| Demo prototype | Done (this repo) |
| Theoretical framework (HD3 t-design rotation -> NTT coefficient reduction) | Done |
| Empirical experiments on real Llama-3 KV vectors | In progress |
| Comparison against production Kyber (liboqs) baseline | In progress |
| Paper draft | Planned |
| arXiv submission | Planned |

---

## Key Differences: Demo vs Full Research

| Aspect | This demo | Full research |
|--------|-----------|---------------|
| Rotation | QR decomposition (random orthogonal) | HD3 t-design (structured, with theoretical guarantees) |
| NTT | NumPy FFT simulation (float64) | Real Kyber NTT via liboqs |
| KV vectors | Synthetic np.random.randn | Real Llama-3 8B inference traces |
| Quantization | Scalar int8 | Per-channel, group-wise |
| Evaluation | Latency + memory (simulated) | Latency + memory + cryptographic correctness |

---

## Why This Matters

NIST finalized ML-KEM (Kyber) in August 2024. Enterprises running LLM APIs at scale will need PQC-compliant inference pipelines within the next 3-5 years. The overhead problem is real and unsolved. QuantRot-PQC is a direction toward solving it.

---

## Related Work

- **CRYSTALS-Kyber / ML-KEM**: Bos et al., 2018; NIST FIPS 203 (2024)
- **QuantRot**: Rotation-based quantization for transformers (QuaRot, SpinQuant)
- **KV-cache compression**: StreamingLLM, H2O, ScissorHands
- **PQC in ML systems**: Ongoing NIST/ETSI work on PQC migration guidance

---

*For benchmark numbers from this demo, see [RESULTS.md](RESULTS.md).*
