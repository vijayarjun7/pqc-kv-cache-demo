# Research Context

This repository contains a runnable demo. It is not a research artifact and
does not itself validate any research claim. This document exists to draw a
clear line between what the demo shows and what the underlying research
(QuantRot-PQC) investigates, so the two are not conflated.

## What This Demo Shows

The demo measures the *shape* of overhead you'd expect when applying
post-quantum cryptographic operations to an LLM KV-cache, and the *shape* of
recovery from a rotate-and-quantize optimization step. Concretely:

- A baseline projection cost with no cryptography involved.
- A simulated PQC cost, using NumPy FFT as a stand-in for Kyber's NTT, applied
  to synthetic KV-cache-shaped tensors.
- An optimized PQC path that applies a QR-based rotation and int8
  quantization before the same simulated NTT step, showing a recovered
  latency and memory profile.

All three numbers are real measurements of real NumPy operations — nothing in
the demo is faked or hardcoded — but the operations themselves are
simulations of cryptographic and optimization primitives, not the primitives
themselves. See [Simulation Methodology in RESULTS.md](RESULTS.md#simulation-methodology)
for the specifics.

## What the Research Investigates

The QuantRot-PQC research direction asks a narrower and more rigorous
question than the demo can answer: whether a specific, theoretically
motivated random rotation (an HD³ t-design construction), applied to real KV
vectors from a real LLM, measurably reduces coefficient overflow/reduction
pressure in a real, instrumented Kyber NTT implementation (liboqs) — and
whether any such reduction translates into a net cycle-count improvement once
the cost of the rotation itself is accounted for.

That question is empirical and is not yet answered. The hypotheses below
state what is being tested, not what has been shown.

## The Gap Between Demo and Research

| | Demo | Research |
|---|---|---|
| NTT | Simulated using NumPy FFT — shows overhead pattern | Real Kyber-512 via liboqs — measures real cycles |
| Rotation | QR rotation + int8 — QuantRot-*inspired*, not HD³ | HD³ t-design rotation — theoretically grounded |
| Data | Synthetic KV vectors from `np.random.randn` | Real KV vectors extracted from Llama-3.2-1B |

The demo's rotation and quantization steps are inspired by the research
direction but are a simplified stand-in, not an implementation of it. Numbers
in [RESULTS.md](RESULTS.md) should not be read as evidence for or against the
hypotheses below.

## Exact Research Claim

> We propose QuantRot-PQC, a t-design-based random unitary pre-rotation
> applied to LLM KV-cache vectors prior to CRYSTALS-Kyber polynomial
> encoding. By redistributing vector energy across dimensions, the HD³
> rotation produces concentrated coordinate distributions with variance proxy
> approximately 1/d, reducing the centered coefficient dynamic range (L∞
> spread in centered Zq representation) after scaling into polynomial form.
> This coefficient concentration may reduce input-stage coefficient overflow
> and reduction pressure in certain NTT implementations. Because NTT
> reduction behavior depends on implementation details, the magnitude of any
> performance benefit is evaluated experimentally rather than assumed
> theoretically. The pre-rotation is public, invertible, input-independent,
> and operates entirely in the plaintext domain prior to Kyber's algebraic
> operations. It does not alter the Module-LWE problem instance, nor does it
> modify the distributions of the public matrix A, secret vector s, or error
> vector e. Consequently, the construction is expected to inherit the
> security properties of the underlying CRYSTALS-Kyber scheme under standard
> Module-LWE assumptions. A formal security reduction is left for future
> work.

## Hypotheses

### H1 (Mathematical)

HD³ rotation reduces empirical L∞ norm and centered coefficient dynamic range
of KV vectors from Llama-3 by ≥20% compared to unrotated vectors.

### H1-efficiency (Systems)

Reduced centered coefficient magnitude decreases input-stage conditional
reduction frequency in Kyber NTT. Exact cycle improvement is
implementation-dependent and measured experimentally using instrumented
liboqs NTT.

### H2 (Systems)

Net latency improvement is measurable and consistent at d=256 on commodity
hardware after accounting for rotation overhead — compared to vanilla
Kyber-512 baseline (~45,200 AVX2 cycles per encapsulation).

## References

- CRYSTALS-Kyber specification v3.02
  https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
  https://arxiv.org/abs/2504.19874
- Brandão, Harrow, Horodecki — Local Random Quantum Circuits are Approximate
  Polynomial-Designs (2016)
  https://arxiv.org/abs/1208.0692
- Harrow, Mehraban — Approximate Unitary t-Designs by Short Random Quantum
  Circuits (2018)
  https://arxiv.org/abs/1809.06957
- Albrecht, Player, Scott — On the Concrete Hardness of Learning with Errors
  (2015)
  https://eprint.iacr.org/2015/046

## Status

| Component | Status |
|---|---|
| Demo | ✅ Complete |
| Theory | ✅ Complete |
| Experiments | 🔄 In progress |
| Paper | ⬜ Planned — IACR ePrint first, then arXiv cs.CR |

Target: Month 5. Sequence: IACR ePrint (crypto community) → arXiv cs.CR
cross-post (broader ML community).
