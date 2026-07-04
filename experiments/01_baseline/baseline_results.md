# Baseline Experiment Results

## Environment
- Hardware: [fill in — CPU, RAM, OS]
- Python: [fill in]
- KV dimension d: 256
- Kyber variant: Kyber-512 (q=3329, NIST Level 1)
- Scale factor α: 5235
- Theory L∞ bound (d=256, δ=0.001): 0.3205

## Kyber-512 Encapsulation Latency
- Trials: 100
- Mean:   6.66 μs
- Median: 6.50 μs
- Std:    0.91 μs
- Min:    6.33 μs
- Max:    13.38 μs
- Ciphertext size: 768 bytes (spec: 768)
- Shared secret:   32 bytes (spec: 32)

## KV Vector Distribution (Before Rotation)
- Trials: 1000 synthetic unit-norm Gaussian vectors

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| L∞ norm (float) | 0.1898 | 0.0215 | 0.1461 | 0.2786 |
| Variance | 0.0039 | 0.0000 | 0.0037 | 0.0039 |
| Kurtosis | -0.0319 | 0.2993 | -0.7646 | 1.6078 |
| Coeff L∞ (centered Zq) | 993.5460 | 112.3224 | 765.0000 | 1459.0000 |
| Coeff variance | 106654.0383 | 555.6431 | 102688.8201 | 107079.5321 |
| % coeffs within ±q/2 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |

## H1 Baseline Reference
- Raw L∞ mean before rotation: 0.1898
- Theory predicts after HD³ rotation: 0.3205
- H1 requires ≥20% reduction: L∞ after rotation must be ≤ 0.1518

## Status
- Baseline: ✅ Complete
- Next: experiments/02_h1_rotation/rotation_experiment.py