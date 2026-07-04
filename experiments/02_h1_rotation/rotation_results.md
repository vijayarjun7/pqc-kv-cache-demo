# H1 Rotation Experiment Results

## H1 Hypothesis
HD³ rotation reduces empirical L∞ norm of KV vectors from Llama-3 by ≥20% compared to unrotated vectors.

## H1 Verdict
**NOT SUPPORTED ❌**

- L∞ before rotation: 0.1907
- L∞ after rotation:  0.1904
- Reduction:          0.2%
- Required:           ≥20%
- Theory bound:       0.3205

## Invertibility Check
- Max reconstruction error: 2.98e-08
- Mean reconstruction error: 1.01e-08
- Rotation is exactly invertible: ✅ Yes

## Full Metrics Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| L∞ norm | 0.1907 | 0.1904 | -0.2% |
| Variance | 0.0039 | 0.0039 | -0.0% |
| Kurtosis | -0.0168 | -0.0154 | +8.2% |
| Coeff L∞ (centered Zq) | 998.3640 | 996.8280 | -0.2% |
| Coeff variance | 106653.9654 | 106620.2878 | -0.0% |
| % coeffs within ±q/2 | 100.0000 | 100.0000 | +0.0% |

## Experimental Setup
- Trials: 1000
- Dimension d: 256
- HD³ applications: 3
- Scale factor α: 5235
- Kyber modulus q: 3329
- Input: synthetic unit-norm Gaussian vectors
- Seed: np.random.seed(42)

## Next Step
Run experiments/03_real_kv/extract_kv.py to test H1 on real KV vectors from Llama-3.