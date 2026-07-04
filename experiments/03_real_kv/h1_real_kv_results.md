# H1 Experiment — Real KV Vectors

**Model:** meta-llama/Llama-3.2-1B
**Vectors:** 500
**Head dimension d:** 64

## H1 Verdict
**SUPPORTED ✅**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| L∞ norm (mean) | 0.5407 | 0.3239 | +40.1% |
| Kurtosis (mean) | 6.15 | -0.09 | -101.5% |
| % within ±q/2 | 100.0% | 100.0% | +0.0pp |

## Key Observations
- Real KV L∞ before rotation: 0.5407 (synthetic was 0.1898)
- Real KV kurtosis before: 6.15 (Gaussian = 0.0 — heavy tails confirm non-Gaussian structure)
- Theory bound (d=64): 0.6062
- H1 required: ≥20% L∞ reduction
- H1 result: 40.1% reduction

## Notes
Real KV vectors extracted via past_key_values from Llama-3.2-1B attention layers. Unit-normalized per vector before analysis. HD³ rotation applied per-vector with fresh random signs each trial.

## Next Step
H1 supported on real KV vectors. Proceed to 04_full_benchmark for end-to-end Kyber latency measurement.