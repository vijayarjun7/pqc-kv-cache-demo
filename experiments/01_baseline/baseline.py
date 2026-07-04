# experiments/01_baseline/baseline.py
#
# Measures Kyber-512 encapsulation latency and KV vector
# coefficient distribution BEFORE any rotation.
# This establishes the baseline numbers that H1 rotation
# experiment will compare against.
#
# Run after verify_kyber.py confirms your environment.
#
# Output: baseline_results.md (auto-generated)

import oqs
import numpy as np
import scipy.stats
import time
import json
from pathlib import Path

# ── Constants ────────────────────────────────────────────
D = 256          # KV vector dimension (per attention head)
Q = 3329         # Kyber modulus
Q_HALF = 1664    # q/2 — centered coefficient threshold
ALPHA = 5235     # Scale factor: q / (2 × 0.318) ≈ 5235
N_KV_TRIALS = 1000   # KV vector samples for distribution stats
N_KYBER_TRIALS = 100  # Kyber encapsulation timing trials
THEORY_BOUND = np.sqrt(2 * np.log(2 * D / 0.001) / D)  # 0.318 for d=256

# ── KV Vector Generation ─────────────────────────────────
def generate_kv_vector(d=D, seed=None):
    """
    Synthetic KV vector: Gaussian, unit norm (post-layer-norm surrogate).
    """
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)

# ── Coefficient Mapping ───────────────────────────────────
def to_centered_coefficients(v, alpha=ALPHA, q=Q, q_half=Q_HALF):
    """
    Map float32 vector to centered Zq coefficients.
    Centered: values in [-q/2, q/2] = [-1664, 1664]
    """
    raw = np.round(alpha * v).astype(np.int64)
    mod = raw % q
    centered = np.where(mod > q_half, mod - q, mod)
    return centered

# ── Distribution Statistics ───────────────────────────────
def measure_distribution(v):
    """
    Measure key distribution properties of a vector.
    """
    coeffs = to_centered_coefficients(v)
    return {
        "linf_norm":          float(np.max(np.abs(v))),
        "variance":           float(np.var(v)),
        "kurtosis":           float(scipy.stats.kurtosis(v)),
        "coeff_linf":         int(np.max(np.abs(coeffs))),
        "coeff_variance":     float(np.var(coeffs)),
        "pct_within_qhalf":   float(np.mean(np.abs(coeffs) <= Q_HALF) * 100),
    }

# ── Kyber Latency Measurement ─────────────────────────────
def measure_kyber_latency(n_trials=N_KYBER_TRIALS):
    """
    Measure Kyber-512 encapsulation latency over n_trials.
    Returns latency in microseconds.
    """
    kem = oqs.KeyEncapsulation('Kyber512')
    pk = kem.generate_keypair()

    # Warmup — 5 runs not counted
    for _ in range(5):
        kem.encap_secret(pk)

    times_us = []
    for _ in range(n_trials):
        start = time.perf_counter_ns()
        ct, ss = kem.encap_secret(pk)
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        times_us.append(elapsed_us)

    return {
        "mean_us":   float(np.mean(times_us)),
        "median_us": float(np.median(times_us)),
        "std_us":    float(np.std(times_us)),
        "min_us":    float(np.min(times_us)),
        "max_us":    float(np.max(times_us)),
        "n_trials":  n_trials,
        "ct_bytes":  len(ct),
        "ss_bytes":  len(ss),
    }

# ── KV Distribution Baseline ──────────────────────────────
def measure_kv_baseline(n_trials=N_KV_TRIALS):
    """
    Measure KV vector distribution properties over n_trials.
    No rotation applied — this is the raw baseline.
    """
    results = []
    for i in range(n_trials):
        v = generate_kv_vector()
        results.append(measure_distribution(v))

    keys = results[0].keys()
    summary = {}
    for k in keys:
        vals = [r[k] for r in results]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }
    return summary

# ── Results Writer ────────────────────────────────────────
def write_results(kyber_stats, kv_stats, output_path):
    """
    Write baseline results to markdown file.
    """
    lines = []
    lines.append("# Baseline Experiment Results")
    lines.append("")
    lines.append("## Environment")
    lines.append("- Hardware: [fill in — CPU, RAM, OS]")
    lines.append("- Python: [fill in]")
    lines.append(f"- KV dimension d: {D}")
    lines.append(f"- Kyber variant: Kyber-512 (q={Q}, NIST Level 1)")
    lines.append(f"- Scale factor α: {ALPHA}")
    lines.append(f"- Theory L∞ bound (d={D}, δ=0.001): {THEORY_BOUND:.4f}")
    lines.append("")
    lines.append("## Kyber-512 Encapsulation Latency")
    lines.append(f"- Trials: {kyber_stats['n_trials']}")
    lines.append(f"- Mean:   {kyber_stats['mean_us']:.2f} μs")
    lines.append(f"- Median: {kyber_stats['median_us']:.2f} μs")
    lines.append(f"- Std:    {kyber_stats['std_us']:.2f} μs")
    lines.append(f"- Min:    {kyber_stats['min_us']:.2f} μs")
    lines.append(f"- Max:    {kyber_stats['max_us']:.2f} μs")
    lines.append(f"- Ciphertext size: {kyber_stats['ct_bytes']} bytes (spec: 768)")
    lines.append(f"- Shared secret:   {kyber_stats['ss_bytes']} bytes (spec: 32)")
    lines.append("")
    lines.append("## KV Vector Distribution (Before Rotation)")
    lines.append(f"- Trials: {N_KV_TRIALS} synthetic unit-norm Gaussian vectors")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|--------|------|-----|-----|-----|")

    metric_labels = {
        "linf_norm":        "L∞ norm (float)",
        "variance":         "Variance",
        "kurtosis":         "Kurtosis",
        "coeff_linf":       "Coeff L∞ (centered Zq)",
        "coeff_variance":   "Coeff variance",
        "pct_within_qhalf": "% coeffs within ±q/2",
    }

    for k, label in metric_labels.items():
        s = kv_stats[k]
        lines.append(
            f"| {label} | {s['mean']:.4f} | "
            f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |"
        )

    lines.append("")
    lines.append("## H1 Baseline Reference")
    lines.append(
        f"- Raw L∞ mean before rotation: "
        f"{kv_stats['linf_norm']['mean']:.4f}"
    )
    lines.append(
        f"- Theory predicts after HD³ rotation: {THEORY_BOUND:.4f}"
    )
    lines.append(
        f"- H1 requires ≥20% reduction: "
        f"L∞ after rotation must be ≤ "
        f"{kv_stats['linf_norm']['mean'] * 0.8:.4f}"
    )
    lines.append("")
    lines.append("## Status")
    lines.append("- Baseline: ✅ Complete")
    lines.append("- Next: experiments/02_h1_rotation/rotation_experiment.py")

    output_path.write_text("\n".join(lines))
    print(f"\nResults written to {output_path}")

# ── Main ──────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("QuantRot-PQC — Baseline Experiment")
    print("=" * 50)

    print(f"\n[1/2] Measuring Kyber-512 latency ({N_KYBER_TRIALS} trials)...")
    kyber_stats = measure_kyber_latency()
    print(f"      Mean: {kyber_stats['mean_us']:.2f} μs")
    print(f"      Median: {kyber_stats['median_us']:.2f} μs")
    print(f"      Std: {kyber_stats['std_us']:.2f} μs")

    print(f"\n[2/2] Measuring KV distribution ({N_KV_TRIALS} vectors, d={D})...")
    kv_stats = measure_kv_baseline()
    print(f"      L∞ norm mean:       {kv_stats['linf_norm']['mean']:.4f}")
    print(f"      Theory bound:       {THEORY_BOUND:.4f}")
    print(f"      % within ±q/2:      {kv_stats['pct_within_qhalf']['mean']:.1f}%")

    print("\n" + "=" * 50)
    print("H1 Reference:")
    print(f"  Raw L∞ before rotation: {kv_stats['linf_norm']['mean']:.4f}")
    print(f"  Theory after HD³:       {THEORY_BOUND:.4f}")
    h1_threshold = kv_stats['linf_norm']['mean'] * 0.8
    print(f"  H1 target (≥20% drop):  ≤{h1_threshold:.4f}")
    print("=" * 50)

    output_path = Path(__file__).parent / "baseline_results.md"
    write_results(kyber_stats, kv_stats, output_path)

if __name__ == "__main__":
    main()
