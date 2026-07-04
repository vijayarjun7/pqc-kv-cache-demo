# experiments/02_h1_rotation/rotation_experiment.py
#
# Tests H1: Does HD³ rotation reduce L∞ norm of KV vectors by ≥20%?
#
# Compares:
#   - Raw synthetic KV vectors (baseline from 01_baseline)
#   - Same vectors after HD³ rotation
#
# Measures:
#   - L∞ norm before vs after
#   - Coefficient distribution before vs after
#   - % coefficients within ±q/2
#   - Variance and kurtosis
#   - Theory vs empirical comparison
#
# H1 decision:
#   SUPPORTED     if mean L∞ reduction ≥ 20%
#   NOT SUPPORTED if mean L∞ reduction < 20%

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ── Constants ─────────────────────────────────────────────
D = 256
Q = 3329
Q_HALF = 1664
ALPHA = 5235
N_TRIALS = 1000
THEORY_BOUND = np.sqrt(2 * np.log(2 * D / 0.001) / D)
H1_THRESHOLD_PCT = 20.0  # minimum % reduction to support H1

# Baseline numbers from 01_baseline
BASELINE_LINF_MEAN = 0.1898
H1_TARGET = BASELINE_LINF_MEAN * 0.80  # ≤ 0.1518

# ── HD³ Rotation ──────────────────────────────────────────
def hadamard_transform(x):
    """
    Fast Walsh-Hadamard Transform.
    Requires len(x) to be a power of 2.
    O(d log d) in-place.
    """
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2
    return x / np.sqrt(n)

def hd3_rotation(x, signs=None):
    """
    HD³ rotation: apply (Hadamard × random diagonal) 3 times.

    Each application:
      1. Multiply by random ±1 diagonal (D)
      2. Apply Hadamard transform (H)

    Three applications approximate a unitary 2-design.
    Produces sub-Gaussian coordinates with variance ≈ 1/d.

    Args:
        x:     input vector (unit norm, float32)
        signs: optional list of 3 sign arrays for reproducibility
               if None, generated fresh each call

    Returns:
        rotated vector (unit norm preserved)
        signs used (for reproducibility / inverse rotation)
    """
    d = len(x)
    x = x.copy().astype(np.float64)
    used_signs = []

    for i in range(3):
        if signs is not None:
            s = signs[i]
        else:
            s = np.random.choice([-1, 1], size=d)
        used_signs.append(s)
        x = x * s
        x = hadamard_transform(x)

    # Renormalize to unit norm
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    return x.astype(np.float32), used_signs

def inverse_hd3(x_rotated, signs):
    """
    Inverse HD³ rotation: U⁻¹ = Uᵀ for orthogonal U.
    Apply operations in reverse order with transposed signs.
    Verifies exact invertibility of the rotation.
    """
    x = x_rotated.copy().astype(np.float64)
    for s in reversed(signs):
        x = hadamard_transform(x)
        x = x * s
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x.astype(np.float32)

# ── Distribution Measurement ──────────────────────────────
def to_centered_coefficients(v):
    raw = np.round(ALPHA * v).astype(np.int64)
    mod = raw % Q
    return np.where(mod > Q_HALF, mod - Q, mod)

def measure_vector(v):
    coeffs = to_centered_coefficients(v)
    return {
        "linf":             float(np.max(np.abs(v))),
        "variance":         float(np.var(v)),
        "kurtosis":         float(scipy.stats.kurtosis(v)),
        "coeff_linf":       int(np.max(np.abs(coeffs))),
        "coeff_variance":   float(np.var(coeffs)),
        "pct_within_qhalf": float(np.mean(np.abs(coeffs) <= Q_HALF) * 100),
    }

# ── Main Experiment ───────────────────────────────────────
def run_h1_experiment(n_trials=N_TRIALS):
    """
    Run H1 experiment over n_trials KV vectors.
    Returns before/after measurements and reduction stats.
    """
    before_list = []
    after_list  = []
    recon_errors = []

    print(f"Running H1 experiment: {n_trials} trials, d={D}")
    print(f"Theory bound:  {THEORY_BOUND:.4f}")
    print(f"H1 target:     L∞ ≤ {H1_TARGET:.4f} (≥20% reduction from {BASELINE_LINF_MEAN})")
    print()

    for i in range(n_trials):
        # Generate raw KV vector
        v = np.random.randn(D).astype(np.float32)
        v = v / np.linalg.norm(v)

        # Measure before rotation
        before_list.append(measure_vector(v))

        # Apply HD³ rotation
        v_rot, signs = hd3_rotation(v)

        # Measure after rotation
        after_list.append(measure_vector(v_rot))

        # Verify invertibility
        v_recovered = inverse_hd3(v_rot, signs)
        recon_error = float(np.max(np.abs(v - v_recovered)))
        recon_errors.append(recon_error)

    return before_list, after_list, recon_errors

# ── Summary Statistics ────────────────────────────────────
def summarize(results, label):
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

# ── Plot ──────────────────────────────────────────────────
def plot_results(before_list, after_list, output_path):
    """
    Generate comparison plots:
      - L∞ norm distribution before vs after
      - Coefficient distribution before vs after
    """
    before_linf = [r["linf"] for r in before_list]
    after_linf  = [r["linf"] for r in after_list]
    before_pct  = [r["pct_within_qhalf"] for r in before_list]
    after_pct   = [r["pct_within_qhalf"] for r in after_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"H1 Experiment: HD³ Rotation Effect on KV Vectors (d={D}, n={N_TRIALS})",
        fontsize=13
    )

    # Plot 1 — L∞ norm distribution
    axes[0].hist(before_linf, bins=40, alpha=0.6,
                 color="#4C72B0", label=f"Before (mean={np.mean(before_linf):.4f})")
    axes[0].hist(after_linf, bins=40, alpha=0.6,
                 color="#55A868", label=f"After  (mean={np.mean(after_linf):.4f})")
    axes[0].axvline(THEORY_BOUND, color="red", linestyle="--",
                    label=f"Theory bound ({THEORY_BOUND:.4f})")
    axes[0].axvline(H1_TARGET, color="orange", linestyle="--",
                    label=f"H1 target ({H1_TARGET:.4f})")
    axes[0].set_xlabel("L∞ norm")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("L∞ Norm: Before vs After HD³ Rotation")
    axes[0].legend(fontsize=8)

    # Plot 2 — % coefficients within ±q/2
    axes[1].hist(before_pct, bins=20, alpha=0.6,
                 color="#4C72B0", label=f"Before (mean={np.mean(before_pct):.1f}%)")
    axes[1].hist(after_pct, bins=20, alpha=0.6,
                 color="#55A868", label=f"After  (mean={np.mean(after_pct):.1f}%)")
    axes[1].set_xlabel("% Coefficients within ±q/2")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Centered Coefficient Range: Before vs After")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()

# ── Results Writer ────────────────────────────────────────
def write_results(before_summary, after_summary,
                  recon_errors, output_path):

    linf_before = before_summary["linf"]["mean"]
    linf_after  = after_summary["linf"]["mean"]
    reduction   = (linf_before - linf_after) / linf_before * 100
    h1_status   = "SUPPORTED ✅" if reduction >= H1_THRESHOLD_PCT else "NOT SUPPORTED ❌"

    lines = []
    lines.append("# H1 Rotation Experiment Results")
    lines.append("")
    lines.append("## H1 Hypothesis")
    lines.append(
        "HD³ rotation reduces empirical L∞ norm of KV vectors "
        "from Llama-3 by ≥20% compared to unrotated vectors."
    )
    lines.append("")
    lines.append("## H1 Verdict")
    lines.append(f"**{h1_status}**")
    lines.append("")
    lines.append(f"- L∞ before rotation: {linf_before:.4f}")
    lines.append(f"- L∞ after rotation:  {linf_after:.4f}")
    lines.append(f"- Reduction:          {reduction:.1f}%")
    lines.append(f"- Required:           ≥{H1_THRESHOLD_PCT:.0f}%")
    lines.append(f"- Theory bound:       {THEORY_BOUND:.4f}")
    lines.append("")
    lines.append("## Invertibility Check")
    lines.append(
        f"- Max reconstruction error: "
        f"{np.max(recon_errors):.2e}"
    )
    lines.append(
        f"- Mean reconstruction error: "
        f"{np.mean(recon_errors):.2e}"
    )
    lines.append(
        f"- Rotation is exactly invertible: "
        f"{'✅ Yes' if np.max(recon_errors) < 1e-5 else '⚠️ Check'}"
    )
    lines.append("")
    lines.append("## Full Metrics Comparison")
    lines.append("")
    lines.append("| Metric | Before | After | Change |")
    lines.append("|--------|--------|-------|--------|")

    metric_labels = {
        "linf":             "L∞ norm",
        "variance":         "Variance",
        "kurtosis":         "Kurtosis",
        "coeff_linf":       "Coeff L∞ (centered Zq)",
        "coeff_variance":   "Coeff variance",
        "pct_within_qhalf": "% coeffs within ±q/2",
    }

    for k, label in metric_labels.items():
        b = before_summary[k]["mean"]
        a = after_summary[k]["mean"]
        if b != 0:
            change = f"{(a - b) / abs(b) * 100:+.1f}%"
        else:
            change = "—"
        lines.append(f"| {label} | {b:.4f} | {a:.4f} | {change} |")

    lines.append("")
    lines.append("## Experimental Setup")
    lines.append(f"- Trials: {N_TRIALS}")
    lines.append(f"- Dimension d: {D}")
    lines.append(f"- HD³ applications: 3")
    lines.append(f"- Scale factor α: {ALPHA}")
    lines.append(f"- Kyber modulus q: {Q}")
    lines.append(f"- Input: synthetic unit-norm Gaussian vectors")
    lines.append(f"- Seed: np.random.seed(42)")
    lines.append("")
    lines.append("## Next Step")
    lines.append(
        "Run experiments/03_real_kv/extract_kv.py "
        "to test H1 on real KV vectors from Llama-3."
    )

    output_path.write_text("\n".join(lines))
    print(f"Results written to {output_path}")

# ── Main ──────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("QuantRot-PQC — H1 Rotation Experiment")
    print("=" * 50)

    before_list, after_list, recon_errors = run_h1_experiment()

    before_summary = summarize(before_list, "before")
    after_summary  = summarize(after_list,  "after")

    # H1 verdict
    linf_before = before_summary["linf"]["mean"]
    linf_after  = after_summary["linf"]["mean"]
    reduction   = (linf_before - linf_after) / linf_before * 100

    print("=" * 50)
    print("H1 RESULT:")
    print(f"  L∞ before: {linf_before:.4f}")
    print(f"  L∞ after:  {linf_after:.4f}")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Required:  ≥{H1_THRESHOLD_PCT:.0f}%")
    print(f"  Theory:    {THEORY_BOUND:.4f}")
    print(f"  Status:    {'SUPPORTED ✅' if reduction >= H1_THRESHOLD_PCT else 'NOT SUPPORTED ❌'}")
    print("=" * 50)

    print(f"\nInvertibility:")
    print(f"  Max recon error: {np.max(recon_errors):.2e}")
    print(f"  Rotation invertible: {'✅' if np.max(recon_errors) < 1e-5 else '⚠️'}")

    # Save outputs
    out_dir = Path(__file__).parent
    plot_results(
        before_list, after_list,
        out_dir / "linf_comparison.png"
    )
    write_results(
        before_summary, after_summary,
        recon_errors,
        out_dir / "rotation_results.md"
    )

if __name__ == "__main__":
    main()
