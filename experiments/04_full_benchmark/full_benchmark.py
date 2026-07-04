# experiments/04_full_benchmark/full_benchmark.py
#
# Tests H2: is a net latency improvement measurable at d=64 on real
# KV vectors, once rotation overhead is accounted for?
#
# Pipeline under test:
#   rotate KV vector (HD³) -> quantize (int8) -> encode (centered Zq)
#   -> Kyber-512 encapsulate
# vs.
#   vanilla Kyber-512 encapsulate (no rotation/quantization/encoding)
#
# IMPORTANT CAVEAT — read before interpreting results:
#   liboqs's public KeyEncapsulation.encap_secret() is a black box: it
#   generates its own internal randomness and does not accept the
#   rotated KV vector as input. This script cannot make Kyber's actual
#   NTT run *on* the rotated data — that would require an instrumented
#   liboqs build (the H1-efficiency hypothesis already flags this as
#   separate future work). What this script measures instead is the
#   practical end-to-end pipeline cost: rotation + quantization +
#   encoding overhead, stacked on top of an unchanged Kyber-512 call,
#   compared against calling Kyber-512 alone. That is a meaningful
#   but different question from "does rotation reduce Kyber's
#   internal reduction pressure" (H1-efficiency) — it answers
#   "does this pipeline cost more or less than vanilla Kyber, in
#   practice, on this hardware."
#
# Requirements:
#   Run after experiments/03_real_kv/extract_kv.py has produced
#   real_kv_vectors.npy (500 real Llama-3.2-1B KV vectors, d=64).
#
# Output:
#   h2_full_benchmark_results.md
#   latency_comparison.png

import oqs
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ─────────────────────────────────────────────
D = 64            # real KV head dimension (Llama-3.2-1B)
Q = 3329
Q_HALF = 1664
ALPHA = 5235
N_WARMUP = 5
KV_VECTORS_PATH = Path(__file__).parent.parent / "03_real_kv" / "real_kv_vectors.npy"

# ── HD³ Rotation (same construction as 02_h1_rotation / 03_real_kv,
#     vectorized here to remove pure-Python loop overhead) ──
def hadamard_transform(x):
    """
    Vectorized Fast Walsh-Hadamard Transform.
    Uses numpy reshape — no Python loops.
    O(d log d), fully vectorized.
    Requires len(x) to be a power of 2.
    """
    n = len(x)
    h = 1
    while h < n:
        x = x.reshape(-1, 2 * h)
        left  = x[:, :h].copy()
        right = x[:, h:].copy()
        x[:, :h] = left + right
        x[:, h:] = left - right
        x = x.reshape(-1)
        h *= 2
    return x / np.sqrt(n)

def hd3_rotation(x):
    d = len(x)
    x = x.copy().astype(np.float64)
    for _ in range(3):
        s = np.random.choice([-1, 1], size=d)
        x = x * s
        x = hadamard_transform(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x.astype(np.float32)

# ── Quantize + Encode ─────────────────────────────────────
def quantize_int8(v):
    scale = float(np.max(np.abs(v)))
    if scale == 0:
        scale = 1.0
    scale = scale / 127.0
    q = np.round(v / scale).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return q.astype(np.float32) * scale

def to_centered_coefficients(v, alpha=ALPHA, q=Q, q_half=Q_HALF):
    raw = np.round(alpha * v).astype(np.int64)
    mod = raw % q
    return np.where(mod > q_half, mod - q, mod)

# ── Pipeline Timing ───────────────────────────────────────
def measure_vanilla_kyber(kem, pk, n_trials):
    """
    Vanilla Kyber-512 encapsulation — no preprocessing.
    """
    for _ in range(N_WARMUP):
        kem.encap_secret(pk)

    times_us = []
    for _ in range(n_trials):
        start = time.perf_counter_ns()
        kem.encap_secret(pk)
        times_us.append((time.perf_counter_ns() - start) / 1000)
    return np.array(times_us)

def measure_rotated_pipeline(kem, pk, vectors):
    """
    Full pipeline per KV vector: rotate -> quantize -> encode -> encapsulate.
    Returns arrays of (preprocessing_us, kyber_us, total_us) per trial.
    """
    # Warmup — run the full pipeline N_WARMUP times, uncounted
    for i in range(N_WARMUP):
        v = vectors[i % len(vectors)]
        v_rot = hd3_rotation(v)
        q, scale = quantize_int8(v_rot)
        v_dq = dequantize(q, scale)
        to_centered_coefficients(v_dq)
        kem.encap_secret(pk)

    preprocessing_us = []
    kyber_us = []
    total_us = []

    for v in vectors:
        t0 = time.perf_counter_ns()
        v_rot = hd3_rotation(v)
        q, scale = quantize_int8(v_rot)
        v_dq = dequantize(q, scale)
        to_centered_coefficients(v_dq)
        t1 = time.perf_counter_ns()

        kem.encap_secret(pk)
        t2 = time.perf_counter_ns()

        pre = (t1 - t0) / 1000
        kyb = (t2 - t1) / 1000
        preprocessing_us.append(pre)
        kyber_us.append(kyb)
        total_us.append(pre + kyb)

    return np.array(preprocessing_us), np.array(kyber_us), np.array(total_us)

# ── Plot ──────────────────────────────────────────────────
def plot_results(vanilla_us, total_us, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vanilla_us, bins=40, alpha=0.7, density=True, color="#4C72B0",
            label=f"Vanilla Kyber-512 (mean={np.mean(vanilla_us):.2f} μs)")
    ax.hist(total_us, bins=40, alpha=0.6, density=True, color="#C44E52",
            label=f"Rotated pipeline (mean={np.mean(total_us):.2f} μs)")
    ax.set_xlabel("Latency (μs)")
    ax.set_ylabel("Density")
    ax.set_title(f"H2: Vanilla Kyber-512 vs Rotated Pipeline Latency (d={D})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()

# ── Results Writer ────────────────────────────────────────
def write_results(vanilla_us, preprocessing_us, kyber_us, total_us, output_path):
    vanilla_mean = float(np.mean(vanilla_us))
    total_mean   = float(np.mean(total_us))
    pre_mean     = float(np.mean(preprocessing_us))
    kyber_mean   = float(np.mean(kyber_us))
    net_delta_pct = (vanilla_mean - total_mean) / vanilla_mean * 100
    h2_status = "SUPPORTED ✅" if net_delta_pct > 0 else "NOT SUPPORTED ❌"

    lines = []
    lines.append("# H2 Experiment — Full Pipeline Benchmark")
    lines.append("")
    lines.append("## H2 Hypothesis")
    lines.append(
        "Net latency improvement is measurable and consistent at d=256 on "
        "commodity hardware after accounting for rotation overhead — compared "
        "to vanilla Kyber-512 baseline (~45,200 AVX2 cycles per encapsulation)."
    )
    lines.append("")
    lines.append(
        f"**Note:** this run tests d={D} (real Llama-3.2-1B head dimension "
        "from experiments/03_real_kv), not d=256 as in the original H2 "
        "statement. Cycle counts are not reported — only wall-clock μs, "
        "measured via `time.perf_counter_ns()` — since no hardware cycle "
        "counter (e.g. rdtsc) is used here, μs and the cited AVX2 cycle "
        "figure are not directly comparable."
    )
    lines.append("")
    lines.append("## Important Caveat")
    lines.append(
        "liboqs's public `encap_secret()` is a black box: it generates its "
        "own internal randomness and does not accept the rotated KV vector "
        "as input. This benchmark cannot make Kyber's actual NTT run *on* "
        "the rotated data — that requires an instrumented liboqs build "
        "(flagged as separate future work under H1-efficiency). What is "
        "measured here is the **practical end-to-end pipeline cost**: "
        "rotation + quantization + encoding overhead, stacked on top of an "
        "unchanged Kyber-512 call, compared against calling Kyber-512 alone. "
        "A negative result here does not disconfirm H1-efficiency's internal "
        "reduction-pressure claim — it only shows that, via the public "
        "Python bindings, added preprocessing cannot make a fixed-cost "
        "black-box operation faster."
    )
    lines.append("")
    lines.append("## H2 Verdict (practical pipeline version)")
    lines.append(f"**{h2_status}**")
    lines.append("")
    lines.append(f"- Vanilla Kyber-512 mean:        {vanilla_mean:.2f} μs")
    lines.append(f"- Rotated pipeline mean (total): {total_mean:.2f} μs")
    lines.append(f"  - Preprocessing (rotate+quantize+encode): {pre_mean:.2f} μs")
    lines.append(f"  - Kyber-512 encapsulate component:        {kyber_mean:.2f} μs")
    lines.append(f"- Net delta: {net_delta_pct:+.1f}% ({'improvement' if net_delta_pct > 0 else 'regression'})")
    lines.append("")
    lines.append("## Full Latency Stats (μs)")
    lines.append("")
    lines.append("| Stage | Mean | Median | Std | Min | Max |")
    lines.append("|-------|------|--------|-----|-----|-----|")
    for label, arr in [
        ("Vanilla Kyber-512", vanilla_us),
        ("Preprocessing (rotate+quantize+encode)", preprocessing_us),
        ("Kyber-512 component (pipeline)", kyber_us),
        ("Total pipeline", total_us),
    ]:
        lines.append(
            f"| {label} | {np.mean(arr):.2f} | {np.median(arr):.2f} | "
            f"{np.std(arr):.2f} | {np.min(arr):.2f} | {np.max(arr):.2f} |"
        )

    lines.append("")
    lines.append("## Experimental Setup")
    lines.append(f"- Trials: {len(vanilla_us)} (one real KV vector per trial)")
    lines.append(f"- KV vectors: {KV_VECTORS_PATH.relative_to(Path(__file__).parent.parent.parent)}")
    lines.append(f"- Dimension d: {D}")
    lines.append("- HD³ applications: 3")
    lines.append(f"- Warmup trials (uncounted): {N_WARMUP}")
    lines.append("- Timer: time.perf_counter_ns()")
    lines.append("")
    lines.append("## Interpretation")
    if net_delta_pct > 0:
        lines.append(
            "The rotated pipeline was faster end-to-end than vanilla "
            "Kyber-512 alone in this run. Given the caveat above, treat "
            "this as a measurement to reproduce rather than a settled "
            "result — rerun across hardware before relying on it."
        )
    else:
        lines.append(
            "The rotated pipeline costs more end-to-end than vanilla "
            "Kyber-512 alone, as expected: rotation/quantization/encoding "
            "add real Python-side overhead on top of an unchanged, "
            "fixed-cost Kyber call. This does not test — and cannot "
            "disconfirm — H1-efficiency's claim about reduced internal "
            "NTT reduction pressure, which requires an instrumented "
            "liboqs build to observe directly."
        )
    lines.append("")
    lines.append("## Next Step")
    lines.append(
        "To test H1-efficiency and the cycle-level version of H2 properly, "
        "instrument liboqs's Kyber NTT implementation directly (e.g. count "
        "conditional Barrett/Montgomery reductions per call) and compare "
        "rotated vs unrotated coefficient inputs at that level."
    )

    output_path.write_text("\n".join(lines))
    print(f"Results written to {output_path}")

# ── Main ──────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("QuantRot-PQC — H2 Full Pipeline Benchmark")
    print("=" * 55)

    if not KV_VECTORS_PATH.exists():
        print(f"Missing {KV_VECTORS_PATH} — run experiments/03_real_kv/extract_kv.py first.")
        return

    vectors = np.load(KV_VECTORS_PATH)
    print(f"Loaded {len(vectors)} real KV vectors, d={vectors.shape[1]}")

    kem = oqs.KeyEncapsulation('Kyber512')
    pk = kem.generate_keypair()

    print(f"\n[1/2] Measuring vanilla Kyber-512 ({len(vectors)} trials)...")
    vanilla_us = measure_vanilla_kyber(kem, pk, n_trials=len(vectors))
    print(f"      Mean: {np.mean(vanilla_us):.2f} μs")

    print(f"\n[2/2] Measuring rotated pipeline ({len(vectors)} trials)...")
    preprocessing_us, kyber_us, total_us = measure_rotated_pipeline(kem, pk, vectors)
    print(f"      Preprocessing mean: {np.mean(preprocessing_us):.2f} μs")
    print(f"      Kyber component mean: {np.mean(kyber_us):.2f} μs")
    print(f"      Total mean: {np.mean(total_us):.2f} μs")

    vanilla_mean = np.mean(vanilla_us)
    total_mean = np.mean(total_us)
    net_delta_pct = (vanilla_mean - total_mean) / vanilla_mean * 100

    print("\n" + "=" * 55)
    print("H2 RESULT:")
    print(f"  Vanilla Kyber-512:  {vanilla_mean:.2f} μs")
    print(f"  Rotated pipeline:   {total_mean:.2f} μs")
    print(f"  Net delta:          {net_delta_pct:+.1f}%")
    print(f"  Status:             {'SUPPORTED ✅' if net_delta_pct > 0 else 'NOT SUPPORTED ❌'}")
    print("=" * 55)

    out_dir = Path(__file__).parent
    plot_results(vanilla_us, total_us, out_dir / "latency_comparison.png")
    write_results(
        vanilla_us, preprocessing_us, kyber_us, total_us,
        out_dir / "h2_full_benchmark_results.md"
    )

if __name__ == "__main__":
    main()
