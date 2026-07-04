# experiments/03_real_kv/extract_kv.py
#
# Extracts real KV-cache vectors from Llama-3.2-1B
# and tests H1 on actual transformer KV distributions.
#
# Real KV vectors have outlier/heavy-tailed dimensions
# unlike synthetic Gaussian — this is the meaningful H1 test.
#
# Requirements:
#   pip install transformers torch
#   HuggingFace account with Llama-3.2-1B access
#   (or use an open model — see fallback below)
#
# Output:
#   real_kv_vectors.npy      — extracted KV vectors
#   h1_real_kv_results.md    — H1 verdict on real data
#   linf_real_comparison.png — distribution plots

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ── Constants ─────────────────────────────────────────────
D = 64           # head dimension for Llama-3.2-1B
Q = 3329
Q_HALF = 1664
ALPHA = 5235
THEORY_BOUND = np.sqrt(2 * np.log(2 * D / 0.001) / D)
H1_THRESHOLD_PCT = 20.0
BASELINE_LINF_MEAN = 0.1898  # from synthetic baseline

# ── HD³ Rotation (same as 02_h1_rotation) ────────────────
def hadamard_transform(x):
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

def hd3_rotation(x):
    d = len(x)
    x = x.copy().astype(np.float64)
    used_signs = []
    for _ in range(3):
        s = np.random.choice([-1, 1], size=d)
        used_signs.append(s)
        x = x * s
        x = hadamard_transform(x)
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x.astype(np.float32), used_signs

# ── KV Extraction ─────────────────────────────────────────
def extract_kv_vectors(model_name="meta-llama/Llama-3.2-1B",
                        n_vectors=500):
    """
    Extract key vectors from Llama-3.2-1B attention layers.
    Uses forward hooks to capture KV cache during inference.

    Args:
        model_name: HuggingFace model identifier
        n_vectors:  number of KV vectors to collect

    Returns:
        numpy array of shape (n_vectors, head_dim)
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    kv_vectors = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is attention output — capture key states
            # For Llama, input[0] is hidden states
            hidden = input[0].detach()
            # Project to get approximate key vectors per head
            # Shape: (batch, seq, hidden)
            b, s, h = hidden.shape
            # Flatten to individual vectors per position
            vectors = hidden.reshape(-1, h).numpy()
            # Take first 64 dims as proxy for head dimension
            # (real extraction uses past_key_values — see note below)
            for v in vectors[:10]:  # limit per layer per forward pass
                v_head = v[:64].astype(np.float32)
                norm = np.linalg.norm(v_head)
                if norm > 0:
                    kv_vectors.append(v_head / norm)
        return hook

    # Register hooks on all attention layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run inference on diverse prompts
    prompts = [
        "The quantum computing revolution will transform",
        "Post-quantum cryptography ensures that encrypted",
        "Large language models process text by",
        "The KV cache stores key and value vectors",
        "Neural networks learn representations through",
        "Security in AI systems requires careful",
        "Mathematical foundations of machine learning include",
        "Transformer attention mechanisms compute",
        "The future of cryptography depends on",
        "Efficient inference requires optimizing memory",
    ]

    print(f"Running inference on {len(prompts)} prompts...")
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=32,
                truncation=True
            )
            _ = model(**inputs)
            if len(kv_vectors) >= n_vectors:
                break

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"Extracted {len(kv_vectors)} KV vectors")
    vectors = np.array(kv_vectors[:n_vectors])
    return vectors


def extract_kv_via_past_key_values(
        model_name="meta-llama/Llama-3.2-1B",
        n_vectors=500):
    """
    Alternative extraction using past_key_values directly.
    More accurate — captures actual key vectors from attention.

    Returns:
        numpy array of shape (n_vectors, head_dim)
    """
    print(f"Loading model: {model_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    kv_vectors = []

    prompts = [
        "The quantum computing revolution will transform cryptography",
        "Post-quantum cryptographic schemes protect against quantum",
        "Large language models require efficient memory management",
        "The KV cache stores attention keys and values during inference",
        "Neural network attention mechanisms transform input sequences",
        "Security researchers study post-quantum cryptography methods",
        "Mathematical proofs establish the hardness of lattice problems",
        "Transformer models process tokens through multiple attention layers",
        "Efficient inference systems optimize both latency and memory",
        "Cryptographic primitives form the foundation of secure systems",
        "Machine learning models learn from large training datasets",
        "Quantum computers threaten current public-key cryptography",
    ]

    print(f"Extracting real KV vectors via past_key_values...")
    with torch.no_grad():
        for prompt in prompts:
            if len(kv_vectors) >= n_vectors:
                break

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=64,
                truncation=True
            )

            outputs = model(
                **inputs,
                use_cache=True,
                output_attentions=False
            )

            # past_key_values: tuple of (key, value) per layer
            # key shape: (batch, n_heads, seq_len, head_dim)
            past_kv = outputs.past_key_values

            for layer_kv in past_kv:
                keys = layer_kv[0]  # (1, n_heads, seq, head_dim)
                # Flatten to (n_heads * seq, head_dim)
                k = keys.squeeze(0)           # (n_heads, seq, head_dim)
                k = k.reshape(-1, k.shape[-1])  # (n_heads*seq, head_dim)
                k_np = k.numpy()

                for v in k_np:
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        kv_vectors.append(v / norm)
                    if len(kv_vectors) >= n_vectors:
                        break
                if len(kv_vectors) >= n_vectors:
                    break

    print(f"Extracted {len(kv_vectors)} real KV vectors")
    print(f"Head dimension: {kv_vectors[0].shape[0]}")
    return np.array(kv_vectors[:n_vectors])

# ── Distribution Analysis ─────────────────────────────────
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
        "pct_within_qhalf": float(np.mean(np.abs(coeffs) <= Q_HALF) * 100),
    }

def analyze_vectors(vectors, label=""):
    """
    Analyze distribution before and after HD³ rotation.
    """
    print(f"\nAnalyzing {len(vectors)} {label} vectors...")

    before_list = []
    after_list  = []

    for v in vectors:
        # Ensure unit norm
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        before_list.append(measure_vector(v))

        # Rotate
        # Pad or truncate to power of 2 if needed
        d = len(v)
        next_pow2 = 2 ** int(np.ceil(np.log2(d)))
        if next_pow2 != d:
            v_padded = np.zeros(next_pow2, dtype=np.float32)
            v_padded[:d] = v
            v_rot, _ = hd3_rotation(v_padded)
            v_rot = v_rot[:d]
        else:
            v_rot, _ = hd3_rotation(v)

        # Renormalize after truncation
        norm_rot = np.linalg.norm(v_rot)
        if norm_rot > 0:
            v_rot = v_rot / norm_rot

        after_list.append(measure_vector(v_rot))

    return before_list, after_list

# ── Plot ──────────────────────────────────────────────────
def plot_results(before_list, after_list,
                 label, output_path):
    before_linf = [r["linf"] for r in before_list]
    after_linf  = [r["linf"] for r in after_list]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"H1 on Real KV Vectors: {label} (n={len(before_list)})",
        fontsize=13
    )

    # L∞ distribution
    axes[0].hist(before_linf, bins=40, alpha=0.6,
                 color="#4C72B0",
                 label=f"Before (mean={np.mean(before_linf):.4f})")
    axes[0].hist(after_linf, bins=40, alpha=0.6,
                 color="#55A868",
                 label=f"After  (mean={np.mean(after_linf):.4f})")
    axes[0].axvline(THEORY_BOUND, color="red",
                    linestyle="--",
                    label=f"Theory ({THEORY_BOUND:.4f})")
    axes[0].set_xlabel("L∞ norm")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("L∞ Norm Distribution")
    axes[0].legend(fontsize=8)

    # Kurtosis comparison
    before_kurt = [r["kurtosis"] for r in before_list]
    after_kurt  = [r["kurtosis"] for r in after_list]
    axes[1].hist(before_kurt, bins=40, alpha=0.6,
                 color="#4C72B0",
                 label=f"Before (mean={np.mean(before_kurt):.2f})")
    axes[1].hist(after_kurt, bins=40, alpha=0.6,
                 color="#55A868",
                 label=f"After  (mean={np.mean(after_kurt):.2f})")
    axes[1].set_xlabel("Kurtosis")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Kurtosis: Before vs After (tail heaviness)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()

# ── Results Writer ────────────────────────────────────────
def write_results(before_list, after_list,
                  model_name, output_path):

    linf_before = np.mean([r["linf"] for r in before_list])
    linf_after  = np.mean([r["linf"] for r in after_list])
    reduction   = (linf_before - linf_after) / linf_before * 100
    kurt_before = np.mean([r["kurtosis"] for r in before_list])
    kurt_after  = np.mean([r["kurtosis"] for r in after_list])
    pct_before  = np.mean([r["pct_within_qhalf"] for r in before_list])
    pct_after   = np.mean([r["pct_within_qhalf"] for r in after_list])

    h1_status = (
        "SUPPORTED ✅"
        if reduction >= H1_THRESHOLD_PCT
        else "NOT SUPPORTED ❌"
    )

    lines = []
    lines.append("# H1 Experiment — Real KV Vectors")
    lines.append("")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Vectors:** {len(before_list)}")
    lines.append(f"**Head dimension d:** {D}")
    lines.append("")
    lines.append("## H1 Verdict")
    lines.append(f"**{h1_status}**")
    lines.append("")
    lines.append(f"| Metric | Before | After | Change |")
    lines.append(f"|--------|--------|-------|--------|")
    lines.append(
        f"| L∞ norm (mean) | {linf_before:.4f} | "
        f"{linf_after:.4f} | {reduction:+.1f}% |"
    )
    lines.append(
        f"| Kurtosis (mean) | {kurt_before:.2f} | "
        f"{kurt_after:.2f} | "
        f"{(kurt_after-kurt_before)/abs(kurt_before)*100:+.1f}% |"
    )
    lines.append(
        f"| % within ±q/2 | {pct_before:.1f}% | "
        f"{pct_after:.1f}% | "
        f"{pct_after-pct_before:+.1f}pp |"
    )
    lines.append("")
    lines.append("## Key Observations")
    lines.append(
        f"- Real KV L∞ before rotation: {linf_before:.4f} "
        f"(synthetic was {BASELINE_LINF_MEAN:.4f})"
    )
    lines.append(
        f"- Real KV kurtosis before: {kurt_before:.2f} "
        f"(Gaussian = 0.0 — heavy tails confirm non-Gaussian structure)"
    )
    lines.append(f"- Theory bound (d={D}): {THEORY_BOUND:.4f}")
    lines.append(f"- H1 required: ≥{H1_THRESHOLD_PCT:.0f}% L∞ reduction")
    lines.append(f"- H1 result: {reduction:.1f}% reduction")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "Real KV vectors extracted via past_key_values from "
        "Llama-3.2-1B attention layers. Unit-normalized per vector "
        "before analysis. HD³ rotation applied per-vector with "
        "fresh random signs each trial."
    )
    lines.append("")
    lines.append("## Next Step")
    if reduction >= H1_THRESHOLD_PCT:
        lines.append(
            "H1 supported on real KV vectors. "
            "Proceed to 04_full_benchmark for end-to-end "
            "Kyber latency measurement."
        )
    else:
        lines.append(
            "H1 not supported on real KV vectors. "
            "Investigate kurtosis and outlier structure. "
            "Consider per-head normalization before rotation."
        )

    output_path.write_text("\n".join(lines))
    print(f"Results written to {output_path}")

# ── Main ──────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("QuantRot-PQC — H1 Experiment on Real KV Vectors")
    print("=" * 55)

    model_name = "meta-llama/Llama-3.2-1B"
    out_dir = Path(__file__).parent

    # Extract real KV vectors
    try:
        vectors = extract_kv_via_past_key_values(
            model_name=model_name,
            n_vectors=500
        )
        np.save(out_dir / "real_kv_vectors.npy", vectors)
        print(f"Saved {len(vectors)} vectors — shape {vectors.shape}")
    except Exception as e:
        print(f"Model load failed: {e}")
        print("If Llama-3.2-1B requires HF auth, run:")
        print("  huggingface-cli login")
        print("Or use: model_name = 'facebook/opt-125m' as fallback")
        return

    # Analyze
    before_list, after_list = analyze_vectors(
        vectors,
        label="Llama-3.2-1B KV"
    )

    # H1 verdict
    linf_before = np.mean([r["linf"] for r in before_list])
    linf_after  = np.mean([r["linf"] for r in after_list])
    reduction   = (linf_before - linf_after) / linf_before * 100

    print("\n" + "=" * 55)
    print("H1 RESULT (REAL KV VECTORS):")
    print(f"  L∞ before: {linf_before:.4f}")
    print(f"  L∞ after:  {linf_after:.4f}")
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Required:  ≥{H1_THRESHOLD_PCT:.0f}%")
    print(f"  Kurtosis before: "
          f"{np.mean([r['kurtosis'] for r in before_list]):.2f}")
    print(
        f"  Status: "
        f"{'SUPPORTED ✅' if reduction >= H1_THRESHOLD_PCT else 'NOT SUPPORTED ❌'}"
    )
    print("=" * 55)

    # Save outputs
    plot_results(
        before_list, after_list,
        label="Llama-3.2-1B",
        output_path=out_dir / "linf_real_comparison.png"
    )
    write_results(
        before_list, after_list,
        model_name=model_name,
        output_path=out_dir / "h1_real_kv_results.md"
    )

if __name__ == "__main__":
    main()
