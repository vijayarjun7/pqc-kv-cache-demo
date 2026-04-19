"""
Simulation engine for the Secure AI Inference API Load demo.
All computation lives here — no Streamlit imports.
"""

import time
import tracemalloc
import dataclasses

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
HEADS       = 8    # attention heads per request payload
HEAD_DIM    = 32   # per-head dimension
NTT_BLOCK   = 32   # polynomial block size for vectorised NTT
KYBER_Q     = 3329 # Kyber modulus (mod reduction step)
EXPANSION   = 2    # simulated ciphertext expansion factor (~Kyber-768)

# Fraction of max_requests at which we sample checkpoints
CHECKPOINT_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RequestMetrics:
    mode:            str
    n_requests:      int
    avg_latency_ms:  float   # ms per request
    total_time_s:    float
    peak_memory_mb:  float
    throughput_rps:  float   # requests per second


# ---------------------------------------------------------------------------
# Request payload generation
# ---------------------------------------------------------------------------

def generate_request_payload(payload_scale: int, seed: int) -> np.ndarray:
    """
    Synthetic KV-cache-style tensor for one inference request.
    Shape: (payload_scale * HEADS, HEAD_DIM) — flattened attention keys/values.
    payload_scale controls workload size (1=tiny, 20=large).
    """
    rng = np.random.default_rng(seed)
    rows = payload_scale * HEADS
    return rng.standard_normal((rows, HEAD_DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorised NTT simulation — batch FFT, no Python loop per block
# ---------------------------------------------------------------------------

def _batch_ntt(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate NTT-style polynomial arithmetic on all rows simultaneously.
    matrix shape: (n_blocks, NTT_BLOCK)
    Uses numpy vectorised rfft — same per-block arithmetic, no Python loop.
    """
    freq = np.fft.rfft(matrix.astype(np.float64), axis=1)
    key  = rng.standard_normal(freq.shape) + 1j * rng.standard_normal(freq.shape)
    result = np.fft.irfft(freq * key, n=NTT_BLOCK, axis=1)
    return np.mod(result * 1000, KYBER_Q).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-request handlers — three modes
# ---------------------------------------------------------------------------

def handle_baseline_request(payload: np.ndarray, rng: np.random.Generator) -> float:
    """
    Normal AI inference: projection matmul + layer-norm + activation.
    No security overhead. Returns elapsed seconds.
    """
    t0 = time.perf_counter()
    W         = rng.standard_normal((payload.shape[1], payload.shape[1])).astype(np.float32)
    projected = payload @ W
    norms     = np.linalg.norm(projected, axis=-1, keepdims=True)
    normalized = projected / (norms + 1e-6)
    _ = np.tanh(normalized)
    return time.perf_counter() - t0


def handle_secure_request(payload: np.ndarray, rng: np.random.Generator) -> float:
    """
    Simulated PQC-style overhead (NOT real cryptography).
    NTT polynomial arithmetic + ciphertext expansion inspired by Kyber-768.
    Overhead: vectorised FFT transforms + 2× memory allocation.
    """
    t0 = time.perf_counter()

    flat = payload.flatten()
    pad  = (-len(flat)) % NTT_BLOCK
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks    = flat.reshape(-1, NTT_BLOCK)
    encrypted = _batch_ntt(blocks, rng)

    # Simulate ciphertext expansion: allocate u + v components
    n = blocks.shape[0]
    ciphertext = np.empty((n * EXPANSION, NTT_BLOCK), dtype=np.float32)
    ciphertext[:n]  = encrypted
    ciphertext[n:]  = encrypted * 0.5          # error/noise term

    # Partial decapsulation pass (inverse NTT on u component)
    _ = np.fft.irfft(
        np.fft.rfft(ciphertext[:n].astype(np.float64), axis=1),
        n=NTT_BLOCK, axis=1,
    ).astype(np.float32)

    return time.perf_counter() - t0


def handle_optimized_secure_request(payload: np.ndarray, rng: np.random.Generator) -> float:
    """
    Optimized secure processing: int8 quantization + chunk compression before PQC.

    Optimization steps:
    1. Int8 quantization — clips dynamic range, reduces representational noise.
       Equivalent to W8A8 quantization used in production LLM serving.
    2. 4× mean-pool compression — groups 4 adjacent values into one representative
       float32, reducing total NTT block count by 4×.
    Then runs the same NTT simulation on the smaller compressed tensor.
    """
    t0 = time.perf_counter()

    # Step 1: int8 quantization
    scale      = float(np.max(np.abs(payload))) / 127.0 + 1e-9
    quantized  = np.clip(np.round(payload / scale), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale

    # Step 2: 4× mean-pool chunk compression
    flat   = dequantized.flatten()
    trim   = (len(flat) // 4) * 4
    compressed = flat[:trim].reshape(-1, 4).mean(axis=1).astype(np.float32)

    # Pad and run NTT on the compressed (4× smaller) representation
    pad = (-len(compressed)) % NTT_BLOCK
    if pad:
        compressed = np.concatenate([compressed, np.zeros(pad, dtype=np.float32)])
    blocks    = compressed.reshape(-1, NTT_BLOCK)
    encrypted = _batch_ntt(blocks, rng)

    n = blocks.shape[0]
    ciphertext = np.empty((n * EXPANSION, NTT_BLOCK), dtype=np.float32)
    ciphertext[:n] = encrypted
    ciphertext[n:] = encrypted * 0.5

    _ = np.fft.irfft(
        np.fft.rfft(ciphertext[:n].astype(np.float64), axis=1),
        n=NTT_BLOCK, axis=1,
    ).astype(np.float32)

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Batch simulation runner
# ---------------------------------------------------------------------------

_HANDLERS = {
    "Baseline":          handle_baseline_request,
    "Secure":            handle_secure_request,
    "Optimized Secure":  handle_optimized_secure_request,
}


def run_batch_simulation(
    n_requests:    int,
    payload_scale: int,
    base_seed:     int = 42,
) -> list:
    """
    Run all three modes for n_requests simulated API calls.
    Returns list[RequestMetrics] — one per mode, in order.
    Each request uses a deterministic per-request seed for reproducibility.
    """
    results = []
    for mode, handler in _HANDLERS.items():
        latencies = []
        tracemalloc.start()

        for i in range(n_requests):
            rng     = np.random.default_rng(base_seed + i)
            payload = generate_request_payload(payload_scale, seed=base_seed + i)
            latencies.append(handler(payload, rng))

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total      = sum(latencies)
        avg_ms     = (total / n_requests) * 1000
        peak_mb    = peak_bytes / (1024 ** 2)
        throughput = n_requests / total if total > 0 else 0.0

        results.append(RequestMetrics(
            mode=mode,
            n_requests=n_requests,
            avg_latency_ms=avg_ms,
            total_time_s=total,
            peak_memory_mb=peak_mb,
            throughput_rps=throughput,
        ))

    return results


# ---------------------------------------------------------------------------
# Multi-checkpoint sweep — returns tidy DataFrame
# ---------------------------------------------------------------------------

def run_full_sweep(
    max_requests:  int,
    payload_scale: int,
    progress_cb=None,   # optional callable(step, total, n_requests)
) -> pd.DataFrame:
    """
    Sweep six checkpoints from 1% to 100% of max_requests.
    progress_cb is called after each checkpoint for UI progress updates.
    Returns a tidy DataFrame, one row per (checkpoint, mode).
    """
    checkpoints = sorted(set(
        max(1, int(f * max_requests)) for f in CHECKPOINT_FRACTIONS
    ))

    rows = []
    for step, n in enumerate(checkpoints):
        metrics_list = run_batch_simulation(n, payload_scale)

        base_lat = next(m.avg_latency_ms  for m in metrics_list if m.mode == "Baseline")
        sec_lat  = next(m.avg_latency_ms  for m in metrics_list if m.mode == "Secure")
        base_mem = next(m.peak_memory_mb  for m in metrics_list if m.mode == "Baseline")
        sec_mem  = next(m.peak_memory_mb  for m in metrics_list if m.mode == "Secure")

        for m in metrics_list:
            overhead_lat = (m.avg_latency_ms - base_lat) / base_lat * 100 if m.mode != "Baseline" else 0.0
            overhead_mem = (m.peak_memory_mb - base_mem) / base_mem * 100 if m.mode != "Baseline" else 0.0
            recovery_lat = (sec_lat - m.avg_latency_ms) / sec_lat * 100 if m.mode == "Optimized Secure" else 0.0

            rows.append({
                "Request Count":      m.n_requests,
                "Mode":               m.mode,
                "Avg Latency (ms)":   round(m.avg_latency_ms, 4),
                "Total Time (s)":     round(m.total_time_s, 4),
                "Peak Memory (MB)":   round(m.peak_memory_mb, 2),
                "Throughput (req/s)": round(m.throughput_rps, 1),
                "Overhead %":         round(overhead_lat, 1),
                "Mem Overhead %":     round(overhead_mem, 1),
                "Recovery %":         round(recovery_lat, 1),
            })

        if progress_cb:
            progress_cb(step + 1, len(checkpoints), n)

    return pd.DataFrame(rows)
