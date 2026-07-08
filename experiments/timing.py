"""
EXPERIMENT 4 — Decoding latency (Table V, Fig 6).

UPGRADES:
  * Median of 30 repetitions with warm-up 
  * Reports median, 5th and 95th percentile per decoder
  * NEW: GABPNN measured on BOTH CPU (numpy) and GPU (torch) 
  * Uses process-priority-safe monotonic clock

"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.surface_code import SurfaceCode
from src.exact_mwpm_nx import ExactMWPM
from src.classical_decoders import UnionFind
from src.hybrid_decoder import HybridGABPNN

try:
    import torch
    from src.gabpnn_torch import GABPNNTorch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

DS       = [3, 5, 7, 9]
P        = 0.05
N_BATCH  = 1000
REPS     = 30
P_TRAIN  = [0.01, 0.03, 0.05, 0.08]
N_TRAIN  = 5000        # small: we only need a trained net for timing
EPOCHS   = 20
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_timing.json")


def bench(fn, reps=REPS):
    fn()  # warm up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) / N_BATCH * 1e6)
    a = np.array(times)
    return {"median_us": float(np.median(a)),
            "p05_us": float(np.percentile(a, 5)),
            "p95_us": float(np.percentile(a, 95))}


def main():
    res = {"distances": DS, "n_batch": N_BATCH, "reps": REPS,
           "timing": {}}
    for d in DS:
        print(f"d={d}...", flush=True)
        code = SurfaceCode(d)
        rng = np.random.default_rng(2024)
        s, _, _, _ = code.generate_samples(N_BATCH, P, rng=rng)

        mwpm = ExactMWPM(code)
        uf = UnionFind(code)

        # quick-train a CPU GABPNN for realistic weights
        ts, tl = [], []
        for p in P_TRAIN:
            a, b, _, _ = code.generate_samples(N_TRAIN, p, rng=rng)
            ts.append(a); tl.append(b)
        g_cpu = HybridGABPNN(code, hidden=256, bp_rounds=3, lr=7e-4,
                             rng=np.random.default_rng(2024))
        g_cpu.train(np.vstack(ts), np.concatenate(tl),
                    epochs=EPOCHS, batch_size=256, verbose=False)

        entry = {
            "greedy_mwpm": bench(lambda: mwpm.decode_batch(s, p=P)),
            "union_find": bench(lambda: uf.decode_batch(s)),
            "gabpnn_cpu": bench(lambda: g_cpu.decode_batch(s)),
        }
        if USE_GPU:
            g_gpu = GABPNNTorch(code, hidden=256, bp_rounds=3,
                                lr=7e-4, seed=2024)
            g_gpu.train(np.vstack(ts), np.concatenate(tl),
                        epochs=EPOCHS, batch=1024, verbose=False)

            def gpu_call():
                g_gpu.decode_batch(s)
                torch.cuda.synchronize()
            entry["gabpnn_gpu"] = bench(gpu_call)

        res["timing"][str(d)] = entry
        line = f"  MWPM={entry['greedy_mwpm']['median_us']:.1f}us  " \
               f"UF={entry['union_find']['median_us']:.1f}us  " \
               f"GABPNN-CPU={entry['gabpnn_cpu']['median_us']:.1f}us"
        if USE_GPU:
            line += f"  GABPNN-GPU={entry['gabpnn_gpu']['median_us']:.2f}us"
        print(line)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
