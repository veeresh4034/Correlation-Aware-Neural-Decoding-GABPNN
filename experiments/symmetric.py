
"""
EXPERIMENT 1 — Symmetric depolarising channel (Tables I, II; Figs 1-3, 7, 8).

  * N_TEST   = 100,000 per (d, p) point  (was 8,000)  -> CI +/-0.14% at 5%
  * N_TRAIN  = 40,000 per p-value        (was 8,000)  -> 280k total per d
  * SEEDS    = 5 independent train/eval runs           -> mean +/- std on
               
  


"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.surface_code import SurfaceCode
from src.exact_mwpm_nx import ExactMWPM
from src.classical_decoders import UnionFind

try:
    import torch
    from src.gabpnn_torch import GABPNNTorch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False
if not USE_GPU:
    from src.hybrid_decoder import HybridGABPNN

# Config (edit for quick runs) 
DS       = [3, 5, 7]
P_TEST   = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03,
            0.05, 0.07, 0.10, 0.13, 0.15]
P_TRAIN  = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
N_TRAIN  = 40_000      # per p-value  (280k total per distance)
N_TEST   = 100_000     # per (d, p) point
SEEDS    = [2024, 2025, 2026, 2027, 2028]
EPOCHS   = 100
HIDDEN   = 256
K_BP     = 3
LR       = 7e-4
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_symmetric.json")
# -----------------------------------------------------------------------


def make_gabpnn(code, seed):
    if USE_GPU:
        return GABPNNTorch(code, hidden=HIDDEN, bp_rounds=K_BP,
                           lr=LR, seed=seed)
    return HybridGABPNN(code, hidden=HIDDEN, bp_rounds=K_BP, lr=LR,
                        rng=np.random.default_rng(seed))


def main():
    print(f"Device: {'CUDA GPU' if USE_GPU else 'CPU (numpy)'}")
    results = {"p_values": P_TEST, "distances": DS, "seeds": SEEDS,
               "sym": {}}

    for d in DS:
        code = SurfaceCode(d)
        mwpm = ExactMWPM(code)
        uf = UnionFind(code)
        acc = {"mwpm": [], "uf": [],
               "gabpnn_mean": [], "gabpnn_std": []}

        # Classical decoders are deterministic -> evaluate once
        print(f"\n=== d={d} | classical decoders on {N_TEST:,} samples ===",
              flush=True)
        rng_eval = np.random.default_rng(999)
        test_sets = {}
        for p in P_TEST:
            s, l, _, _ = code.generate_samples(N_TEST, p, rng=rng_eval)
            test_sets[p] = (s, l)
            acc["mwpm"].append(float((mwpm.decode_batch(s, p=p) != l).mean()))
            acc["uf"].append(float((uf.decode_batch(s) != l).mean()))
            print(f"  p={p:.3f}: MWPM={acc['mwpm'][-1]*100:.3f}%  "
                  f"UF={acc['uf'][-1]*100:.3f}%", flush=True)

        # GABPNN: 5 independent seeds
        per_seed = np.zeros((len(SEEDS), len(P_TEST)))
        for si, seed in enumerate(SEEDS):
            print(f"  seed {seed}: training...", flush=True)
            t0 = time.time()
            rng_tr = np.random.default_rng(seed)
            ts, tl = [], []
            for p in P_TRAIN:
                s, l, _, _ = code.generate_samples(N_TRAIN, p, rng=rng_tr)
                ts.append(s); tl.append(l)
            g = make_gabpnn(code, seed)
            if USE_GPU:
                g.train(np.vstack(ts), np.concatenate(tl),
                        epochs=EPOCHS, batch=1024, verbose=False)
            else:
                g.train(np.vstack(ts), np.concatenate(tl),
                        epochs=EPOCHS, batch_size=256, verbose=False)
            for pi, p in enumerate(P_TEST):
                s, l = test_sets[p]
                per_seed[si, pi] = float((g.decode_batch(s) != l).mean())
            print(f"    done in {time.time()-t0:.0f}s | "
                  f"p=0.05 LER={per_seed[si, P_TEST.index(0.05)]*100:.3f}%",
                  flush=True)

        acc["gabpnn_mean"] = per_seed.mean(axis=0).tolist()
        acc["gabpnn_std"] = per_seed.std(axis=0).tolist()
        acc["gabpnn_per_seed"] = per_seed.tolist()
        results["sym"][str(d)] = acc

        pi5 = P_TEST.index(0.05)
        m, gm, gs = acc["mwpm"][pi5], acc["gabpnn_mean"][pi5], \
            acc["gabpnn_std"][pi5]
        print(f"  SUMMARY d={d} p=0.05: MWPM={m*100:.3f}%  "
              f"GABPNN={gm*100:.3f}+/-{gs*100:.3f}%  "
              f"improvement={(m-gm)/m*100:.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
