"""
EXPERIMENT 5 — NEW: large code distances d = 9, 11 .

With your RTX 3050 this experiment extends every
symmetric-channel result to d=9 (1.2M params) and d=11 (2.0M params).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.surface_code import SurfaceCode
from src.exact_mwpm_nx import ExactMWPM
from src.classical_decoders import UnionFind

import torch
from src.gabpnn_torch import GABPNNTorch

assert torch.cuda.is_available(), \
    "Experiment 5 requires CUDA. Install torch with CUDA support."

DS       = [9, 11]
P_TEST   = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03,
            0.05, 0.07, 0.10, 0.13, 0.15]
P_TRAIN  = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
N_TRAIN  = 60_000     # per p -> 420k per distance (GPU handles it)
N_TEST   = 50_000
SEEDS    = [2024, 2025, 2026]
EPOCHS   = 120
HIDDEN   = 384        # larger hidden dim for larger codes
K_BP     = 3
LR       = 5e-4
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_large_d.json")


def main():
    res = {"p_values": P_TEST, "distances": DS, "seeds": SEEDS,
           "hidden": HIDDEN, "sym": {}}
    for d in DS:
        code = SurfaceCode(d)
        mwpm = ExactMWPM(code)
        uf = UnionFind(code)
        n_params = (code.n_stab * (K_BP + 3)) * HIDDEN * 2 \
            + HIDDEN * HIDDEN * 2 + HIDDEN * 4
        print(f"\n=== d={d} ({code.n_stab} stabilisers, "
              f"~{n_params/1e6:.1f}M params) ===", flush=True)

        acc = {"mwpm": [], "uf": [],
               "gabpnn_mean": [], "gabpnn_std": []}

        rng_eval = np.random.default_rng(999)
        test_sets = {}
        print("  Classical evaluation (slow at large d)...", flush=True)
        for p in P_TEST:
            s, l, _, _ = code.generate_samples(N_TEST, p, rng=rng_eval)
            test_sets[p] = (s, l)
            acc["mwpm"].append(float((mwpm.decode_batch(s, p=p) != l).mean()))
            acc["uf"].append(float((uf.decode_batch(s) != l).mean()))
            print(f"    p={p:.3f}: MWPM={acc['mwpm'][-1]*100:.3f}%",
                  flush=True)

        per_seed = np.zeros((len(SEEDS), len(P_TEST)))
        for si, seed in enumerate(SEEDS):
            t0 = time.time()
            rng = np.random.default_rng(seed)
            ts, tl = [], []
            for p in P_TRAIN:
                s, l, _, _ = code.generate_samples(N_TRAIN, p, rng=rng)
                ts.append(s); tl.append(l)
            g = GABPNNTorch(code, hidden=HIDDEN, bp_rounds=K_BP,
                            lr=LR, seed=seed)
            g.train(np.vstack(ts), np.concatenate(tl),
                    epochs=EPOCHS, batch=1024, verbose=True)
            for pi, p in enumerate(P_TEST):
                s, l = test_sets[p]
                per_seed[si, pi] = float((g.decode_batch(s) != l).mean())
            print(f"  seed {seed}: {time.time()-t0:.0f}s  "
                  f"p=0.05 LER="
                  f"{per_seed[si, P_TEST.index(0.05)]*100:.3f}%",
                  flush=True)
            g.save(os.path.join(os.path.dirname(OUT),
                                f"gabpnn_d{d}_seed{seed}.pt"))

        acc["gabpnn_mean"] = per_seed.mean(0).tolist()
        acc["gabpnn_std"] = per_seed.std(0).tolist()
        res["sym"][str(d)] = acc

        pi5 = P_TEST.index(0.05)
        m, gm = acc["mwpm"][pi5], acc["gabpnn_mean"][pi5]
        print(f"  SUMMARY d={d} p=0.05: MWPM={m*100:.3f}%  "
              f"GABPNN={gm*100:.3f}%  imp={(m-gm)/m*100:.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
