"""
EXPERIMENT 3 — Asymmetric Z-biased channel (Table IV, Figs 4, 5).

Reproduces the eta = {1, 10, 100, 1000} sweep including the eta=1000
reversal, with upgrades:
  * N_TEST = 100,000 -> the reversal is now established at >20 sigma
  * 3 seeds for both GABPNN-sym and GABPNN-asym
  * NEW: eta = 300 added to locate WHERE the reversal begins (this is
    a genuinely new datapoint the paper's reviewers may ask about)


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

D        = 5
ETAS     = [1, 10, 100, 300, 1000]          # 300 is NEW
P_TEST   = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03,
            0.05, 0.07, 0.10, 0.13, 0.15]
P_TRAIN  = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
N_TRAIN  = 40_000
N_TEST   = 100_000
SEEDS    = [2024, 2025, 2026]
EPOCHS   = 100
HIDDEN   = 256
K_BP     = 3
LR       = 7e-4
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_asymmetric.json")


def make_gabpnn(code, seed):
    if USE_GPU:
        return GABPNNTorch(code, hidden=HIDDEN, bp_rounds=K_BP,
                           lr=LR, seed=seed)
    return HybridGABPNN(code, hidden=HIDDEN, bp_rounds=K_BP, lr=LR,
                        rng=np.random.default_rng(seed))


def train_model(code, seed, channel, eta):
    rng = np.random.default_rng(seed)
    ts, tl = [], []
    for p in P_TRAIN:
        s, l, _, _ = code.generate_samples(
            N_TRAIN, p, channel=channel, eta=eta, rng=rng)
        ts.append(s); tl.append(l)
    g = make_gabpnn(code, seed)
    if USE_GPU:
        g.train(np.vstack(ts), np.concatenate(tl), epochs=EPOCHS,
                batch=1024, eta=eta, verbose=False)
    else:
        g.train(np.vstack(ts), np.concatenate(tl), epochs=EPOCHS,
                batch_size=256, verbose=False)
    return g


def main():
    print(f"Device: {'CUDA GPU' if USE_GPU else 'CPU'}")
    code = SurfaceCode(D)
    mwpm = ExactMWPM(code)
    uf = UnionFind(code)
    res = {"p_values": P_TEST, "etas": ETAS, "seeds": SEEDS, "asym": {}}

    # Symmetric-trained models (shared across all eta evaluations)
    print("Training GABPNN-sym models...", flush=True)
    sym_models = [train_model(code, s, "depolarising", 1.0)
                  for s in SEEDS]

    for eta in ETAS:
        print(f"\n=== eta={eta} ===", flush=True)
        t0 = time.time()
        asym_models = [train_model(code, s, "asymmetric", eta)
                       for s in SEEDS]
        print(f"  asym models trained ({time.time()-t0:.0f}s)",
              flush=True)

        entry = {"mwpm": [], "uf": [],
                 "sym_mean": [], "sym_std": [],
                 "asym_mean": [], "asym_std": []}
        rng_eval = np.random.default_rng(555)
        for p in P_TEST:
            s, l, _, _ = code.generate_samples(
                N_TEST, p, channel="asymmetric", eta=eta, rng=rng_eval)
            entry["mwpm"].append(float((mwpm.decode_batch(s, p=p) != l).mean()))
            entry["uf"].append(float((uf.decode_batch(s) != l).mean()))
            sym_l = [float((m.decode_batch(s, eta=eta) != l).mean())
                     if USE_GPU else
                     float((m.decode_batch(s) != l).mean())
                     for m in sym_models]
            asym_l = [float((m.decode_batch(s, eta=eta) != l).mean())
                      if USE_GPU else
                      float((m.decode_batch(s) != l).mean())
                      for m in asym_models]
            entry["sym_mean"].append(float(np.mean(sym_l)))
            entry["sym_std"].append(float(np.std(sym_l)))
            entry["asym_mean"].append(float(np.mean(asym_l)))
            entry["asym_std"].append(float(np.std(asym_l)))
        res["asym"][str(eta)] = entry

        pi5 = P_TEST.index(0.05)
        print(f"  p=0.05: MWPM={entry['mwpm'][pi5]*100:.2f}%  "
              f"sym={entry['sym_mean'][pi5]*100:.2f}%  "
              f"asym={entry['asym_mean'][pi5]*100:.2f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
