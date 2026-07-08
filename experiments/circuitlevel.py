"""
EXPERIMENT 7 — Circuit-level noise with Stim .

pip install stim pymatching

Trains GABPNN directly on Stim circuit-level syndromes (single round,
rotated_memory_z) with the BP graph derived from Stim's detector error
model, and compares against pymatching on the same DEM.

"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import stim
import pymatching

import torch
import torch.nn as nn

USE_GPU = torch.cuda.is_available()
DEV = torch.device("cuda" if USE_GPU else "cpu")

DS       = [3, 5]
P_TEST   = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01]
P_TRAIN  = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012]
N_TRAIN  = 50_000
N_TEST   = 100_000
EPOCHS   = 100
HIDDEN   = 256
K_BP     = 3
LR       = 5e-4
SEED     = 2024
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_circuit_level.json")


def circ(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=1, distance=d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p / 2,
        before_measure_flip_probability=p / 2)


def dem_adjacency(circuit):
    """BP graph from Stim's detector error model."""
    dem = circuit.detector_error_model(decompose_errors=True)
    n = circuit.num_detectors
    A = np.zeros((n, n), dtype=np.float32)
    for inst in dem.flattened():
        if inst.type == "error":
            dets = [t.val for t in inst.targets_copy()
                    if t.is_relative_detector_id()]
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    A[dets[i], dets[j]] += 1
                    A[dets[j], dets[i]] += 1
    deg = A.sum(axis=1, keepdims=True); deg[deg == 0] = 1
    return A / deg, (A.sum(axis=1) / (A.max() + 1e-8)).astype(np.float32)


class Net(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.skip = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, 1)     # binary: logical flip or not
        self.act = nn.ReLU()

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2) + self.skip(x))
        return self.out(h3).squeeze(-1)


def features(S, A_norm, degf, K=K_BP):
    Sf = torch.tensor(S, dtype=torch.float32, device=DEV)
    An = torch.tensor(A_norm, device=DEV)
    dg = torch.tensor(degf, device=DEV)
    N = Sf.shape[0]
    feats = [Sf]; msg = Sf
    for _ in range(K):
        msg = msg @ An.T
        feats.append(msg)
    feats.append(dg.expand(N, -1))
    return torch.cat(feats, dim=1)


def main():
    torch.manual_seed(SEED)
    res = {"p_values": P_TEST, "distances": DS, "data": {}}
    for d in DS:
        print(f"\n=== d={d} (circuit-level, Stim) ===", flush=True)
        A_norm, degf = dem_adjacency(circ(d, 0.005))

        # training data across P_TRAIN
        ts, tl = [], []
        for p in P_TRAIN:
            c = circ(d, p)
            det, obs = c.compile_detector_sampler(seed=SEED).sample(
                shots=N_TRAIN, separate_observables=True)
            ts.append(det.astype(np.uint8))
            tl.append(obs[:, 0].astype(np.float32))
        Xtr = features(np.vstack(ts), A_norm, degf)
        ytr = torch.tensor(np.concatenate(tl), device=DEV)

        net = Net(Xtr.shape[1], HIDDEN).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        # weighted BCE against class imbalance
        pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1),
                             device=DEV)
        lf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        N = len(ytr)
        print(f"  training on {N:,} samples "
              f"({float(ytr.mean())*100:.1f}% positive)...", flush=True)
        t0 = time.time()
        for ep in range(EPOCHS):
            perm = torch.randperm(N, device=DEV)
            for i in range(0, N, 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad()
                loss = lf(net(Xtr[idx]), ytr[idx])
                loss.backward(); opt.step()
        print(f"  trained in {time.time()-t0:.0f}s", flush=True)

        entry = {"pymatching": [], "gabpnn": []}
        net.eval()
        for p in P_TEST:
            c = circ(d, p)
            dem = c.detector_error_model(decompose_errors=True)
            mat = pymatching.Matching.from_detector_error_model(dem)
            det, obs = c.compile_detector_sampler(seed=SEED + 7).sample(
                shots=N_TEST, separate_observables=True)
            det = det.astype(np.uint8)
            lbl = obs[:, 0].astype(np.int64)

            pm = float((mat.decode_batch(det)[:, 0] != lbl).mean())
            with torch.no_grad():
                X = features(det, A_norm, degf)
                pr = []
                for i in range(0, len(X), 65536):
                    pr.append((net(X[i:i+65536]) > 0).long())
                gp = float((torch.cat(pr).cpu().numpy() != lbl).mean())
            entry["pymatching"].append(pm)
            entry["gabpnn"].append(gp)
            print(f"  p={p:.3f}: pymatching={pm*100:.4f}%  "
                  f"GABPNN={gp*100:.4f}%", flush=True)
        res["data"][str(d)] = entry

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
