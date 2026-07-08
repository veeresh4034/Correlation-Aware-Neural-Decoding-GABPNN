"""
EXPERIMENT 2 — Ablation study (Table III, Fig 9).

Three controlled variants, identical training data per seed:
  V1: raw syndrome -> plain MLP (no BP, no skip)
  V2: BP features  -> plain MLP (no skip)
  V3: BP features  -> residual MLP  (= full GABPNN)

The incremental BP gain and skip gain get error bars,
which turns the paper's ablation from suggestive into rigorous. The
per-point noise we saw at N=8000 disappears at N=100k.

"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.surface_code import SurfaceCode
from src.exact_mwpm_nx import ExactMWPM

try:
    import torch, torch.nn as nn
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

D        = 5
P_TEST   = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03,
            0.05, 0.07, 0.10, 0.13, 0.15]
P_TRAIN  = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
N_TRAIN  = 40_000
N_TEST   = 100_000
SEEDS    = [2024, 2025, 2026, 2027, 2028]
EPOCHS   = 100
HIDDEN   = 256
K_BP     = 3
LR       = 7e-4
OUT      = os.path.join(os.path.dirname(__file__), "..",
                        "results", "results_ablation.json")


def build_graph(code):
    H = np.vstack([code.H_z, code.H_x]).astype(np.float32)
    A = H @ H.T
    np.fill_diagonal(A, 0)
    deg = A.sum(axis=1, keepdims=True); deg[deg == 0] = 1
    return (A / deg), (A.sum(axis=1) / (A.max() + 1e-8)).astype(np.float32)


def features(S, variant, A_norm, degf, K=K_BP):
    """variant 'raw' -> just syndrome; 'bp' -> full BP feature vector."""
    Sf = S.astype(np.float32)
    if variant == "raw":
        return Sf
    N = len(S); feats = [Sf]; msg = Sf
    for _ in range(K):
        msg = msg @ A_norm.T
        feats.append(msg)
    feats.append(np.tile(degf, (N, 1)))
    feats.append(np.zeros((N, S.shape[1]), dtype=np.float32))  # eta=1
    return np.hstack(feats)


if USE_GPU:
    class MLP(nn.Module):
        def __init__(self, in_dim, hidden, residual):
            super().__init__()
            self.residual = residual
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, hidden)
            if residual:
                self.skip = nn.Linear(in_dim, hidden)
            self.out = nn.Linear(hidden, 4)
            self.act = nn.ReLU()
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            h1 = self.act(self.fc1(x))
            h2 = self.act(self.fc2(h1))
            pre3 = self.fc3(h2)
            if self.residual:
                pre3 = pre3 + self.skip(x)
            return self.out(self.act(pre3))

    def train_eval(Xtr, ytr, tests, residual, seed):
        dev = torch.device("cuda")
        torch.manual_seed(seed)
        net = MLP(Xtr.shape[1], HIDDEN, residual).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        lf = nn.CrossEntropyLoss()
        X = torch.tensor(Xtr, device=dev)
        y = torch.tensor(ytr, dtype=torch.long, device=dev)
        N = len(y)
        net.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad()
                loss = lf(net(X[idx]), y[idx])
                loss.backward(); opt.step()
        net.eval(); lers = []
        with torch.no_grad():
            for Xte, yte in tests:
                Xt = torch.tensor(Xte, device=dev)
                preds = []
                for i in range(0, len(Xt), 65536):
                    preds.append(net(Xt[i:i+65536]).argmax(1))
                p = torch.cat(preds).cpu().numpy()
                lers.append(float((p != yte).mean()))
        return lers
else:
    from src.hybrid_decoder import ResidualMLP, Adam as NpAdam

    def train_eval(Xtr, ytr, tests, residual, seed):
        # numpy fallback: uses ResidualMLP; for V1/V2 the skip weights
        # are zeroed and frozen (approximation of a plain MLP)
        rng = np.random.default_rng(seed)
        mlp = ResidualMLP(Xtr.shape[1], hidden=HIDDEN, rng=rng)
        if not residual:
            mlp.params["Ws"][:] = 0
        opt = NpAdam(lr=LR)
        N = len(ytr); rng2 = np.random.default_rng(99)
        for _ in range(EPOCHS):
            idx = rng2.permutation(N)
            for s in range(0, N, 256):
                xb = Xtr[idx[s:s+256]]; yb = ytr[idx[s:s+256]]
                probs = mlp.forward(xb)
                grads = mlp.backward(yb)
                if not residual:
                    grads["Ws"][:] = 0; grads["bs"][:] = 0
                opt.step(mlp.params, grads)
        lers = []
        for Xte, yte in tests:
            p = mlp.forward(Xte, training=False).argmax(1)
            lers.append(float((p != yte).mean()))
        return lers


def main():
    print(f"Device: {'CUDA GPU' if USE_GPU else 'CPU'}")
    code = SurfaceCode(D)
    A_norm, degf = build_graph(code)
    mwpm = ExactMWPM(code)

    rng_eval = np.random.default_rng(999)
    raw_tests, bp_tests, mwpm_ler = [], [], []
    for p in P_TEST:
        s, l, _, _ = code.generate_samples(N_TEST, p, rng=rng_eval)
        raw_tests.append((features(s, "raw", A_norm, degf), l))
        bp_tests.append((features(s, "bp", A_norm, degf), l))
        mwpm_ler.append(float((mwpm.decode_batch(s, p=p) != l).mean()))
    print("Classical baseline done.", flush=True)

    variants = {"raw_mlp": ("raw", False), "bp_mlp": ("bp", False),
                "bp_res_mlp": ("bp", True)}
    res = {"p_values": P_TEST, "mwpm": mwpm_ler}

    for name, (feat, residual) in variants.items():
        per_seed = np.zeros((len(SEEDS), len(P_TEST)))
        for si, seed in enumerate(SEEDS):
            t0 = time.time()
            rng = np.random.default_rng(seed)
            ts, tl = [], []
            for p in P_TRAIN:
                s, l, _, _ = code.generate_samples(N_TRAIN, p, rng=rng)
                ts.append(s); tl.append(l)
            Xtr = features(np.vstack(ts), feat, A_norm, degf)
            tests = raw_tests if feat == "raw" else bp_tests
            per_seed[si] = train_eval(Xtr, np.concatenate(tl),
                                      tests, residual, seed)
            print(f"  {name} seed {seed}: {time.time()-t0:.0f}s  "
                  f"p=0.05 LER="
                  f"{per_seed[si, P_TEST.index(0.05)]*100:.3f}%",
                  flush=True)
        res[name] = per_seed.mean(0).tolist()
        res[name + "_std"] = per_seed.std(0).tolist()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {OUT}")

    pi5 = P_TEST.index(0.05)
    print(f"\np=0.05: MWPM={mwpm_ler[pi5]*100:.3f}%  "
          f"V1={res['raw_mlp'][pi5]*100:.3f}%  "
          f"V2={res['bp_mlp'][pi5]*100:.3f}%  "
          f"V3={res['bp_res_mlp'][pi5]*100:.3f}%")


if __name__ == "__main__":
    main()
