"""
GABPNN — PyTorch GPU implementation.


Architecture (Definition 1 in the paper):
  BP pre-processor (parameter-free, K=3 rounds on Tanner graph)
  -> feature vector x in R^{m(K+3)}
  -> residual MLP (3 hidden layers, H units, skip x->h3)
  -> softmax over 4 logical cosets {I, X_L, Z_L, Y_L}
"""
import numpy as np
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BPFeaturesTorch:
    """Parameter-free BP pre-processor on GPU (Algorithm 1)."""

    def __init__(self, code, K=3, device=None):
        self.K = K
        self.device = device or get_device()
        H = np.vstack([code.H_z, code.H_x]).astype(np.float32)
        A = (H @ H.T)
        np.fill_diagonal(A, 0)
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1
        self.A_norm = torch.tensor(A / deg, device=self.device)
        self.degree = torch.tensor(
            (A.sum(axis=1) / (A.max() + 1e-8)).astype(np.float32),
            device=self.device)
        self.m = code.n_stab

    @torch.no_grad()
    def extract(self, syndromes, eta=1.0):
        """syndromes: (N, m) uint8 numpy or torch -> (N, m(K+3)) float32 GPU."""
        if isinstance(syndromes, np.ndarray):
            s = torch.tensor(syndromes, dtype=torch.float32,
                             device=self.device)
        else:
            s = syndromes.to(self.device, dtype=torch.float32)
        N = s.shape[0]
        eta_v = float(np.log10(max(eta, 1.0)) / 3.0)
        feats = [s]
        msg = s
        for _ in range(self.K):
            msg = msg @ self.A_norm.T
            feats.append(msg)
        feats.append(self.degree.expand(N, -1))
        feats.append(torch.full((N, self.m), eta_v, device=self.device))
        return torch.cat(feats, dim=1)


class ResidualMLPTorch(nn.Module):
    """3-hidden-layer MLP with skip x->h3 (Eq. 5-7)."""

    def __init__(self, in_dim, hidden=256, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.skip = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, n_classes)
        self.act = nn.ReLU()
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        h3 = self.act(self.fc3(h2) + self.skip(x))   # residual skip
        return self.out(h3)


class GABPNNTorch:
    """Full GABPNN decoder (GPU). Drop-in analogue of HybridGABPNN."""

    def __init__(self, code, hidden=256, bp_rounds=3, lr=7e-4,
                 seed=2024, device=None):
        self.device = device or get_device()
        torch.manual_seed(seed)
        self.bp = BPFeaturesTorch(code, K=bp_rounds, device=self.device)
        in_dim = code.n_stab * (bp_rounds + 3)
        self.net = ResidualMLPTorch(in_dim, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, syndromes, labels, epochs=100, batch=1024,
              eta=1.0, verbose=True):
        X = self.bp.extract(syndromes, eta)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
        N = len(y)
        self.net.train()
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)
            tot = 0.0
            nb = 0
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                self.opt.zero_grad()
                logits = self.net(X[idx])
                loss = self.loss_fn(logits, y[idx])
                loss.backward()
                self.opt.step()
                tot += loss.item()
                nb += 1
            if verbose and (ep + 1) % 20 == 0:
                print(f"    epoch {ep+1}/{epochs}  loss={tot/nb:.4f}",
                      flush=True)

    @torch.no_grad()
    def decode_batch(self, syndromes, eta=1.0):
        self.net.eval()
        X = self.bp.extract(syndromes, eta)
        preds = []
        for i in range(0, X.shape[0], 65536):
            preds.append(self.net(X[i:i + 65536]).argmax(dim=1))
        return torch.cat(preds).cpu().numpy()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(
            torch.load(path, map_location=self.device))
