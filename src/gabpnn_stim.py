"""
GABPNNStim — GABPNN decoder fully consistent with Stim circuit-level noise.

Key design:
- BP graph built directly from Stim's Detector Error Model (DEM)
  so syndrome bits perfectly match graph edges
- Binary output (logical error / no error) matching Stim observables
- Residual MLP with skip connection, trained with Adam
- Channel conditioning scalar for asymmetric noise support
"""
import numpy as np


class GABPNNStim:

    def __init__(self, circuit, hidden=256, bp_rounds=3, lr=5e-4, rng=None):
        self.n   = circuit.num_detectors
        self.K   = bp_rounds
        self.lr  = lr
        self.rng = rng if rng is not None else np.random.default_rng(2024)

        # Build normalised adjacency from DEM
        dem = circuit.detector_error_model(decompose_errors=True)
        A   = np.zeros((self.n, self.n), dtype=np.float32)
        for inst in dem.flattened():
            if inst.type == 'error':
                dets = [t.val for t in inst.targets_copy()
                        if t.is_relative_detector_id()]
                for i in range(len(dets)):
                    for j in range(i + 1, len(dets)):
                        A[dets[i], dets[j]] += 1
                        A[dets[j], dets[i]] += 1
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1
        self.A_norm = (A / deg).astype(np.float32)
        self.degree = (A.sum(axis=1) / (A.max() + 1e-8)).astype(np.float32)

        # MLP initialisation
        feat_dim = self.n * (bp_rounds + 3)
        H = hidden

        def W(a, b):
            return self.rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2. / a)

        def bv(n_):
            return np.zeros(n_, dtype=np.float32)

        self.p = {
            'W1': W(feat_dim, H), 'b1': bv(H),
            'W2': W(H, H),        'b2': bv(H),
            'W3': W(H, H),        'b3': bv(H),
            'Ws': W(feat_dim, H), 'bs': bv(H),   # skip connection
            'W4': W(H, 1),        'b4': bv(1),   # binary logit
        }
        self.m  = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v_ = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t  = 0

    # ── Feature extraction ────────────────────────────────────────────────────
    def _features(self, syndromes, eta=1.0):
        N    = len(syndromes)
        s    = syndromes.astype(np.float32)
        eta_v = float(np.log10(max(eta, 1)) / 3.)
        feats = [s]
        msg   = s.copy()
        for _ in range(self.K):
            msg = msg @ self.A_norm.T
            feats.append(msg)
        feats.append(np.tile(self.degree, (N, 1)))
        feats.append(np.full((N, self.n), eta_v, dtype=np.float32))
        return np.hstack(feats)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, X, train=True):
        p  = self.p
        h1 = np.maximum(0, X  @ p['W1'] + p['b1'])
        h2 = np.maximum(0, h1 @ p['W2'] + p['b2'])
        h3 = np.maximum(0, h2 @ p['W3'] + p['b3']
                            + X @ p['Ws'] + p['bs'])   # residual skip
        logit = h3 @ p['W4'] + p['b4']
        prob  = 1. / (1. + np.exp(-np.clip(logit, -20, 20)))
        if train:
            self._cache = dict(X=X, h1=h1, h2=h2, h3=h3, prob=prob)
        return prob

    # ── Backward pass (binary cross-entropy) ─────────────────────────────────
    def _backward(self, labels):
        c = self._cache; p = self.p
        N = len(labels); g = {}
        dL = (c['prob'].squeeze() - labels.astype(np.float32)) / N
        dL = dL[:, None]
        g['W4'] = c['h3'].T @ dL;  g['b4'] = dL.sum(0)
        dh3     = (dL @ p['W4'].T) * (c['h3'] > 0)
        g['W3'] = c['h2'].T @ dh3; g['b3'] = dh3.sum(0)
        g['Ws'] = c['X'].T  @ dh3; g['bs'] = dh3.sum(0)
        dh2     = (dh3 @ p['W3'].T) * (c['h2'] > 0)
        g['W2'] = c['h1'].T @ dh2; g['b2'] = dh2.sum(0)
        dh1     = (dh2 @ p['W2'].T) * (c['h1'] > 0)
        g['W1'] = c['X'].T  @ dh1; g['b1'] = dh1.sum(0)
        for k in g:
            np.clip(g[k], -5, 5, out=g[k])
        return g

    # ── Adam update ───────────────────────────────────────────────────────────
    def _adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            self.m[k]  = b1  * self.m[k]  + (1 - b1) * grads[k]
            self.v_[k] = b2  * self.v_[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k]  / (1 - b1 ** self.t)
            vh = self.v_[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    # ── Public: train ─────────────────────────────────────────────────────────
    def train(self, syndromes, labels, epochs=50, batch=512,
              eta=1.0, verbose=False):
        X   = self._features(syndromes, eta)
        N   = len(labels)
        rng2 = np.random.default_rng(99)
        for ep in range(epochs):
            idx = rng2.permutation(N)
            loss_sum = 0.; nb = 0
            for s in range(0, N, batch):
                xb = X[idx[s:s + batch]]
                yb = labels[idx[s:s + batch]]
                pr = self._forward(xb)
                pr_c = np.clip(pr.squeeze(), 1e-7, 1 - 1e-7)
                loss_sum += (-yb * np.log(pr_c)
                             - (1 - yb) * np.log(1 - pr_c)).mean()
                g = self._backward(yb)
                self._adam(g)
                nb += 1
            if verbose and (ep + 1) % 10 == 0:
                print(f'    epoch {ep+1}/{epochs}  loss={loss_sum/nb:.4f}')

    # ── Public: decode ────────────────────────────────────────────────────────
    def decode(self, syndromes, eta=1.0):
        X    = self._features(syndromes, eta)
        prob = self._forward(X, train=False).squeeze()
        return (prob >= 0.5).astype(np.int64)
