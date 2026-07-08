"""
Hybrid Neural-Network Decoder

Architecture: Graph-Augmented Belief-Propagation Neural Network (GABPNN)

Novel contributions:
  1. Syndrome graph features (node degree, adjacency weights) computed via
     a single BP-inspired message-passing pre-processing step.
  2. Channel-conditioning: bias indicator concatenated to every feature vector,
     enabling a single model to handle both symmetric and asymmetric channels.
  3. Residual MLP tower on top of the graph features for final coset prediction.

Implemented entirely in numpy/scipy — no PyTorch required.
Training uses mini-batch SGD with Adam optimiser, implemented from scratch.
"""

import numpy as np
from scipy.special import expit as sigmoid   # numerically stable σ(x)


# ══════════════════════════════════════════════════════════════════════════════
# Adam optimiser (per-parameter, in-place)
# ══════════════════════════════════════════════════════════════════════════════

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = {}
        self.v     = {}
        self.t     = 0

    def step(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ══════════════════════════════════════════════════════════════════════════════
# Activations & losses
# ══════════════════════════════════════════════════════════════════════════════

def relu(x):       return np.maximum(0, x)
def relu_grad(x):  return (x > 0).astype(float)

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, labels):
    n = len(labels)
    return -np.log(probs[np.arange(n), labels] + 1e-12).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Graph feature extractor (BP message-passing pre-processor)
# ══════════════════════════════════════════════════════════════════════════════

class BPFeatureExtractor:
    """
    Given a syndrome vector s ∈ {0,1}^n_stab, run K rounds of belief-
    propagation-style message passing on the Tanner graph of the code.
    Outputs enriched node features: [s_i, sum_j s_j·A_ij, degree_i, channel_bias]
    """

    def __init__(self, code, K=3):
        self.code = code
        self.K    = K
        self._build_adjacency()

    def _build_adjacency(self):
        """Syndrome-to-syndrome adjacency via shared data qubits."""
        H = np.vstack([self.code.H_z, self.code.H_x]).astype(float)   # (n_stab, n_data)
        # A[i,j] = number of shared data qubits between stabiliser i and j
        A = H @ H.T
        np.fill_diagonal(A, 0)
        # Normalise rows
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1
        self.A_norm = A / deg
        self.degree = A.sum(axis=1) / (A.max() + 1e-8)   # normalised degree

    def extract(self, syndromes, eta_feat):
        """
        syndromes : (N, n_stab) float
        eta_feat  : (N, 1)      log10(eta) normalised to [0,1]
        Returns   : (N, n_stab * (K+1) + n_stab + 1)  feature matrix
        """
        N, ns = syndromes.shape
        s = syndromes.astype(float)

        features = [s]   # raw syndrome
        msg = s.copy()
        for _ in range(self.K):
            msg = msg @ self.A_norm.T    # (N, ns) message aggregation
            features.append(msg)

        # Degree features (broadcast over batch)
        deg_feat = np.tile(self.degree, (N, 1))      # (N, ns)
        features.append(deg_feat)

        # Channel conditioning
        features.append(np.tile(eta_feat, (1, ns)))  # (N, ns)

        return np.hstack(features)   # (N, ns*(K+3))


# ══════════════════════════════════════════════════════════════════════════════
# Residual MLP
# ══════════════════════════════════════════════════════════════════════════════

class ResidualMLP:
    """
    3-hidden-layer MLP with skip connections.
    Layers: input → H1 → H2 → H3 → 4 classes
    Skip:   input (projected) → H3 output
    """

    def __init__(self, input_dim, hidden=256, n_classes=4, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        self.n_classes = n_classes

        def W(a, b):
            return rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0/a)
        def b_(n):
            return np.zeros(n, dtype=np.float32)

        self.params = {
            'W1': W(input_dim, hidden),   'b1': b_(hidden),
            'W2': W(hidden,    hidden),   'b2': b_(hidden),
            'W3': W(hidden,    hidden),   'b3': b_(hidden),
            'W4': W(hidden,    n_classes),'b4': b_(n_classes),
            'Ws': W(input_dim, hidden),   'bs': b_(hidden),   # skip projection
        }
        self.cache = {}

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x, training=True):
        p = self.params
        x = x.astype(np.float32)

        h1_pre = x @ p['W1'] + p['b1']
        h1     = relu(h1_pre)

        h2_pre = h1 @ p['W2'] + p['b2']
        h2     = relu(h2_pre)

        h3_pre = h2 @ p['W3'] + p['b3']
        skip   = x @ p['Ws'] + p['bs']       # residual skip
        h3     = relu(h3_pre + skip)

        logits = h3 @ p['W4'] + p['b4']
        probs  = softmax(logits)

        if training:
            self.cache = dict(x=x, h1_pre=h1_pre, h1=h1,
                              h2_pre=h2_pre, h2=h2,
                              h3_pre=h3_pre, skip=skip, h3=h3,
                              logits=logits, probs=probs)
        return probs

    # ── Backward ─────────────────────────────────────────────────────────────
    def backward(self, labels):
        c = self.cache
        p = self.params
        N = len(labels)
        grads = {}

        # Output layer gradient
        dL = c['probs'].copy()
        dL[np.arange(N), labels] -= 1
        dL /= N

        grads['W4'] = c['h3'].T @ dL
        grads['b4'] = dL.sum(0)
        dh3         = dL @ p['W4'].T

        # h3 = relu(h3_pre + skip)
        dh3_pre_skip = dh3 * relu_grad(c['h3_pre'] + c['skip'])
        grads['W3']  = c['h2'].T  @ dh3_pre_skip
        grads['b3']  = dh3_pre_skip.sum(0)
        grads['Ws']  = c['x'].T   @ dh3_pre_skip
        grads['bs']  = dh3_pre_skip.sum(0)
        dh2          = dh3_pre_skip @ p['W3'].T

        dh2_pre     = dh2 * relu_grad(c['h2_pre'])
        grads['W2'] = c['h1'].T @ dh2_pre
        grads['b2'] = dh2_pre.sum(0)
        dh1         = dh2_pre @ p['W2'].T

        dh1_pre     = dh1 * relu_grad(c['h1_pre'])
        grads['W1'] = c['x'].T @ dh1_pre
        grads['b1'] = dh1_pre.sum(0)

        # Clip gradients
        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads


# ══════════════════════════════════════════════════════════════════════════════
# Full Hybrid Decoder
# ══════════════════════════════════════════════════════════════════════════════

class HybridGABPNN:
    """
    Graph-Augmented Belief-Propagation Neural Network (GABPNN) decoder.

    Pipeline:
      syndrome → BPFeatureExtractor → ResidualMLP → logical coset label
    """

    def __init__(self, code, hidden=256, bp_rounds=3, lr=1e-3, rng=None):
        self.code = code
        self.bp   = BPFeatureExtractor(code, K=bp_rounds)

        # Compute input dimension
        ns = code.n_stab
        feat_dim = ns * (bp_rounds + 3)   # raw + K aggregations + degree + eta_feat

        self.mlp  = ResidualMLP(feat_dim, hidden=hidden, rng=rng)
        self.opt  = Adam(lr=lr)
        self.rng  = rng if rng else np.random.default_rng(42)

    def _eta_to_feat(self, eta, N):
        """Normalise eta ∈ {1,10,100,1000} → [0,1]."""
        return np.full((N, 1), np.log10(max(eta, 1)) / 3.0, dtype=np.float32)

    def _featurise(self, syndromes, eta=1.0):
        N = len(syndromes)
        eta_f = self._eta_to_feat(eta, N)
        return self.bp.extract(syndromes, eta_f).astype(np.float32)

    def train(self, syndromes, labels, epochs=30, batch_size=256, eta=1.0, verbose=False):
        N = len(syndromes)
        X = self._featurise(syndromes, eta)
        losses = []

        for ep in range(epochs):
            idx = self.rng.permutation(N)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                bi = idx[start:start+batch_size]
                xb, yb = X[bi], labels[bi]
                probs = self.mlp.forward(xb, training=True)
                loss  = cross_entropy(probs, yb)
                grads = self.mlp.backward(yb)
                self.opt.step(self.mlp.params, grads)
                ep_loss += loss
                n_batches += 1
            avg_loss = ep_loss / n_batches
            losses.append(avg_loss)
            if verbose and (ep+1) % 10 == 0:
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={avg_loss:.4f}")

        return losses

    def decode_batch(self, syndromes, eta=1.0):
        X     = self._featurise(syndromes, eta)
        probs = self.mlp.forward(X, training=False)
        return probs.argmax(axis=1).astype(np.int64)

    def predict_proba(self, syndromes, eta=1.0):
        X = self._featurise(syndromes, eta)
        return self.mlp.forward(X, training=False)
