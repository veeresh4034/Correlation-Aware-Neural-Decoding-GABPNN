"""
Exact MWPM decoder — thin wrapper over PyMatching (Blossom V, C++).

v2 CORRECTNESS FIX: the previous NetworkX implementation extracted
logical parity against a non-invariant label convention and mishandled
multi-defect boundary matching. This version delegates matching AND
observable prediction entirely to pymatching with the stabilizer-
invariant fault observables:

  Z-syndromes (X errors):  faults_matrix = L_z   (predicts x_err @ L_z)
  X-syndromes (Z errors):  faults_matrix = L_x   (predicts z_err @ L_x)


"""
import numpy as np
import pymatching


class ExactMWPM:
    """Exact minimum-weight perfect matching via PyMatching."""

    def __init__(self, code):
        self.code = code
        self.nz = code.n_z_stab
        # Uniform weights: for iid single-round noise the optimal
        # matching is weight-scale-invariant, so p is not needed here.
        self.m_x = pymatching.Matching(
            code.H_z, weights=np.ones(code.n_data),
            faults_matrix=code.L_z.reshape(1, -1))
        self.m_z = pymatching.Matching(
            code.H_x, weights=np.ones(code.n_data),
            faults_matrix=code.L_x.reshape(1, -1))

    def decode_batch(self, syndromes, p=0.05,
                     channel='depolarising', eta=1.0):
        s = np.asarray(syndromes, dtype=np.uint8)
        lx = self.m_x.decode_batch(s[:, :self.nz])[:, 0]
        lz = self.m_z.decode_batch(s[:, self.nz:])[:, 0]
        return (lx + 2 * lz).astype(np.int64)
