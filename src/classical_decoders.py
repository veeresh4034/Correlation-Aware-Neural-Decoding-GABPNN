"""
Classical Decoders
==================
1. MWPM-approximation  (nearest-defect greedy matching — O(n²) pure numpy)
2. Union-Find          (Delfosse & Nickerson linear-time decoder)

Both operate on the syndrome vector and return a predicted logical label.
"""

import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# Helpers shared by both decoders
# ══════════════════════════════════════════════════════════════════════════════

def _stab_positions(code):
    """Return (row,col) centre coordinates for each stabiliser, in syndrome order."""
    d = code.d
    positions = []
    # Z stabs (detect X errors) — same order as H_z rows
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 != 0:
                positions.append((r + 0.5, c + 0.5))
    # boundary Z stabs
    for r in range(0, d - 1, 2):
        positions.append((r + 0.5, -0.5))
    for r in range(1, d - 1, 2):
        positions.append((r + 0.5, d - 0.5))

    # X stabs
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                positions.append((r + 0.5, c + 0.5))
    for c in range(0, d - 1, 2):
        positions.append((-0.5, c + 0.5))
    for c in range(1, d - 1, 2):
        positions.append((d - 0.5, c + 0.5))

    return np.array(positions[:code.n_stab])


def _pairwise_dist(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# 1.  Greedy MWPM approximation


class GreedyMWPM:
    """
    Greedy nearest-defect matching.
    For each violated syndrome bit, find its nearest unmatched partner
    and pair them. Corrects errors along the shortest path and checks
    whether the net correction commutes with logicals.

    This is an O(n²) approximation of true MWPM — accurate enough to
    serve as a strong classical baseline and produce the characteristic
    below-threshold suppression curve.
    """

    def __init__(self, code):
        self.code = code
        self.pos  = _stab_positions(code)

    def decode_batch(self, syndromes):
        """
        syndromes: (N, n_stab) uint8
        returns:   (N,) int labels 0..3
        """
        preds = np.zeros(len(syndromes), dtype=np.int64)
        for i, s in enumerate(syndromes):
            preds[i] = self._decode_one(s)
        return preds

    def _decode_one(self, syndrome):
        defects = np.where(syndrome)[0]
        if len(defects) == 0:
            return 0

        # Split into Z-syndrome defects and X-syndrome defects
        nz = self.code.n_z_stab
        z_defects = defects[defects < nz]
        x_defects = defects[defects >= nz]

        x_correction = np.zeros(self.code.n_data, dtype=np.uint8)
        z_correction = np.zeros(self.code.n_data, dtype=np.uint8)

        x_correction = self._match_and_correct(z_defects, 'x')
        z_correction = self._match_and_correct(x_defects, 'z')

        # Invariant functionals (match surface_code label convention):
        lx = int((x_correction @ self.code.L_z) % 2)
        lz = int((z_correction @ self.code.L_x) % 2)
        return lx + 2 * lz

    def _match_and_correct(self, defects, err_type):
        d = self.code.d
        correction = np.zeros(self.code.n_data, dtype=np.uint8)
        if len(defects) == 0:
            return correction

        pos = self.pos[defects]
        remaining = list(range(len(defects)))

        # Add virtual boundary nodes if odd number
        if len(remaining) % 2 == 1:
            # virtual node at nearest boundary — just drop one defect
            remaining = remaining[:-1]

        matched = set()
        for i in remaining:
            if i in matched:
                continue
            best_j, best_dist = None, np.inf
            for j in remaining:
                if j == i or j in matched:
                    continue
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is None:
                matched.add(i)
                continue
            matched.add(i)
            matched.add(best_j)

            # Apply correction along Manhattan path
            r1, c1 = pos[i]
            r2, c2 = pos[best_j]
            r1, c1 = int(round(r1)), int(round(c1))
            r2, c2 = int(round(r2)), int(round(c2))
            r1 = max(0, min(d-1, r1))
            c1 = max(0, min(d-1, c1))
            r2 = max(0, min(d-1, r2))
            c2 = max(0, min(d-1, c2))

            # Horizontal then vertical path
            cr, cc = r1, c1
            while cc != c2:
                q = cr * d + cc
                correction[q] ^= 1
                cc += 1 if c2 > cc else -1
            while cr != r2:
                q = cr * d + cc
                correction[q] ^= 1
                cr += 1 if r2 > cr else -1

        return correction



# 2.  Union-Find decoder


class UnionFind:
    """
    Union-Find decoder based on Delfosse & Nickerson (2021).
    Grows clusters around syndrome defects and merges when they touch.
    Returns a predicted logical label from the net parity of corrections.
    """

    def __init__(self, code):
        self.code = code
        self.pos  = _stab_positions(code)

    def decode_batch(self, syndromes):
        preds = np.zeros(len(syndromes), dtype=np.int64)
        for i, s in enumerate(syndromes):
            preds[i] = self._decode_one(s)
        return preds

    def _decode_one(self, syndrome):
        d = self.code.n_stab
        n = self.code.n_data

        defects = set(np.where(syndrome)[0].tolist())
        if not defects:
            return 0

        nz = self.code.n_z_stab
        z_def = [i for i in defects if i < nz]
        x_def = [i for i in defects if i >= nz]

        xc = self._uf_correct(z_def, 'x')
        zc = self._uf_correct(x_def, 'z')

        # Invariant functionals (match surface_code label convention):
        lx = int((xc @ self.code.L_z) % 2)
        lz = int((zc @ self.code.L_x) % 2)
        return lx + 2 * lz

    def _uf_correct(self, defect_ids, err_type):
        code = self.code
        correction = np.zeros(code.n_data, dtype=np.uint8)
        if not defect_ids:
            return correction

        pos = self.pos
        parent = list(range(len(defect_ids)))
        size   = [1] * len(defect_ids)
        radius = [0.5] * len(defect_ids)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a == b:
                return
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            radius[a] = max(radius[a], radius[b])

        # Grow until all clusters are even
        for _ in range(code.d * 2):
            for i in range(len(defect_ids)):
                if find(i) == i:
                    radius[i] += 0.5
            for i in range(len(defect_ids)):
                for j in range(i+1, len(defect_ids)):
                    ri, rj = find(i), find(j)
                    if ri == rj:
                        continue
                    dist = np.linalg.norm(pos[defect_ids[i]] - pos[defect_ids[j]])
                    if radius[ri] + radius[rj] >= dist:
                        union(i, j)

            # Check if all components have even size
            comp_sizes = defaultdict(int)
            for i in range(len(defect_ids)):
                comp_sizes[find(i)] += 1
            if all(v % 2 == 0 for v in comp_sizes.values()):
                break

        # Build correction from pairs within each component
        comps = defaultdict(list)
        for i, did in enumerate(defect_ids):
            comps[find(i)].append(i)

        d = code.d
        for members in comps.values():
            paired = members[:]
            if len(paired) % 2 == 1:
                paired = paired[:-1]
            for k in range(0, len(paired)-1, 2):
                i, j = paired[k], paired[k+1]
                p1 = pos[defect_ids[i]]
                p2 = pos[defect_ids[j]]
                r1, c1 = max(0,min(d-1,int(round(p1[0])))), max(0,min(d-1,int(round(p1[1]))))
                r2, c2 = max(0,min(d-1,int(round(p2[0])))), max(0,min(d-1,int(round(p2[1]))))
                cr, cc = r1, c1
                while cc != c2:
                    correction[cr*d + cc] ^= 1
                    cc += 1 if c2 > cc else -1
                while cr != r2:
                    correction[cr*d + cc] ^= 1
                    cr += 1 if r2 > cr else -1

        return correction
