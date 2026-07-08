"""
EXPERIMENT 6 — Exact MWPM (pymatching) comparison (Discussion Sec. V-B).

pip install pymatching

Approach: pymatching decodes X and Z syndromes independently using the
check matrices and uniform weights -log(p/3), exactly mirroring the
Greedy MWPM baseline's noise prior, so the comparison isolates
matching quality (greedy vs exact Blossom V).


"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.surface_code import SurfaceCode
from src.exact_mwpm_nx import ExactMWPM

import pymatching

DS      = [3, 5, 7]
P_TEST  = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
N_TEST  = 100_000
OUT     = os.path.join(os.path.dirname(__file__), "..",
                       "results", "results_pymatching.json")


def build_matchers(code):
    """Two independent matchers: Z-stabs decode X errors (predict L_z
    logical flip? -> use L_x support as fault ids is subtle; we use
    the direct approach: faults = data qubits, observable = logical)."""
    mx = pymatching.Matching(code.H_z,
                             weights=np.ones(code.n_data),
                             faults_matrix=code.L_x.reshape(1, -1))
    mz = pymatching.Matching(code.H_x,
                             weights=np.ones(code.n_data),
                             faults_matrix=code.L_z.reshape(1, -1))
    return mx, mz


def main():
    res = {"p_values": P_TEST, "distances": DS, "n_test": N_TEST,
           "data": {}}
    for d in DS:
        code = SurfaceCode(d)
        greedy = ExactMWPM(code)
        mx, mz = build_matchers(code)
        nz = code.n_z_stab
        entry = {"greedy": [], "pymatching": [], "pm_time_us": []}
        print(f"\n=== d={d} ===", flush=True)
        rng = np.random.default_rng(999)
        for p in P_TEST:
            s, l, xe, ze = code.generate_samples(N_TEST, p, rng=rng)
            g_ler = float((greedy.decode_batch(s, p=p) != l).mean())

            sx, sz = s[:, :nz], s[:, nz:]
            t0 = time.perf_counter()
            px = mx.decode_batch(sx)[:, 0]     # predicted L_x flip
            pz = mz.decode_batch(sz)[:, 0]     # predicted L_z flip
            pm_time = (time.perf_counter() - t0) / N_TEST * 1e6

            pred = (px + 2 * pz).astype(np.int64)
            pm_ler = float((pred != l).mean())

            entry["greedy"].append(g_ler)
            entry["pymatching"].append(pm_ler)
            entry["pm_time_us"].append(pm_time)
            print(f"  p={p:.3f}: Greedy={g_ler*100:.3f}%  "
                  f"pymatching={pm_ler*100:.3f}%  "
                  f"({pm_time:.2f} us/sample)", flush=True)
        res["data"][str(d)] = entry

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {OUT}")
    print("\nUse these exact numbers in Discussion Section V-B in place")
    print("of the current approximate values.")


if __name__ == "__main__":
    main()
