"""
generates the 7 manuscript figures (IEEE Access spec: 11 pt fonts,
vector PDF, 7.16 in / 3.5 in widths) from results/*.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results") + os.sep
OUT = os.path.join(HERE, "..", "paper_figures") + os.sep
os.makedirs(OUT, exist_ok=True)
S = json.load(open(R + "results_symmetric.json"))
L = json.load(open(R + "results_large_d.json"))
A = json.load(open(R + "results_ablation.json"))
Y = json.load(open(R + "results_asymmetric.json"))
T = json.load(open(R + "results_timing.json"))
P = np.array(S["p_values"])

FULL_W, SINGLE_W = 7.16, 3.5
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlepad": 7,
    "axes.linewidth": 1.0, "axes.grid": True,
    "grid.alpha": 0.20, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.fontsize": 10, "legend.framealpha": 0.97,
    "legend.edgecolor": "#bbbbbb",
    "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "lines.linewidth": 1.8, "lines.markersize": 6.5,
})
CM, CU, CG = "#C0392B", "#E07B39", "#1A7A4A"
DCOL = {3: "#C0392B", 5: "#E07B39", 7: "#1A7A4A",
        9: "#2980B9", 11: "#7B2FBE"}
MEW, MFC, LW = 1.4, "white", 1.8


def ci95(y, n):
    y = np.asarray(y, float); z = 1.96
    lo = (y+z*z/(2*n)-z*np.sqrt(y*(1-y)/n+z*z/(4*n*n)))/(1+z*z/n)
    hi = (y+z*z/(2*n)+z*np.sqrt(y*(1-y)/n+z*z/(4*n*n)))/(1+z*z/n)
    return np.maximum(lo, 1e-7), np.minimum(hi, 1.0)


def logax(ax, ylabel=True, ymin=5e-6):
    ax.set_xlabel(r"Physical error rate, $p$")
    if ylabel:
        ax.set_ylabel(r"Logical error rate, $P_L$")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.tick_params(which="both", top=True, right=True)
    ax.set_xlim(-0.004, 0.155); ax.set_ylim(ymin, 1.0)


def ptag(ax, t):
    ax.text(0.05, 0.97, t, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#aaaaaa", alpha=0.9, linewidth=0.8))


# FIG 1
fig = plt.figure(figsize=(FULL_W, 3.1))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)
for col, d in enumerate(["3", "5", "7"]):
    ax = fig.add_subplot(gs[col])
    D = S["sym"][d]
    for y, lab, c, mk, ls in [(D["mwpm"], "Exact MWPM", CM, "o", "-"),
                              (D["uf"], "Union-Find", CU, "s", "--")]:
        y = np.array(y); m = y > 0
        lo, hi = ci95(y, 100000)
        ax.fill_between(P[m], lo[m], hi[m], alpha=0.10, color=c, lw=0)
        ax.plot(P[m], y[m], marker=mk, color=c, lw=LW, ms=6,
                markerfacecolor=MFC, markeredgecolor=c,
                markeredgewidth=MEW, linestyle=ls, label=lab)
    g = np.array(D["gabpnn_mean"]); gs_ = np.array(D["gabpnn_std"])
    m = g > 0
    ax.fill_between(P[m], np.maximum(g[m]-gs_[m], 1e-7), g[m]+gs_[m],
                    alpha=0.18, color=CG, lw=0)
    ax.plot(P[m], g[m], marker="D", color=CG, lw=2.0, ms=6,
            markerfacecolor=MFC, markeredgecolor=CG,
            markeredgewidth=MEW, label="GABPNN (ours)", zorder=4)
    logax(ax, ylabel=(col == 0))
    if col:
        ax.set_yticklabels([])
    ax.set_title(f"$d={d}$")
    ptag(ax, f"({chr(97+col)})")
    if col == 0:
        ax.legend(loc="lower right", fontsize=9)
plt.savefig(OUT+"fig1_sym_ler.pdf"); plt.close(); print("fig1")

# FIG 2
fig, ax = plt.subplots(figsize=(SINGLE_W, 3.0))
ds, deltas = [], []
for d in ["3", "5", "7"]:
    i = list(P).index(0.05)
    m0, g0 = S["sym"][d]["mwpm"][i], S["sym"][d]["gabpnn_mean"][i]
    ds.append(int(d)); deltas.append((m0-g0)/m0*100)
for d in ["9", "11"]:
    i = L["p_values"].index(0.05)
    m0, g0 = L["sym"][d]["mwpm"][i], L["sym"][d]["gabpnn_mean"][i]
    ds.append(int(d)); deltas.append((m0-g0)/m0*100)
cols = [CG if v > 0 else CM for v in deltas]
bars = ax.bar([str(d) for d in ds], deltas, color=cols, alpha=0.85,
              edgecolor="white", width=0.6, zorder=3)
for b, v in zip(bars, deltas):
    ax.annotate(f"{v:+.0f}%", (b.get_x()+b.get_width()/2, v),
                textcoords="offset points",
                xytext=(0, 5 if v > 0 else -13),
                ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(0, color="#333", lw=1.0)
ax.set_xlabel(r"Code distance, $d$")
ax.set_ylabel(r"$P_L$ change vs.\ exact MWPM (%)")
ax.set_ylim(-1750, 130)
ax.set_yscale("symlog", linthresh=20)
ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT+"fig2_crossover.pdf"); plt.close(); print("fig2")

# FIG 3
fig = plt.figure(figsize=(FULL_W*0.72, 3.1))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
for col, (key, title) in enumerate([("mwpm", "Exact MWPM"),
                                    ("gabpnn_mean", "GABPNN")]):
    ax = fig.add_subplot(gs[col])
    for d in [3, 5, 7, 9, 11]:
        src = S["sym"][str(d)] if d <= 7 else L["sym"][str(d)]
        y = np.array(src[key]); m = y > 0
        ax.plot(P[m], y[m], marker="o", color=DCOL[d], lw=LW, ms=5.5,
                markerfacecolor=MFC, markeredgecolor=DCOL[d],
                markeredgewidth=MEW, label=f"$d={d}$")
    logax(ax, ylabel=(col == 0), ymin=5e-6)
    if col:
        ax.set_yticklabels([])
    ax.set_title(title)
    ptag(ax, f"({chr(97+col)})")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2,
              columnspacing=0.8, handletextpad=0.4)
plt.savefig(OUT+"fig3_scaling.pdf"); plt.close(); print("fig3")

# FIG 4
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2))
ax = axes[0]
sty = [(np.array(A["mwpm"]), None, "Exact MWPM", CM, "o", "-"),
       (np.array(A["raw_mlp"]), np.array(A["raw_mlp_std"]),
        "V1: Raw+MLP", "#8E44AD", "v", "--"),
       (np.array(A["bp_mlp"]), np.array(A["bp_mlp_std"]),
        "V2: BP+MLP", "#2980B9", "s", "-."),
       (np.array(A["bp_res_mlp"]), np.array(A["bp_res_mlp_std"]),
        "V3: GABPNN", CG, "D", "-")]
for y, s, lab, c, mk, ls in sty:
    m = y > 0
    if s is not None:
        ax.fill_between(P[m], np.maximum(y[m]-s[m], 1e-7), y[m]+s[m],
                        alpha=0.13, color=c, lw=0)
    ax.plot(P[m], y[m], marker=mk, color=c, lw=LW, ms=6,
            markerfacecolor=MFC, markeredgecolor=c,
            markeredgewidth=MEW, linestyle=ls, label=lab)
logax(ax, ymin=1e-5)
ax.set_title("(a) Ablation: $P_L$ comparison, $d=5$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, fontsize=9.5)
ax2 = axes[1]
v1 = np.array(A["raw_mlp"]); v3 = np.array(A["bp_res_mlp"])
m = v1 > 0
gain = np.where(v1 > 0, (v1-v3)/np.maximum(v1, 1e-12)*100, 0)
s1 = np.array(A["raw_mlp_std"]); s3 = np.array(A["bp_res_mlp_std"])
band = np.where(v1 > 0, (s1+s3)/np.maximum(v1, 1e-12)*100, 0)
ax2.fill_between(P[m], gain[m]-band[m], gain[m]+band[m],
                 alpha=0.15, color=CG, lw=0,
                 label=r"$\pm$1 std (5 seeds)")
ax2.plot(P[m], gain[m], "-D", color=CG, lw=LW, ms=6,
         markerfacecolor=MFC, markeredgecolor=CG, markeredgewidth=MEW,
         label="V3 over V1 (BP+skip gain)")
ax2.axhline(0, color="#444", lw=0.9, ls=":")
ax2.set_xlabel(r"Physical error rate, $p$")
ax2.set_ylabel("Relative gain (%)")
ax2.set_title("(b) Architecture gain at full training budget")
ax2.set_ylim(-14, 14)
ax2.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax2.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax2.tick_params(which="both", direction="in", top=True, right=True)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
           fontsize=9.5)
fig.subplots_adjust(bottom=0.30, wspace=0.32)
plt.savefig(OUT+"fig4_ablation.pdf"); plt.close(); print("fig4")

# FIG 5
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))
etas = [1, 10, 100, 300, 1000]
i5 = Y["p_values"].index(0.05); i1 = Y["p_values"].index(0.01)
for ax, ii, pv, tag in [(axes[0], i5, 0.05, "(a)"),
                        (axes[1], i1, 0.01, "(b)")]:
    mw = [Y["asym"][str(e)]["mwpm"][ii]*100 for e in etas]
    sy = [Y["asym"][str(e)]["sym_mean"][ii]*100 for e in etas]
    ss = [Y["asym"][str(e)]["sym_std"][ii]*100 for e in etas]
    am = [Y["asym"][str(e)]["asym_mean"][ii]*100 for e in etas]
    as_ = [Y["asym"][str(e)]["asym_std"][ii]*100 for e in etas]
    x = np.arange(len(etas))
    ax.plot(x, mw, "-o", color=CM, lw=LW, ms=6, markerfacecolor=MFC,
            markeredgecolor=CM, markeredgewidth=MEW,
            label="Exact MWPM")
    ax.errorbar(x, sy, yerr=ss, fmt="--s", color="#7B2FBE", lw=LW,
                ms=6, markerfacecolor=MFC, markeredgecolor="#7B2FBE",
                markeredgewidth=MEW, capsize=3, label="GABPNN-sym")
    ax.errorbar(x, am, yerr=as_, fmt="-D", color=CG, lw=LW, ms=6,
                markerfacecolor=MFC, markeredgecolor=CG,
                markeredgewidth=MEW, capsize=3, label="GABPNN-asym")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\eta={e}$" for e in etas], fontsize=9)
    ax.set_ylabel(r"$P_L$ (%)")
    ax.set_title(f"$p={pv}$, $d=5$")
    ptag(ax, tag)
    ax.grid(True, alpha=0.2, ls="--", lw=0.5)
    ax.tick_params(direction="in")
axes[0].legend(fontsize=9, loc="center left")
plt.tight_layout(w_pad=2.5)
plt.savefig(OUT+"fig5_asymmetric.pdf"); plt.close(); print("fig5")

# FIG 6
fig, ax = plt.subplots(figsize=(SINGLE_W, 3.0))
ds = [3, 5, 7, 9]
pm = [T["timing"][str(d)]["greedy_mwpm"]["median_us"] for d in ds]
uf = [T["timing"][str(d)]["union_find"]["median_us"] for d in ds]
gc = [T["timing"][str(d)]["gabpnn_cpu"]["median_us"] for d in ds]
gg = [T["timing"][str(d)]["gabpnn_gpu"]["median_us"] for d in ds]
ax.plot(ds, pm, "-o", color=CM, lw=LW, ms=6.5, markerfacecolor=MFC,
        markeredgecolor=CM, markeredgewidth=MEW,
        label="Exact MWPM (PyMatching, C++)")
ax.plot(ds, uf, "--s", color=CU, lw=LW, ms=6.5, markerfacecolor=MFC,
        markeredgecolor=CU, markeredgewidth=MEW,
        label="Union-Find (Py)")
ax.plot(ds, gc, "-.v", color="#2980B9", lw=LW, ms=6.5,
        markerfacecolor=MFC, markeredgecolor="#2980B9",
        markeredgewidth=MEW, label="GABPNN (CPU)")
ax.plot(ds, gg, "-D", color=CG, lw=2.0, ms=6.5, markerfacecolor=MFC,
        markeredgecolor=CG, markeredgewidth=MEW,
        label="GABPNN (GPU, batched)")
ax.set_yscale("log")
ax.set_xticks(ds)
ax.set_xlabel(r"Code distance, $d$")
ax.set_ylabel(r"Decode time ($\mu$s / sample)")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(True, which="both", alpha=0.2, ls="--", lw=0.5)
ax.tick_params(which="both", direction="in", top=True, right=True)
plt.tight_layout()
plt.savefig(OUT+"fig6_timing.pdf"); plt.close(); print("fig6")

# FIG 7 — v1 archived (pre-fix) d=5 numbers for the case study
v1_p = [0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05,
        0.07, 0.10, 0.13, 0.15]
v1_mwpm = [0.00126, 0.00232, 0.00672, 0.01106, 0.01383, 0.0277,
           0.04218, 0.07137, 0.10387, 0.15044, 0.20034, 0.23218]
v1_gab = [0.000662, 0.00133, 0.003528, 0.005608, 0.00672, 0.01431,
          0.023198, 0.043058, 0.066716, 0.1088, 0.154862, 0.191186]
v1_imp = [36.6, 39.7, 47.3, 53.7, 60.8]   # apparent gain, d=3..11
v2_mwpm = np.array(S["sym"]["5"]["mwpm"])
v2_gab = np.array(S["sym"]["5"]["gabpnn_mean"])
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.2))
ax = axes[0]
ax.plot(v1_p, v1_mwpm, "--o", color=CM, lw=LW, ms=5.5, alpha=0.55,
        markerfacecolor=MFC, markeredgecolor=CM, markeredgewidth=1.2,
        label="MWPM, non-invariant labels")
ax.plot(v1_p, v1_gab, "--D", color=CG, lw=LW, ms=5.5, alpha=0.55,
        markerfacecolor=MFC, markeredgecolor=CG, markeredgewidth=1.2,
        label="GABPNN, non-invariant labels")
m = v2_mwpm > 0
ax.plot(P[m], v2_mwpm[m], "-o", color=CM, lw=2.0, ms=6,
        markerfacecolor=CM, label="MWPM, corrected labels")
m = v2_gab > 0
ax.plot(P[m], v2_gab[m], "-D", color=CG, lw=2.0, ms=6,
        markerfacecolor=CG, label="GABPNN, corrected labels")
logax(ax, ymin=1e-5)
ax.set_title("(a) Label convention effect, $d=5$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, fontsize=8.5)
ax2 = axes[1]
ds_all = [3, 5, 7, 9, 11]
v2_imp = []
for d in ds_all:
    if d <= 7:
        i = list(P).index(0.05)
        m0 = S["sym"][str(d)]["mwpm"][i]
        g0 = S["sym"][str(d)]["gabpnn_mean"][i]
    else:
        i = L["p_values"].index(0.05)
        m0 = L["sym"][str(d)]["mwpm"][i]
        g0 = L["sym"][str(d)]["gabpnn_mean"][i]
    v2_imp.append((m0-g0)/m0*100)
x = np.arange(len(ds_all)); w = 0.38
ax2.bar(x-w/2, v1_imp, w, color="#999999", alpha=0.8,
        label="Non-invariant labels", edgecolor="white", zorder=3)
ax2.bar(x+w/2, v2_imp, w, color=CG, alpha=0.9,
        label="Corrected labels", edgecolor="white", zorder=3)
ax2.axhline(0, color="#333", lw=1.0)
ax2.set_xticks(x)
ax2.set_xticklabels([f"$d={d}$" for d in ds_all])
ax2.set_ylabel(r"Apparent gain vs.\ MWPM (%)")
ax2.set_title("(b) Manufactured improvement at $p=0.05$")
ax2.set_yscale("symlog", linthresh=50)
ax2.set_ylim(-1800, 110)
ax2.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5)
ax2.set_axisbelow(True)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=2, fontsize=9)
fig.subplots_adjust(bottom=0.30, wspace=0.32)
plt.savefig(OUT+"fig7_pitfall.pdf"); plt.close(); print("fig7")
print("All 7 paper figures regenerated.")
