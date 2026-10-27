"""Step 3 — produce every figure for the review deck from the saved results.

  python make_figures.py --arch edgenet
Figures go to results/figures/*.png (200 dpi, slide-ready).
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

p = argparse.ArgumentParser(); p.add_argument("--arch", default="edgenet"); a = p.parse_args()
R = json.load(open(f"results/results_{a.arch}.json"))
T = json.load(open(f"results/train_{a.arch}.json"))
D = pd.read_csv(f"results/samples_{a.arch}.csv")
OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)

# ---- palette (validated reference palette, light mode) ----
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlecolor": INK, "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2,
    "axes.axisbelow": True})
METHODS = ["MSP (softmax)", "Entropy", "Energy", "Mahalanobis", [k for k in R["failure_detection_all_ID"] if k.startswith("MC")][0], "EdgeGuard CARS"]
MCOL = {m: c for m, c in zip(METHODS, [MUTED, "#b9b7ae", S[3], S[2], S[6], S[0]])}


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig); print("saved", name)


def bar_labels(ax, bars, fmt="{:.2f}", color=INK2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=color)


# 1 training curves
h = pd.DataFrame(T["history"])
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(h.epoch, h.train_acc * 100, color=S[0], label="Train")
ax.plot(h.epoch, h.val_acc * 100, color=S[1], label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Base model training ({a.arch}, {T['params']/1e6:.2f}M params)")
ax.text(h.epoch.iloc[-1], h.val_acc.iloc[-1] * 100 - 4, f"{h.val_acc.iloc[-1]*100:.1f}%", color=INK2, ha="right")
ax.legend(loc="lower right"); save(fig, "fig01_training_curves")

# 2 accuracy vs severity
pcs = R["accuracy"]["per_corruption_severity"]; clean = R["accuracy"]["clean"] * 100
groups = [("Corruptions seen during calibration", ["gaussian_noise", "defocus_blur", "low_light", "fog"]),
          ("Held-out corruptions (never seen)", ["shot_noise", "motion_blur", "contrast", "jpeg"])]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (title, cs) in zip(axs, groups):
    ax.axhline(clean, color=MUTED, ls="--", lw=1.2); ax.text(1, clean + 1.5, f"clean {clean:.1f}%", color=MUTED, fontsize=9)
    for i, c in enumerate(cs):
        ys = np.array(pcs[c]) * 100
        ax.plot(range(1, 6), ys, marker="o", ms=6, color=S[i], label=c.replace("_", " "))
    ax.set_title(title, fontsize=12); ax.set_xlabel("Severity"); ax.set_xticks(range(1, 6)); ax.legend(fontsize=9)
axs[0].set_ylabel("Top-1 accuracy (%)"); axs[0].set_ylim(0, 100)
fig.suptitle("Accuracy collapses under real-world degradation", fontweight="bold", color=INK)
save(fig, "fig02_accuracy_vs_severity")

# 3 AUROC comparison
scen = [("Failure detection\n(all corrupted + clean)", R["failure_detection_all_ID"]),
        ("Failure detection\n(held-out corruptions)", R["failure_detection_heldout_corruptions"])]
for on, v in R["ood_detection"].items():
    scen.append((f"OOD detection\n({on.replace('_', ' ')})", v))
fig, ax = plt.subplots(figsize=(12, 4.6))
w = 0.13; x = np.arange(len(scen))
for i, m in enumerate(METHODS):
    vals = [s[1][m]["AUROC"] for s in scen]
    bs = ax.bar(x + (i - 2.5) * w, vals, w * 0.9, color=MCOL[m], label=m)
    if m == "EdgeGuard CARS":
        bar_labels(ax, bs, color=INK)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in scen]); ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC (higher is better)")
ax.set_title("Which score best separates reliable from unreliable predictions?")
ax.legend(ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17))
save(fig, "fig03_auroc_comparison")

# 4 ROC curves (all ID failure detection)
ID = D[~D.ood]
sc = {"MSP (softmax)": -ID.msp, "Energy": ID.energy, "Mahalanobis": ID.maha, METHODS[4]: ID.mc_entropy, "EdgeGuard CARS": -ID.cars}
fig, ax = plt.subplots(figsize=(5.6, 5))
for m, s in sc.items():
    fpr, tpr, _ = roc_curve(ID.fail, s)
    ax.plot(fpr, tpr, color=MCOL[m], lw=2.6 if "CARS" in m else 1.8,
            label=f"{m}  {R['failure_detection_all_ID'][m]['AUROC']:.3f}")
ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":")
ax.set_xlabel("False alarm rate (correct predictions flagged)"); ax.set_ylabel("Failures caught (recall)")
ax.set_title("ROC — detecting wrong predictions"); ax.legend(fontsize=8.5, loc="lower right", title="AUROC", title_fontsize=9)
save(fig, "fig04_roc_failure_detection")

# 5 risk coverage
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for m in ["MSP (softmax)", "Mahalanobis", METHODS[4], "EdgeGuard CARS"]:
    rc = R["risk_coverage"][m]
    ax.plot(np.array(rc["coverage"]) * 100, np.array(rc["risk"]) * 100, color=MCOL[m],
            lw=2.6 if "CARS" in m else 1.8, label=f"{m} (AURC {rc['AURC']*100:.1f})")
ax.set_xlabel("Coverage: % of inputs the system acts on"); ax.set_ylabel("Error rate on accepted inputs (%)")
ax.set_title("Risk–coverage: act on the safest inputs first"); ax.legend(fontsize=9, loc="upper left")
save(fig, "fig05_risk_coverage")

# 6 reliability diagram (corrupted)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, key, title in [(axs[0], "softmax", "Plain softmax confidence"), (axs[1], "edgeguard", "EdgeGuard calibrated P(correct)")]:
    b = R["calibration_ece"]["corrupted"][key]["bins"]
    conf = [x["conf"] for x in b]; acc = [x["acc"] for x in b]
    ax.bar(conf, acc, width=1 / 15 * 0.9, color=S[0] if key == "edgeguard" else MUTED, alpha=0.9)
    ax.plot([0, 1], [0, 1], color=CRIT, lw=1.2, ls="--")
    ax.set_title(f"{title}\nECE = {R['calibration_ece']['corrupted'][key]['ece']*100:.1f}%", fontsize=12)
    ax.set_xlabel("Predicted confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axs[0].set_ylabel("Actual accuracy")
fig.suptitle("Calibration on corrupted inputs (dashed = perfect)", fontweight="bold", color=INK)
save(fig, "fig06_reliability_diagram")

# 7 CARS distributions
fig, ax = plt.subplots(figsize=(7.4, 4.2))
grp = [("Correct predictions", D[D.correct], S[0]), ("Wrong predictions (ID)", D[(~D.correct) & (~D.ood)], S[1]),
       ("Out-of-distribution", D[D.ood], S[6])]
bins = np.linspace(0, 100, 41)
for nm, d, c in grp:
    ax.hist(d.cars, bins=bins, density=True, histtype="step", lw=2.2, color=c, label=f"{nm} (median {d.cars.median():.0f})")
ax.set_xlabel("CARS reliability score (0–100)"); ax.set_ylabel("Density")
ax.set_title("CARS separates trustworthy from untrustworthy predictions"); ax.legend(fontsize=9, loc="upper center")
save(fig, "fig07_cars_distribution")

# 8 decision breakdown by condition (adaptive)
cats = [("Clean", D.set == "clean")]
for fam, lab in [("noise", "Noise"), ("blur", "Blur"), ("low_light", "Low light"), ("fog", "Fog/contrast"), ("compression", "JPEG")]:
    cats.append((lab + " (s≥3)", (D.family == fam) & (D.severity >= 3)))
for on in R["ood_detection"]:
    cats.append((on.replace("_proxy", "*").replace("_", " "), D.set == on))
fig, ax = plt.subplots(figsize=(10.5, 4.4))
bottom = np.zeros(len(cats))
for di, (nm, c) in enumerate([("ACCEPT", GOOD), ("WARN", WARN), ("DEFER", CRIT)]):
    vals = np.array([(D[m].decision == di).mean() * 100 for _, m in cats])
    ax.bar(range(len(cats)), vals, 0.62, bottom=bottom, color=c, label=nm, edgecolor=SURF, linewidth=2)
    for i, v in enumerate(vals):
        if v > 7:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color=INK)
    bottom += vals
accs = [D[m].correct.mean() * 100 if not D[m].ood.all() else None for _, m in cats]
ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{n}\nacc {a_:.0f}%" if a_ is not None else f"{n}\nacc n/a" for (n, _), a_ in zip(cats, accs)], fontsize=9)
ax.set_ylabel("% of inputs"); ax.set_ylim(0, 100)
ax.set_title("EdgeGuard decisions track how hard the input really is")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "fig08_decision_breakdown")

# 9 fixed vs adaptive threshold
fa = R["fixed_vs_adaptive"]; fams = [f for f in ["normal", "noise", "blur", "low_light", "fog", "compression"] if f in fa]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(fams)); w = 0.36
for ax, key, yl in [(axs[0], "error_on_accepted", "Error on ACCEPTED inputs (%)"), (axs[1], "coverage", "Inputs accepted (%)")]:
    v1 = [fa[f]["fixed"][key] * 100 for f in fams]; v2 = [fa[f]["adaptive"][key] * 100 for f in fams]
    b1 = ax.bar(x - w / 2, v1, w * 0.92, color=MUTED, label="Fixed threshold")
    b2 = ax.bar(x + w / 2, v2, w * 0.92, color=S[0], label="Adaptive threshold")
    lab = {"blur": "blur\n(incl. unseen\nmotion)", "compression": "JPEG\n(unseen)", "fog": "fog/contrast"}
    ax.set_xticks(x); ax.set_xticklabels([lab.get(f, f.replace("_", " ")) for f in fams], fontsize=9); ax.set_ylabel(yl)
    if key == "error_on_accepted":
        ax.axhline(R["calibration"].get("risk_target", 0.03) * 100 if "risk_target" in R["calibration"] else 3, color=CRIT, ls="--", lw=1.2)
        ax.text(-0.45, 3.4, "target 3%", color=CRIT, ha="left", fontsize=9)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}", ha="center", fontsize=8, color=INK2)
axs[0].legend(fontsize=9)
axs[0].set_title("Safety: errors that slip through"); axs[1].set_title("Cost: how much work is still automated")
save(fig, "fig09_fixed_vs_adaptive")

# 10 explanation confusion
cm = pd.DataFrame(R["explanation"]["confusion"]).fillna(0)
order_t = [t for t in ["noise", "blur", "low_light", "fog", "ood"] if t in cm.index]
order_p = [t for t in ["noise", "blur", "low_light", "fog", "ood", "ambiguous", "memory"] if t in cm.columns]
cm = cm.reindex(index=order_t, columns=order_p).fillna(0)
cmn = cm.div(cm.sum(1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.4, 4.6))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
im = ax.imshow(cmn.values, cmap=cmap, vmin=0, vmax=100)
ax.grid(False)
for i in range(cmn.shape[0]):
    for j in range(cmn.shape[1]):
        v = cmn.values[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10, color="white" if v > 55 else INK)
ax.set_xticks(range(len(order_p))); ax.set_xticklabels([t.replace("_", " ") for t in order_p], rotation=20)
ax.set_yticks(range(len(order_t))); ax.set_yticklabels([t.replace("_", " ") for t in order_t])
ax.set_xlabel("Reason EdgeGuard gave"); ax.set_ylabel("True cause")
ax.set_title(f"Explainable failure reason — accuracy {R['explanation']['accuracy']*100:.1f}%")
fig.colorbar(im, ax=ax, fraction=0.04, label="% of flagged inputs")
save(fig, "fig10_explanation_confusion")

# 11 ablation
ab = R["ablation"]; names = list(ab.keys())
fig, ax = plt.subplots(figsize=(10, 4.2))
mets = [("AUROC_unreliable_all", "Unreliable (all)"), ("AUROC_failure_heldout", "Failure (held-out shifts)"), ("AUROC_ood", "OOD (avg)")]
w = 0.2; x = np.arange(len(mets))
for i, n in enumerate(names):
    bs = ax.bar(x + (i - 1.5) * w, [ab[n][m] for m, _ in mets], w * 0.9, color=[MUTED, S[2], S[0], S[3]][i], label=n)
    bar_labels(ax, bs, fmt="{:.3f}")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets]); ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUROC")
ax.set_title("Ablation: each signal family adds information"); ax.legend(fontsize=9, ncol=2, loc="lower right")
save(fig, "fig11_ablation")

# 12 latency
E = R["efficiency"]
items = [("Base model\n(no reliability)", E["latency_base_ms"], MUTED), ("EdgeGuard\n(1 pass)", E["latency_edgeguard_ms"], S[0]),
         ("MC-Dropout\n(10 passes)", E["latency_mc_dropout10_ms"], S[6]), ("Deep Ensemble\n(5 models)", E["latency_deep_ensemble5_ms"], S[3])]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bs = ax.bar(range(4), [v for _, v, _ in items], 0.6, color=[c for *_, c in items])
for b, (_, v, _) in zip(bs, items):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05 * max(i[1] for i in items) / 4, f"{v:.2f} ms", ha="center", fontsize=10, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([n for n, *_ in items]); ax.set_ylabel("Latency per image (ms)")
ax.set_title(f"Edge latency ({E['device']}) — EdgeGuard overhead {E['overhead_pct']:.1f}%")
save(fig, "fig12_latency")

# 13 blind spot memory
st = R["blind_spot_memory"]["streams"]
fig, axs = plt.subplots(1, len(st), figsize=(11, 4), sharey=True)
for ax, (shift, res) in zip(np.atleast_1d(axs), st.items()):
    for key, c, lab in [("no_memory", MUTED, "Without memory"), ("with_memory", S[0], "With blind-spot memory")]:
        cv = res[key]["curve"]
        ax.plot([q["window"] for q in cv], [q["missed_failure_rate"] * 100 for q in cv], marker="o", ms=6, color=c,
                label=f"{lab} (overall {res[key]['missed_failure_rate']*100:.1f}%)")
    ax.set_title(f"Unseen shift: {shift.replace('_', ' ')}", fontsize=12); ax.set_xlabel("Stream window (300 inputs each)")
    ax.legend(fontsize=9)
np.atleast_1d(axs)[0].set_ylabel("Failures silently accepted (%)")
rel = [(1 - r["with_memory"]["missed_failure_rate"] / r["no_memory"]["missed_failure_rate"]) * 100 for r in st.values()]
fa = [(r["with_memory"]["false_alarm"] - r["no_memory"]["false_alarm"]) * 100 for r in st.values()]
np.atleast_1d(axs)[0].set_ylim(bottom=0)
fig.suptitle(f"Blind-spot memory: {min(rel):.0f}–{max(rel):.0f}% fewer silent misses on unseen shifts "
             f"(cost: +{min(fa):.0f}–{max(fa):.0f} pts false alarms)", fontweight="bold", color=INK)
save(fig, "fig13_blind_spot_memory")
print("all figures done")
