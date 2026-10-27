"""Illustrative examples figure (fig00): plain model vs EdgeGuard on the same inputs.
Examples are selected automatically by criterion (stated in the figure), not hand-picked pixels.
  python examples.py --arch edgenet
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax
from edgeguard.data import load_cifar10, corrupt, load_ood, CLASSES
from edgeguard.model import build_model, infer
from edgeguard.core import EdgeGuard, signal_matrix, REASON_TEXT, DECISIONS

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="edgenet"); A = ap.parse_args()
ck = torch.load(f"checkpoints/{A.arch}.pt", map_location="cpu")
m = build_model(A.arch); m.load_state_dict(ck["state"]); m.eval()
eg = EdgeGuard.load(f"checkpoints/edgeguard_{A.arch}.pkl")
_, _, xte, yte = load_cifar10()
idx = np.arange(3000, 3600)


def analyse(imgs):
    lg, ft = infer(m, imgs)
    s = signal_matrix(lg, ft, imgs, eg.maha)
    o = eg.decide(s, use_memory=False)
    return softmax(lg, 1), o


panels = []
p, o = analyse(xte[idx])
ok = np.where((p.argmax(1) == yte[idx]) & (o["decision"] == 0))[0][0]
panels.append(("Clean input", xte[idx][ok], p[ok], yte[idx][ok], o, ok))
for name, sev in [("low_light", 5), ("motion_blur", 5), ("fog", 5), ("gaussian_noise", 5)]:
    imgs = corrupt(xte[idx], name, sev, seed=5)
    p, o = analyse(imgs)
    wrong_conf = np.where((p.argmax(1) != yte[idx]) & (p.max(1) > 0.8) & (o["decision"] > 0))[0]
    j = wrong_conf[0] if len(wrong_conf) else np.where(p.argmax(1) != yte[idx])[0][0]
    panels.append((name.replace("_", " ") + f" (s{sev})", imgs[j], p[j], yte[idx][j], o, j))
for on in ["digits_proxy", "textures_proxy"]:
    imgs = load_ood(on, 300, seed=11)
    p, o = analyse(imgs)
    cand = np.where((p.max(1) > 0.8) & (o["decision"] > 0))[0]
    j = cand[0] if len(cand) else int(np.argmax(p.max(1)))
    panels.append(("unknown object: " + on.split("_")[0], imgs[j], p[j], -1, o, j))

col = {"ACCEPT": "#0ca30c", "WARN": "#fab219", "DEFER": "#d03b3b"}
fig, axs = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 4.1), facecolor="#fcfcfb")
for ax, (title, img, pr, y, o, j) in zip(axs, panels):
    ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    d = DECISIONS[o["decision"][j]]
    for sp in ax.spines.values():
        sp.set_edgecolor(col[d]); sp.set_linewidth(4)
    truth = CLASSES[y] if y >= 0 else "not a CIFAR class"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(f"truth: {truth}\nplain model: {CLASSES[pr.argmax()]} {pr.max()*100:.0f}%\n"
                  f"EdgeGuard: {d}  (CARS {o['cars'][j]:.0f})\n{REASON_TEXT[o['reason'][j]] if d!='ACCEPT' else 'reliable'}",
                  fontsize=8.5)
fig.suptitle("Same model, same single forward pass — EdgeGuard catches confident mistakes", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/fig00_examples.png", dpi=200); print("saved fig00")
