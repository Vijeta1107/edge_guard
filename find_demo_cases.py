"""Finds test images that make each demo scene work reliably (so nothing is improvised
in front of the panel). Writes results/demo_cases.json, which app.py loads as 'Demo scenes'.
  python find_demo_cases.py --arch edgenet
"""
import argparse, json
import numpy as np
import torch
from scipy.special import softmax
from edgeguard.data import load_cifar10, apply_custom, load_ood, CLASSES
from edgeguard.model import build_model, to_tensor
from edgeguard.core import EdgeGuard, BlindSpotMemory, signal_matrix, DECISIONS, REASON_TEXT

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="edgenet"); A = ap.parse_args()
m = build_model(A.arch); m.load_state_dict(torch.load(f"checkpoints/{A.arch}.pt", map_location="cpu")["state"]); m.eval()
eg = EdgeGuard.load(f"checkpoints/edgeguard_{A.arch}.pkl")
_, _, xte, yte = load_cifar10()


@torch.no_grad()
def run(img, mem=None):
    lg, ft = m(to_tensor(img[None])); lg, ft = lg.numpy(), ft.numpy()
    s = signal_matrix(lg, ft, img[None], eg.maha)
    eg.memory = mem
    o = eg.decide(s, ft, lg.argmax(1), use_memory=mem is not None)
    p = softmax(lg, 1)[0]
    return dict(pred=int(p.argmax()), conf=float(p.max()), cars=float(o["cars"][0]), dec=DECISIONS[o["decision"][0]],
                reason=o["reason"][0], mem=bool(o["mem_hit"][0])), ft


SCENES = {"night": dict(dark=0.85), "fog": dict(fog_amt=0.85), "blur": dict(blur=0.8), "noise": dict(noise=0.7)}
REASON = {"night": "low_light", "fog": "fog", "blur": "blur", "noise": "noise"}
cases = {"silent_failure": {}, "fog_ramp": None, "ood": {}, "memory": None}
for i in range(2000):
    c, _ = run(xte[i])
    if not (c["pred"] == yte[i] and c["dec"] == "ACCEPT" and c["conf"] > 0.95):
        continue
    for sc, kw in SCENES.items():
        if sc in cases["silent_failure"]:
            continue
        r, _ = run(apply_custom(xte[i], seed=i, **{k: kw.get(k, 0) for k in ["blur", "noise", "dark", "fog_amt"]}))
        if r["pred"] != yte[i] and r["conf"] > 0.8 and r["dec"] == "DEFER" and r["reason"] == REASON[sc]:
            cases["silent_failure"][sc] = dict(idx=i, truth=CLASSES[yte[i]], clean=dict(c, pred=CLASSES[c["pred"]]),
                                               degraded=dict(r, pred=CLASSES[r["pred"]]), sliders=kw)
    if cases["fog_ramp"] is None:
        ramp = []
        for f in np.linspace(0, 1, 11):
            r, _ = run(apply_custom(xte[i], fog_amt=f, seed=i)); ramp.append((round(float(f), 1), r["dec"], round(r["cars"], 1), r["pred"] == yte[i]))
        decs = [d for _, d, _, _ in ramp]
        if decs[0] == "ACCEPT" and "WARN" in decs and decs[-1] == "DEFER":
            cases["fog_ramp"] = dict(idx=i, truth=CLASSES[yte[i]], ramp=ramp)
    if len(cases["silent_failure"]) == 4 and cases["fog_ramp"]:
        break

for kind, on, seed in [("Digits (SVHN-like)", "digits_proxy", 7), ("Textures", "textures_proxy", 7)]:
    ims = load_ood(on, 64, seed=seed)
    for j in range(64):
        r, _ = run(ims[j])
        if r["conf"] > 0.75 and r["dec"] == "DEFER" and r["reason"] == "ood":
            cases["ood"][kind] = dict(sample=j, plain=f"{CLASSES[r['pred']]} {r['conf']*100:.0f}%", cars=r["cars"], reason=r["reason"]); break

# blind-spot memory: a failure EdgeGuard ACCEPTS; operator marks it; the same scene seen again is caught
for i in range(2000, 6000):
    for b in [0.35, 0.45, 0.55]:
        r, ft = run(apply_custom(xte[i], blur=b, seed=i))
        if r["pred"] != yte[i] and r["dec"] == "ACCEPT":
            mem = BlindSpotMemory(ft.shape[1]); mem.add(ft, np.array([r["pred"]]))
            r2, _ = run(apply_custom(xte[i], blur=b + 0.05, seed=i + 1), mem)
            ok_other = []
            for k in range(6000, 6200):     # memory must not fire on unrelated correct images
                rk, _ = run(xte[k], mem); ok_other.append(rk["mem"])
            if r2["mem"] and r2["dec"] != "ACCEPT" and np.mean(ok_other) < 0.02:
                cases["memory"] = dict(idx=i, truth=CLASSES[yte[i]], blur=b, plain=f"{CLASSES[r['pred']]} {r['conf']*100:.0f}%",
                                       first=r["dec"], second=r2["dec"], unrelated_false_hits=float(np.mean(ok_other)))
                break
    if cases["memory"]:
        break
eg.memory = None
json.dump(cases, open("results/demo_cases.json", "w"), indent=1, default=str)
print(json.dumps(cases, indent=1, default=str))
