"""Step 2 — fit EdgeGuard on the calibration set and run the full evaluation.

  python evaluate.py --arch edgenet                 # offline proxies for OOD
  python evaluate.py --arch resnet18 --real_ood     # Colab: CIFAR-100 + SVHN

Outputs: results/results_<arch>.json, results/samples_<arch>.csv,
         checkpoints/edgeguard_<arch>.pkl
"""
import argparse, json, time, os
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from edgeguard.data import (load_cifar10, load_ood, corrupt, CORRUPTIONS, FAMILY, CONDITIONS,
                            FIT_CORRUPTIONS, HELDOUT_CORRUPTIONS)
from edgeguard.model import build_model, infer, to_tensor
from edgeguard.core import (EdgeGuard, BlindSpotMemory, signal_matrix, MODEL_KEYS, SHIFT_KEYS,
                            ALL_KEYS, REASON_TEXT, DECISIONS)
from edgeguard.signals import QUALITY_KEYS, quality_signals

p = argparse.ArgumentParser()
p.add_argument("--arch", default="edgenet")
p.add_argument("--real_ood", action="store_true")
p.add_argument("--n_corrupt", type=int, default=2000, help="test images per corruption x severity")
p.add_argument("--quick", action="store_true")
p.add_argument("--risk_target", type=float, default=0.03)
a = p.parse_args()
os.makedirs("results", exist_ok=True)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
R = {}

# ------------------------------------------------------------------ setup
xtr, ytr, xte, yte = load_cifar10()
val_idx = np.load("data/val_idx.npy")
tr_mask = np.ones(len(xtr), bool); tr_mask[val_idx] = False
ck = torch.load(f"checkpoints/{a.arch}.pt", map_location=dev)
model = build_model(a.arch).to(dev); model.load_state_dict(ck["state"]); model.eval()

print("features on train set ...", flush=True)
tr_ids = np.random.default_rng(0).permutation(np.where(tr_mask)[0])[:5000] if a.quick else np.where(tr_mask)[0]
_, ftr = infer(model, xtr[tr_ids], device=dev)
eg = EdgeGuard(risk_target=a.risk_target)
eg.fit_maha(ftr, ytr[tr_ids])

# ---------------------------------------------- calibration (validation) set
rng = np.random.default_rng(1)
if a.quick:
    val_idx = val_idx[:1000]
xv, yv = xtr[val_idx], ytr[val_idx]
V_imgs, V_y, V_cond, V_sev = [xv], [yv], [np.zeros(len(xv), int)], [np.zeros(len(xv), int)]
for ci, cname in enumerate(FIT_CORRUPTIONS):
    sev = rng.integers(1, 6, len(xv))
    imgs = np.stack([corrupt(xv[i:i + 1], cname, int(sev[i]), seed=10 * ci + i)[0] for i in range(len(xv))])
    V_imgs.append(imgs); V_y.append(yv); V_sev.append(sev)
    V_cond.append(np.full(len(xv), CONDITIONS.index(FAMILY[cname])))
V_imgs = np.concatenate(V_imgs); V_y = np.concatenate(V_y)
V_cond = np.concatenate(V_cond); V_sev = np.concatenate(V_sev)
lv, fv = infer(model, V_imgs, device=dev)
sv = signal_matrix(lv, fv, V_imgs, eg.maha)
fail_v = (lv.argmax(1) != V_y).astype(int)
cond_fit = np.where(V_sev == 1, -1, V_cond)       # severity-1 is visually ~clean: not used for condition clf
eg.fit(sv, fail_v, cond_fit)
R["calibration"] = dict(n=int(len(V_y)), risk_target=a.risk_target, fail_rate=float(fail_v.mean()), tau_fixed=eg.tau_fixed,
                        tau_by_condition=eg.tau_cond, maha_tau=eg.maha_tau,
                        fusion_weights=dict(zip(eg.keys, map(float, eg.fusion.coef_[0]))))
print("tau fixed", eg.tau_fixed, "adaptive", eg.tau_cond, flush=True)

# ------------------------------------------------------------- test sets
sets = []
nc = 2000 if a.quick else len(xte)
sets.append(dict(set="clean", family="normal", severity=0, imgs=xte[:nc], y=yte[:nc], ood=False))
cidx = np.random.default_rng(2).permutation(len(xte))[:a.n_corrupt]
for cname in CORRUPTIONS:
    for s in range(1, 6):
        sets.append(dict(set=cname, family=FAMILY[cname], severity=s,
                         imgs=corrupt(xte[cidx], cname, s, seed=100 + s), y=yte[cidx], ood=False))
ood_names = ["cifar100", "svhn"] if a.real_ood else ["digits_proxy", "textures_proxy"]
for on in ood_names:
    oi = load_ood(on, n=2000)
    sets.append(dict(set=on, family="ood", severity=0, imgs=oi, y=np.full(len(oi), -1), ood=True))

rows = []
mc_T = 20
for S in sets:
    lg, ft = infer(model, S["imgs"], device=dev)
    s = signal_matrix(lg, ft, S["imgs"], eg.maha)
    out = eg.decide(s, adaptive=True, use_memory=False)
    outf = eg.decide(s, adaptive=False, use_memory=False)
    # MC-Dropout baseline (T stochastic passes through the dropout head)
    with torch.no_grad():
        f_t = torch.from_numpy(ft).to(dev)
        model.drop.train()
        probs = torch.stack([torch.softmax(model.fc(model.drop(f_t)), 1) for _ in range(mc_T)]).mean(0)
        model.drop.eval()
        probs = probs.cpu().numpy()
    mc_ent = -(probs * np.log(probs + 1e-12)).sum(1)
    pred = lg.argmax(1)
    df = pd.DataFrame(dict(set=S["set"], family=S["family"], severity=S["severity"], ood=S["ood"],
                           y=S["y"], pred=pred, correct=(pred == S["y"]) & (not S["ood"]),
                           cars=out["cars"], decision=out["decision"], decision_fixed=outf["decision"],
                           tau=out["tau"], reason=out["reason"], detected=[CONDITIONS[d] for d in eg.condition(s)[0]],
                           mc_entropy=mc_ent, p_fail=eg.p_fail(s), ood_flag=out["ood_flag"]))
    for k in ALL_KEYS:
        df[k] = s[k]
    rows.append(df)
    print(f"{S['set']:16s} s{S['severity']} acc {(df.correct.mean() if not S['ood'] else float('nan')):.3f} "
          f"accept {(df.decision==0).mean():.2f} defer {(df.decision==2).mean():.2f}", flush=True)
D = pd.concat(rows, ignore_index=True)
D["correct"] = D.correct.astype(bool)
D["fail"] = (~D.correct).astype(int)
D.to_csv(f"results/samples_{a.arch}.csv", index=False)

# ------------------------------------------------------------- metrics
def fpr_at_95tpr(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, 0.95)])


SCORES = {"MSP (softmax)": lambda d: -d.msp, "Entropy": lambda d: d.entropy, "Energy": lambda d: d.energy,
          "Mahalanobis": lambda d: d.maha, f"MC-Dropout (T={mc_T})": lambda d: d.mc_entropy,
          "EdgeGuard CARS": lambda d: -d.cars}


def score_table(d, target):
    out = {}
    for nm, fn in SCORES.items():
        sc = fn(d).values
        out[nm] = dict(AUROC=float(roc_auc_score(target, sc)), AUPR=float(average_precision_score(target, sc)),
                       FPR95=fpr_at_95tpr(target, sc))
    return out


ID = D[~D.ood]
clean = D[D.set == "clean"]
R["accuracy"] = dict(clean=float(clean.correct.mean()),
                     per_corruption={c: float(ID[ID.set == c].correct.mean()) for c in CORRUPTIONS},
                     per_severity={int(s): float(ID[ID.severity == s].correct.mean()) for s in range(0, 6)},
                     per_corruption_severity={c: [float(ID[(ID.set == c) & (ID.severity == s)].correct.mean())
                                                  for s in range(1, 6)] for c in CORRUPTIONS})
# accuracy preserved: EdgeGuard never changes the prediction
R["accuracy"]["note"] = "EdgeGuard does not modify the base model's predictions, so accuracy is preserved by construction."

R["failure_detection_all_ID"] = score_table(ID, ID.fail.values)
seen = ID[ID.set.isin(["clean"] + FIT_CORRUPTIONS)]
held = ID[ID.set.isin(["clean"] + HELDOUT_CORRUPTIONS)]
R["failure_detection_seen_corruptions"] = score_table(seen, seen.fail.values)
R["failure_detection_heldout_corruptions"] = score_table(held, held.fail.values)
R["ood_detection"] = {}
for on in ood_names:
    d = pd.concat([clean, D[D.set == on]])
    R["ood_detection"][on] = score_table(d, d.ood.astype(int).values)

# decisions: precision / recall / F1 of "flag" (WARN or DEFER) for unreliable predictions
def flag_metrics(d, col="decision"):
    flag = (d[col] > 0).values; f = d.fail.values.astype(bool)
    tp, fp, fn = (flag & f).sum(), (flag & ~f).sum(), (~flag & f).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    acc_kept = 1 - f[~flag].mean() if (~flag).sum() else float("nan")
    return dict(precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / max(prec + rec, 1e-9)),
                false_alarm_rate=float((flag & ~f).sum() / max((~f).sum(), 1)),
                defer_false_alarm_rate=float(((d[col] == 2).values & ~f).sum() / max((~f).sum(), 1)),
                coverage=float((~flag).mean()), error_on_accepted=float(1 - acc_kept),
                accept=float((d[col] == 0).mean()), warn=float((d[col] == 1).mean()), defer=float((d[col] == 2).mean()))


R["decision_metrics"] = {"overall_adaptive": flag_metrics(D), "overall_fixed": flag_metrics(D, "decision_fixed"),
                         "clean_adaptive": flag_metrics(clean), "clean_fixed": flag_metrics(clean, "decision_fixed")}
R["plain_model_silent_failures"] = dict(
    note="fraction of wrong predictions the plain model makes with softmax confidence >= 0.9",
    all=float(((D.fail == 1) & (D.msp >= 0.9)).sum() / max((D.fail == 1).sum(), 1)),
    edgeguard_accepted_wrong=float(((D.fail == 1) & (D.decision == 0)).sum() / max((D.fail == 1).sum(), 1)))

# fixed vs adaptive threshold, per true condition
fa = {}
for fam in ["normal", "noise", "blur", "low_light", "fog", "compression", "ood"]:
    d = D[D.family == fam]
    if fam == "normal":
        d = clean
    else:
        d = d[d.severity.isin([0, 3, 4, 5])]
    row = {}
    for col, nm in [("decision_fixed", "fixed"), ("decision", "adaptive")]:
        acc = d[d[col] == 0]
        row[nm] = dict(coverage=float((d[col] == 0).mean()),
                       error_on_accepted=float(acc.fail.mean()) if len(acc) else float("nan"))
    row["mean_tau_adaptive"] = float(d.tau.mean())
    fa[fam] = row
R["fixed_vs_adaptive"] = fa

# risk-coverage (selective prediction) on all ID
def risk_coverage(score_reliable, fail):
    o = np.argsort(-score_reliable); f = fail[o]
    n = np.arange(1, len(f) + 1)
    risk = np.cumsum(f) / n
    return n / len(f), risk, float(risk.mean())


rc = {}
for nm, fn in SCORES.items():
    cov, risk, aurc = risk_coverage(-fn(ID).values, ID.fail.values)
    sel = np.linspace(0, len(cov) - 1, 200).astype(int)
    rc[nm] = dict(AURC=aurc, coverage=cov[sel].tolist(), risk=risk[sel].tolist())
R["risk_coverage"] = rc

# calibration (ECE): raw softmax confidence vs EdgeGuard (CARS/100) as P(correct)
def ece(conf, correct, nb=15):
    b = np.clip((conf * nb).astype(int), 0, nb - 1); e = 0; bins = []
    for i in range(nb):
        m = b == i
        if m.sum():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
            bins.append(dict(bin=i, conf=float(conf[m].mean()), acc=float(correct[m].mean()), n=int(m.sum())))
    return float(e), bins


R["calibration_ece"] = {}
for nm, d in [("clean", clean), ("corrupted", ID[ID.set != "clean"])]:
    e1, b1 = ece(d.msp.values, d.correct.values.astype(float))
    e2, b2 = ece(1 - d.p_fail.values, d.correct.values.astype(float))   # calibrated P(correct) before the OOD gate
    R["calibration_ece"][nm] = {"softmax": dict(ece=e1, bins=b1), "edgeguard": dict(ece=e2, bins=b2)}

# explanation accuracy (flagged, severity>=3 corruptions in known families, and OOD)
ex = D[(D.decision > 0) & (((D.family.isin(["noise", "blur", "low_light", "fog"])) & (D.severity >= 3)) | D.ood)].copy()
ex["true_reason"] = np.where(ex.ood, "ood", ex.family)
cm = pd.crosstab(ex.true_reason, ex.reason)
R["explanation"] = dict(accuracy=float((ex.true_reason == ex.reason).mean()), n=int(len(ex)),
                        confusion=cm.to_dict(),
                        per_class={t: float((ex[ex.true_reason == t].reason == t).mean()) for t in ex.true_reason.unique()},
                        ood_mentioned_rate=float(ex[ex.true_reason == "ood"].ood_flag.mean()) if (ex.true_reason == "ood").any() else None)

# ------------------------------------------------------- ablation (signals)
abl = {}
fit_imgs_s = sv
for nm, keys in [("Model uncertainty only", MODEL_KEYS), ("+ Context shift (Mahalanobis)", MODEL_KEYS + SHIFT_KEYS),
                 ("+ Input quality = full CARS", ALL_KEYS), ("Input quality only", QUALITY_KEYS)]:
    g = EdgeGuard(risk_target=a.risk_target); g.maha = eg.maha
    g.fit(sv, fail_v, cond_fit, keys=keys)
    sc = -g.cars({k: D[k].values for k in ALL_KEYS})
    abl[nm] = dict(
        AUROC_unreliable_all=float(roc_auc_score(D.fail, sc)),
        AUROC_failure_heldout=float(roc_auc_score(D.fail[D.set.isin(["clean"] + HELDOUT_CORRUPTIONS)],
                                                  sc[D.set.isin(["clean"] + HELDOUT_CORRUPTIONS).values])),
        AUROC_ood=float(np.mean([roc_auc_score(D.ood[D.set.isin(["clean", on])],
                                               sc[D.set.isin(["clean", on]).values]) for on in ood_names])))
R["ablation"] = abl

# ------------------------------------------------ blind-spot memory stream
# calibrate memory threshold on validation: false-hit rate on correct samples ~5%
mem_cal = BlindSpotMemory(fv.shape[1], capacity=512)
pv = lv.argmax(1)
fidx = np.where(fail_v == 1)[0]; cidx_v = np.where(fail_v == 0)[0]
pick = np.random.default_rng(3).choice(fidx, min(512, len(fidx)), replace=False)
mem_cal.add(fv[pick], pv[pick])
sim_ok = mem_cal.similarity(fv[cidx_v], pv[cidx_v])
eg.mem_tau = float(np.percentile(sim_ok, 95))

stream_res = {}
for shift in ["motion_blur", "jpeg"]:      # held-out shifts never seen by the fusion
    r3 = np.random.default_rng(4)
    NS = 600 if a.quick else 3000
    sidx = r3.permutation(len(xte))[:NS]
    half = np.zeros(NS, bool); half[NS // 2:] = True
    imgs = xte[sidx].copy()
    sev = r3.integers(3, 6, NS)
    imgs[half] = np.stack([corrupt(imgs[i:i + 1], shift, int(sev[i]), seed=i)[0] for i in np.where(half)[0]])
    order = r3.permutation(NS); imgs, ys = imgs[order], yte[sidx][order]
    lg, ft = infer(model, imgs, device=dev)
    s = signal_matrix(lg, ft, imgs, eg.maha); pred = lg.argmax(1); fail = pred != ys
    res = {}
    for use_mem in [False, True]:
        eg.memory = BlindSpotMemory(ft.shape[1], capacity=512) if use_mem else None
        dec = np.zeros(NS, int)
        for i in range(NS):
            si = {k: v[i:i + 1] for k, v in s.items()}
            o = eg.decide(si, ft[i:i + 1], pred[i:i + 1], adaptive=True, use_memory=use_mem)
            dec[i] = o["decision"][0]
            # operator feedback: flagged failures get confirmed; 30% of silent misses are found by audit
            if use_mem and fail[i] and (dec[i] > 0 or r3.random() < 0.3):
                eg.memory.add(ft[i:i + 1], pred[i:i + 1])
        win = 300; curve = []
        for w in range(0, NS, win):
            m = slice(w, w + win)
            f, d_ = fail[m], dec[m]
            curve.append(dict(window=w // win + 1, missed_failure_rate=float(((d_ == 0) & f).sum() / max(f.sum(), 1)),
                              false_alarm=float(((d_ > 0) & ~f).sum() / max((~f).sum(), 1))))
        res["with_memory" if use_mem else "no_memory"] = dict(
            missed_failure_rate=float(((dec == 0) & fail).sum() / max(fail.sum(), 1)),
            false_alarm=float(((dec > 0) & ~fail).sum() / max((~fail).sum(), 1)), curve=curve,
            memory_kb=float(eg.memory.size_kb()) if use_mem else 0.0,
            memory_entries=int(len(eg.memory.bank)) if use_mem else 0)
    stream_res[shift] = res
eg.memory = None
R["blind_spot_memory"] = dict(mem_tau=eg.mem_tau, streams=stream_res)

# --------------------------------------------------------------- efficiency
from edgeguard.bench import run_benchmark
R["efficiency"] = run_benchmark(model.to("cpu"), eg, xte[:64], a.arch)
print(json.dumps(R["efficiency"], indent=1))

R["headline"] = dict(plain_error_all_inputs=float(D.fail.mean()), plain_error_ID=float(ID.fail.mean()),
                     edgeguard_error_on_accepted=R["decision_metrics"]["overall_adaptive"]["error_on_accepted"],
                     failures_flagged=R["decision_metrics"]["overall_adaptive"]["recall"], n_test=int(len(D)))
json.dump(R, open(f"results/results_{a.arch}.json", "w"), indent=1, default=float)
eg.save(f"checkpoints/edgeguard_{a.arch}.pkl")
print("done")
