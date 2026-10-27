"""Step 4 — live demo.   python app.py   ->  open http://127.0.0.1:7860

Tabs
  1. Live check     : pick/upload an image, degrade it with sliders, compare the
                      plain model vs EdgeGuard (CARS, decision, reason).
  2. Unknown objects: OOD inputs the model was never trained on.
  3. Blind-spot memory: mark a failure, watch similar inputs get caught.
  4. Results        : all benchmark figures.
"""
import argparse, glob, json, os
import numpy as np
import cv2
import torch
import gradio as gr
from scipy.special import softmax

from edgeguard.data import load_cifar10, apply_custom, load_ood, corrupt, CLASSES, CONDITIONS
from edgeguard.model import build_model, to_tensor
from edgeguard.core import EdgeGuard, BlindSpotMemory, signal_matrix, REASON_TEXT, DECISIONS

ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="edgenet"); ap.add_argument("--share", action="store_true")
A = ap.parse_args()
ck = torch.load(f"checkpoints/{A.arch}.pt", map_location="cpu")
MODEL = build_model(A.arch); MODEL.load_state_dict(ck["state"]); MODEL.eval()
EG = EdgeGuard.load(f"checkpoints/edgeguard_{A.arch}.pkl")
EG.memory = BlindSpotMemory(MODEL.feat_dim, capacity=512)
_, _, XTE, YTE = load_cifar10()
OOD = {"Digits (SVHN-like)": load_ood("digits_proxy", 64, seed=7), "Textures": load_ood("textures_proxy", 64, seed=7)}
DEC_COL = {"ACCEPT": "#0ca30c", "WARN": "#fab219", "DEFER": "#d03b3b"}
DEC_ICON = {"ACCEPT": "✔", "WARN": "⚠", "DEFER": "⛔"}
DEC_ACTION = {"ACCEPT": "Act on the prediction", "WARN": "Act cautiously / request more sensing",
              "DEFER": "Do not act — hand over to backup model or human"}
STATE = {}
try:
    CASES = json.load(open("results/demo_cases.json"))
except Exception:
    CASES = {}


def _scenes():
    sc = {}
    sf = CASES.get("silent_failure", {})
    first = next(iter(sf.values()), {"idx": 7})
    sc["A · Clean baseline"] = (first["idx"], 0, 0, 0, 0)
    if "night" in sf:
        sc["B · Night drive: confident mistake caught"] = (sf["night"]["idx"], 0, 0, 0.85, 0)
    if "blur" in sf:
        sc["C · Motion blur: confident mistake caught"] = (sf["blur"]["idx"], 0.8, 0, 0, 0)
    if "noise" in sf:
        sc["C2 · Sensor noise: confident mistake caught"] = (sf["noise"]["idx"], 0, 0.7, 0, 0)
    if CASES.get("fog_ramp"):
        sc["D · Fog ramp: drag the Fog slider 0 → 1"] = (CASES["fog_ramp"]["idx"], 0, 0, 0, 0)
    if CASES.get("memory"):
        m = CASES["memory"]
        sc["E · Blind-spot memory (step 1: mark wrong)"] = (m["idx"], m["blur"], 0, 0, 0)
        sc["E · Blind-spot memory (step 2: same scene again)"] = (m["idx"], m["blur"] + 0.05, 0, 0, 0)
    return sc


SCENES = _scenes()


def to32(img):
    if img is None:
        return None
    img = np.asarray(img)[..., :3]
    h, w = img.shape[:2]
    if (h, w) != (32, 32):
        s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
        img = cv2.resize(img[y0:y0 + s, x0:x0 + s], (32, 32), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def big(img):
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def run(img32, adaptive=True):
    lg, ft = MODEL(to_tensor(img32[None]))
    lg, ft = lg.numpy(), ft.numpy()
    s = signal_matrix(lg, ft, img32[None], EG.maha)
    pred = lg.argmax(1)
    o = EG.decide(s, ft, pred, adaptive=adaptive, use_memory=True)
    contrib = EG.contributions(s)
    det, pr = EG.condition(s)
    STATE["last"] = (ft, pred)
    return dict(probs=softmax(lg, 1)[0], pred=int(pred[0]), cars=float(o["cars"][0]), tau=float(o["tau"][0]),
                decision=DECISIONS[o["decision"][0]], reason=REASON_TEXT[o["reason"][0]],
                detected=CONDITIONS[det[0]], det_p=float(pr[0, det[0]]), mem=bool(o["mem_hit"][0]),
                ood=bool(o["ood_flag"][0]) and o["reason"][0] != "ood",
                contrib={k: float(v[0]) + (3.0 * float(EG.ood_gate(s)[0]) if k == "Context shift" else 0.0)
                         for k, v in contrib.items()})


def plain_html(r):
    top = np.argsort(-r["probs"])[:3]
    rows = "".join(f"<div style='display:flex;gap:8px;align-items:center;margin:4px 0'><span style='width:90px'>{CLASSES[i]}</span>"
                   f"<div style='flex:1;background:#e1e0d9;border-radius:4px;height:14px'><div style='width:{r['probs'][i]*100:.0f}%;"
                   f"background:#898781;height:14px;border-radius:4px'></div></div><span style='width:50px;text-align:right'>"
                   f"{r['probs'][i]*100:.1f}%</span></div>" for i in top)
    return (f"<div style='font-family:system-ui;padding:12px;border:1px solid #e1e0d9;border-radius:10px'>"
            f"<div style='color:#52514e;font-size:13px'>PLAIN MODEL (no reliability check)</div>"
            f"<div style='font-size:26px;font-weight:700;margin:6px 0'>{CLASSES[r['pred']]} — {r['probs'][r['pred']]*100:.1f}% confident</div>"
            f"{rows}<div style='color:#52514e;font-size:13px;margin-top:6px'>Always acts on its top prediction.</div></div>")


def eg_html(r):
    d = r["decision"]; c = DEC_COL[d]
    tot = sum(max(v, 0) for v in r["contrib"].values()) or 1
    bars = "".join(f"<div style='display:flex;gap:8px;align-items:center;margin:3px 0'><span style='width:140px'>{k}</span>"
                   f"<div style='flex:1;background:#e1e0d9;border-radius:4px;height:12px'><div style='width:{max(v,0)/tot*100:.0f}%;"
                   f"background:#2a78d6;height:12px;border-radius:4px'></div></div><span style='width:44px;text-align:right'>"
                   f"{max(v,0)/tot*100:.0f}%</span></div>" for k, v in r["contrib"].items())
    mem = "<div style='color:#d03b3b;font-weight:600'>Blind-spot memory hit: similar to a recorded failure</div>" if r["mem"] else ""
    return (f"<div style='font-family:system-ui;padding:12px;border:2px solid {c};border-radius:10px'>"
            f"<div style='color:#52514e;font-size:13px'>EDGEGUARD</div>"
            f"<div style='display:flex;align-items:baseline;gap:14px'><span style='font-size:44px;font-weight:800'>{r['cars']:.0f}</span>"
            f"<span style='color:#52514e'>/ 100 CARS reliability &nbsp;·&nbsp; accept threshold {r['tau']:.0f} "
            f"(condition: {r['detected'].replace('_',' ')})</span></div>"
            f"<div style='display:inline-block;background:{c};color:#0b0b0b;font-weight:800;padding:6px 14px;border-radius:8px;font-size:20px'>"
            f"{DEC_ICON[d]} {d}</div> <span style='margin-left:8px'>{DEC_ACTION[d]}</span>"
            f"<div style='margin-top:10px'><b>Why:</b> {r['reason'] if d != 'ACCEPT' else 'All reliability signals within normal range'}"
            f"{' + unfamiliar content (out-of-distribution)' if r['ood'] else ''}</div>{mem}"
            f"<div style='margin-top:10px;color:#52514e;font-size:13px'>Risk contribution</div>{bars}</div>")


def live(idx, upload, blur, noise, dark, fog, adaptive):
    base = to32(upload) if upload is not None else XTE[int(idx)]
    img = apply_custom(base, blur, noise, dark, fog, seed=int(idx))
    r = run(img, adaptive)
    truth = f"Ground truth: **{CLASSES[YTE[int(idx)]]}**" if upload is None else "Uploaded image"
    return big(img), plain_html(r), eg_html(r), truth


def preset(name):
    return {"Clean": (0, 0, 0, 0), "Motion blur": (0.8, 0, 0, 0), "Night / low light": (0, 0, 0.85, 0),
            "Sensor noise": (0, 0.7, 0, 0), "Dense fog": (0, 0, 0, 0.85)}[name]


def ood(kind, i):
    img = OOD[kind][int(i)]
    r = run(img, True)
    return big(img), plain_html(r), eg_html(r)


def mark_failure():
    if "last" in STATE:
        EG.memory.add(*STATE["last"])
    return f"Memory: {len(EG.memory.bank)} failure fingerprints stored ({EG.memory.size_kb():.1f} KB, no images kept)"


def clear_mem():
    EG.memory = BlindSpotMemory(MODEL.feat_dim, capacity=512)
    return "Memory cleared"


with gr.Blocks(title="EdgeGuard demo") as demo:
    gr.Markdown("# EdgeGuard — uncertainty-aware edge AI\nSingle forward pass → prediction **+** reliability score (CARS) **+** decision **+** reason.")
    with gr.Tab("1 · Live reliability check"):
        with gr.Row():
            with gr.Column(scale=1):
                scene = gr.Dropdown(list(SCENES), label="Demo scene (pre-verified)", value=None)
                idx = gr.Slider(0, 9999, value=7, step=1, label="CIFAR-10 test image #")
                upload = gr.Image(label="…or upload any photo", type="numpy")
                pre = gr.Radio(["Clean", "Motion blur", "Night / low light", "Sensor noise", "Dense fog"], value="Clean", label="Scenario preset")
                blur = gr.Slider(0, 1, 0, label="Blur"); noise = gr.Slider(0, 1, 0, label="Sensor noise")
                dark = gr.Slider(0, 1, 0, label="Darkness"); fog = gr.Slider(0, 1, 0, label="Fog")
                adaptive = gr.Checkbox(True, label="Adaptive threshold (per detected condition)")
            with gr.Column(scale=1):
                im = gr.Image(label="What the edge camera sees (32×32, upscaled)", height=256)
                truth = gr.Markdown()
            with gr.Column(scale=2):
                ph = gr.HTML(); eh = gr.HTML()
                with gr.Row():
                    mk = gr.Button("Operator: this prediction was WRONG → store in blind-spot memory")
                    cl = gr.Button("Clear memory")
                ms = gr.Markdown()
        ins = [idx, upload, blur, noise, dark, fog, adaptive]
        for c in ins:
            c.change(live, ins, [im, ph, eh, truth])
        pre.change(preset, pre, [blur, noise, dark, fog])
        scene.change(lambda k: (gr.update(value=SCENES[k][0]), None, *SCENES[k][1:]) if k else (gr.update(), None, 0, 0, 0, 0),
                     scene, [idx, upload, blur, noise, dark, fog])
        mk.click(mark_failure, None, ms); cl.click(clear_mem, None, ms)
        demo.load(live, ins, [im, ph, eh, truth])
    with gr.Tab("2 · Unknown objects (OOD)"):
        gr.Markdown("Images from categories the model has **never** seen. The plain model still names one of its 10 classes, often confidently.")
        with gr.Row():
            _o = CASES.get("ood", {}).get("Digits (SVHN-like)", {}).get("sample", 0)
            kind = gr.Radio(list(OOD), value="Digits (SVHN-like)", label="OOD source"); oi = gr.Slider(0, 63, _o, step=1, label="Sample")
        with gr.Row():
            oim = gr.Image(height=256); oph = gr.HTML(); oeh = gr.HTML()
        kind.change(ood, [kind, oi], [oim, oph, oeh]); oi.change(ood, [kind, oi], [oim, oph, oeh])
    with gr.Tab("3 · Results"):
        figs = sorted(glob.glob("results/figures/fig*.png"))
        gr.Gallery(figs, columns=2, height="auto", label="Benchmark figures")

try:
    demo.launch(share=A.share, theme=gr.themes.Soft())   # Gradio >= 6
except TypeError:
    demo.launch(share=A.share)                            # older Gradio
