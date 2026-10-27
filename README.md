# EdgeGuard - Uncertainty-Aware Edge AI for Reliable Decision Making

Senior Design Project, School of Computer Science and Engineering (AI), KLE Technological University, Hubballi.
Team: Vijeta Hegde, Rakshita R Talegaon, Tejaswini M V, Manjula Nayak · Guide: Maheshwari Kittur

EdgeGuard wraps an unmodified image classifier and, from its **single forward pass**, returns
**prediction + reliability score (CARS, 0–100) + decision (ACCEPT / WARN / DEFER) + reason**.

```
image ──► base CNN ──► logits ─────► MSP · entropy · energy                   (model uncertainty)
              └──────► features ───► Mahalanobis on L2-normalised features      (context shift)
image ──► 5 cheap image stats ─────► blur · noise · brightness · contrast · haze (input quality)
                     │
                     ▼
   CARS = 100 · (1 − p_fail) · (1 − shift gate)     p_fail = logistic fusion of the 10 signals
   condition classifier → adaptive threshold τ(condition), Clopper–Pearson bound, 3 % risk target
   decision: CARS ≥ τ → ACCEPT · 50 ≤ CARS < τ → WARN · CARS < 50 → DEFER   (+ reason)
   blind-spot memory: 512 float16 feature fingerprints of confirmed failures (no images stored)
```

## Results (reference run: EdgeNet, 0.29 M params, CPU)

Test set: 94,000 images = 10k clean CIFAR-10 + 80k corrupted (8 types × 5 severities × 2k) + 4k OOD proxies.

| Metric | Baseline | EdgeGuard |
|---|---|---|
| Error on predictions acted on (all inputs) | 43.5 % (plain model) | **3.5 %** (accepts 22.7 %) |
| Wrong predictions flagged | 0 % | **98.2 %** |
| Failure-detection AUROC | MSP 0.837 · MC dropout 0.836 | **0.842** |
| OOD AUROC (digits / textures) | MSP 0.866 / 0.842 | **0.960 / 0.910** |
| ECE on corrupted inputs | 12.5 % | **4.2 %** |
| Extra latency per image (1 CPU thread) | MC dropout (10 passes) +17 ms | **+0.14 ms** |
| Extra memory | — | 70 KB (+132 KB memory bank) |

All figures are in `results/figures/`; every number is in `results/results_edgenet.json`.

## Repository layout

| Path | Contents |
|---|---|
| `edgeguard/` | Library: `data.py` (CIFAR-10, corruptions, OOD sets), `model.py` (EdgeNet, ResNet-18), `signals.py`, `core.py` (CARS, thresholds, decisions, reasons, memory), `bench.py` |
| `train.py` | Step 1 — train the base model |
| `evaluate.py` | Step 2 — calibrate EdgeGuard and run the full evaluation |
| `make_figures.py`, `examples.py` | Step 3 — all figures |
| `find_demo_cases.py` | Pre-verifies the demo scenes for the app |
| `app.py` | Step 4 — live Gradio demo |
| `benchmark.py`, `summarize.py` | Latency benchmark; print headline numbers |
| `EdgeGuard_Colab.ipynb` | Full GPU pipeline (ResNet-18 + real CIFAR-100/SVHN) |
| `checkpoints/` | Trained EdgeNet, fitted EdgeGuard, ONNX export |
| `results/` | Metrics JSON, figures, demo scenes |
| `paper/` | IEEE conference paper (LaTeX, compile on Overleaf) |
| `docs/` | Review 1 and Review 2 presentations |

## Quick start (laptop, uses the trained model in `checkpoints/`)

```bash
git clone https://github.com/Vijeta1107/edge_guard.git
cd edge_guard
python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1     Mac/Linux: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python find_demo_cases.py      # re-verifies demo scenes on your machine (~2 min)
python app.py                  # open http://127.0.0.1:7860   (--share for a public link)
```

The first run downloads CIFAR-10 (~170 MB). If the download is slow, use
`git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images data/c10` instead; the loader uses those folders automatically.

**Windows:** if `import torch` fails with `WinError 1114 (c10.dll)`, install the
[Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), or use
`pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu` on Python 3.9–3.12.

## Reproduce everything

```bash
python train.py --arch edgenet --epochs 20             # ~30 min on CPU
python evaluate.py --arch edgenet                      # ~10 min; offline OOD proxies
python make_figures.py --arch edgenet && python examples.py --arch edgenet
python benchmark.py --arch edgenet                     # run this on the edge board too
python summarize.py edgenet
```

### GPU run for final numbers (Colab or Kaggle, T4)

```bash
git clone https://github.com/Vijeta1107/edge_guard.git && cd edge_guard
pip install gradio onnx onnxruntime
git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images data/c10   # faster than the official server
python train.py --arch resnet18 --epochs 60 --lr 0.1   # ~40–50 min on a T4
python evaluate.py --arch resnet18 --real_ood          # real CIFAR-100 + SVHN as OOD
python make_figures.py --arch resnet18 && python examples.py --arch resnet18 && python find_demo_cases.py --arch resnet18
python app.py --arch resnet18 --share
```
On Kaggle, turn on Internet and choose GPU T4 x2 in the notebook settings.

## Notes on scope

* Corruptions are re-implemented in the style of CIFAR-10-C (Hendrycks & Dietterich, 2019) and generated on the fly; the official `.npy` files can be loaded into the `sets` list in `evaluate.py`.
* Fusion and thresholds are fitted only on the 5k calibration split with four corruption types; shot noise, motion blur, contrast, JPEG and all OOD data are unseen during fitting.
* The reference OOD sets (rendered digits, texture crops) are proxies; `--real_ood` uses CIFAR-100 and SVHN.
* The MC-dropout baseline samples the dropout head (AUROC); its latency is measured as 10 full forward passes.
* EdgeGuard never changes the base prediction, so clean accuracy is preserved by construction.
