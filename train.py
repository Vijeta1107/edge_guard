"""Step 1 — train the base perception model on CIFAR-10 (in-distribution).

  CPU / quick : python train.py --arch edgenet --epochs 15
  Colab GPU   : python train.py --arch resnet18 --epochs 60
"""
import argparse, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from edgeguard.data import load_cifar10
from edgeguard.model import build_model, to_tensor, infer

p = argparse.ArgumentParser()
p.add_argument("--arch", default="edgenet")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--out", default="checkpoints")
a = p.parse_args()

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
xtr, ytr, xte, yte = load_cifar10()
rng = np.random.default_rng(0)
perm = rng.permutation(len(xtr))
val_idx, tr_idx = perm[:5000], perm[5000:]          # 45k train / 5k calibration-val
np.save("data/val_idx.npy", val_idx)
x, y = xtr[tr_idx], ytr[tr_idx]

model = build_model(a.arch).to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"{a.arch}: {nparams/1e6:.3f}M params on {dev}")
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
steps = a.epochs * (len(x) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.25)


def augment(xb):
    # random crop with 4px reflect padding + horizontal flip
    xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    i, j = np.random.randint(0, 9, 2)
    xb = xb[:, :, i:i + 32, j:j + 32]
    flip = torch.rand(xb.shape[0]) < 0.5
    xb[flip] = xb[flip].flip(3)
    return xb


hist = []
for ep in range(a.epochs):
    model.train()
    t0, tl, tc = time.time(), 0.0, 0
    order = np.random.permutation(len(x))
    for k in range(len(x) // a.bs):
        b = order[k * a.bs:(k + 1) * a.bs]
        xb = augment(to_tensor(x[b])).to(dev)
        yb = torch.from_numpy(y[b]).to(dev)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        tl += loss.item() * len(b); tc += (logits.argmax(1) == yb).sum().item()
    lg, _ = infer(model, xtr[val_idx], device=dev)
    vacc = (lg.argmax(1) == ytr[val_idx]).mean()
    n = (len(x) // a.bs) * a.bs
    hist.append(dict(epoch=ep + 1, loss=tl / n, train_acc=tc / n, val_acc=float(vacc)))
    print(f"ep {ep+1:2d} loss {tl/n:.3f} train {tc/n:.3f} val {vacc:.3f} ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(a.out, exist_ok=True)
torch.save({"arch": a.arch, "state": model.state_dict()}, f"{a.out}/{a.arch}.pt")
lg, _ = infer(model, xte, device=dev)
test_acc = float((lg.argmax(1) == yte).mean())
print("clean test acc", test_acc)
os.makedirs("results", exist_ok=True)
json.dump({"arch": a.arch, "params": nparams, "history": hist, "clean_test_acc": test_acc},
          open(f"results/train_{a.arch}.json", "w"), indent=2)
