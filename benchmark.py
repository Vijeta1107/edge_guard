"""Re-run only the efficiency benchmark (e.g. on a Raspberry Pi) and update the results file.
    python benchmark.py --arch edgenet [--threads 4]"""
import argparse, json
import torch
from edgeguard.data import load_cifar10
from edgeguard.model import build_model
from edgeguard.core import EdgeGuard
from edgeguard.bench import run_benchmark

p = argparse.ArgumentParser(); p.add_argument("--arch", default="edgenet"); p.add_argument("--threads", type=int, default=1)
a = p.parse_args()
m = build_model(a.arch); m.load_state_dict(torch.load(f"checkpoints/{a.arch}.pt", map_location="cpu")["state"])
eg = EdgeGuard.load(f"checkpoints/edgeguard_{a.arch}.pkl")
E = run_benchmark(m, eg, load_cifar10()[2][:64], a.arch, a.threads)
print(json.dumps(E, indent=1))
path = f"results/results_{a.arch}.json"
R = json.load(open(path)); R["efficiency"] = E; json.dump(R, open(path, "w"), indent=1)
