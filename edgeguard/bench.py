"""Latency / memory benchmark — run on the actual edge device (e.g. Raspberry Pi):
    python benchmark.py --arch edgenet
Overhead is measured directly (time of the post-forward EdgeGuard step), using
medians over many single-image runs, so noise in the CNN timing does not leak in."""
import os, time
import numpy as np
import torch
from .model import to_tensor
from .core import signal_matrix


def _median_ms(fn, xs, reps=3):
    for x in xs[:10]:
        fn(x)
    ts = []
    for _ in range(reps):
        for x in xs:
            t = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


@torch.no_grad()
def run_benchmark(model, eg, imgs, arch, threads=1):
    torch.set_num_threads(threads)
    model.eval()
    xs = [imgs[i:i + 1] for i in range(len(imgs))]
    t_base = _median_ms(lambda x: model(to_tensor(x)), xs)
    cache = {}
    for i, x in enumerate(xs):
        lg, ft = model(to_tensor(x)); cache[i] = (lg.numpy(), ft.numpy(), x)
    keys = list(cache)
    def post(i):
        lg, ft, x = cache[i]
        eg.decide(signal_matrix(lg, ft, x, eg.maha), adaptive=True, use_memory=False)
    t_post = _median_ms(post, keys)
    from .signals import quality_signals
    t_q = _median_ms(quality_signals, xs)
    def mc(x):
        model.drop.train()
        for _ in range(10):
            model(to_tensor(x))
        model.drop.eval()
    t_mc = _median_ms(mc, xs[:32], reps=1)
    t_onnx, onnx_mb = None, None
    try:
        import onnxruntime as ort
        path = f"checkpoints/{arch}.onnx"
        torch.onnx.export(model, to_tensor(xs[0]), path, input_names=["x"], output_names=["logits", "feat"],
                          opset_version=17, dynamo=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        t_onnx = _median_ms(lambda x: sess.run(None, {"x": to_tensor(x).numpy()}), xs)
        onnx_mb = os.path.getsize(path) / 1e6
    except Exception as e:  # onnx optional
        print("onnx skipped:", e)
    nparams = sum(p.numel() for p in model.parameters())
    D = eg.maha.mu.shape[1]
    module_kb = (eg.maha.mu.nbytes + eg.maha.P.nbytes) / 1024 + 1.0
    return dict(device=f"CPU, {threads} thread, batch=1", params=int(nparams), model_mb_fp32=nparams * 4 / 1e6,
                onnx_mb=onnx_mb, latency_base_ms=t_base, latency_edgeguard_ms=t_base + t_post,
                overhead_ms=t_post, overhead_pct=100 * t_post / t_base,
                latency_quality_signals_ms=t_q, latency_mc_dropout10_ms=t_mc, latency_deep_ensemble5_ms=5 * t_base,
                latency_onnx_base_ms=t_onnx,
                overhead_pct_vs_onnx_base=(100 * t_post / t_onnx) if t_onnx else None,
                edgeguard_module_kb=module_kb, feat_dim=int(D), blind_spot_memory_kb_at_512=512 * D * 2 / 1024)
