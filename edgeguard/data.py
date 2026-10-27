"""Data loading, corruption generation (CIFAR-10-C style) and OOD sets.

All images are handled as uint8 RGB arrays of shape (N, 32, 32, 3).
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# --------------------------------------------------------------------------
# CIFAR-10 loading
# --------------------------------------------------------------------------

def _load_folder(root):
    xs, ys = [], []
    for ci, c in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(root, c, "*")))
        for f in files:
            xs.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))
            ys.append(ci)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_cifar10(data_dir="data", cache="data/cifar10.npz"):
    """Loads CIFAR-10. Uses torchvision download if available (Colab), else an
    image-folder copy at data/c10/{train,test}/<class>/*.jpg."""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    folder = os.path.join(data_dir, "c10")
    if os.path.isdir(folder):
        xtr, ytr = _load_folder(os.path.join(folder, "train"))
        xte, yte = _load_folder(os.path.join(folder, "test"))
    else:
        import torchvision
        tr = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        te = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
        xtr, ytr = tr.data, np.array(tr.targets)
        xte, yte = te.data, np.array(te.targets)
    np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


# --------------------------------------------------------------------------
# Corruptions (re-implementation in the spirit of Hendrycks & Dietterich 2019)
# severity s in 1..5. Input/Output uint8 HxWx3.
# --------------------------------------------------------------------------

def _f(x):
    return x.astype(np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def gaussian_noise(x, s, rng):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][s - 1]
    return _u8(_f(x) + rng.normal(0, c, x.shape))


def shot_noise(x, s, rng):
    c = [500, 250, 100, 75, 50][s - 1]
    return _u8(rng.poisson(_f(x) * c) / c)


def defocus_blur(x, s, rng):
    sig = [0.6, 0.8, 1.0, 1.4, 1.8][s - 1]
    return cv2.GaussianBlur(x, (0, 0), sig)


def motion_blur(x, s, rng):
    k = [3, 5, 7, 9, 11][s - 1]
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    ang = rng.uniform(-45, 45)
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= ker.sum() + 1e-8
    return cv2.filter2D(x, -1, ker, borderType=cv2.BORDER_REFLECT)


def low_light(x, s, rng):
    # darker scene + amplified sensor noise, as in real low-light capture
    g = [0.55, 0.40, 0.28, 0.18, 0.12][s - 1]
    n = [0.005, 0.01, 0.015, 0.02, 0.025][s - 1]
    y = _f(x) ** 1.2 * g
    return _u8(y + rng.normal(0, n, x.shape))


def fog(x, s, rng):
    a = [0.25, 0.4, 0.55, 0.65, 0.75][s - 1]
    h, w = x.shape[:2]
    small = rng.uniform(0.6, 1.0, (4, 4)).astype(np.float32)
    fmap = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    y = _f(x)
    y = y * (1 - a) + a * fmap * 0.9
    return _u8(y)


def contrast(x, s, rng):
    c = [0.4, 0.3, 0.2, 0.12, 0.07][s - 1]
    y = _f(x)
    m = y.mean(axis=(0, 1), keepdims=True)
    return _u8((y - m) * c + m)


def jpeg(x, s, rng):
    q = [60, 40, 25, 15, 8][s - 1]
    ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "fog": fog,
    "contrast": contrast,
    "jpeg": jpeg,
}

# Corruption -> human-readable failure family (used for explanation labels)
FAMILY = {
    "clean": "normal",
    "gaussian_noise": "noise", "shot_noise": "noise",
    "defocus_blur": "blur", "motion_blur": "blur",
    "low_light": "low_light",
    "fog": "fog", "contrast": "fog",
    "jpeg": "compression",
}
CONDITIONS = ["normal", "noise", "blur", "low_light", "fog"]

# Corruptions seen when fitting the fusion / thresholds (validation set).
# The rest are held out to test generalisation to unseen degradations.
FIT_CORRUPTIONS = ["gaussian_noise", "defocus_blur", "low_light", "fog"]
HELDOUT_CORRUPTIONS = ["shot_noise", "motion_blur", "contrast", "jpeg"]


def corrupt(xs, name, severity, seed=0):
    rng = np.random.default_rng(seed)
    fn = CORRUPTIONS[name]
    return np.stack([fn(x, severity, rng) for x in xs])


def apply_custom(x, blur=0.0, noise=0.0, dark=0.0, fog_amt=0.0, seed=0):
    """Continuous controls in [0,1] used by the live demo sliders."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    if blur > 0:
        y = cv2.GaussianBlur(y, (0, 0), 0.1 + 2.0 * blur)
    if fog_amt > 0:
        a = 0.8 * fog_amt
        h, w = y.shape[:2]
        fmap = cv2.resize(rng.uniform(0.6, 1, (4, 4)).astype(np.float32), (w, h))[..., None]
        y = _u8(_f(y) * (1 - a) + a * fmap * 0.9)
    if dark > 0:
        g = 1.0 - 0.9 * dark
        y = _u8(_f(y) ** (1 + 0.3 * dark) * g + rng.normal(0, 0.025 * dark, y.shape))
    if noise > 0:
        y = _u8(_f(y) + rng.normal(0, 0.12 * noise, y.shape))
    return y


# --------------------------------------------------------------------------
# OOD sets
# --------------------------------------------------------------------------

def load_ood(name, n=2000, data_dir="data", seed=0):
    """Real OOD sets via torchvision when downloadable (Colab):
       'cifar100', 'svhn'.  Offline proxies: 'digits_proxy', 'textures_proxy'."""
    rng = np.random.default_rng(seed)
    if name == "cifar100":
        import torchvision
        d = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)
        return d.data[rng.permutation(len(d.data))[:n]]
    if name == "svhn":
        import torchvision
        d = torchvision.datasets.SVHN(data_dir, split="test", download=True)
        x = d.data.transpose(0, 2, 3, 1)
        return x[rng.permutation(len(x))[:n]]
    if name == "digits_proxy":
        return _synthetic_digits(n, rng)
    if name == "textures_proxy":
        return _texture_patches(n, rng)
    raise ValueError(name)


def _synthetic_digits(n, rng):
    """SVHN-like proxy: rendered house-number digits on coloured backgrounds."""
    fonts = (glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) + glob.glob("C:/Windows/Fonts/arial*.ttf")
             + glob.glob("/System/Library/Fonts/Supplemental/Arial*.ttf") + glob.glob("/Library/Fonts/*.ttf"))
    out = []
    for i in range(n):
        bg = tuple(int(v) for v in rng.integers(0, 255, 3))
        fg = tuple(int(255 - v) for v in bg)
        im = Image.new("RGB", (64, 64), bg)
        dr = ImageDraw.Draw(im)
        txt = str(rng.integers(0, 999))
        try:
            font = ImageFont.truetype(fonts[rng.integers(len(fonts))], int(rng.integers(30, 46)))
        except Exception:
            font = ImageFont.load_default()
        dr.text((int(rng.integers(0, 12)), int(rng.integers(0, 14))), txt, fill=fg, font=font)
        im = im.resize((32, 32), Image.BILINEAR)
        out.append(np.asarray(im))
    return np.stack(out)


def _texture_patches(n, rng):
    """DTD-like proxy: random crops of texture / material photos bundled with scikit-image."""
    from skimage import data as skd
    srcs = []
    for nm in ["brick", "grass", "gravel", "rough_wall", "coffee", "moon",
               "text", "camera", "page", "coins", "rocket", "hubble_deep_field"]:
        try:
            im = getattr(skd, nm)()
        except Exception:
            continue
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        srcs.append(im[..., :3].astype(np.uint8))
    out = []
    for i in range(n):
        im = srcs[rng.integers(len(srcs))]
        h, w = im.shape[:2]
        sz = int(rng.integers(32, min(h, w) // 3))
        y0, x0 = rng.integers(0, h - sz), rng.integers(0, w - sz)
        crop = cv2.resize(im[y0:y0 + sz, x0:x0 + sz], (32, 32), interpolation=cv2.INTER_AREA)
        out.append(crop)
    return np.stack(out)
