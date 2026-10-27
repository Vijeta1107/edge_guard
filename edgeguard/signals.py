"""Single-pass reliability signals.

Model-side   (from the logits + penultimate features of ONE forward pass):
    msp, entropy, energy, mahalanobis distance (context shift / OOD)
Input-side   (cheap image statistics, no neural network):
    blur (log Laplacian variance), noise sigma (Immerkaer), brightness,
    RMS contrast, haze (dark-channel prior)
"""
import numpy as np
import cv2
from scipy.special import logsumexp, softmax


# ---------------------------- model-side --------------------------------

def logit_signals(logits):
    mx = logits.max(1, keepdims=True)
    e = np.exp(logits - mx)
    z = e.sum(1, keepdims=True)
    p = e / z
    msp = p.max(1)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    energy = -(np.log(z[:, 0]) + mx[:, 0])          # -logsumexp; higher = more OOD-like
    return dict(msp=msp, entropy=ent, energy=energy)


def l2n(f):
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance (Lee et al., 2018)
    fitted once, offline, on L2-normalised training features. Normalising first
    matters: on raw features degraded images collapse toward the origin and look
    'close' to every class (AUROC < 0.5 in our runs). Cost at inference:
    one D x D and one D x C matmul — negligible next to the CNN."""

    def fit(self, feats, labels, num_classes=10, shrink=1e-3):
        feats = l2n(feats)
        D = feats.shape[1]
        self.mu = np.stack([feats[labels == c].mean(0) for c in range(num_classes)])
        centered = feats - self.mu[labels]
        cov = centered.T @ centered / len(feats)
        cov += shrink * np.trace(cov) / D * np.eye(D)
        self.P = np.linalg.inv(cov).astype(np.float32)
        self.mu = self.mu.astype(np.float32)
        return self

    def score(self, feats):
        # ||f-mu_c||^2_P = f'Pf - 2 f'P mu_c + mu_c'P mu_c   (one D x D and one D x C matmul)
        if not hasattr(self, "Pmu"):
            self.Pmu = self.P @ self.mu.T
            self.muPmu = (self.mu.T * self.Pmu).sum(0)
        f = l2n(feats.astype(np.float32))
        fPf = ((f @ self.P) * f).sum(1, keepdims=True)
        m = fPf - 2 * f @ self.Pmu + self.muPmu[None]
        return np.sqrt(np.maximum(m.min(1), 0))               # distance to closest class

    def state(self):
        return dict(mu=self.mu, P=self.P)

    def load(self, st):
        self.mu, self.P = st["mu"], st["P"]
        return self


# ---------------------------- input-side --------------------------------

_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
_ERODE = np.ones((3, 3), np.uint8)
cv2.setNumThreads(1)   # tiny images: threading overhead dominates


def quality_one(img):
    """~35 us per 32x32 image on one CPU core."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m, sd = cv2.meanStdDev(g)
    _, ls = cv2.meanStdDev(cv2.Laplacian(g, cv2.CV_32F))
    nz = cv2.filter2D(g, cv2.CV_32F, _NOISE_K)[1:-1, 1:-1]
    noise = cv2.norm(nz, cv2.NORM_L1) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    dark = cv2.erode(cv2.min(cv2.min(img[..., 0], img[..., 1]), img[..., 2]), _ERODE)
    return (np.log1p(ls[0, 0] ** 2), noise, m[0, 0] / 255.0, sd[0, 0] / 255.0, cv2.mean(dark)[0] / 255.0)


QUALITY_KEYS = ["blur_lapvar", "noise_sigma", "brightness", "contrast", "haze"]


def quality_signals(imgs):
    imgs = np.ascontiguousarray(imgs)
    q = np.array([quality_one(im) for im in imgs], dtype=np.float32)
    return {k: q[:, i] for i, k in enumerate(QUALITY_KEYS)}
