"""EdgeGuard core: CARS fusion, adaptive thresholds, decision policy,
explainable failure reason, and blind-spot memory."""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .signals import logit_signals, quality_signals, Mahalanobis, QUALITY_KEYS
from .data import CONDITIONS

MODEL_KEYS = ["msp", "entropy", "energy"]
SHIFT_KEYS = ["maha", "feat_norm"]
ALL_KEYS = MODEL_KEYS + SHIFT_KEYS + QUALITY_KEYS
GROUPS = {"Model uncertainty": MODEL_KEYS,
          "Context shift": SHIFT_KEYS,
          "Input quality": QUALITY_KEYS}
REASON_TEXT = {
    "noise": "Sensor noise",
    "blur": "Motion / defocus blur",
    "low_light": "Low illumination",
    "fog": "Fog / low contrast",
    "ood": "Unknown object (out-of-distribution)",
    "ambiguous": "Ambiguous input (high model uncertainty)",
    "memory": "Matches a past failure (blind-spot memory)",
    "none": "-",
}
DECISIONS = ["ACCEPT", "WARN", "DEFER"]


def signal_matrix(logits, feats, imgs, maha):
    s = logit_signals(logits)
    s["maha"] = np.log(maha.score(feats) + 1e-6)
    s["feat_norm"] = np.log(np.linalg.norm(feats, axis=1) + 1e-6)
    s.update(quality_signals(imgs))
    return s


def _stack(s, keys):
    return np.stack([s[k] for k in keys], 1)


def risk_threshold(cars, fail, target, conf=0.9, min_n=30):
    """Lowest CARS threshold whose accepted set has error <= target, using a
    Clopper-Pearson upper confidence bound on the error, so conditions with few
    calibration samples automatically get a stricter (higher) threshold."""
    from scipy.stats import beta
    order = np.argsort(-cars)
    c, f = cars[order], fail[order]
    n = np.arange(1, len(f) + 1); k = np.cumsum(f)
    ucb = np.where(k < n, beta.ppf(conf, k + 1, np.maximum(n - k, 1e-9)), 1.0)
    ok = np.where((ucb <= target) & (n >= min_n))[0]
    if len(ok) == 0:
        return 100.0
    return float(c[ok.max()])


class BlindSpotMemory:
    """Stores only compact, L2-normalised feature vectors (float16) of
    confirmed failures — never raw images. Lookup = one matrix-vector product."""

    def __init__(self, dim, capacity=512):
        self.capacity, self.dim = capacity, dim
        self.bank = np.zeros((0, dim), np.float16)
        self.cls = np.zeros((0,), np.int64)

    @staticmethod
    def _norm(f):
        return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)

    def add(self, feats, pred_cls):
        f = self._norm(np.atleast_2d(feats)).astype(np.float16)
        self.bank = np.concatenate([self.bank, f])[-self.capacity:]
        self.cls = np.concatenate([self.cls, np.atleast_1d(pred_cls)])[-self.capacity:]

    def similarity(self, feats, pred_cls):
        """Max cosine similarity to a stored failure with the same predicted class."""
        f = self._norm(np.atleast_2d(feats)).astype(np.float32)
        if len(self.bank) == 0:
            return np.zeros(len(f))
        sim = f @ self.bank.astype(np.float32).T
        same = np.atleast_1d(pred_cls)[:, None] == self.cls[None]
        sim = np.where(same, sim, -1.0)
        return sim.max(1)

    def size_kb(self):
        return self.bank.nbytes / 1024 + self.cls.nbytes / 1024


class EdgeGuard:
    def __init__(self, risk_target=0.03, defer_below=50.0):
        self.risk_target = risk_target
        self.defer_below = defer_below
        self.memory = None
        self.mem_tau = 1.01

    # ----------------------------------------------------------------- fit
    def fit_maha(self, train_feats, train_labels):
        self.maha = Mahalanobis().fit(train_feats, train_labels)
        return self

    def fit(self, s_val, fail_val, cond_val, keys=ALL_KEYS):
        """s_val: signal dict on the validation set (clean + FIT corruptions)
        fail_val: 1 if the base model's prediction was wrong
        cond_val: true condition label (index into CONDITIONS)"""
        self.keys = keys
        self.use_gate = "maha" in keys
        X = _stack(s_val, keys)
        self.scaler = StandardScaler().fit(X)
        self.fusion = LogisticRegression(C=1.0, max_iter=2000).fit(self.scaler.transform(X), fail_val)
        # condition classifier (input-quality only) for adaptive threshold + explanation
        Q = _stack(s_val, QUALITY_KEYS)
        self.qscaler = StandardScaler().fit(Q)
        ok = cond_val >= 0                      # -1 = ambiguous (e.g. severity-1), skipped
        self.cond_clf = LogisticRegression(C=1.0, max_iter=3000).fit(self.qscaler.transform(Q[ok]), cond_val[ok])
        self._compile()
        # OOD rule threshold: 95th percentile of maha on clean val
        clean = cond_val == 0
        mc = s_val["maha"][clean]
        self.maha_tau = float(np.percentile(mc, 95))
        # context-shift gate: CARS is pulled to 0 once the feature distance leaves the
        # range seen on clean calibration data (99th percentile). Needs NO OOD data.
        self.gate_q = float(np.percentile(mc, 99))
        self.gate_s = float(0.25 * np.subtract(*np.percentile(mc, [75, 25])))
        # thresholds
        cars = self.cars(s_val)
        self.tau_fixed = risk_threshold(cars, fail_val, self.risk_target)
        det = self.condition(s_val)[0]
        self.tau_cond = {}
        for ci, c in enumerate(CONDITIONS):
            m = det == ci
            self.tau_cond[c] = risk_threshold(cars[m], fail_val[m], self.risk_target) if m.sum() >= 50 else self.tau_fixed
        return self

    def _compile(self):
        """Freeze the fitted sklearn models into plain numpy weights so that
        on-device inference is a couple of tiny dot products (no sklearn overhead)."""
        self.w_f = (self.fusion.coef_[0] / self.scaler.scale_).astype(np.float32)
        self.b_f = float(self.fusion.intercept_[0] - (self.fusion.coef_[0] * self.scaler.mean_ / self.scaler.scale_).sum())
        W = self.cond_clf.coef_ / self.qscaler.scale_[None]
        self.W_c = W.astype(np.float32)
        self.b_c = (self.cond_clf.intercept_ - (self.cond_clf.coef_ * self.qscaler.mean_[None] / self.qscaler.scale_[None]).sum(1)).astype(np.float32)

    # ------------------------------------------------------------- scoring
    def p_fail(self, s):
        z = _stack(s, self.keys) @ self.w_f + self.b_f
        return 1.0 / (1.0 + np.exp(-z))

    def ood_gate(self, s):
        if not getattr(self, "use_gate", False) or not hasattr(self, "gate_q"):
            return np.zeros(len(s["maha"]))
        return 1.0 / (1.0 + np.exp(-(s["maha"] - self.gate_q) / self.gate_s))

    def cars(self, s):
        """Context-Aware Reliability Score, 0 (unreliable) .. 100 (reliable)
        = P(prediction correct | signals) x (1 - context-shift gate)."""
        return 100.0 * (1.0 - self.p_fail(s)) * (1.0 - self.ood_gate(s))

    def condition(self, s):
        z = _stack(s, QUALITY_KEYS) @ self.W_c.T + self.b_c
        z = np.exp(z - z.max(1, keepdims=True))
        pr = z / z.sum(1, keepdims=True)
        return pr.argmax(1), pr

    def contributions(self, s):
        """Per-group share of the risk logit (for the explanation bar chart)."""
        z = self.scaler.transform(_stack(s, self.keys))
        contrib = z * self.fusion.coef_[0][None]
        out = {}
        for g, ks in GROUPS.items():
            idx = [self.keys.index(k) for k in ks if k in self.keys]
            out[g] = contrib[:, idx].sum(1) if idx else np.zeros(len(z))
        return out

    def thresholds(self, s, adaptive=True):
        det = self.condition(s)[0]
        if not adaptive:
            return np.full(len(det), self.tau_fixed)
        return np.array([self.tau_cond[CONDITIONS[d]] for d in det])

    # ------------------------------------------------------------ decision
    def decide(self, s, feats=None, preds=None, adaptive=True, use_memory=True):
        cars = self.cars(s)
        det, pr = self.condition(s)
        tau = (np.array([self.tau_cond[CONDITIONS[d]] for d in det]) if adaptive
               else np.full(len(det), self.tau_fixed))
        dec = np.where(cars >= tau, 0, np.where(cars >= self.defer_below, 1, 2))
        mem_hit = np.zeros(len(cars), bool)
        if use_memory and self.memory is not None and feats is not None and len(self.memory.bank):
            sim = self.memory.similarity(feats, preds)
            mem_hit = sim >= self.mem_tau
            dec = np.where(mem_hit & (dec == 0), 1, dec)
        reasons = self.explain(s, dec, mem_hit, (det, pr))
        ood_flag = (self.ood_gate(s) > 0.5) & (dec > 0)
        return dict(cars=cars, tau=tau, decision=dec, reason=reasons, mem_hit=mem_hit, ood_flag=ood_flag)

    def explain(self, s, dec, mem_hit=None, cond=None):
        det, pr = cond if cond is not None else self.condition(s)
        out = []
        for i in range(len(dec)):
            if dec[i] == 0:
                out.append("none"); continue
            if mem_hit is not None and mem_hit[i] and det[i] == 0:
                out.append("memory"); continue
            if det[i] != 0 and pr[i, det[i]] >= 0.5:
                out.append(CONDITIONS[det[i]])
            elif s["maha"][i] > self.maha_tau or self.ood_gate({"maha": s["maha"][i:i + 1]})[0] > 0.5:
                out.append("ood")
            else:
                out.append("ambiguous")
        return np.array(out)

    # ---------------------------------------------------------------- io
    def save(self, path):
        st = {k: v for k, v in self.__dict__.items() if k != "maha"}
        st["maha_state"] = self.maha.state()
        pickle.dump(st, open(path, "wb"))

    @classmethod
    def load(cls, path):
        st = pickle.load(open(path, "rb"))
        g = cls()
        ms = st.pop("maha_state")
        g.__dict__.update(st)
        g.maha = Mahalanobis().load(ms)
        if not hasattr(g, "w_f"):
            g._compile()
        return g
