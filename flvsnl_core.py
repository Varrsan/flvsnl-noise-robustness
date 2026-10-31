"""
flvsnl_core.py  --  Core library for the loss-family characterisation study (v2).

Replaces: FLvsNL_AFL.py, FLvsNL_AFL_v2_patch.py, FLvsNL_AFL_baselines_patch.py

What is new in v2 (relative to the code used for the first draft):
  1. MEAN-MATCHED adaptive losses. V1/V2 previously applied a strictly smaller
     average epsilon than Static LS, so "adaptive vs static" was confounded with
     "less smoothing vs more smoothing". AdaptiveFuzzyLoss(match_mean=True)
     rescales eps(x) so its running mean equals eps_target.
  2. ORACLE adaptive losses. eps(x) driven by ground-truth clean/noisy membership.
     Upper bound on what ANY target-side adaptivity could achieve.
  3. Per-sample DIAGNOSTICS. Per epoch, AUROC of each candidate fuzziness signal
     (entropy, disagreement, small-loss) for detecting mislabelled samples, plus
     mean signal separately for clean and noisy samples.
  4. Noise realisation is tied to the seed (shared across losses within a seed),
     so seed-level variance reflects noise-draw variance too.
  5. Full per-epoch curves recorded; both final-epoch and best-epoch reported.
  6. Extra losses: MAE, NCE+RCE (APL), negative label smoothing.
  7. Asymmetric (pair-flip) noise in addition to symmetric.
  8. Every run cached to JSON -> a timed-out Kaggle session loses nothing.

Usage:  %run -i flvsnl_core.py   then   %run -i flvsnl_run.py
"""

import os
import json
import math
import time
import random
import hashlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score

# =============================================================================
# 0. Global config
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = os.environ.get("FLVSNL_RESULTS_DIR", "/kaggle/working/results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "runs.json")

EPS_CAP = 0.9          # hard ceiling on eps(x); keeps mu a valid distribution
DIAG_SUBSET = 10000    # samples used for per-epoch diagnostics (0 = disable)

print(f"[core] device={DEVICE}  results={RESULTS_PATH}")


# =============================================================================
# 1. Reproducibility
# =============================================================================

def set_seed(seed):
    """Seed every RNG we touch. Called once per (experiment, seed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _worker_init(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


# =============================================================================
# 2. Datasets with tracked noise
# =============================================================================

class NoisyLabelDataset(Dataset):
    """
    Wraps a base dataset and overrides its labels with a corrupted copy, while
    RECORDING which samples were flipped. Returns (x, y_noisy, idx).

    The original get_noisy_data() materialised 60k transformed tensors into a
    Python list (slow, ~190 MB) and used `i in list` for membership (O(n^2)).
    This version stores labels as an int64 array and leaves image decoding to
    the base dataset, so construction is effectively instantaneous.
    """

    def __init__(self, base, noisy_labels, clean_labels, is_noisy):
        self.base = base
        self.noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
        self.clean_labels = np.asarray(clean_labels, dtype=np.int64)
        self.is_noisy = np.asarray(is_noisy, dtype=np.int64)

    def __len__(self):
        return len(self.noisy_labels)

    def __getitem__(self, idx):
        # The base may be a torchvision dataset -> (x, y), or one of our own
        # ArrayDatasets -> (x, y, idx). Accept either, so tabular/text data can
        # be corrupted with the same code path as images.
        item = self.base[idx]
        x = item[0]
        return x, int(self.noisy_labels[idx]), idx


def _base_labels(dataset):
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return np.asarray(t.numpy() if torch.is_tensor(t) else t, dtype=np.int64)
    if hasattr(dataset, "y"):
        t = dataset.y
        return np.asarray(t.numpy() if torch.is_tensor(t) else t, dtype=np.int64)
    return np.asarray([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)


def make_noisy_dataset(base, noise_rate, num_classes, seed, noise_type="symmetric"):
    """
    Build a corrupted copy of `base`.

    noise_type:
      'symmetric' -- flip to a uniformly random OTHER class (paper's protocol).
      'asymmetric'-- pair flip y -> (y+1) % K, applied to a fraction of EACH class.

    The RNG is seeded by `seed`, so:
      - all losses within one seed see the IDENTICAL corrupted set (pairing kept),
      - different seeds see different corruptions (variance is honest).
    """
    clean = _base_labels(base)
    n = len(clean)
    rng = np.random.RandomState(10_000 + seed)
    noisy = clean.copy()
    is_noisy = np.zeros(n, dtype=np.int64)

    if noise_rate <= 0:
        return NoisyLabelDataset(base, noisy, clean, is_noisy)

    if noise_type == "symmetric":
        flip_idx = rng.choice(n, size=int(round(n * noise_rate)), replace=False)
        for i in flip_idx:
            choices = [c for c in range(num_classes) if c != clean[i]]
            noisy[i] = choices[rng.randint(len(choices))]
        is_noisy[flip_idx] = 1

    elif noise_type == "asymmetric":
        for c in range(num_classes):
            idx_c = np.where(clean == c)[0]
            k = int(round(len(idx_c) * noise_rate))
            if k == 0:
                continue
            flip_idx = rng.choice(idx_c, size=k, replace=False)
            noisy[flip_idx] = (c + 1) % num_classes
            is_noisy[flip_idx] = 1
    else:
        raise ValueError(f"unknown noise_type={noise_type}")

    actual = is_noisy.mean()
    print(f"  [noise] type={noise_type} requested={noise_rate:.2f} "
          f"actual={actual:.4f} flipped={int(is_noisy.sum())}/{n}")
    return NoisyLabelDataset(base, noisy, clean, is_noisy)


class SubsetNoisy(Dataset):
    """
    A slice of a NoisyLabelDataset that renumbers indices locally, so the
    per-sample is_noisy lookup inside the training loop stays correct.

    use_clean_labels=True produces the SAME samples carrying their true labels.
    That gives a clean validation set, which is an ORACLE model-selection signal
    (nobody has clean labels at training time in the setting we are studying);
    it is included only to bound how much early stopping could ever recover.
    """

    def __init__(self, parent, indices, use_clean_labels=False):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)
        self.use_clean = bool(use_clean_labels)
        self.clean_labels = parent.clean_labels[self.indices]
        self.noisy_labels = (self.clean_labels if use_clean_labels
                             else parent.noisy_labels[self.indices])
        self.is_noisy = (np.zeros(len(self.indices), dtype=np.int64)
                         if use_clean_labels else parent.is_noisy[self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.parent.base[int(self.indices[i])][0]
        return x, int(self.noisy_labels[i]), i


def split_train_val(noisy_ds, val_frac=0.1, seed=0):
    """
    Carve a validation set out of the corrupted training data.

    Returns (train_sub, val_noisy, val_clean).

      val_noisy -- labels are corrupted at the same rate as training. This is
                   the REALISTIC model-selection signal and the one the paper
                   should report, because it is all a practitioner has.
      val_clean -- same samples, true labels. Oracle selection; report as an
                   upper bound only.

    The split is stratified by is_noisy so the validation set carries the same
    noise rate as the training remainder.
    """
    rng = np.random.RandomState(50_000 + seed)
    n = len(noisy_ds)
    idx = np.arange(n)
    val_idx = []
    for grp in (0, 1):                       # stratify on clean/noisy
        g = idx[noisy_ds.is_noisy == grp]
        rng.shuffle(g)
        val_idx.append(g[:int(round(len(g) * val_frac))])
    val_idx = np.sort(np.concatenate(val_idx))
    train_idx = np.setdiff1d(idx, val_idx)

    print(f"  [split] train={len(train_idx)} val={len(val_idx)} "
          f"(val noise rate {noisy_ds.is_noisy[val_idx].mean():.3f}, "
          f"train {noisy_ds.is_noisy[train_idx].mean():.3f})")
    return (SubsetNoisy(noisy_ds, train_idx),
            SubsetNoisy(noisy_ds, val_idx, use_clean_labels=False),
            SubsetNoisy(noisy_ds, val_idx, use_clean_labels=True))


class ArrayDataset(Dataset):
    """Tabular / TF-IDF data. Returns (x, y, idx) to match NoisyLabelDataset."""

    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.targets = self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx]), idx


# =============================================================================
# 3. Models  (architectures preserved EXACTLY from v1 for comparability)
# =============================================================================

class CNNClassifier(nn.Module):
    """
    conv(1->32,3x3) - relu - conv(32->64,3x3) - relu - pool2 - pool2
    - dropout(0.25) - flatten - fc(->128) - relu - linear(->K)

    NOTE for the write-up: Section 4.3 of the draft says "two conv layers each
    followed by ReLU and 2x2 max-pooling". The code (v1 and here) applies both
    pools after conv2. Total downsampling is the same (/4), so results are
    unaffected, but the prose should be corrected to match the code.
    """

    def __init__(self, in_channels, image_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        final_dim = image_dim // 4
        self.fc = nn.Linear(64 * final_dim * final_dim, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.head(x)


class MLPClassifier(nn.Module):
    """fc(->256) - relu - drop(.3) - fc(->128) - relu - drop(.3) - linear(->K)"""

    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop2 = nn.Dropout(0.3)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.head(x)


# =============================================================================
# 4. Losses
#    Uniform interface:  loss(logits, targets, is_noisy=None) -> scalar
# =============================================================================

def _soft_target_loss(logits, targets, eps):
    """
    Cross-entropy against the fuzzy membership vector
        mu_y = 1 - eps ,  mu_{i != y} = eps / (K - 1)
    `eps` is a (B,) tensor (already detached). Works for negative eps too.
    """
    logp = F.log_softmax(logits, dim=1)
    B, K = logp.shape
    off = (eps / (K - 1)).unsqueeze(1).expand(B, K).contiguous()
    mu = off.scatter(1, targets.unsqueeze(1), (1.0 - eps).unsqueeze(1))
    return -(mu * logp).sum(dim=1).mean()


class _EpsTracker:
    """
    Mixin: accumulate how the smoothing budget is split between correctly- and
    incorrectly-labelled samples.

    This is the direct measurement of the allocation hypothesis. A signal can
    RANK noisy samples near-perfectly (AUROC ~0.997) and still send a large
    share of the total smoothing mass to clean samples, because ranking is
    scale-free and allocation is not. `share_to_clean` is that share; the oracle
    achieves 0.0 by construction.
    """

    def reset_stats(self):
        self._eps_sum, self._eps_n = 0.0, 0
        self._eps_sum_clean, self._n_clean = 0.0, 0
        self._eps_sum_noisy, self._n_noisy = 0.0, 0

    def _track(self, eps, is_noisy):
        self._eps_sum += float(eps.sum())
        self._eps_n += eps.numel()
        if is_noisy is None:
            return
        m = is_noisy.bool()
        self._eps_sum_noisy += float(eps[m].sum())
        self._n_noisy += int(m.sum())
        self._eps_sum_clean += float(eps[~m].sum())
        self._n_clean += int((~m).sum())

    def epoch_mean_eps(self):
        return self._eps_sum / max(self._eps_n, 1)

    def epoch_eps_split(self):
        """-> dict with mean eps on clean / on noisy, and the budget share."""
        tot = self._eps_sum_clean + self._eps_sum_noisy
        if self._n_clean == 0 and self._n_noisy == 0:
            return None
        return {
            "eps_clean": self._eps_sum_clean / max(self._n_clean, 1),
            "eps_noisy": self._eps_sum_noisy / max(self._n_noisy, 1),
            "share_to_clean": (self._eps_sum_clean / tot) if tot > 0 else None,
        }


class StaticFuzzyLoss(nn.Module, _EpsTracker):
    """Label Smoothing. eps < 0 gives negative label smoothing."""

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = float(epsilon)
        self.reset_stats()

    def forward(self, logits, targets, is_noisy=None):
        eps = torch.full((logits.size(0),), self.epsilon,
                         device=logits.device, dtype=logits.dtype)
        self._track(eps, is_noisy)
        return _soft_target_loss(logits, targets, eps)


class AdaptiveFuzzyLoss(nn.Module, _EpsTracker):
    """
    Target-side modification with a per-sample fuzziness coefficient.

    signal:
      'entropy'      -- V1: normalised Shannon entropy of p(x)          in [0,1]
      'disagreement' -- V2: (1 - p_y(x)) ** gamma                       in [0,1]
      'oracle'       -- 1.0 if the sample is actually mislabelled else 0
      'topk'         -- binarised disagreement: within each batch, the top
                        `noise_rate` fraction by (1 - p_y) get eps, the rest 0.
                        This mimics the ORACLE'S ALLOCATION using only the
                        model's own ranking. Requires a known noise rate, which
                        is standard in this literature (Co-teaching, DivideMix
                        both assume it). It is the decisive test of the
                        allocation hypothesis: if the ranking really is
                        near-perfect, binarising it should recover most of the
                        oracle's advantage.

    match_mean:
      False -> eps(x) = eps_target * s(x).  This is the v1 behaviour, and it
               makes E[eps] STRICTLY LESS than eps_target, confounding
               "adaptive vs static" with "less smoothing vs more smoothing".
      True  -> eps(x) = eps_target * s(x) / EMA[s], clamped to [0, EPS_CAP].
               Average smoothing now matches Static LS, so the comparison
               isolates adaptivity. THIS IS THE COMPARISON THE PAPER NEEDS.

    The coefficient is always computed from a detached copy of p(x); no gradient
    flows through eps, so d(loss)/d(logit_j) = p_j - mu_j.
    """

    def __init__(self, signal="entropy", eps_target=0.1, gamma=2.0,
                 match_mean=False, ema_momentum=0.9, eps_cap=EPS_CAP,
                 noise_rate=0.4):
        super().__init__()
        assert signal in ("entropy", "disagreement", "oracle", "topk")
        self.signal = signal
        self.eps_target = float(eps_target)
        self.gamma = float(gamma)
        self.match_mean = bool(match_mean)
        self.ema_momentum = float(ema_momentum)
        self.eps_cap = float(eps_cap)
        self.noise_rate = float(noise_rate)
        self._ema = None
        self.reset_stats()

    def _raw_signal(self, logits, targets, is_noisy):
        with torch.no_grad():
            if self.signal == "oracle":
                if is_noisy is None:
                    raise ValueError("oracle signal requires is_noisy")
                return is_noisy.to(logits.dtype)

            p = F.softmax(logits, dim=1)
            if self.signal == "entropy":
                K = logits.size(1)
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
                return (ent / math.log(K)).clamp(0.0, 1.0)

            py = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            dis = (1.0 - py).clamp(0.0, 1.0)

            if self.signal == "topk":
                B = logits.size(0)
                k = int(round(self.noise_rate * B))
                s = torch.zeros_like(dis)
                if k > 0:
                    s[torch.topk(dis, k).indices] = 1.0
                return s

            return dis ** self.gamma

    def forward(self, logits, targets, is_noisy=None):
        s = self._raw_signal(logits, targets, is_noisy)

        if self.match_mean:
            batch_mean = float(s.mean())
            self._ema = batch_mean if self._ema is None else (
                self.ema_momentum * self._ema + (1 - self.ema_momentum) * batch_mean)
            scale = self.eps_target / max(self._ema, 1e-6)
            eps = (scale * s).clamp(0.0, self.eps_cap)
        else:
            eps = (self.eps_target * s).clamp(0.0, self.eps_cap)

        self._track(eps, is_noisy)
        return _soft_target_loss(logits, targets, eps.detach())


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, weights=None):
        super().__init__()
        self.gamma, self.alpha, self.weights = gamma, alpha, weights

    def forward(self, logits, targets, is_noisy=None):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.weights is not None:
            loss = loss * self.weights.to(logits.device)[targets]
        return loss.mean()


class GCELoss(nn.Module):
    """Zhang & Sabuncu (2018):  (1 - p_y^q) / q.   Bounded above by 1/q."""

    def __init__(self, q=0.7):
        super().__init__()
        self.q = float(q)

    def forward(self, logits, targets, is_noisy=None):
        p = F.softmax(logits, dim=1)
        py = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
        return ((1.0 - py ** self.q) / self.q).mean()


class SCELoss(nn.Module):
    """Wang et al. (2019):  alpha * CE + beta * RCE, with log(0) -> A."""

    def __init__(self, alpha=0.1, beta=1.0, A=-4.0):
        super().__init__()
        self.alpha, self.beta, self.A = alpha, beta, A

    def forward(self, logits, targets, is_noisy=None):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p = F.softmax(logits, dim=1)
        py = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        rce = -self.A * (1.0 - py)          # = -sum_{k!=y} p_k * A
        return (self.alpha * ce + self.beta * rce).mean()


class MAELoss(nn.Module):
    """MAE on the one-hot target, scaled to coincide exactly with GCE at q=1."""

    def forward(self, logits, targets, is_noisy=None):
        p = F.softmax(logits, dim=1)
        py = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (1.0 - py).mean()


class NCEandRCELoss(nn.Module):
    """Ma et al. (2020) Active-Passive Loss: alpha * NCE + beta * RCE."""

    def __init__(self, alpha=1.0, beta=1.0, A=-4.0):
        super().__init__()
        self.alpha, self.beta, self.A = alpha, beta, A

    def forward(self, logits, targets, is_noisy=None):
        logp = F.log_softmax(logits, dim=1)
        nce = logp.gather(1, targets.unsqueeze(1)).squeeze(1) / logp.sum(dim=1).clamp_max(-1e-7)
        p = F.softmax(logits, dim=1)
        py = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        rce = -self.A * (1.0 - py)
        return (self.alpha * nce + self.beta * rce).mean()


class SoftBootstrapLoss(nn.Module):
    """Reed et al. (2015):  target = beta * onehot + (1 - beta) * p(x)."""

    def __init__(self, beta=0.95):
        super().__init__()
        self.beta = float(beta)

    def forward(self, logits, targets, is_noisy=None):
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            p = F.softmax(logits, dim=1)
        onehot = torch.zeros_like(logp).scatter_(1, targets.unsqueeze(1), 1.0)
        tgt = self.beta * onehot + (1.0 - self.beta) * p
        return -(tgt * logp).sum(dim=1).mean()


class DiscardTopKLoss(nn.Module, _EpsTracker):
    """
    Cross-entropy computed only on the samples NOT selected as likely-noisy.

    Within each batch, the top `noise_rate` fraction by disagreement (1 - p_y)
    is dropped entirely and the loss is averaged over the remainder. This is
    the single-network form of the small-loss sample-selection criterion used
    by Co-teaching and DivideMix.

    Why it is here: pushing target-side smoothing to its limit (eps -> uniform
    on selected samples) makes the target carry no label information, which is
    functionally the same intervention as dropping the sample. Including the
    explicit discard baseline turns that from a rhetorical concession into a
    measurement -- we can report how much of the high-budget target-side gain is
    simply sample selection arrived at from the loss-function side.

    The tracked "eps" is the discard mask, so share_to_clean reports the
    fraction of discarded mass that fell on correctly-labelled samples.
    """

    def __init__(self, noise_rate=0.4):
        super().__init__()
        self.noise_rate = float(noise_rate)
        self.reset_stats()

    def forward(self, logits, targets, is_noisy=None):
        ce = F.cross_entropy(logits, targets, reduction="none")
        with torch.no_grad():
            p = F.softmax(logits, dim=1)
            dis = 1.0 - p.gather(1, targets.unsqueeze(1)).squeeze(1)
            B = logits.size(0)
            k = int(round(self.noise_rate * B))
            drop = torch.zeros(B, device=logits.device, dtype=logits.dtype)
            if k > 0:
                drop[torch.topk(dis, k).indices] = 1.0
        self._track(drop, is_noisy)
        keep = 1.0 - drop
        return (ce * keep).sum() / keep.sum().clamp_min(1.0)


class CELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, logits, targets, is_noisy=None):
        w = self.weights.to(logits.device) if self.weights is not None else None
        return F.cross_entropy(logits, targets, weight=w)


# --- registry ----------------------------------------------------------------

def build_loss(name, eps=0.1, gamma=2.0, class_weights=None, noise_rate=0.4,
               hp=None):
    """
    `hp` carries loss-specific hyperparameters so they can be swept and cached:
        gce      -> {"q": float}
        sce      -> {"alpha": float, "beta": float, "A": float}
        nce_rce  -> {"alpha": float, "beta": float, "A": float}
    Anything absent falls back to the published default.
    """
    hp = hp or {}
    """`name` -> criterion. Any new loss must be registered here."""
    n = name.lower()
    if n == "ce":
        return CELoss()
    if n == "weighted_ce":
        return CELoss(weights=class_weights)
    if n == "focal":
        return FocalLoss(gamma=2.0)
    if n == "focal_weighted":
        return FocalLoss(gamma=2.0, weights=class_weights)
    if n in ("ls_static", "fuzzy_static"):
        return StaticFuzzyLoss(epsilon=eps)
    if n == "ls_negative":
        return StaticFuzzyLoss(epsilon=-abs(eps))
    if n == "bootstrap":
        return SoftBootstrapLoss(beta=0.95)
    if n == "gce":
        return GCELoss(q=hp.get("q", 0.7))
    if n == "sce":
        return SCELoss(alpha=hp.get("alpha", 0.1), beta=hp.get("beta", 1.0),
                       A=hp.get("A", -4.0))
    if n == "mae":
        return MAELoss()
    if n == "nce_rce":
        return NCEandRCELoss(alpha=hp.get("alpha", 1.0), beta=hp.get("beta", 1.0),
                             A=hp.get("A", -4.0))

    # --- adaptive target-side family ---
    if n == "afl_v1":                 # v1 behaviour: NOT mean matched
        return AdaptiveFuzzyLoss("entropy", eps, gamma, match_mean=False)
    if n == "afl_v2":
        return AdaptiveFuzzyLoss("disagreement", eps, gamma, match_mean=False)
    if n == "afl_v1_matched":         # the comparison the paper needs
        return AdaptiveFuzzyLoss("entropy", eps, gamma, match_mean=True)
    if n == "afl_v2_matched":
        return AdaptiveFuzzyLoss("disagreement", eps, gamma, match_mean=True)
    if n == "afl_oracle_matched":     # oracle at the same average smoothing
        return AdaptiveFuzzyLoss("oracle", eps, gamma, match_mean=True)
    if n == "afl_oracle_max":         # unconstrained oracle: the ceiling
        return AdaptiveFuzzyLoss("oracle", EPS_CAP, gamma, match_mean=False)
    if n == "afl_v2_topk":            # oracle-style ALLOCATION, model's ranking
        return AdaptiveFuzzyLoss("topk", eps, gamma, match_mean=True,
                                 noise_rate=noise_rate)
    if n == "ce_discard_topk":        # sample selection, for the honest contrast
        return DiscardTopKLoss(noise_rate=noise_rate)

    raise ValueError(f"unknown loss '{name}'")


LOSS_FAMILY = {
    "ce": "unmodified CE",
    "weighted_ce": "unmodified CE",
    "focal": "hard-sample reweighting",
    "focal_weighted": "hard-sample reweighting",
    "ls_static": "target-side (static)",
    "ls_negative": "target-side (static)",
    "bootstrap": "target-side (model-output)",
    "afl_v1": "target-side (model-output)",
    "afl_v2": "target-side (model-output)",
    "afl_v1_matched": "target-side (model-output)",
    "afl_v2_matched": "target-side (model-output)",
    "afl_v2_topk": "target-side (model-output)",
    "ce_discard_topk": "sample selection",
    "afl_oracle_matched": "target-side (oracle)",
    "afl_oracle_max": "target-side (oracle)",
    "gce": "robust loss reformulation",
    "sce": "robust loss reformulation",
    "mae": "robust loss reformulation",
    "nce_rce": "robust loss reformulation",
}


# =============================================================================
# 5. Train / evaluate / diagnose
# =============================================================================

def train_one_epoch(model, criterion, optimizer, loader, is_noisy_tensor):
    model.train()
    if hasattr(criterion, "reset_stats"):
        criterion.reset_stats()

    tot_loss, correct, total = 0.0, 0, 0
    for x, y, idx in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        noisy = is_noisy_tensor[idx].to(DEVICE) if is_noisy_tensor is not None else None

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y, noisy)
        loss.backward()
        optimizer.step()

        tot_loss += float(loss.detach()) * x.size(0)
        correct += int((logits.detach().argmax(1) == y).sum())
        total += x.size(0)

    mean_eps = criterion.epoch_mean_eps() if hasattr(criterion, "epoch_mean_eps") else None
    eps_split = criterion.epoch_eps_split() if hasattr(criterion, "epoch_eps_split") else None
    return tot_loss / total, correct / total, mean_eps, eps_split


@torch.no_grad()
def evaluate(model, loader):
    """Always evaluated on the CLEAN test set."""
    model.eval()
    preds_all, y_all = [], []
    for batch in loader:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        preds_all.append(model(x).argmax(1).cpu())
        y_all.append(y.cpu())
    preds = torch.cat(preds_all).numpy()
    ys = torch.cat(y_all).numpy()
    return float((preds == ys).mean()), float(f1_score(ys, preds, average="macro", zero_division=0))


@torch.no_grad()
def diagnose_signals(model, diag_loader, is_noisy_tensor):
    """
    For a fixed subset of the TRAINING set, compute the three candidate
    fuzziness signals and score how well each separates clean from mislabelled
    samples.

    Returns AUROC for:
      entropy       -- V1's signal
      disagreement  -- V2's signal (1 - p_y)
      loss          -- the small-loss criterion (Co-teaching / DivideMix)

    AUROC ~ 0.5 means the signal carries no information about which labels are
    wrong; that is the direct evidence for the self-referential failure claim.
    An AUROC well above 0.5 for `loss` but not for the others is ALSO an
    important finding -- it would mean the signal is informative and the failure
    is in how the adaptive losses USE it.

    IMPORTANT ANALYTIC NOTE (surfaced by these diagnostics, worth a sentence in
    the paper):  auroc_disagreement and auroc_loss are provably IDENTICAL.
    V2's signal is (1 - p_y) and the small-loss criterion ranks by -log(p_y);
    both are strictly monotone in p_y, so they induce the same ordering of
    samples and therefore the same AUROC. Consequence: V2 is driven by exactly
    the signal that Co-teaching / DivideMix use for sample selection. If V2
    still fails to beat Static LS, the failure cannot be blamed on the signal
    being uninformative -- it must lie in how a continuous target modulation
    uses that ordering. That is a much sharper claim than the current draft
    makes, and it pre-empts the obvious reviewer objection.
    """
    model.eval()
    ent_l, dis_l, loss_l, noisy_l = [], [], [], []

    for x, y, idx in diag_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        K = logits.size(1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1) / math.log(K)
        py = p.gather(1, y.unsqueeze(1)).squeeze(1)
        ce = F.cross_entropy(logits, y, reduction="none")

        ent_l.append(ent.cpu()); dis_l.append((1 - py).cpu())
        loss_l.append(ce.cpu()); noisy_l.append(is_noisy_tensor[idx].cpu())

    ent = torch.cat(ent_l).numpy()
    dis = torch.cat(dis_l).numpy()
    ce = torch.cat(loss_l).numpy()
    noisy = torch.cat(noisy_l).numpy()

    out = {"n_diag": int(len(noisy)), "frac_noisy": float(noisy.mean())}
    if noisy.min() == noisy.max():
        return out  # clean run: AUROC undefined

    for key, sig in (("entropy", ent), ("disagreement", dis), ("loss", ce)):
        out[f"auroc_{key}"] = float(roc_auc_score(noisy, sig))
        out[f"mean_{key}_clean"] = float(sig[noisy == 0].mean())
        out[f"mean_{key}_noisy"] = float(sig[noisy == 1].mean())
    return out


# =============================================================================
# 6. Single-run driver with caching
# =============================================================================

def _load_cache():
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                return json.load(f)
        except Exception as e:
            print(f"[core] WARNING: cache unreadable ({e}); starting fresh.")
    return {}


RUNS = _load_cache()
print(f"[core] cache holds {len(RUNS)} completed runs")


def _save_cache():
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RUNS, f)
    os.replace(tmp, RESULTS_PATH)   # atomic; survives a kill mid-write


def run_key(exp, loss, seed, **kw):
    payload = json.dumps({"exp": exp, "loss": loss, "seed": seed, **kw}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


# --- dry-run planning ---------------------------------------------------------
# Set DRY_RUN = True before a stage to count the work instead of doing it.
DRY_RUN = False
PLANNED = []


def budget_report(sec_per_run=300):
    """
    Print how many runs a stage would launch and what that costs.
    Calibrate sec_per_run from the 'wall_sec' of your first few real runs --
    on a Kaggle T4 with this CNN, 20 epochs on KMNIST is roughly 4-6 minutes.
    """
    if not PLANNED:
        print("[budget] nothing planned (all runs already cached?)")
        return
    by_exp = defaultdict(int)
    for p in PLANNED:
        by_exp[p["exp"]] += 1
    print(f"\n[budget] {len(PLANNED)} runs still to do "
          f"@ {sec_per_run}s -> {len(PLANNED)*sec_per_run/3600:.1f} GPU-hours")
    for e, n in sorted(by_exp.items()):
        print(f"    {e:<16} {n:>4} runs  ~{n*sec_per_run/3600:>5.1f} h")
    print("[budget] Kaggle gives ~30 T4-hours/week. Split stages across "
          "sessions if needed;\n         runs.json makes resuming free.")


def run_single(exp, loss_name, seed, train_ds, test_ds, num_classes,
               model_fn, epochs=20, lr=1e-3, batch_size=128, eval_batch=512,
               eps=0.1, gamma=2.0, class_weights=None, diag_every=0,
               num_workers=2, meta=None, force=False,
               val_noisy_ds=None, val_clean_ds=None, noise_rate=0.4, hp=None):
    """
    Train ONE model with ONE loss under ONE seed. Cached by a hash of the full
    configuration, so re-running a stage after a crash skips completed work.

    Records the whole accuracy/F1 curve, so both final-epoch and best-epoch
    numbers are available at analysis time without retraining.
    """
    meta = meta or {}
    hp = hp or {}
    key = run_key(exp, loss_name, seed, epochs=epochs, lr=lr, bs=batch_size,
                  eps=eps, gamma=gamma,
                  **({"hp": json.dumps(hp, sort_keys=True)} if hp else {}),
                  **meta)
    if key in RUNS and not force:
        r = RUNS[key]
        print(f"  [skip] {exp} | {loss_name} | seed {seed} "
              f"-> acc={r['final_acc']:.4f} (cached)")
        return r

    if DRY_RUN:
        PLANNED.append({"exp": exp, "loss": loss_name, "seed": seed})
        return None

    set_seed(seed)
    g = _loader_generator(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"),
                              generator=g, worker_init_fn=_worker_init, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=eval_batch, shuffle=False,
                             num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))

    is_noisy_tensor = None
    diag_loader = None
    if hasattr(train_ds, "is_noisy"):
        is_noisy_tensor = torch.as_tensor(train_ds.is_noisy, dtype=torch.long)
        if diag_every and DIAG_SUBSET:
            rs = np.random.RandomState(777)
            sub = rs.choice(len(train_ds), size=min(DIAG_SUBSET, len(train_ds)),
                            replace=False)
            diag_loader = DataLoader(torch.utils.data.Subset(train_ds, sub.tolist()),
                                     batch_size=eval_batch, shuffle=False,
                                     num_workers=num_workers)

    val_noisy_loader = val_clean_loader = None
    if val_noisy_ds is not None:
        val_noisy_loader = DataLoader(val_noisy_ds, batch_size=eval_batch,
                                      shuffle=False, num_workers=num_workers)
    if val_clean_ds is not None:
        val_clean_loader = DataLoader(val_clean_ds, batch_size=eval_batch,
                                      shuffle=False, num_workers=num_workers)

    model = model_fn(num_classes).to(DEVICE)
    if loss_name.lower() == "afl_v2_topk":
        # top-k puts eps_target/noise_rate on each selected sample, so the
        # largest achievable budget is noise_rate * EPS_CAP. Past that the
        # clamp silently caps eps and the run is no longer mean-matched.
        max_budget = noise_rate * EPS_CAP
        if eps > max_budget + 1e-9:
            print(f"  [WARN] afl_v2_topk eps_target={eps:.3f} exceeds the "
                  f"achievable budget {max_budget:.3f} (= noise_rate x eps_cap). "
                  f"Mean matching WILL be broken; check mean_eps_avg before "
                  f"using this point.")

    criterion = build_loss(loss_name, eps=eps, gamma=gamma,
                           class_weights=class_weights, noise_rate=noise_rate,
                           hp=hp)
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    curve = {"acc": [], "f1": [], "train_acc": [], "train_loss": [], "mean_eps": [],
             "val_noisy_acc": [], "val_clean_acc": [],
             "eps_clean": [], "eps_noisy": [], "share_to_clean": []}
    diagnostics = []
    t0 = time.time()

    for ep in range(epochs):
        tr_loss, tr_acc, mean_eps, eps_split = train_one_epoch(
            model, criterion, optimizer, train_loader, is_noisy_tensor)
        acc, f1 = evaluate(model, test_loader)

        curve["acc"].append(acc); curve["f1"].append(f1)
        curve["train_acc"].append(tr_acc); curve["train_loss"].append(tr_loss)
        curve["mean_eps"].append(mean_eps)
        curve["eps_clean"].append(eps_split["eps_clean"] if eps_split else None)
        curve["eps_noisy"].append(eps_split["eps_noisy"] if eps_split else None)
        curve["share_to_clean"].append(eps_split["share_to_clean"] if eps_split else None)

        curve["val_noisy_acc"].append(
            evaluate(model, val_noisy_loader)[0] if val_noisy_loader else None)
        curve["val_clean_acc"].append(
            evaluate(model, val_clean_loader)[0] if val_clean_loader else None)

        if diag_loader is not None and ((ep + 1) % diag_every == 0 or ep == 0):
            d = diagnose_signals(model, diag_loader, is_noisy_tensor)
            d["epoch"] = ep + 1
            diagnostics.append(d)

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            eps_s = f" eps={mean_eps:.4f}" if mean_eps is not None else ""
            print(f"    [{loss_name}] ep {ep+1:>2}/{epochs} "
                  f"train_acc={tr_acc:.4f} test_acc={acc:.4f} f1={f1:.4f}{eps_s}")

    # ---- model selection under three protocols -----------------------------
    # final       : train to a fixed budget, take the last epoch (the draft's
    #               protocol). Measures memorisation resistance.
    # sel_noisyval: early stop on a held-out set with the SAME label noise.
    #               This is the realistic protocol -- the only one a
    #               practitioner can actually run -- and it is what the revised
    #               paper should report as its headline.
    # sel_cleanval: early stop on clean labels. ORACLE selection; upper bound.
    # best (test) : peak on the test set. LEAKS -- diagnostic only, never a
    #               headline number.
    def _sel(curve_key):
        v = curve.get(curve_key) or []
        v = [x for x in v if x is not None]
        if not v:
            return None, None, None
        i = int(np.argmax(v))
        return curve["acc"][i], curve["f1"][i], i + 1

    best_i = int(np.argmax(curve["acc"]))
    sel_nv_acc, sel_nv_f1, sel_nv_ep = _sel("val_noisy_acc")
    sel_cv_acc, sel_cv_f1, sel_cv_ep = _sel("val_clean_acc")

    rec = {
        "sel_noisyval_acc": sel_nv_acc, "sel_noisyval_f1": sel_nv_f1,
        "sel_noisyval_epoch": sel_nv_ep,
        "sel_cleanval_acc": sel_cv_acc, "sel_cleanval_f1": sel_cv_f1,
        "sel_cleanval_epoch": sel_cv_ep,
        "share_to_clean_avg": (float(np.mean([x for x in curve["share_to_clean"]
                                              if x is not None]))
                               if any(x is not None for x in curve["share_to_clean"])
                               else None),
        "exp": exp, "loss": loss_name, "seed": seed,
        "family": LOSS_FAMILY.get(loss_name, "?"),
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "eps": eps, "gamma": gamma, "meta": meta, "hp": hp,
        "noise_rate": noise_rate,
        "final_acc": curve["acc"][-1], "final_f1": curve["f1"][-1],
        "best_acc": curve["acc"][best_i], "best_f1": curve["f1"][best_i],
        "best_epoch": best_i + 1,
        "mean_eps_last": curve["mean_eps"][-1],
        "mean_eps_avg": (float(np.mean([e for e in curve["mean_eps"] if e is not None]))
                         if any(e is not None for e in curve["mean_eps"]) else None),
        "curve": curve, "diagnostics": diagnostics,
        "wall_sec": round(time.time() - t0, 1),
    }
    RUNS[key] = rec
    _save_cache()

    sel_s = (f"  sel={rec['sel_noisyval_acc']:.4f}(ep {rec['sel_noisyval_epoch']})"
             if rec["sel_noisyval_acc"] is not None else "")
    shr_s = (f"  clean_share={rec['share_to_clean_avg']:.3f}"
             if rec["share_to_clean_avg"] is not None else "")
    print(f"  [done] {exp} | {loss_name} | seed {seed} -> "
          f"final={rec['final_acc']:.4f}{sel_s}  peak={rec['best_acc']:.4f}"
          f"{shr_s}  {rec['wall_sec']}s")
    return rec


def query(**filters):
    """Fetch cached runs, e.g. query(exp='S1_sym40', loss='ls_static')."""
    out = []
    for r in RUNS.values():
        ok = True
        for k, v in filters.items():
            got = r.get(k, r.get("meta", {}).get(k))
            if got != v:
                ok = False
                break
        if ok:
            out.append(r)
    return sorted(out, key=lambda r: (r["loss"], r["seed"]))
