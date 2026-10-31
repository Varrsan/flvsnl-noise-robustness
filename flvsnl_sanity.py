"""
flvsnl_sanity.py  --  Run this FIRST, before any experiment.

Twelve assertions that catch the failure modes that would silently invalidate a
whole sweep. Every one of these has bitten someone. If any assertion fires,
stop and fix it -- do not start Stage 1.

    %run -i flvsnl_core.py
    %run -i flvsnl_sanity.py
"""

import math
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
_PASS, _FAIL = [], []


def check(name, cond, detail=""):
    (_PASS if cond else _FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


print("\n" + "=" * 72)
print("  SANITY CHECKS")
print("=" * 72)

B, K = 64, 10
logits = torch.randn(B, K)
y = torch.randint(0, K, (B,))
is_noisy = (torch.rand(B) < 0.4).long()
ce_ref = float(F.cross_entropy(logits, y))

# -- 1. eps = 0 must reduce exactly to cross-entropy --------------------------
L = float(StaticFuzzyLoss(0.0)(logits, y))
check("1  StaticFuzzy(eps=0) == CE", abs(L - ce_ref) < 1e-5, f"{L:.6f} vs {ce_ref:.6f}")

# -- 2. static must equal torch's own label smoothing -------------------------
L = float(StaticFuzzyLoss(0.10)(logits, y))
ref = float(F.cross_entropy(logits, y, label_smoothing=0.10))
# torch smooths as (1-e)*onehot + e/K (mass on ALL K); ours puts e/(K-1) on the
# K-1 others. The two differ by a known constant factor on the off-diagonal.
mu = torch.full((B, K), 0.10 / (K - 1)); mu[torch.arange(B), y] = 0.90
manual = float(-(mu * F.log_softmax(logits, 1)).sum(1).mean())
check("2  StaticFuzzy(0.1) == hand-computed soft-target CE",
      abs(L - manual) < 1e-6, f"{L:.6f} vs {manual:.6f} (torch LS={ref:.6f})")

# -- 3. adaptive losses must produce a non-zero gradient ----------------------
for nm, crit in [("afl_v1", build_loss("afl_v1")),
                 ("afl_v2", build_loss("afl_v2")),
                 ("afl_v1_matched", build_loss("afl_v1_matched")),
                 ("afl_oracle_max", build_loss("afl_oracle_max"))]:
    lg = logits.detach().clone().requires_grad_(True)
    crit(lg, y, is_noisy).backward()
    gn = float(lg.grad.norm())
    check(f"3  {nm} grad norm > 0", gn > 1e-6, f"{gn:.4f}")

# -- 4. no gradient may flow through the fuzziness coefficient ----------------
# d loss / d logit_j must equal p_j - mu_j exactly.
lg = logits.detach().clone().requires_grad_(True)
crit = build_loss("afl_v1")
loss = crit(lg, y, is_noisy)
loss.backward()
with torch.no_grad():
    p = F.softmax(lg, 1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1) / math.log(K)
    eps = (0.1 * ent).clamp(0, EPS_CAP)
    mu = (eps / (K - 1)).unsqueeze(1).expand(B, K).contiguous()
    mu = mu.scatter(1, y.unsqueeze(1), (1 - eps).unsqueeze(1))
    expected = (p - mu) / B
err = float((lg.grad - expected).abs().max())
check("4  stop-gradient holds: dL/dz == (p - mu)/B", err < 1e-6, f"max err {err:.2e}")

# -- 5. the confound the paper must rule out ---------------------------------
# Unmatched adaptive applies a strictly SMALLER average eps than static.
crit_u = build_loss("afl_v1"); crit_u(logits, y, is_noisy)
crit_m = build_loss("afl_v1_matched")
for _ in range(60):                      # let the EMA settle
    crit_m(torch.randn(B, K), y, is_noisy)
crit_m.reset_stats(); crit_m(logits, y, is_noisy)
eu, em = crit_u.epoch_mean_eps(), crit_m.epoch_mean_eps()
check("5a unmatched afl_v1 mean eps < 0.10 (THE CONFOUND)", eu < 0.10,
      f"mean eps = {eu:.4f} vs static 0.1000")
check("5b matched afl_v1 mean eps ~= 0.10", abs(em - 0.10) < 0.02,
      f"mean eps = {em:.4f}")

# V2 on CONFIDENT predictions (what the model looks like after a few epochs) is
# where the confound is most extreme: eps = eps_max * (1-p_y)^2 collapses to
# near zero once the model starts fitting the data.
conf = torch.zeros(B, K); conf[torch.arange(B), y] = 4.0   # p_y ~ 0.98
crit_u2 = build_loss("afl_v2"); crit_u2(conf, y, is_noisy)
e2 = crit_u2.epoch_mean_eps()
check("5c unmatched afl_v2 on confident preds: mean eps << 0.10", e2 < 0.01,
      f"mean eps = {e2:.5f} vs static 0.10000 -> {0.10/max(e2,1e-9):.0f}x less smoothing")

# -- 6. oracle must smooth noisy samples and only noisy samples ---------------
crit = build_loss("afl_oracle_max")
s = crit._raw_signal(logits, y, is_noisy)
check("6  oracle signal == is_noisy exactly",
      bool(torch.equal(s.long(), is_noisy)), f"frac noisy {float(is_noisy.float().mean()):.3f}")

# -- 7. known limiting cases of the robust losses -----------------------------
check("7a Bootstrap(beta=1) == CE",
      abs(float(SoftBootstrapLoss(1.0)(logits, y)) - ce_ref) < 1e-5)
check("7b Focal(gamma=0) == CE",
      abs(float(FocalLoss(gamma=0.0)(logits, y)) - ce_ref) < 1e-5)
gce_small = float(GCELoss(q=1e-4)(logits, y))
check("7c GCE(q->0) -> CE", abs(gce_small - ce_ref) < 5e-3,
      f"{gce_small:.5f} vs {ce_ref:.5f}")
check("7d GCE(q=1) == MAE", abs(float(GCELoss(q=1.0)(logits, y)) -
                               float(MAELoss()(logits, y))) < 1e-6)
check("7e GCE bounded by 1/q", float(GCELoss(q=0.7)(logits, y)) <= 1 / 0.7)

# -- 8. SCE / RCE algebra -----------------------------------------------------
p = F.softmax(logits, 1); py = p.gather(1, y.unsqueeze(1)).squeeze(1)
rce_manual = float((4.0 * (1 - py)).mean())
sce = float(SCELoss(alpha=0.0, beta=1.0, A=-4.0)(logits, y))
check("8  SCE(alpha=0) == RCE = -A*(1-p_y)", abs(sce - rce_manual) < 1e-6,
      f"{sce:.6f} vs {rce_manual:.6f}")

# -- 9. NCE is bounded in [0,1] ----------------------------------------------
logp = F.log_softmax(logits, 1)
nce = (logp.gather(1, y.unsqueeze(1)).squeeze(1) / logp.sum(1))
check("9  NCE in [0,1]", bool((nce >= 0).all() and (nce <= 1).all()),
      f"range [{float(nce.min()):.4f}, {float(nce.max()):.4f}]")

# -- 10. noise construction: rate, mask, and never-a-self-flip ----------------
class _Toy(torch.utils.data.Dataset):
    def __init__(self, n=2000, k=10):
        self.targets = np.arange(n) % k
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return torch.zeros(1), int(self.targets[i])

toy = _Toy()
ds = make_noisy_dataset(toy, 0.40, 10, seed=0)
rate_err = abs(ds.is_noisy.mean() - 0.40)
check("10a symmetric noise rate is exact", rate_err < 0.005, f"actual {ds.is_noisy.mean():.4f}")
flipped_ok = bool((ds.noisy_labels[ds.is_noisy == 1] != ds.clean_labels[ds.is_noisy == 1]).all())
check("10b flipped samples really changed class", flipped_ok)
kept_ok = bool((ds.noisy_labels[ds.is_noisy == 0] == ds.clean_labels[ds.is_noisy == 0]).all())
check("10c unflipped samples untouched", kept_ok)

ds_a = make_noisy_dataset(toy, 0.40, 10, seed=0, noise_type="asymmetric")
pair_ok = bool(np.all(ds_a.noisy_labels[ds_a.is_noisy == 1] ==
                      (ds_a.clean_labels[ds_a.is_noisy == 1] + 1) % 10))
check("10d asymmetric noise is a pair flip y -> y+1", pair_ok)

# -- 11. noise is shared within a seed, different across seeds ---------------
same = make_noisy_dataset(toy, 0.40, 10, seed=0)
diff = make_noisy_dataset(toy, 0.40, 10, seed=1)
check("11a same seed -> identical corruption (pairing preserved)",
      bool(np.array_equal(ds.noisy_labels, same.noisy_labels)))
check("11b different seed -> different corruption (honest variance)",
      not bool(np.array_equal(ds.noisy_labels, diff.noisy_labels)))

# -- 12. diagnostics AUROC is well behaved on a synthetic oracle signal -------
from sklearn.metrics import roc_auc_score
rs = np.random.RandomState(0)
lbl = rs.randint(0, 2, 500)
check("12a AUROC of a perfect signal == 1.0",
      abs(roc_auc_score(lbl, lbl.astype(float)) - 1.0) < 1e-9)
check("12b AUROC of pure noise ~= 0.5",
      abs(roc_auc_score(lbl, rs.rand(500)) - 0.5) < 0.10)


# -- 13. top-k allocation: binarised ranking at the known noise rate ---------
crit = build_loss("afl_v2_topk", eps=0.1, noise_rate=0.4)
sig = crit._raw_signal(logits, y, is_noisy)
check("13a topk signal is binary", bool(((sig == 0) | (sig == 1)).all()))
check("13b topk selects exactly noise_rate fraction",
      abs(float(sig.mean()) - 0.4) < 1.0 / B, f"selected {float(sig.mean()):.3f}")
for _ in range(60):
    crit(torch.randn(B, K), y, is_noisy)
crit.reset_stats(); crit(logits, y, is_noisy)
check("13c topk mean eps ~= 0.10", abs(crit.epoch_mean_eps() - 0.10) < 0.02,
      f"mean eps = {crit.epoch_mean_eps():.4f}")

# -- 14. epsilon allocation tracking ----------------------------------------
# The oracle must spend NOTHING on correctly-labelled samples; V2 must spend a
# substantial share on them. This is the quantity Stage 1b is measuring.
crit = build_loss("afl_oracle_matched")
for _ in range(60): crit(torch.randn(B, K), y, is_noisy)
crit.reset_stats(); crit(logits, y, is_noisy)
sp = crit.epoch_eps_split()
check("14a oracle spends 0% of budget on clean samples",
      sp["share_to_clean"] < 1e-9, f"share = {sp['share_to_clean']:.4f}")

crit = build_loss("afl_v2_matched")
for _ in range(60): crit(torch.randn(B, K), y, is_noisy)
crit.reset_stats(); crit(torch.randn(B, K), y, is_noisy)
sp2 = crit.epoch_eps_split()
check("14b V2 spends a nonzero share on clean samples",
      sp2["share_to_clean"] > 0.05, f"share = {sp2['share_to_clean']:.4f}")

crit = build_loss("ls_static")
crit(logits, y, is_noisy); sp3 = crit.epoch_eps_split()
check("14c static share == class balance (no adaptivity)",
      abs(sp3["share_to_clean"] - float((1 - is_noisy).float().mean())) < 1e-6,
      f"share = {sp3['share_to_clean']:.4f}")

# -- 15. train/val split integrity ------------------------------------------
ds = make_noisy_dataset(_Toy(), 0.40, 10, seed=0)
tr, vn, vc = split_train_val(ds, val_frac=0.1, seed=0)
check("15a split sizes sum to the whole", len(tr) + len(vn) == len(ds),
      f"{len(tr)} + {len(vn)} = {len(ds)}")
check("15b no overlap between train and val indices",
      len(np.intersect1d(tr.indices, vn.indices)) == 0)
check("15c val noise rate matches train (stratified)",
      abs(vn.is_noisy.mean() - tr.is_noisy.mean()) < 0.02,
      f"val {vn.is_noisy.mean():.3f} vs train {tr.is_noisy.mean():.3f}")
check("15d clean val really carries true labels",
      bool((vc.noisy_labels == vc.clean_labels).all()) and vc.is_noisy.sum() == 0)
check("15e noisy val really carries corrupted labels",
      int((vn.noisy_labels != vn.clean_labels).sum()) == int(vn.is_noisy.sum()))
check("15f subset returns LOCAL indices (is_noisy lookup stays aligned)",
      tr[0][2] == 0 and tr[5][2] == 5)


# -- 16. budget saturation: the ceiling on top-k's achievable mean eps -------
# top-k puts eps_target/noise_rate on each selected sample, so the largest
# budget it can actually deliver is noise_rate * EPS_CAP. Stage 1c sweeps up to
# that ceiling; asking for more silently breaks mean matching.
max_budget = 0.4 * EPS_CAP
crit = build_loss("afl_v2_topk", eps=max_budget, noise_rate=0.4)
for _ in range(60): crit(torch.randn(B, K), y, is_noisy)
crit.reset_stats(); crit(torch.randn(B, K), y, is_noisy)
check("16a topk reaches its ceiling budget exactly",
      abs(crit.epoch_mean_eps() - max_budget) < 0.02,
      f"target {max_budget:.3f} achieved {crit.epoch_mean_eps():.3f}")

crit = build_loss("afl_v2_topk", eps=0.60, noise_rate=0.4)
for _ in range(60): crit(torch.randn(B, K), y, is_noisy)
crit.reset_stats(); crit(torch.randn(B, K), y, is_noisy)
check("16b beyond the ceiling, topk under-delivers (must be detected)",
      crit.epoch_mean_eps() < 0.60 - 0.05,
      f"asked 0.600 delivered {crit.epoch_mean_eps():.3f}")

# -- 17. oracle_matched at the ceiling budget == oracle_max ------------------
# Both put EPS_CAP on every mislabelled sample and nothing elsewhere. If these
# diverge, the magnitude ladder does not join up with Stage 1b's oracle_max.
# They coincide only when the EMPIRICAL noise fraction equals the nominal rate:
# oracle_matched normalises by what it actually observes, so with 0.406 noisy in
# a batch it puts 0.36/0.406 = 0.887 on each rather than 0.900. That is correct
# behaviour, so the check uses a batch with exactly 40% mislabelled.
exact = torch.zeros(B, dtype=torch.long); exact[: int(0.4 * B)] = 1
a = build_loss("afl_oracle_matched", eps=max_budget)
for _ in range(200): a(torch.randn(B, K), y, exact)
b_ = build_loss("afl_oracle_max")
la = float(a(logits, y, exact)); lb = float(b_(logits, y, exact))
check("17a oracle_matched(eps=noise_rate*cap) == oracle_max (exact rate)",
      abs(la - lb) < 1e-3, f"{la:.6f} vs {lb:.6f}")
a2 = build_loss("afl_oracle_matched", eps=max_budget)
for _ in range(200): a2(torch.randn(B, K), y, is_noisy)
a2.reset_stats(); a2(logits, y, is_noisy)
check("17b oracle_matched normalises by the OBSERVED noise rate",
      abs(a2.epoch_mean_eps() - max_budget) < 0.02,
      f"budget {a2.epoch_mean_eps():.4f} (batch rate "
      f"{float(is_noisy.float().mean()):.3f}, nominal 0.400)")

# -- 18. discard baseline ---------------------------------------------------
crit = build_loss("ce_discard_topk", noise_rate=0.4)
lg = logits.detach().clone().requires_grad_(True)
loss = crit(lg, y, is_noisy); loss.backward()
sp = crit.epoch_eps_split()
check("18a discard drops the noise_rate fraction",
      abs(crit.epoch_mean_eps() - 0.4) < 1.0 / B,
      f"dropped {crit.epoch_mean_eps():.3f}")
check("18b discard still produces gradient", float(lg.grad.norm()) > 1e-6)
# with a perfectly ordered signal, every dropped sample would be mislabelled
crit2 = build_loss("ce_discard_topk", noise_rate=0.4)
perfect = torch.zeros(B, K); perfect[torch.arange(B), y] = 6.0
perfect[is_noisy.bool()] = 0.0            # noisy samples get uniform -> high dis
crit2(perfect, y, is_noisy)
sp2 = crit2.epoch_eps_split()
check("18c on a perfect signal, discards land on mislabelled samples",
      sp2["share_to_clean"] < 0.05, f"clean share {sp2['share_to_clean']:.3f}")


# -- 19. hyperparameter plumbing (Stage 2 tuning depends on this) ------------
# If hp did not reach the loss, every tuning arm would silently train the
# default and the whole sweep would be wasted.
la = float(build_loss("gce", hp={"q": 0.3})(logits, y))
lb = float(build_loss("gce", hp={"q": 0.9})(logits, y))
lc = float(build_loss("gce")(logits, y))
check("19a gce q actually reaches the loss", abs(la - lb) > 1e-4,
      f"q=0.3 -> {la:.4f}, q=0.9 -> {lb:.4f}")
check("19b gce default is still q=0.7",
      abs(lc - float(GCELoss(q=0.7)(logits, y))) < 1e-9)
sa = float(build_loss("sce", hp={"alpha": 1.0, "beta": 1.0})(logits, y))
sb = float(build_loss("sce", hp={"alpha": 0.1, "beta": 1.0})(logits, y))
check("19c sce alpha/beta reach the loss", abs(sa - sb) > 1e-4,
      f"{sa:.4f} vs {sb:.4f}")
na = float(build_loss("nce_rce", hp={"alpha": 10.0, "beta": 1.0})(logits, y))
nb = float(build_loss("nce_rce", hp={"alpha": 1.0, "beta": 1.0})(logits, y))
check("19d nce_rce alpha/beta reach the loss", abs(na - nb) > 1e-4)

# -- 20. hp must change the cache key, or tuning arms overwrite each other ---
k1 = run_key("X", "gce", 0, hp=json.dumps({"q": 0.3}, sort_keys=True))
k2 = run_key("X", "gce", 0, hp=json.dumps({"q": 0.9}, sort_keys=True))
k3 = run_key("X", "gce", 0)
check("20a different hp -> different cache key", k1 != k2)
check("20b hp-free key is distinct from hp keys", k3 != k1 and k3 != k2)

# -- 21. top-k ceiling scales with the noise rate ----------------------------
# In the noise sweep the ceiling is eta * eps_cap, so it MOVES with eta. The
# sweep therefore parameterises by eps-per-selected-sample, not by budget.
for eta in (0.1, 0.3, 0.7):
    ceil_ = eta * EPS_CAP
    c = build_loss("afl_v2_topk", eps=ceil_, noise_rate=eta)
    for _ in range(80): c(torch.randn(B, K), y, is_noisy)
    c.reset_stats(); c(torch.randn(B, K), y, is_noisy)
    check(f"21 topk ceiling at eta={eta} is eta*cap={ceil_:.3f}",
          abs(c.epoch_mean_eps() - ceil_) < 0.02,
          f"delivered {c.epoch_mean_eps():.4f}")

print("-" * 72)
print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
if _FAIL:
    print("  FAILED: " + ", ".join(_FAIL))
    raise SystemExit("Sanity checks failed -- do not start the experiments.")
print("  All checks passed. Safe to run flvsnl_run.py")
print("=" * 72)
