"""
flvsnl_analyze1b.py  --  Analysis for Stage 1b.

Run after flvsnl_stage1b.py. CPU only; works on a laptop given runs.json.

Produces the three tables the revised paper needs:
  Table A  allocation: does concentrating the smoothing budget close the gap
           to the oracle?
  Table B  protocol: how much of the loss-side advantage survives honest
           early stopping?
  Table C  the mechanism chain, per loss: clean share -> accuracy
"""

import os
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METRICS = [("final_acc", "final"),
           ("sel_noisyval_acc", "noisy-val ES"),
           ("sel_cleanval_acc", "clean-val ES"),
           ("best_acc", "test peak")]


def _get(exp, metric, loss, gamma=None):
    out = {}
    for r in RUNS.values():
        if r["exp"] != exp or r["loss"] != loss:
            continue
        if gamma is not None and abs(r.get("gamma", -1) - gamma) > 1e-9:
            continue
        v = r.get(metric)
        if v is not None:
            if r["seed"] in out and out[r["seed"]] != v:
                # Several configurations (eps/tag/hp) share this (exp, loss,
                # seed). Keeping one silently would make the arm an arbitrary
                # pick -- the failure mode that mislabelled Stage 2's Table H.
                print(f"  [OVERWRITE] {exp}/{loss}/seed {r['seed']}: multiple "
                      f"configurations match; showing an ARBITRARY one. "
                      f"Filter on eps/tag/hp.")
            out[r["seed"]] = v
    return np.array([out[s] for s in sorted(out)]) if out else np.array([])


def _pair(a, b):
    """Paired delta a-b with bootstrap CI and t-test. None if underpowered."""
    n = min(len(a), len(b))
    if n < 3:
        return None
    d = a[:n] - b[:n]
    rs = np.random.RandomState(0)
    boots = d[rs.randint(0, n, size=(20000, n))].mean(axis=1)
    p = stats.ttest_rel(a[:n], b[:n]).pvalue if d.std(ddof=1) > 0 else np.nan
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), p


# =============================================================================
# Table A -- the allocation hypothesis
# =============================================================================

def table_allocation():
    print("\n" + "=" * 96)
    print("  TABLE A -- ALLOCATION: does concentrating the budget close the "
          "gap to the oracle?")
    print("=" * 96)

    arms = [("ls_static", "S1b_ref", None, "static, no adaptivity")]
    arms += [("afl_v2_matched", "S1b_ref", 2.0, "V2, gamma=2")]
    arms += [("afl_v2_matched", "S1b_gamma", g, f"V2, gamma={g:g}")
             for g in [4.0, 8.0, 16.0, 32.0]]
    arms += [("afl_v2_topk", "S1b_ref", None, "V2 ranking, oracle-shape alloc"),
             ("afl_oracle_matched", "S1b_ref", None, "ORACLE, matched mean"),
             ("afl_oracle_max", "S1b_ref", None, "ORACLE, unconstrained")]

    base = _get("S1b_ref", "sel_noisyval_acc", "ls_static")
    print(f"{'arm':<34}{'clean share':>13}{'acc (ES)':>11}{'std':>8}"
          f"{'vs LS':>10}{'p':>8}")
    print("-" * 96)
    for loss, exp, g, label in arms:
        acc = _get(exp, "sel_noisyval_acc", loss, g)
        shr = _get(exp, "share_to_clean_avg", loss, g)
        if len(acc) == 0:
            continue
        s = f"{shr.mean():>13.3f}" if len(shr) else f"{'--':>13}"
        r = _pair(acc, base) if len(base) else None
        cmp_s = (f"{r[0]:>+10.4f}{r[3]:>8.3f}" if r else f"{'--':>10}{'--':>8}")
        print(f"{label:<34}{s}{acc.mean():>11.4f}"
              f"{acc.std(ddof=1) if len(acc) > 1 else 0:>8.4f}{cmp_s}")

    print("\n  HOW TO READ THIS")
    print("  The allocation hypothesis predicts accuracy rises as clean share")
    print("  falls, approaching the oracle. If afl_v2_topk (which uses only the")
    print("  model's own ranking, binarised) lands near afl_oracle_matched, the")
    print("  hypothesis is confirmed: ranking was never the problem, allocation")
    print("  was. If topk stays near ls_static despite a clean share close to")
    print("  the oracle's, the hypothesis is WRONG and the mechanism section")
    print("  must be rewritten around something else -- report that honestly.")


# =============================================================================
# Table B -- protocol
# =============================================================================

def table_protocol(exp="S1b_ref"):
    print("\n" + "=" * 96)
    print("  TABLE B -- PROTOCOL: how much of the loss-side advantage survives "
          "early stopping?")
    print("=" * 96)
    losses = sorted({r["loss"] for r in RUNS.values() if r["exp"] == exp})
    order = [L for L in ["ce", "ls_static", "afl_v1_matched", "afl_v2_matched",
                         "afl_v2_topk", "afl_oracle_matched", "afl_oracle_max",
                         "gce", "sce", "nce_rce"] if L in losses]

    print(f"{'loss':<22}" + "".join(f"{lab:>15}" for _, lab in METRICS)
          + f"{'ES epoch':>10}")
    print("-" * 96)
    for L in order:
        row = f"{L:<22}"
        for m, _ in METRICS:
            v = _get(exp, m, L)
            row += f"{v.mean():>15.4f}" if len(v) else f"{'--':>15}"
        ep = _get(exp, "sel_noisyval_epoch", L)
        row += f"{ep.mean():>10.1f}" if len(ep) else f"{'--':>10}"
        print(row)

    print(f"\n{'--- GCE minus LS under each protocol ---':<50}")
    for m, lab in METRICS:
        r = _pair(_get(exp, m, "gce"), _get(exp, m, "ls_static"))
        if r:
            print(f"  {lab:<16} {r[0]:>+8.4f}  CI [{r[1]:>+7.4f},{r[2]:>+7.4f}]"
                  f"  p={r[3]:.4f}")
    print("\n  The abstract's headline number is the 'final' row. If the")
    print("  'noisy-val ES' row is much smaller, the claim must be restated as")
    print("  a claim about fixed-budget training, not about the losses per se.")


# =============================================================================
# Table C -- mechanism chain
# =============================================================================

def table_mechanism(exp="S1b_ref"):
    print("\n" + "=" * 96)
    print("  TABLE C -- MECHANISM CHAIN: epsilon allocation over training")
    print("=" * 96)
    losses = ["ls_static", "afl_v1_matched", "afl_v2_matched", "afl_v2_topk",
              "afl_oracle_matched", "afl_oracle_max"]
    eps_ep = [1, 2, 5, 10, 15, 20]
    print(f"{'loss':<22}{'eps on clean':>14}{'eps on noisy':>14}{'ratio':>8}"
          + "".join(f"{'ep' + str(e):>8}" for e in eps_ep) + "   <- share to clean")
    print("-" * 96)
    for L in losses:
        runs = [r for r in RUNS.values() if r["exp"] == exp and r["loss"] == L]
        if not runs:
            continue
        c = np.mean([np.mean([x for x in r["curve"]["eps_clean"] if x is not None])
                     for r in runs])
        n = np.mean([np.mean([x for x in r["curve"]["eps_noisy"] if x is not None])
                     for r in runs])
        row = f"{L:<22}{c:>14.4f}{n:>14.4f}{(n / c if c > 1e-9 else float('inf')):>8.1f}"
        for e in eps_ep:
            vals = [r["curve"]["share_to_clean"][e - 1] for r in runs
                    if len(r["curve"]["share_to_clean"]) >= e
                    and r["curve"]["share_to_clean"][e - 1] is not None]
            row += f"{np.mean(vals):>8.3f}" if vals else f"{'--':>8}"
        print(row)
    print("\n  The oracle rows are the reference: share_to_clean = 0.000 by")
    print("  construction. Everything else is budget spent on samples that")
    print("  did not need smoothing.")


# =============================================================================
# Figures
# =============================================================================

def plot_allocation(fname="s1b_allocation.png"):
    xs, ys, es, labs = [], [], [], []
    for loss, exp, g, lab in [("afl_v2_matched", "S1b_ref", 2.0, "g=2"),
                              ("afl_v2_matched", "S1b_gamma", 4.0, "g=4"),
                              ("afl_v2_matched", "S1b_gamma", 8.0, "g=8"),
                              ("afl_v2_matched", "S1b_gamma", 16.0, "g=16"),
                              ("afl_v2_matched", "S1b_gamma", 32.0, "g=32"),
                              ("afl_v2_topk", "S1b_ref", None, "top-k"),
                              ("afl_oracle_matched", "S1b_ref", None, "oracle")]:
        a, s = _get(exp, "sel_noisyval_acc", loss, g), _get(exp, "share_to_clean_avg", loss, g)
        if len(a) and len(s):
            xs.append(s.mean()); ys.append(a.mean())
            es.append(a.std(ddof=1) if len(a) > 1 else 0); labs.append(lab)
    if not xs:
        print("  (no allocation runs to plot)")
        return
    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, ys, yerr=es, fmt="o", capsize=3)
    for x, y, l in zip(xs, ys, labs):
        plt.annotate(l, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    b = _get("S1b_ref", "sel_noisyval_acc", "ls_static")
    if len(b):
        plt.axhline(b.mean(), ls="--", color="grey", lw=1, label="static LS")
    plt.xlabel("share of smoothing budget spent on correctly-labelled samples")
    plt.ylabel("test accuracy (early stopped on noisy val)")
    plt.title("Allocation, not ranking, governs target-side adaptivity")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG_DIR, fname); plt.savefig(p, dpi=150); plt.close()
    print(f"  saved {p}")


def plot_protocol(exp="S1b_ref", fname="s1b_protocol.png"):
    losses = [L for L in ["ce", "ls_static", "afl_v2_matched", "afl_v2_topk",
                          "afl_oracle_max", "gce", "sce", "nce_rce"]
              if len(_get(exp, "final_acc", L))]
    if not losses:
        return
    w, x = 0.2, np.arange(len(losses))
    plt.figure(figsize=(9, 5))
    for i, (m, lab) in enumerate(METRICS):
        vals = [_get(exp, m, L).mean() if len(_get(exp, m, L)) else np.nan
                for L in losses]
        plt.bar(x + (i - 1.5) * w, vals, w, label=lab)
    plt.xticks(x, losses, rotation=30, ha="right", fontsize=8)
    plt.ylabel("Clean test accuracy"); plt.ylim(0.6, 1.0)
    plt.title("Loss ranking depends on the model-selection protocol")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG_DIR, fname); plt.savefig(p, dpi=150); plt.close()
    print(f"  saved {p}")


table_allocation()
table_protocol()
table_mechanism()
plot_allocation()
plot_protocol()
print("\n" + "=" * 96)
