"""
flvsnl_analyze2.py  --  Analysis for Stage 2.  (v2: hardened)

WHAT WENT WRONG IN v1, AND WHAT CHANGED
---------------------------------------
v1 resolved the tuned baselines like this:

    gce_t = _by_seed(_find(exp="S2_tuned40", ...)) or _by_seed(_find(exp="S1b_ref", ...))

Part C had not run, so S2_tuned40 was empty, the `or` fell through to the
Stage 1b DEFAULTS, and the table printed them under the label "GCE (tuned)".
The numbers were real; the labels were wrong. Nothing crashed, nothing looked
amiss.

Silent fallback is a CLASS of error, so v2 removes the possibility rather than
patching the one line:

  * resolve() NEVER falls back. Missing data yields None and the row prints
    MISSING instead of a plausible number.
  * resolve() REFUSES heterogeneous arms. If the matched runs differ in eps, tag
    or hp it reports MIXED and returns nothing rather than averaging across
    configurations. (S2_tune holds four q values under one (exp, loss) pair --
    exactly the shape that would silently average.)
  * Every table ends with a PROVENANCE block naming the exact eps / tag / hp /
    seed count behind each label, so a mislabelled row is visible.
  * per-seed budget selection CHECKS that every seed had the same candidate
    grid and refuses otherwise. (Part C originally ran only the best budget for
    seeds 5-9, which would have let seeds 0-4 choose from six budgets and seeds
    5-9 from one -- a second bug found in the same audit.)
  * audit_cache() reports any (exp, loss) group spanning multiple
    configurations, so the hazard is findable before it reaches a table.
"""

import os
import json
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

M = "sel_noisyval_acc"      # honest protocol: early stop on noisy validation
PROVENANCE = []

try:
    EPS_CAP
except NameError:      # analysis run without flvsnl_core.py loaded
    EPS_CAP = 0.9


# =============================================================================
# Strict resolution
# =============================================================================

def _val(r):
    """Peak validation accuracy -- the only legitimate selection criterion."""
    v = [x for x in (r["curve"].get("val_noisy_acc") or []) if x is not None]
    return max(v) if v else None


def _config(r):
    return (round(r.get("eps", -1), 6),
            r.get("meta", {}).get("tag", ""),
            json.dumps(r.get("hp") or {}, sort_keys=True))


def _match(exp=None, loss=None, tag=None, eps=None, noise=None):
    out = []
    for r in RUNS.values():
        if exp is not None and r["exp"] != exp:
            continue
        if loss is not None and r["loss"] != loss:
            continue
        if tag is not None and (r.get("meta", {}).get("tag") or "") != tag:
            # Stage 1/1b/1c runs carry NO 'tag' key while Stage 2 runs carry
            # tag=''. Without this normalisation, filtering tag="" would
            # silently drop every pre-Stage-2 run -- 200 of them in the current
            # cache. _config() already normalises the same way.
            continue
        if eps is not None and abs(r.get("eps", -1) - eps) > 1e-9:
            continue
        if noise is not None and abs(r.get("meta", {}).get("noise", -1) - noise) > 1e-9:
            continue
        if r.get(M) is not None:
            out.append(r)
    return out


def resolve(label, quiet=False, **filters):
    """
    Return {seed: run} for exactly one arm, or None.

    Never falls back to another configuration; never averages across
    configurations. Records what it found in PROVENANCE either way.
    """
    runs = _match(**filters)
    if not runs:
        if label:
            PROVENANCE.append((label, "MISSING", 0))
        return None
    cfgs = {_config(r) for r in runs}
    if len(cfgs) > 1:
        if label:
            PROVENANCE.append((label, f"MIXED across {len(cfgs)} configs", len(runs)))
        if not quiet:
            print(f"  [REFUSED] '{label}' matches {len(cfgs)} configurations; "
                  f"not averaging. Tighten the filters.")
            for c in sorted(cfgs):
                print(f"            eps={c[0]} tag={c[1]!r} hp={c[2]}")
        return None
    c = cfgs.pop()
    by_seed = {r["seed"]: r for r in runs}
    if label:
        PROVENANCE.append((label, f"eps={c[0]} tag={c[1]!r} hp={c[2]}", len(by_seed)))
    return by_seed


def resolve_union(label, exps, quiet=False, **filters):
    """
    Gather ONE arm that is split across several experiments (e.g. seeds 0-4 in
    S1b_ref and seeds 5-9 in S1c_budget) and verify the pieces share a
    configuration.

    This exists because `resolve(A) or resolve(B)` returns only the first
    non-empty experiment and silently halves n -- the same fallback bug that
    mislabelled Table H, reappearing in a different disguise.
    """
    merged, cfgs = {}, set()
    for e in exps:
        for r in _match(exp=e, **filters):
            cfgs.add(_config(r))
            merged[r["seed"]] = r
    if not merged:
        if label:
            PROVENANCE.append((label, "MISSING", 0))
        return None
    if len(cfgs) > 1:
        if label:
            PROVENANCE.append((label, f"MIXED across {len(cfgs)} configs", len(merged)))
        if not quiet:
            print(f"  [REFUSED] '{label}' spans {len(cfgs)} configurations "
                  f"across {exps}; not merging.")
        return None
    c = cfgs.pop()
    if label:
        PROVENANCE.append((label, f"eps={c[0]} tag={c[1]!r} hp={c[2]} "
                                  f"(union of {len(exps)} exps)", len(merged)))
    return merged


def vals(arm):
    return np.array([arm[s][M] for s in sorted(arm)]) if arm else np.array([])


def paired(a, b):
    if not a or not b:
        return None
    ks = sorted(set(a) & set(b))
    if len(ks) < 3:
        return None
    x = np.array([a[k][M] for k in ks])
    y = np.array([b[k][M] for k in ks])
    d = x - y
    p = stats.ttest_rel(x, y).pvalue if d.std(ddof=1) > 0 else np.nan
    return d.mean(), p, int((d > 0).sum()), len(ks)


def print_provenance():
    if not PROVENANCE:
        return
    print("\n  --- PROVENANCE (check every label against its configuration) ---")
    for lab, cfg, n in PROVENANCE:
        flag = "   <<< CHECK" if cfg.startswith(("MISSING", "MIXED", "INCONSIST")) else ""
        print(f"    {lab:<30} n={n:<3} {cfg}{flag}")
    PROVENANCE.clear()


# =============================================================================
# Cache audit
# =============================================================================

def audit_cache():
    print("\n" + "=" * 92)
    print("  CACHE AUDIT")
    print("=" * 92)
    groups = {}
    for r in RUNS.values():
        groups.setdefault((r["exp"], r["loss"]), {}) \
              .setdefault(_config(r), set()).add(r["seed"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"  {len(RUNS)} runs | {len(groups)} (exp, loss) groups | "
          f"{len(multi)} span multiple configurations")
    if multi:
        print("\n  These need a tag/eps/hp filter to address a single arm.")
        print("  Expected for sweeps; a hazard only if a table filters on")
        print("  (exp, loss) alone -- resolve() now refuses to, by construction.")
        for (e, l), cfgs in sorted(multi.items()):
            print(f"    {e:<16}{l:<22}{len(cfgs)} configs")
    seeds = {}
    for r in RUNS.values():
        seeds.setdefault(r["exp"], set()).add(r["seed"])
    print("\n  seeds per experiment:")
    for e in sorted(seeds):
        print(f"    {e:<16}{sorted(seeds[e])}")


# =============================================================================
# Table G -- tuning
# =============================================================================

def table_tuning():
    runs = _match(exp="S2_tune")
    if not runs:
        print("\n  (Part A has not been run)")
        return
    print("\n" + "=" * 92)
    print("  TABLE G -- TUNING THE LOSS-SIDE BASELINES (40% noise)")
    print("=" * 92)
    print("  Selected on VALIDATION accuracy; the test column is information")
    print("  only. Noisy-val accuracy is ceilinged near (1 - noise rate), so its")
    print("  dynamic range is compressed. The right-hand column reports whether")
    print("  each setting is actually SEPARABLE from the winner on validation --")
    print("  if not, the selection is a coin flip and must be reported as one.")
    print(f"\n{'loss':<9}{'setting':<14}{'val':>9}{'test':>9}{'std':>8}{'n':>4}"
          f"{'   val gap vs winner':>24}")
    print("-" * 92)
    winners = {}
    for loss in ["gce", "sce", "nce_rce"]:
        tags = sorted({r["meta"]["tag"] for r in runs if r["loss"] == loss})
        arms = {}
        for t in tags:
            a = resolve(f"{loss}/{t}", exp="S2_tune", loss=loss, tag=t)
            if a:
                arms[t] = a
        if not arms:
            continue
        best = max(arms, key=lambda t: np.mean([_val(r) for r in arms[t].values()]))
        winners[loss] = (best, arms[best][sorted(arms[best])[0]].get("hp"))
        n_sep = 0
        for t in tags:
            if t not in arms:
                continue
            va = np.array([_val(arms[t][s]) for s in sorted(arms[t])])
            te = vals(arms[t])
            if t == best:
                cmp_s = "(winner)"
            else:
                ks = sorted(set(arms[best]) & set(arms[t]))
                bw = np.array([_val(arms[best][k]) for k in ks])
                ot = np.array([_val(arms[t][k]) for k in ks])
                d = bw - ot
                p = stats.ttest_rel(bw, ot).pvalue if d.std(ddof=1) > 0 else np.nan
                if p < 0.05:
                    n_sep += 1
                cmp_s = f"{d.mean():+.4f} p={p:.3f} {int((d > 0).sum())}/{len(ks)}"
            print(f"{loss:<9}{t:<14}{va.mean():>9.4f}{te.mean():>9.4f}"
                  f"{te.std(ddof=1) if len(te) > 1 else 0:>8.4f}{len(te):>4}"
                  f"{cmp_s:>24}")
        tt = {t: vals(arms[t]).mean() for t in arms}
        bt = max(tt, key=lambda t: tt[t])
        print(f"{'':<9}-> val picks {best}; test-best is {bt}; "
              f"selection cost {tt[bt] - tt[best]:+.4f}; "
              f"separable from {n_sep}/{len(arms) - 1} rivals")
        print()
    print("  WINNERS (paste into TUNED in flvsnl_stage2.py):")
    for loss, (t, hp) in winners.items():
        print(f'    "{loss}": {hp},        # {t}')
    print("\n  Use the VAL-selected winner even where test-best differs --")
    print("  selecting on test is the practice this paper criticises.")
    print("  If a winner sits at a GRID EDGE, the grid was too narrow: say so")
    print("  rather than adopting an edge value silently.")
    print_provenance()


# =============================================================================
# Table H -- headline
# =============================================================================

def table_selection_regret():
    """
    TABLE K -- how unstable is honest hyperparameter selection?

    Table G tunes once on the pooled validation set across all seeds. A single
    practitioner cannot do that: they have one training run and one noisy
    validation set. This table simulates that situation -- each seed picks its
    own setting using only its own validation accuracy -- and reports the REGRET
    against the setting that was actually best on test.

    The point is not that the regret is large in absolute terms. It is that the
    regret is comparable to the DIFFERENCES BETWEEN LOSS FUNCTIONS that this
    literature reports. If choosing a hyperparameter honestly costs as much as
    switching loss families gains, then reported effect sizes in this area carry
    an unreported error bar of the same magnitude.
    """
    runs = _match(exp="S2_tune")
    if not runs:
        print("\n  (Part A has not been run; no regret analysis)")
        return
    print("\n" + "=" * 92)
    print("  TABLE K -- SELECTION REGRET: what honest per-seed tuning costs")
    print("=" * 92)
    print(f"{'loss':<10}{'settings picked per seed':<44}{'mean':>9}{'worst':>9}"
          f"{'spread':>9}")
    print("-" * 92)
    summary = {}
    for loss in ["gce", "sce", "nce_rce"]:
        tags = sorted({r["meta"]["tag"] for r in runs if r["loss"] == loss})
        arms = {}
        for t in tags:
            a = resolve(f"K/{loss}/{t}", quiet=True, exp="S2_tune",
                        loss=loss, tag=t)
            if a:
                arms[t] = a
        if not arms:
            continue
        te = {t: vals(arms[t]).mean() for t in arms}
        best_t = max(te, key=lambda t: te[t])
        seeds = sorted(set.intersection(*[set(a) for a in arms.values()]))
        picks, regrets = [], []
        for s in seeds:
            cand = {t: _val(arms[t][s]) for t in arms if s in arms[t]}
            p = max(cand, key=lambda t: cand[t])
            picks.append(p)
            regrets.append(te[best_t] - te[p])
        n_distinct = len(set(picks))
        spread = te[best_t] - min(te.values())
        summary[loss] = (float(np.mean(regrets)), float(max(regrets)), spread)
        shown = ", ".join(picks)
        if len(shown) > 42:
            shown = shown[:39] + "..."
        print(f"{loss:<10}{shown:<44}{np.mean(regrets):>+9.4f}"
              f"{max(regrets):>+9.4f}{spread:>9.4f}")
        print(f"{'':<10}{n_distinct} distinct settings chosen across "
              f"{len(seeds)} seeds; test-best was {best_t}")
    PROVENANCE.clear()

    # --- the comparison that makes the point -------------------------------
    print("\n  --- selection noise vs the effects this paper reports ---")
    ls, _b = per_seed_pick("", "ls_static", [0.10, 0.20, 0.30, 0.36, 0.50, 0.70])
    tk, _b2 = per_seed_pick("", "afl_v2_topk", [0.10, 0.20, 0.30, 0.36])
    PROVENANCE.clear()
    effects = []
    r = paired(tk, ls)
    if r:
        effects.append(("top-k over Label Smoothing", r[0]))
    gce_t = resolve("", quiet=True, exp="S2_tuned40", loss="gce", tag="tuned")
    if gce_t is None:
        # fall back is NOT allowed for a reported arm, but for this diagnostic
        # we may read the tuning experiment directly -- labelled as such.
        tags = sorted({x["meta"]["tag"] for x in runs if x["loss"] == "gce"})
        cand = {t: resolve("", quiet=True, exp="S2_tune", loss="gce", tag=t)
                for t in tags}
        cand = {t: a for t, a in cand.items() if a}
        if cand:
            vsel = max(cand, key=lambda t: np.mean([_val(x) for x in cand[t].values()]))
            gce_t = cand[vsel]
            print(f"      (GCE arm read from S2_tune/{vsel}; Part C not yet run)")
    r = paired(gce_t, ls)
    if r:
        effects.append(("GCE over Label Smoothing", r[0]))
    PROVENANCE.clear()

    for lab, eff in effects:
        print(f"    effect: {lab:<36}{eff:>+9.4f}")
    for loss, (mean_r, worst_r, _s) in summary.items():
        print(f"    regret: {loss + ' mean / worst':<36}"
              f"{mean_r:>+9.4f} / {worst_r:+.4f}")
    if effects and summary:
        biggest_eff = max(abs(e) for _l, e in effects)
        worst_regret = max(w for _m, w, _s in summary.values())
        print()
        if worst_regret >= biggest_eff:
            print("    -> Worst-case selection regret EXCEEDS the largest effect")
            print("       reported here. State this in the paper: an honest")
            print("       error bar on any single-run comparison in this")
            print("       literature is at least this wide.")
        else:
            print("    -> Selection regret is smaller than the reported effects;")
            print("       the effects survive, but quote the regret as the")
            print("       implied error bar.")
    print("\n  CAVEAT to state in the paper: Table G tunes once on POOLED")
    print("  validation across seeds, which a single practitioner cannot do.")
    print("  Where the pooled winner is chosen by only a minority of seeds,")
    print("  say so -- it means the pooled result is not reproducible from one")
    print("  run, which is the situation an applied reader is actually in.")


def per_seed_pick(label, loss, budgets, exps=("S1b_ref", "S1c_budget")):
    """
    Choose each seed's budget on ITS OWN validation accuracy.

    Refuses if seeds had different candidate grids -- otherwise a six-way choice
    for some seeds would be compared against a one-way choice for others.
    """
    avail, picked, chosen = {}, {}, {}
    for e in exps:
        for b in budgets:
            for r in _match(exp=e, loss=loss, eps=b):
                avail.setdefault(r["seed"], set()).add(b)
    if not avail:
        PROVENANCE.append((label, "MISSING", 0))
        return None, None
    grids = {frozenset(v) for v in avail.values()}
    if len(grids) > 1:
        print(f"  [REFUSED] '{label}': seeds had DIFFERENT candidate grids.")
        for s in sorted(avail):
            print(f"            seed {s}: {sorted(avail[s])}")
        print("            Run the full budget grid for every seed.")
        PROVENANCE.append((label, "INCONSISTENT GRIDS across seeds", 0))
        return None, None
    grid = sorted(grids.pop())
    for s in sorted(avail):
        best, bv = None, -1
        for e in exps:
            for b in budgets:
                rs = [r for r in _match(exp=e, loss=loss, eps=b) if r["seed"] == s]
                if rs and _val(rs[0]) is not None and _val(rs[0]) > bv:
                    bv, best, chosen[s] = _val(rs[0]), rs[0], b
        if best:
            picked[s] = best
    PROVENANCE.append((label, f"per-seed choice from grid {grid}", len(picked)))
    return picked, chosen


def table_headline():
    print("\n" + "=" * 92)
    print("  TABLE H -- HEADLINE AT 40% NOISE, BOTH FAMILIES TUNED")
    print("=" * 92)

    ls, ls_b = per_seed_pick("LS (per-seed budget)", "ls_static",
                             [0.10, 0.20, 0.30, 0.36, 0.50, 0.70])
    tk, tk_b = per_seed_pick("top-k (per-seed budget)", "afl_v2_topk",
                             [0.10, 0.20, 0.30, 0.36])
    ce = resolve("cross-entropy", exp="S1b_ref", loss="ce")
    orc = resolve("oracle b=0.36", exp="S1c_budget",
                  loss="afl_oracle_matched", eps=0.36)

    # Tuned arms come ONLY from the tuned experiments. No fallback: if Part C
    # has not run these print MISSING, which is the correct visible outcome and
    # exactly the bug that produced v1's mislabelled table.
    gce_t = resolve("GCE (tuned)", exp="S2_tuned40", loss="gce", tag="tuned")
    sce_t = resolve("SCE (tuned)", exp="S2_tuned40", loss="sce", tag="tuned")
    gce_d = resolve("GCE (default q=0.7)", exp="S1b_ref", loss="gce")
    sce_d = resolve("SCE (default a0.1b1.0)", exp="S1b_ref", loss="sce")

    rows = [("cross-entropy", ce), ("LS (per-seed budget)", ls),
            ("top-k (per-seed budget)", tk), ("oracle b=0.36", orc),
            ("GCE (tuned)", gce_t), ("GCE (default)", gce_d),
            ("SCE (tuned)", sce_t), ("SCE (default)", sce_d)]
    print(f"\n{'arm':<30}{'acc':>9}{'std':>8}{'n':>4}   budgets chosen")
    print("-" * 92)
    for lab, d in rows:
        if not d:
            print(f"{lab:<30}{'MISSING':>9}")
            continue
        v = vals(d)
        note = ""
        if d is ls and ls_b:
            note = "  " + str(sorted(ls_b.values()))
        if d is tk and tk_b:
            note = "  " + str(sorted(tk_b.values()))
        print(f"{lab:<30}{v.mean():>9.4f}"
              f"{v.std(ddof=1) if len(v) > 1 else 0:>8.4f}{len(v):>4}{note}")

    print("\n--- key paired comparisons ---")
    for lab, a, b in [("top-k - LS", tk, ls), ("oracle - LS", orc, ls),
                      ("SCE(tuned) - oracle", sce_t, orc),
                      ("SCE(default) - oracle", sce_d, orc),
                      ("SCE(tuned) - top-k", sce_t, tk),
                      ("GCE(tuned) - LS", gce_t, ls),
                      ("SCE(tuned) - GCE(tuned)", sce_t, gce_t),
                      ("GCE tuned - default", gce_t, gce_d),
                      ("SCE tuned - default", sce_t, sce_d)]:
        r = paired(a, b)
        print(f"  {lab:<36}{r[0]:>+9.4f}  p={r[1]:.4f}  signs={r[2]}/{r[3]}"
              if r else f"  {lab:<36}{'unavailable':>9}")
    print_provenance()


# =============================================================================
# Tables I / J -- sweep and prediction
# =============================================================================

def _best_over(label, loss, variants, exp):
    """Validation-best variant of one arm at one noise rate."""
    best, bv, arm = None, -1, None
    for kind, key in variants:
        a = resolve(label and f"{label}/{key}", quiet=True, exp=exp, loss=loss,
                    **({"eps": key} if kind == "eps" else {"tag": key}))
        if not a:
            continue
        v = np.mean([_val(r) for r in a.values()])
        if v > bv:
            bv, best, arm = v, key, a
    return best, arm


def table_ceiling_decomposition():
    """
    TABLE M -- the ceiling decomposition at 40% noise.

    Stage 2 at n=10 found that top-k's advantage over freely-tuned Label
    Smoothing is not significant (+0.0025, p=0.18), REVERSING the n=5 result
    (+0.0053, p=0.008). But at MATCHED budget top-k still wins at three of four
    budgets. Both are true, and the reconciliation is the mechanism:

      targeting is more efficient PER UNIT of smoothing budget,
      but is structurally capped at b <= noise_rate * eps_cap,
      while static smoothing has no ceiling and can simply spend more.

    The two effects cancel. This table shows all three quantities together so
    the paper states the result correctly rather than quoting whichever half
    supports the preferred story.
    """
    print("\n" + "=" * 92)
    print("  TABLE M -- CEILING DECOMPOSITION AT 40% NOISE")
    print("=" * 92)

    def budget_arm(loss, b):
        # UNION, not fallback: seeds 0-4 may sit in S1b_ref and seeds 5-9 in
        # S1c_budget for the very same configuration.
        return resolve_union(f"M/{loss}@{b}", ("S1b_ref", "S1c_budget"),
                             quiet=True, loss=loss, eps=b)

    print("  (a) targeting at MATCHED budget -- top-k minus static")
    print(f"{'budget':<10}{'delta':>10}{'p':>9}{'signs':>9}")
    print("-" * 92)
    for b in [0.10, 0.20, 0.30, 0.36]:
        tk = budget_arm("afl_v2_topk", b)
        st = budget_arm("ls_static", b)
        r = paired(tk, st)
        if r:
            print(f"{b:<10.2f}{r[0]:>+10.4f}{r[1]:>9.4f}{str(r[2]) + '/' + str(r[3]):>9}")

    ceiling = 0.40 * EPS_CAP
    print(f"\n  (b) the ceiling: targeted arms cannot exceed b = "
          f"noise x eps_cap = {ceiling:.2f}")

    def pick(loss, budgets):
        out = {}
        for e in ("S1b_ref", "S1c_budget"):
            for b in budgets:
                for r in _match(exp=e, loss=loss, eps=b):
                    s = r["seed"]
                    v = _val(r)
                    if v is None:
                        continue
                    if s not in out or v > out[s][0]:
                        out[s] = (v, r)
        return {s: r for s, (_v, r) in out.items()}

    free = [0.10, 0.20, 0.30, 0.36, 0.50, 0.70]
    capped = [b for b in free if b <= ceiling + 1e-9]
    ls_free = pick("ls_static", free)
    ls_cap = pick("ls_static", capped)
    tk_free = pick("afl_v2_topk", [0.10, 0.20, 0.30, 0.36])
    chosen = {}
    for s, r in ls_free.items():
        chosen[s] = r["eps"]
    above = sum(1 for b in chosen.values() if b > ceiling + 1e-9)

    print(f"{'comparison':<52}{'delta':>10}{'p':>9}{'signs':>9}")
    print("-" * 92)
    for lab, a, b in [
            ("top-k (free) - LS (free, may exceed ceiling)", tk_free, ls_free),
            ("top-k (free) - LS (capped at the same ceiling)", tk_free, ls_cap),
            ("LS (free) - LS (capped) = value of the extra budget", ls_free, ls_cap)]:
        r = paired(a, b)
        if r:
            print(f"{lab:<52}{r[0]:>+10.4f}{r[1]:>9.4f}"
                  f"{str(r[2]) + '/' + str(r[3]):>9}")
    if chosen:
        print(f"\n  LS per-seed budgets: {sorted(chosen.values())}")
        print(f"  seeds choosing ABOVE the targeted ceiling: {above}/{len(chosen)}")
    print("\n  HOW TO STATE THIS: targeting is more efficient per unit of")
    print("  budget, but is capped at noise x eps_cap because it only touches")
    print("  the selected fraction. Static smoothing has no ceiling. Under free")
    print("  tuning the two effects cancel. Do not quote either half alone.")
    PROVENANCE.clear()


def table_budget_matched_sweep():
    """
    TABLE L -- the noise sweep with every arm at the SAME total budget.

    Table I lets static LS pick from a fixed grid while the targeted arms are
    capped at noise * eps_cap. At 10% noise that is LS at b = 0.5 against an
    oracle capped at b = 0.09 -- a 5.5x mismatch that makes "LS beats the
    oracle" meaningless. Here static LS is run at b = noise * eps_sel (tagged
    'matchedX' by Stage 2b), so each cell compares allocation at equal spend.
    """
    rates = [0.10, 0.30, 0.50, 0.70]
    any_matched = any(_match(exp=f"S2_sweep{int(r * 100)}", loss="ls_static",
                             tag=f"matched{e}")
                      for r in rates for e in [0.5, 0.9])
    if not any_matched:
        print("\n  (TABLE L needs Stage 2b Part B: budget-matched static arms)")
        return
    print("\n" + "=" * 92)
    print("  TABLE L -- BUDGET-MATCHED SWEEP (every arm at b = noise x eps_sel)")
    print("=" * 92)
    for es in [0.5, 0.9]:
        print(f"\n  eps_sel = {es}   (b = noise x {es})")
        print(f"{'noise':<8}{'b':>8}{'static':>10}{'top-k':>10}{'oracle':>10}"
              f"{'topk-static':>14}{'p':>9}{'n':>5}")
        print("-" * 92)
        for rate in rates:
            exp = f"S2_sweep{int(rate * 100)}"
            st = resolve("", quiet=True, exp=exp, loss="ls_static",
                         tag=f"matched{es}")
            tk = resolve("", quiet=True, exp=exp, loss="afl_v2_topk",
                         tag=f"epssel{es}")
            orc = (resolve("", quiet=True, exp=exp, loss="afl_oracle_matched",
                           tag=f"epssel{es}") if es == 0.9 else None)
            row = f"{rate:<8.2f}{rate * es:>8.3f}"
            row += f"{vals(st).mean():>10.4f}" if st else f"{'--':>10}"
            row += f"{vals(tk).mean():>10.4f}" if tk else f"{'--':>10}"
            row += f"{vals(orc).mean():>10.4f}" if orc else f"{'--':>10}"
            # Do not take the budget match on trust: confirm the two arms
            # actually spent the same mean epsilon before differencing them.
            warn = ""
            if st and tk:
                bs = np.mean([x.get("mean_eps_avg") or x["eps"] for x in st.values()])
                bt = np.mean([x.get("mean_eps_avg") or x["eps"] for x in tk.values()])
                if abs(bs - bt) > 0.02:
                    warn = f"  <<< NOT MATCHED ({bs:.3f} vs {bt:.3f})"
            r = paired(tk, st)
            row += (f"{r[0]:>+14.4f}{r[1]:>9.4f}{r[3]:>5}" if r
                    else f"{'--':>14}{'--':>9}{'--':>5}")
            print(row + warn)
    print("\n  This is the comparison Table I cannot make. Any claim about")
    print("  targeting vs static across noise rates should cite THIS table.")
    PROVENANCE.clear()


def table_sweep():
    rates = [0.10, 0.30, 0.50, 0.70]
    exps = [(f"S2_sweep{int(r * 100)}", r) for r in rates]
    if not any(_match(exp=e) for e, _ in exps):
        print("\n  (Part B has not been run)")
        return

    print("\n" + "=" * 92)
    print("  TABLE I -- NOISE SWEEP (each arm at its validation-best setting)")
    print("=" * 92)
    print("  WARNING: the targeted arms (top-k, oracle) are capped at")
    print("  b = noise x eps_cap, while static LS is swept on a fixed grid and")
    print("  is NOT capped. Where the 'LS b' column exceeds the ceiling, LS is")
    print("  spending more than the targeted arms structurally can, and the")
    print("  comparison in that row is NOT about allocation. Use TABLE L for")
    print("  any targeting-vs-static claim across noise rates.")
    print(f"\n{'noise':<8}{'CE':>9}{'LS':>9}{'top-k':>9}{'oracle':>9}"
          f"{'GCE':>9}{'SCE':>9}{'ceiling':>9}   LS b / eps_sel")
    print("-" * 92)
    for exp, rate in exps:
        row, notes = f"{rate:<8.2f}", []
        a = resolve("", quiet=True, exp=exp, loss="ce")
        row += f"{vals(a).mean():>9.4f}" if a else f"{'--':>9}"
        for loss, variants, lab in [
                # tag="" pins the free-grid LS runs; without it a budget-matched
                # run at the same eps would make the arm MIXED and drop the row.
                ("ls_static", [("eps", b) for b in [0.10, 0.30, 0.50]], "LS"),
                ("afl_v2_topk", [("tag", f"epssel{e}") for e in [0.25, 0.5, 0.9]], "topk"),
                ("afl_oracle_matched", [("tag", "epssel0.9")], "oracle")]:
            k, arm = _best_over("", loss, variants, exp)
            row += f"{vals(arm).mean():>9.4f}" if arm else f"{'--':>9}"
            if lab in ("LS", "topk"):
                notes.append(str(k))
        for loss in ["gce", "sce"]:
            a = resolve("", quiet=True, exp=exp, loss=loss, tag="tuned")
            row += f"{vals(a).mean():>9.4f}" if a else f"{'--':>9}"
        ceil_ = rate * EPS_CAP
        row += f"{ceil_:>9.3f}"
        ls_b = notes[0] if notes else None
        flag = ""
        try:
            if ls_b is not None and float(ls_b) > ceil_ + 1e-9:
                flag = "  <<< LS EXCEEDS CEILING"
        except (TypeError, ValueError):
            pass
        print(row + "   " + " / ".join(str(n) for n in notes) + flag)

    print("\n" + "=" * 92)
    print("  TABLE J -- PREDICTION: does top-k's affordable intervention")
    print("             strength FALL as the noise rate rises?")
    print("=" * 92)
    print(f"{'noise':<8}" + "".join(f"{'@' + str(e):>12}" for e in [0.25, 0.5, 0.9])
          + f"{'val-best':>12}{'oracle':>12}")
    print("-" * 92)
    trend = []
    for exp, rate in exps:
        row, best, bv = f"{rate:<8.2f}", None, -1
        for e in [0.25, 0.5, 0.9]:
            a = resolve("", quiet=True, exp=exp, loss="afl_v2_topk", tag=f"epssel{e}")
            if a:
                row += f"{vals(a).mean():>12.4f}"
                v = np.mean([_val(r) for r in a.values()])
                if v > bv:
                    bv, best = v, e
            else:
                row += f"{'--':>12}"
        o = resolve("", quiet=True, exp=exp, loss="afl_oracle_matched", tag="epssel0.9")
        row += f"{best if best else '--':>12}"
        row += f"{vals(o).mean():>12.4f}" if o else f"{'--':>12}"
        print(row)
        if best:
            trend.append((rate, best))
    if len(trend) >= 3:
        r = stats.pearsonr([t[0] for t in trend], [t[1] for t in trend])
        print(f"\n  corr(noise rate, val-best eps_sel) = {r[0]:+.3f}  p={r[1]:.3f}")
        print(f"  Prediction is a NEGATIVE correlation, but this is only "
              f"{len(trend)} points --")
        print("  treat the direction as indicative and lean on the per-rate")
        print("  columns. Flat or positive means the mechanism story needs")
        print("  weakening; report that outcome if it is what you get.")
    PROVENANCE.clear()


def plot_sweep(fname="s2_noise_sweep.png"):
    rates = [0.10, 0.30, 0.50, 0.70]
    got = {k: [] for k in ["CE", "LS", "top-k", "oracle", "GCE", "SCE"]}
    xs = []
    for rate in rates:
        exp = f"S2_sweep{int(rate * 100)}"
        if not _match(exp=exp):
            continue
        xs.append(rate * 100)
        a = resolve("", quiet=True, exp=exp, loss="ce")
        got["CE"].append(vals(a).mean() if a else np.nan)
        for lab, loss, variants in [
                ("LS", "ls_static", [("eps", b) for b in [0.10, 0.30, 0.50]]),
                ("top-k", "afl_v2_topk", [("tag", f"epssel{e}") for e in [0.25, 0.5, 0.9]]),
                ("oracle", "afl_oracle_matched", [("tag", "epssel0.9")])]:
            _k, arm = _best_over("", loss, variants, exp)
            got[lab].append(vals(arm).mean() if arm else np.nan)
        for lab, loss in [("GCE", "gce"), ("SCE", "sce")]:
            a = resolve("", quiet=True, exp=exp, loss=loss, tag="tuned")
            got[lab].append(vals(a).mean() if a else np.nan)
    PROVENANCE.clear()
    if not xs:
        return
    plt.figure(figsize=(7.5, 5))
    for k, v in got.items():
        plt.plot(xs, v, marker="o", label=k)
    plt.xlabel("Label-noise rate (%)")
    plt.ylabel("Test accuracy (early stopped on noisy validation)")
    plt.title("Mechanism families across noise rates, both families tuned")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(FIG_DIR, fname); plt.savefig(p, dpi=150); plt.close()
    print(f"\n  saved {p}")


audit_cache()
table_tuning()
table_selection_regret()
table_headline()
table_ceiling_decomposition()
table_sweep()
table_budget_matched_sweep()
plot_sweep()
print("\n" + "=" * 92)
