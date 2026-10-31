"""
flvsnl_stage2b.py  --  Stage 2b: close the ceiling confound, thicken the sweep.

    %run -i flvsnl_core.py
    %run -i flvsnl_stage2b.py
    %run -i flvsnl_analyze2.py       # Tables I, L, M

WHY THIS STAGE EXISTS
---------------------
Stage 2 delivered two things that need repair before anything is written.

1. THE CEILING CONFOUND IN TABLE I.
   The targeted arms (top-k, oracle) put eps = b/eta on each selected sample,
   and eps is capped at 0.9, so their largest deliverable budget is eta * 0.9.
   Static Label Smoothing has no such limit and was swept on b in
   {0.10, 0.30, 0.50} at every noise rate. At eta = 0.10 that means LS spent
   b = 0.5 while the oracle was structurally capped at b = 0.09 -- a 5.5x budget
   difference. "LS beats the oracle at 10% noise" is an artefact of that
   mismatch, not a finding about targeting.

   Part B runs static LS at budgets DERIVED FROM the same eps-per-selected
   parameterisation, b = eta * eps_sel, so that every arm at a given
   (noise, eps_sel) cell spends the same total budget. Then the comparison
   isolates allocation, which is the thing we are actually measuring.

2. THREE SEEDS IS TOO THIN FOR THE SWEEP.
   The n=10 results at 40% noise overturned the n=5 headline (top-k over LS fell
   from +0.0053, p=0.008 to +0.0025, p=0.183). The sweep is currently n=3 and
   carries the ceiling story, which is now the paper's central mechanism claim.
   Part A takes it to n=5.

WHAT THE SWEEP ALREADY SHOWED, AND WHY IT MATTERS
   At eta = 0.70, eps_sel = 0.9 collapsed to 0.5260 against 0.8280 at
   eps_sel = 0.5. This was checked and is genuine: the budget was delivered
   (0.630), all three seeds are stable, no divergence. With 70% of samples
   receiving near-uniform targets, almost no label information survives. The
   cost-of-imprecision mechanism predicts exactly this, so the collapse is
   evidence rather than an outlier -- but it deserves n=5 before it is quoted.

COST: Part A 80 runs (~5.5 h), Part B 40 runs (~2.8 h). Total ~8.3 GPU-hours.
"""

import numpy as np
import torch
from torchvision import datasets, transforms

# =============================================================================
# CONFIG
# =============================================================================

DRY_RUN = False

RUN_PART_A = True      # extend the sweep from 3 to 5 seeds
RUN_PART_B = True      # budget-matched static arms

# Must match the values used in Stage 2, or the cache will not line up.
TUNED = {
    "gce": {"q": 0.5},
    "sce": {"alpha": 0.1, "beta": 2.0},
    "nce_rce": {"alpha": 1.0, "beta": 1.0},
}

SEEDS_NEW = [3, 4]                 # Part A: seeds the sweep does not yet have
SEEDS_ALL = [0, 1, 2, 3, 4]        # Part B: matched arms for every sweep seed
NOISE_RATES = [0.10, 0.30, 0.50, 0.70]
LS_BUDGETS = [0.10, 0.30, 0.50]    # the free grid, unchanged from Stage 2
EPS_SEL = [0.25, 0.50, 0.90]       # eps on each SELECTED sample
EPS_SEL_MATCHED = [0.50, 0.90]     # budgets to mirror onto static LS

EPOCHS = 20
LR = 1e-3
BATCH = 128
VAL_FRAC = 0.10
DIAG_EVERY = 0
NUM_WORKERS = 2
DATA_ROOT = "./data"

# =============================================================================

PLANNED.clear()
if DRY_RUN:
    print("\n*** DRY RUN: counting work, training nothing. ***")

if any(v is None for v in TUNED.values()):
    raise SystemExit("[GATE] TUNED must carry the Stage 2 Part A winners.")

print("\n--- Setting up KMNIST ---")
kmnist_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
kmnist_train = datasets.KMNIST(root=DATA_ROOT, train=True, transform=kmnist_tf, download=True)
kmnist_test = datasets.KMNIST(root=DATA_ROOT, train=False, transform=kmnist_tf, download=True)

K_CLASSES = len(kmnist_train.classes)
K_CH = kmnist_train[0][0].shape[0]
K_DIM = kmnist_train[0][0].shape[1]

kmnist_test_ds = NoisyLabelDataset(
    kmnist_test, _base_labels(kmnist_test), _base_labels(kmnist_test),
    np.zeros(len(kmnist_test), dtype=np.int64))


def cnn_fn(num_classes):
    return CNNClassifier(K_CH, K_DIM, num_classes)


def _prepare(noise, seed):
    full = make_noisy_dataset(kmnist_train, noise, K_CLASSES, seed)
    return split_train_val(full, val_frac=VAL_FRAC, seed=seed)


def _go(exp, loss, seed, tr, vn, vc, noise, eps=0.10, hp=None, tag=""):
    return run_single(
        exp=exp, loss_name=loss, seed=seed,
        train_ds=tr, test_ds=kmnist_test_ds, num_classes=K_CLASSES,
        model_fn=cnn_fn, epochs=EPOCHS, lr=LR, batch_size=BATCH,
        eps=eps, gamma=2.0, diag_every=DIAG_EVERY, num_workers=NUM_WORKERS,
        val_noisy_ds=vn, val_clean_ds=vc, noise_rate=noise, hp=hp,
        meta={"noise": noise, "noise_type": "symmetric", "dataset": "kmnist",
              "val_frac": VAL_FRAC, "tag": tag},
    )


# =============================================================================
# PART A -- take the existing sweep from 3 seeds to 5
# =============================================================================
if RUN_PART_A:
    print("\n" + "=" * 70)
    print("  PART A: sweep seeds 3-4 (brings the noise sweep to n=5)")
    print("=" * 70)
    for noise in NOISE_RATES:
        for seed in SEEDS_NEW:
            tr, vn, vc = _prepare(noise, seed)
            exp = f"S2_sweep{int(noise * 100)}"
            _go(exp, "ce", seed, tr, vn, vc, noise)
            _go(exp, "gce", seed, tr, vn, vc, noise, hp=TUNED["gce"], tag="tuned")
            _go(exp, "sce", seed, tr, vn, vc, noise, hp=TUNED["sce"], tag="tuned")
            for b in LS_BUDGETS:
                _go(exp, "ls_static", seed, tr, vn, vc, noise, eps=b)
            for es in EPS_SEL:
                _go(exp, "afl_v2_topk", seed, tr, vn, vc, noise,
                    eps=noise * es, tag=f"epssel{es}")
            _go(exp, "afl_oracle_matched", seed, tr, vn, vc, noise,
                eps=noise * 0.90, tag="epssel0.9")

# =============================================================================
# PART B -- static LS at budgets matched to the targeted arms
# =============================================================================
# Tagged 'matchedX' so the analysis can address them without colliding with the
# free-grid LS runs, which keep an empty tag.
if RUN_PART_B:
    print("\n" + "=" * 70)
    print("  PART B: budget-matched static LS (b = noise x eps_sel)")
    print("=" * 70)
    for noise in NOISE_RATES:
        for seed in SEEDS_ALL:
            tr, vn, vc = _prepare(noise, seed)
            exp = f"S2_sweep{int(noise * 100)}"
            for es in EPS_SEL_MATCHED:
                _go(exp, "ls_static", seed, tr, vn, vc, noise,
                    eps=noise * es, tag=f"matched{es}")

if DRY_RUN:
    budget_report(sec_per_run=250)
    print("\n[stage2b] DRY RUN only. Set DRY_RUN = False and re-run.")
else:
    print("\n[stage2b] complete. Now: %run -i flvsnl_analyze2.py")
