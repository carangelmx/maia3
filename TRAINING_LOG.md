# Maia3 Training Log

Track training runs to estimate future training times and performance.

## Training Sessions

| Run | Model | Epochs | Positions | Unfrozen Blocks | Learning Rate | Batch Size | Device | Duration | Final Accuracy | Notes |
|-----|-------|--------|-----------|-----------------|---------------|------------|--------|----------|-----------------|-------|
| v1 | maia3-5m | 1 | 225,360 | 2 | 1e-4 | 128 | CPU | 30 sec | **26.83%** | Baseline, original training |
| v2 | maia3-5m | 10 | 416,320 | 4 | 1e-4 | 256 | CUDA | **25 min** | 26.11% | Unfreezing 4 blocks too aggressive |
| v3 | maia3-5m | 10 | 416,320 | 4 | 5e-5 | 256 | CUDA | ~25 min (est) | TBD | Lower LR for gentler training |
| v4 | maia3-5m | 10 | 416,320 | 3 | 1e-4 | 256 | CUDA | ~25 min (est) | TBD | Revert to 3 blocks |
| v5 | maia3-23m | 10 | 416,320 | 4 | 1e-4 | 128 | CUDA | **61 min** | **35.79%** ✅ | Larger model wins! +33% accuracy |

## Performance Metrics

### Training Speed

- **CPU (v1)**: ~225k positions/30 sec = **7,500 samples/sec**
- **CUDA (v2)**: ~416k positions/25 min = **276 samples/sec**

Wait, that's wrong. Let me recalculate:
- **CUDA (v2)**: 416k × 10 epochs / 1500 sec = ~2,773 samples/sec

### Time Estimation Formula

```
Training Time = (Positions × Epochs × Overhead) / GPU_Throughput

Where:
- GPU_Throughput ≈ 2,800 samples/sec (RTX 5060 Ti, batch 256)
- Overhead = 1.05 (5% for validation, I/O)
```

### Estimated Times (RTX 5060 Ti)

| Positions | Epochs | Model | Estimated Time |
|-----------|--------|-------|-----------------|
| 225k | 1 | 5M | 1 min |
| 225k | 10 | 5M | 10 min |
| 416k | 10 | 5M | 25 min ✓ (measured) |
| 416k | 10 | 23M | 35-40 min |
| 416k | 10 | 79M | 60+ min |

## Learning Points

1. **CUDA is 10x faster** than CPU for training
2. **Unfreezing more blocks requires lower LR** — v2 showed accuracy drop with 4 blocks @ 1e-4
3. **More epochs helps** — training loss decreased 19% across 10 epochs
4. **Larger models need more careful tuning** — capacity vs. data trade-off

## Next Steps

- [ ] Run v3 (lower LR) - validate if lower LR helps with 4 unfrozen blocks
- [ ] Run v4 (3 blocks) - check if reverting improves accuracy
- [ ] Run v5 (maia3-23m) - test if larger model learns better
- [ ] Compare final accuracies and pick best checkpoint

## Checkpoint Locations

- v1: `checkpoints/carangelmx_test.pt` (20.0 MB) — 26.83% accuracy
- v2: `checkpoints/carangelmx_maia3-5m_v2.pt` (20.0 MB) — 26.11% accuracy
- v3: `checkpoints/carangelmx_maia3-5m_v3.pt` (TBD)
- v4: `checkpoints/carangelmx_maia3-5m_v4.pt` (TBD)
- v5: `checkpoints/carangelmx_maia3-23m.pt` (TBD)

---

## Final Implementation

✅ **ELO Tracking Complete (2026-06-08)**

- Extracts median player ELO from training games: **2141**
- Saves to checkpoint metadata (player_elo, base_model)
- Automatically loaded on model init
- Displayed in play_twin_final.py header

### Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Player ELO | 1500 (hardcoded) | 2141 (actual) |
| Difference | -641 ELO | Correct |
| Model | maia3-5m hardcoded | Dynamic (maia3-23m detected) |

**Impact**: Twin now plays at correct skill level, should feel much stronger!

---

**Last Updated**: 2026-06-08
**Total Training Time**: ~122 minutes (v2 + v5 + v5_final)
**Best Checkpoint**: `carangelmx_maia3-23m_v5_final.pt` (87.6 MB, 35.79% accuracy, ELO 2141)

---

## Opening-Fidelity Track (FINE_TUNING_SYSTEM_PROMPT.md)

Primary gate metric: per-color distribution match vs carangelmx's empirical opening
frequencies. Lower JS divergence = closer to the player's repertoire.

### Defects found (2026-06-08)
- **D0 (dominant, inference-side)**: `get_move_probs` fed only the current position, but training
  used an 8-ply history → out-of-distribution queries. Fixed via `_build_board_history` (replay move
  list). **This alone** took aggregate top-1 31.1%→92.8%, JS 0.514→0.082.
- **D1**: training emitted *every* ply, ~50% of targets were the **opponent's** moves. Fixed via
  player-only filtering (default). Validated: 121,461 → 60,722 positions (exactly 50%).
- **D2**: `player_elo` medianed both players (2141). Fixed to carangelmx-only → **2155**.
- **D3**: frozen-block count hardcoded to 2; now `--num-blocks-to-train`.
- **Elo (demoted)**: anchors frozen + single-Elo data ⇒ low tuning leverage. Keep self_elo=2155.

### Opening-fidelity baseline (v5_final 23M checkpoint)
| Metric | broken inference (pre-D0) | **history-fixed** | history-fixed, **T=0.5** |
|--------|---------------------------|-------------------|--------------------------|
| White top1-match | 41.9% | 90.2% | 90.2% |
| Black top1-match | 16.6% | 96.3% | 96.3% |
| Aggregate top1-match | 31.1% | **92.8%** | **92.8%** |
| Aggregate JS | 0.514 | 0.082 | **0.049** |
| top1-mass | 25.4% | 73.4% | **87.9%** |

Key correction: the twin *had* learned carangelmx's Sicilian (1...c5 = 96.3% top-1 with history). The
earlier "wrong opening" read was the D0 inference bug, not training. Residual gap is the **opening
root** (move-1 White: model e4 73.5% vs player 97.6%) and over-softness (handled by T≈0.5).

### Sweep results (opening fidelity, T=1.0, RTX 5060 Ti, Elo 2155)
| name | epochs | lr | blocks | JS (agg, W/B) | top1 | entropy m/p | train |
|------|--------|----|--------|---------------|------|-------------|-------|
| corrected-baseline | 10 | 1e-4 | 2 | 0.0260 (W .023/B .030) | 95.5% | 0.54/0.55 | 31m14s |
| **more-blocks** | 10 | 1e-4 | **4** | **0.0239** (W .023/B .025) | **96.4%** | 0.56/0.55 | 31m58s |
| lower-lr | 10 | 5e-5 | 2 | 0.0257 (W .024/B .028) | 96.3% | 0.58/0.55 | 31m14s |
| longer | 15 | 1e-4 | 2 | 0.0260 (W .023/B .030) | 95.5% | 0.54/0.55 | 46m24s |

**Grid conclusion (2026-06-08).** On openings the field is tight (all JS 0.024–0.026, top-1 95.5–96.4%).
- **more-blocks (4 unfrozen)** is marginally best — JS 0.0239, top-1 96.4%, entropy still matched — at
  the same ~32 min cost. The likely real payoff of extra capacity is **middlegame**, which this grid
  does not measure.
- **lower-lr**: ~tied (top-1 96.3%, JS 0.0257), entropy a touch high (0.58).
- **longer (15 ep)**: identical to corrected-baseline — +5 epochs / +15 min bought nothing on openings.
- **Reproducibility confirmed**: the corrected-baseline re-run reproduced exactly (JS 0.0260, 95.5%,
  31m14s, val_acc identical). The new checkpoints are provenance-stamped (`train_git_sha=6753df4`).

Decision deferred to the **middlegame eval** (ROADMAP frontier): pick between more-blocks and
corrected-baseline once we can score by game phase, not opening top-1 alone.

### Corrected-baseline: what player-only training (D1) bought
Beyond the inference fix, training only on carangelmx's own moves was **not** mere refinement:

| Metric | v5_final (polluted data) + best T | corrected-baseline (player-only), T=1.0 |
|--------|-----------------------------------|------------------------------------------|
| Aggregate JS | 0.049 (needed T=0.5) | **0.026** |
| Aggregate top1-match | 92.8% | **95.5%** |
| top1-mass | 87.9% | 87.2% |
| Model entropy vs player | 1.14–1.38 vs 0.5 (too soft) | **0.54 vs 0.55 (matched)** |
| Move-1 White e4 | 73.5% (player 97.6%) | **99.4%** (root gap closed) |

**Key insight — player-only training obsoleted the temperature hack.** The old checkpoint was too
soft (entropy ~1.2 bits) and needed T≈0.5 to fake carangelmx's sharpness. Training on his moves only
made the model's entropy match his at T=1.0, so sharpening now *overshoots*:

| Temperature | v5_final JS | corrected-baseline JS |
|-------------|-------------|-----------------------|
| 1.0 | 0.082 | **0.026** (best) |
| 0.7 | 0.052 | 0.032 |
| 0.5 | **0.049** (best) | 0.043 |
| 0.3 | 0.062 | — |

Net: D0 (inference history) + D1 (player-only data) together took opening JS **0.514 → 0.026** (20×)
and top-1 match **31% → 95.5%**, with the model now matching carangelmx's entropy and 1.e4 frequency.
No temperature tuning required — keep T=1.0.

### Post-ply-10 gate — the real test (`eval_moves.py`, 2026-06-08)
Per the per-player Maia literature (KDD 2022), openings are ~100% memorized; the genuine test is
move-matching on **plies > 10**. Top-1 / perplexity on a 300-game holdout (note: holdout not yet
excluded in training, so absolute numbers are optimistic — valid for *relative* comparison):

| phase | corrected-baseline (2 blocks) | more-blocks (4 blocks) |
|-------|-------------------------------|------------------------|
| opening ≤ply10 (recall) | 77.4% / ppl 2.20 | 78.4% / ppl 2.18 |
| early-mid 10–29 | 52.2% / 5.04 | 54.3% / 4.71 |
| **middlegame 30–59** | 37.8% / 10.21 | **46.5% / 7.25** |
| endgame 60+ | 19.1% / 26.5 | 18.7% / 33.5 |
| **POST-PLY-10 (gate)** | 36.0% / 11.29 | **39.7% / 10.51** |

**Provisional: more-blocks (4 unfrozen) leads** — but see the confound below; these two models trained
on the holdout, so their numbers are memorization-inflated. Promoted more-blocks as v2 on this basis.

### Full fine-tuning (all 8 blocks) + holdout — overfitting + a confound (2026-06-08)
Ran `--num-blocks-to-train -1 --holdout-frac 0.1 --patience 3` (all blocks, the held-out 10% of games
excluded from training → a TRUE generalization test). Val acc peaked epoch 4 at **41.4%** then declined
(train policy loss 3.9→0.81) → **clear overfitting**; early-stopping kept the epoch-4 best. Consistent
with the literature warning (<~5k games overfits; we have ~6,900 after holdout).

| post-ply-10 gate (300-game holdout) | corrected-baseline 2blk | more-blocks 4blk | full-ft 8blk |
|-------------------------------------|-------------------------|------------------|--------------|
| trained on these games? | yes (inflated) | yes (inflated) | **no (true test)** |
| post-ply-10 top-1 | 36.0% | 39.7% | 32.4% |
| middlegame 30–59 | 37.8% | 46.5% | 29.4% |
| endgame 60+ | 19.1% | 18.7% | **26.5%** |

**Confound:** full-ft is judged on unseen games; the frozen models trained on the whole PGN, so their
holdout numbers are inflated by memorization. The endgame slice (sparse, ~unseen for all) is the
fairest — and full-ft *wins* it (26.5% vs ~19%). **A clean decision needs the frozen candidates
retrained with the same `--holdout-frac 0.1`**, then all three eval'd on the unseen holdout.
Next: retrain corrected-baseline(2blk) and more-blocks(4blk) with holdout excluded.

---

## Model provenance & recovery (go back to any prior model)

`models.json` is the manifest — one entry per checkpoint with its **sha256**, embedded metadata
(base_model, Elo, hyperparams), curated metrics/status, and a hosting URL. Regenerate with
`python build_manifest.py` after any new run.

Two independent ways to restore a prior model:
1. **By weights** — load the `.pt` directly (local `checkpoints/`, or the hosted release asset).
   `NeuralTwinModel` auto-reads `base_model`/`elo` from the checkpoint.
2. **By recipe** — re-train from `train_player.py` + the row's hyperparameters + the raw PGN + `--seed`
   (functionally equivalent; GPU nondeterminism means not bit-exact).

Off-machine copies (survive disk loss):
- **corrected-baseline** (deployed): release `carangelmx-twin-v1`
- **v5_final** (retired): release `carangelmx-twin-v5_final`
- v1/v2/v5/v5_elo: local only (superseded; recoverable by recipe).

**Provenance stamping:** since this commit, every new checkpoint embeds `train_git_sha`, `pgn_sha256`,
`seed`, `trained_at`, and `val_accuracy`, so a model file records exactly the code + data that made it.
(Older checkpoints predate this; their provenance is the `models.json` row + this log.)