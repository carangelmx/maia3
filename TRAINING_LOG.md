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

### Sweep results (appended by run_experiments.py)
| name | model | epochs | lr | blocks | JS (agg, W/B) | top1 | entropy m/p |
|------|-------|--------|----|--------|---------------|------|-------------|
