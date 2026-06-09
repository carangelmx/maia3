#!/usr/bin/env python
"""Generate models.json — a provenance manifest for every twin checkpoint.

Scans the checkpoints directory, reads each checkpoint's embedded metadata,
computes a sha256 of the file, and merges in curated metrics/status/hosting info.
The manifest is the single source of truth for "which model is which" so we can
always go back to a previous model (by file hash) as we move forward.

Run from the maia3 repo root:
    python build_manifest.py
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

# Curated overlay: metrics/status/hosting we know that aren't in the checkpoint.
# Keyed by file stem.
CURATED = {
    "carangelmx_test": {
        "version": "v1", "status": "superseded",
        "notes": "Original 5M fine-tune; hardcoded Elo 1500; ~26.83% next-move acc.",
    },
    "carangelmx_maia3-5m_v2": {
        "version": "v2", "status": "superseded",
        "notes": "5M, 10 epochs, 4 blocks @1e-4; opening accuracy regressed.",
    },
    "carangelmx_maia3-23m_v5": {"version": "v5", "status": "superseded"},
    "carangelmx_maia3-23m_v5_elo": {"version": "v5_elo", "status": "superseded"},
    "carangelmx_maia3-23m_v5_final": {
        "version": "v5_final", "status": "retired",
        "metrics": {"opening_top1_match": 0.928, "opening_js": 0.082},
        "asset_url": "https://github.com/carangelmx/chesspredator_twin/releases/download/"
                     "carangelmx-twin-v5_final/carangelmx_maia3-23m_v5_final.pt",
        "notes": "23M, all-positions (incl. opponents); Elo 2141 (both-players median). "
                 "Pre-D1. Needed T=0.5. Retired in favor of corrected-baseline.",
    },
    "carangelmx_maia3-23m_corrected-baseline": {
        "version": "corrected-baseline", "status": "previous",
        "metrics": {"opening_top1_match": 0.955, "opening_js": 0.026,
                    "post_ply10_top1": 0.360},
        "asset_url": "https://github.com/carangelmx/chesspredator_twin/releases/download/"
                     "carangelmx-twin-v1/carangelmx_maia3-23m_corrected-baseline.pt",
        "notes": "v1: 23M, player-only, 2 unfrozen blocks; Elo 2155. Superseded by more-blocks (v2) "
                 "which wins the post-ply-10 gate.",
    },
    "carangelmx_maia3-23m_v3": {
        "version": "v3", "status": "deployed",
        "metrics": {"post_ply10_top1": 0.294, "post_ply10_perplexity": 15.3,
                    "middlegame_top1": 0.262, "endgame_top1": 0.228},
        "asset_url": "https://github.com/carangelmx/chesspredator_twin/releases/download/"
                     "carangelmx-twin-v3/carangelmx_maia3-23m_v3.pt",
        "notes": "v3 (DEPLOYED): 23M, FULL fine-tune (8 blocks), newest 6,000 games; Elo 2195. Wins "
                 "the clean temporal future-test: post-ply-10 29.4% / ppl 15.3 vs ~25% / ppl 30 for "
                 "frozen 2-4 block variants. Honest unseen-recent fidelity ~29%.",
    },
    "carangelmx_maia3-23m_more-blocks": {
        "version": "more-blocks", "status": "previous",
        "metrics": {"opening_top1_match": 0.964, "opening_js": 0.0239,
                    "post_ply10_top1": 0.397, "middlegame_top1": 0.465},
        "asset_url": "https://github.com/carangelmx/chesspredator_twin/releases/download/"
                     "carangelmx-twin-v2/carangelmx_maia3-23m_more-blocks.pt",
        "notes": "v2 (previous deploy): 23M, 4 unfrozen blocks; Elo 2155. Its post-ply-10 lead was a "
                 "holdout confound (trained on the test games). Superseded by v3 (full fine-tune) on a "
                 "clean temporal test.",
    },
    "carangelmx_maia3-23m_temporal-8blk": {
        "version": "temporal-8blk", "status": "candidate",
        "metrics": {"post_ply10_top1": 0.294, "post_ply10_perplexity": 15.3},
        "notes": "Full fine-tune on the temporal split (newest 600 held out). The clean-test winner; "
                 "v3 is the same recipe retrained on all newest-6,000 for deployment.",
    },
    "carangelmx_maia3-23m_lower-lr": {
        "version": "lower-lr", "status": "candidate",
        "metrics": {"opening_top1_match": 0.963, "opening_js": 0.0257},
        "notes": "Like corrected-baseline but lr 5e-5. ~tied on openings.",
    },
    "carangelmx_maia3-23m_longer": {
        "version": "longer", "status": "candidate",
        "metrics": {"opening_top1_match": 0.955, "opening_js": 0.0260},
        "notes": "Like corrected-baseline but 15 epochs. No opening gain over 10.",
    },
    "carangelmx_maia3-23m_full-ft": {
        "version": "full-ft", "status": "candidate",
        "metrics": {"post_ply10_top1_unseen": 0.324, "val_accuracy": 0.414},
        "notes": "Full fine-tune (all 8 blocks), holdout 10% excluded (true generalization test). "
                 "Overfit: val peaked epoch 4 then declined; train loss 0.81. post-ply-10 32.4% on "
                 "UNSEEN games (not directly comparable to frozen models that trained on the holdout).",
    },
}

# Metadata keys we lift verbatim from a checkpoint when present.
META_KEYS = [
    "player_name", "base_model", "player_elo", "num_blocks_to_train", "player_only",
    "epochs", "lr", "seed", "train_git_sha", "pgn_sha256", "pgn_path", "trained_at",
    "val_accuracy",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: ckpt.get(k) for k in META_KEYS} if isinstance(ckpt, dict) else {}
    entry = {
        "file": str(path).replace("\\", "/"),
        "sha256": sha256(path),
        "size_mb": round(path.stat().st_size / 1048576, 1),
        **{k: v for k, v in meta.items() if v is not None},
    }
    entry.update(CURATED.get(path.stem, {}))
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--out", default="models.json")
    args = ap.parse_args()

    root = Path(args.checkpoints)
    paths = sorted(root.glob("*.pt")) + sorted(root.glob("sweep/*.pt"))
    if not paths:
        print(f"No checkpoints found under {root}/")
        return

    models = []
    for p in paths:
        print(f"  hashing {p.name} ...")
        models.append(describe(p))

    # Stable order: deployed first, then by version/file
    order = {"deployed": 0, "previous": 1, "candidate": 2, "retired": 3, "superseded": 4}
    models.sort(key=lambda m: (order.get(m.get("status"), 9), m.get("version", m["file"])))

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "player": "carangelmx",
        "models": models,
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out} ({len(models)} models)")
    for m in models:
        print(f"  {m.get('version','?'):18} {m.get('status','?'):11} "
              f"{m['sha256'][:12]}  {m['file']}")


if __name__ == "__main__":
    main()
