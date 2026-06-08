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
        "version": "corrected-baseline", "status": "deployed",
        "metrics": {"opening_top1_match": 0.955, "opening_js": 0.026},
        "asset_url": "https://github.com/carangelmx/chesspredator_twin/releases/download/"
                     "carangelmx-twin-v1/carangelmx_maia3-23m_corrected-baseline.pt",
        "notes": "23M, player-only (D1) + per-color Elo (D2); Elo 2155. Deployed sample.",
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
    order = {"deployed": 0, "retired": 1, "superseded": 2}
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
