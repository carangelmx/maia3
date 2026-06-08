"""Fine-tune maia3-5m on a single player's PGN file for digital twin training.

This script:
1. Parses a PGN file and extracts positions + moves + elo values
2. Builds a PyTorch dataset and dataloader
3. Fine-tunes the base model (freeze most blocks, train last 2 + heads)
4. Saves the best checkpoint by validation accuracy
"""

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path
from collections import deque
import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from maia3.models import MAIA3Model
from maia3.model_registry import MODEL_SPECS
from maia3.dataset import tokenize_board, get_legal_moves_mask, get_historical_tokens
from maia3.utils import get_all_possible_moves, seed_everything

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _git_sha() -> str:
    """Short git SHA of this training script's repo, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(path: str) -> str:
    """sha256 of a file (e.g. the training PGN), or 'unknown'."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


class PlayerGameDataset(Dataset):
    """Dataset from PGN file: positions up to move 60 in each game."""

    def __init__(self, pgn_path, all_moves_dict, max_move=60, history_depth=8,
                 player_name=None, player_only=True):
        self.pgn_path = pgn_path
        self.all_moves_dict = all_moves_dict
        self.max_move = max_move
        self.history_depth = history_depth
        # When set, only positions where this player is to move become training
        # targets (D1 fix), and Elo is attributed to this player (D2 fix).
        self.player_name = (player_name or "").lower()
        self.player_only = player_only and bool(self.player_name)
        self.samples = []
        self.player_elo = 1500  # Player's own median, computed in _load_pgn

        self._load_pgn()

    def _player_color(self, headers):
        """Return chess.WHITE/chess.BLACK for the modeled player, or None."""
        if not self.player_name:
            return None
        if self.player_name in headers.get("White", "").lower():
            return chess.WHITE
        if self.player_name in headers.get("Black", "").lower():
            return chess.BLACK
        return None

    def _load_pgn(self):
        """Parse PGN and extract board states, moves, elos, and results."""
        player_elos = []  # the modeled player's own Elo, per game
        skipped_games = 0

        with open(self.pgn_path) as f:
            game_count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_count += 1

                try:
                    white_elo = int(game.headers.get("WhiteElo", 1500))
                    black_elo = int(game.headers.get("BlackElo", 1500))
                    result_header = game.headers.get("Result", "*")
                    result_map = {"1-0": 2, "1/2-1/2": 1, "0-1": 0, "*": 1}
                    result = result_map.get(result_header, 1)
                except (ValueError, KeyError):
                    continue

                # Resolve which color the modeled player has this game (D1/D2).
                player_clr = self._player_color(game.headers)
                if self.player_only and player_clr is None:
                    skipped_games += 1
                    continue

                # Attribute Elo to the modeled player only (D2 fix).
                if self.player_only and player_clr == chess.WHITE and white_elo > 0:
                    player_elos.append(white_elo)
                elif self.player_only and player_clr == chess.BLACK and black_elo > 0:
                    player_elos.append(black_elo)
                else:
                    # Legacy: no per-player attribution, count both sides
                    if white_elo > 0:
                        player_elos.append(white_elo)
                    if black_elo > 0:
                        player_elos.append(black_elo)

                board = game.board()
                board_history = deque([tokenize_board(board)], maxlen=self.history_depth)
                move_count = 0

                for move in game.mainline_moves():
                    if move_count >= self.max_move:
                        break

                    # D1: only learn the modeled player's own decisions.
                    is_player_turn = (not self.player_only) or (board.turn == player_clr)
                    if is_player_turn:
                        self.samples.append({
                            "board_history": deque(board_history, maxlen=self.history_depth),
                            "played_move": move.uci() if board.turn == chess.WHITE else self._mirror_move(move.uci()),
                            "self_elo": white_elo if board.turn == chess.WHITE else black_elo,
                            "oppo_elo": black_elo if board.turn == chess.WHITE else white_elo,
                            "result": result,
                        })

                    board.push(move)
                    board_history.append(tokenize_board(board))
                    move_count += 1

        # Compute the modeled player's median ELO
        if player_elos:
            sorted_elos = sorted(player_elos)
            self.player_elo = sorted_elos[len(sorted_elos) // 2]
            logger.info(f"Player median ELO: {self.player_elo}")

        if self.player_only:
            logger.info(f"player_only=True: kept player-to-move positions; "
                        f"skipped {skipped_games} games without '{self.player_name}'")
        logger.info(f"Loaded {game_count} games, {len(self.samples)} positions")

    @staticmethod
    def _mirror_move(move_uci):
        """Mirror move for black perspective."""
        start, end = move_uci[:2], move_uci[2:4]
        promotion = move_uci[4:] if len(move_uci) > 4 else ""

        def mirror_sq(sq):
            f, r = sq[0], int(sq[1])
            return f + str(9 - r)

        return mirror_sq(start) + mirror_sq(end) + promotion

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        board_history = sample["board_history"]
        played_move = sample["played_move"]
        self_elo = sample["self_elo"]
        oppo_elo = sample["oppo_elo"]
        result = sample["result"]

        # Build historical tokens (matching maia3 inference pipeline)
        historical_tokens = torch.cat(list(board_history), dim=1)
        if len(board_history) < self.history_depth:
            pad = board_history[0].repeat(1, self.history_depth - len(board_history))
            historical_tokens = torch.cat([pad, historical_tokens], dim=1)

        # Add time info stub (zeros)
        historical_tokens = torch.cat([
            historical_tokens,
            torch.zeros((64, 1)),
            torch.zeros((64, 1)),
            torch.zeros((64, 1)),
            torch.zeros((64, 1)),
        ], dim=1)

        move_idx = self.all_moves_dict.get(played_move)
        if move_idx is None:
            raise ValueError(f"Move {played_move} not in vocabulary")

        return {
            "input": historical_tokens,
            "move_idx": move_idx,
            "result": result,
            "self_elo": self_elo,
            "oppo_elo": oppo_elo,
        }


def freeze_all_but_last_blocks(model, num_blocks_to_train=4):
    """Freeze all transformer blocks except the last N, and freeze Elo embeddings."""
    total_blocks = len(model.transformer.layers)
    freeze_until = total_blocks - num_blocks_to_train

    print(f"Freezing blocks 0-{freeze_until-1}, training blocks {freeze_until}-{total_blocks-1}")

    for i, block in enumerate(model.transformer.layers):
        if i < freeze_until:
            for param in block.parameters():
                param.requires_grad = False

    for param in model.elo_embedding_low.parameters():
        param.requires_grad = False
    for param in model.elo_embedding_high.parameters():
        param.requires_grad = False


def train_epoch(model, dataloader, optimizer, device, all_moves_vocab):
    """Train for one epoch."""
    model.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        input_tokens = batch["input"].to(device)  # (B, 64, features)
        move_idx = batch["move_idx"].to(device)
        result = batch["result"].to(device)
        self_elo = batch["self_elo"].to(device)
        oppo_elo = batch["oppo_elo"].to(device)

        batch_size = input_tokens.shape[0]

        optimizer.zero_grad()

        # Forward pass: model expects (B, 64, features)
        move_logits, value_logits, _ = model(input_tokens, self_elo, oppo_elo)

        # Policy loss: (B, 4352) logits vs (B,) target indices
        policy_loss = nn.CrossEntropyLoss()(move_logits, move_idx)

        # Value loss: (B, 3) logits vs (B,) target indices
        value_loss = nn.CrossEntropyLoss()(value_logits, result)

        # Combined loss
        loss = policy_loss + 0.1 * value_loss
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_samples += batch_size

    avg_policy_loss = total_policy_loss / total_samples if total_samples > 0 else 0.0
    avg_value_loss = total_value_loss / total_samples if total_samples > 0 else 0.0
    return avg_policy_loss, avg_value_loss


def eval_epoch(model, dataloader, device):
    """Evaluate policy accuracy on validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_tokens = batch["input"].to(device)
            move_idx = batch["move_idx"].to(device)
            self_elo = batch["self_elo"].to(device)
            oppo_elo = batch["oppo_elo"].to(device)

            move_logits, _, _ = model(input_tokens, self_elo, oppo_elo)

            preds = move_logits.argmax(dim=-1)
            correct += (preds == move_idx).sum().item()
            total += move_idx.shape[0]

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune maia3 on a player's PGN file")
    parser.add_argument("--pgn", required=True, help="Path to PGN file")
    parser.add_argument("--player", default="player", help="Player name for output")
    parser.add_argument("--base-model", default="maia3-5m", help="Base model: maia3-3m-ablation, maia3-5m, maia3-23m, or maia3-79m")
    parser.add_argument("--output", default=None, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-move", type=int, default=60, help="Max move number per game")
    parser.add_argument("--history-depth", type=int, default=8, help="Board history depth")
    parser.add_argument("--num-blocks-to-train", type=int, default=2,
                        help="Number of trailing transformer blocks to unfreeze (D3)")
    parser.add_argument("--all-positions", action="store_true",
                        help="Train on ALL positions incl. opponent moves (legacy). "
                             "Default trains only on the named player's own moves (D1).")

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    # Load model architecture from ModelSpec
    spec = None
    for model_spec in MODEL_SPECS:
        if model_spec.name == args.base_model or args.base_model in model_spec.aliases:
            spec = model_spec
            break

    if spec is None:
        raise ValueError(f"Unknown model: {args.base_model}. Available: "
                        f"{', '.join(s.name for s in MODEL_SPECS)}")

    # Create config namespace from ModelSpec
    class ModelConfig:
        pass

    cfg = ModelConfig()
    for key, value in spec.config.items():
        setattr(cfg, key, value)
    cfg.device = device

    model = MAIA3Model(cfg)
    model = model.to(device)

    # Build dataset
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    dataset = PlayerGameDataset(
        args.pgn,
        all_moves_dict,
        max_move=args.max_move,
        history_depth=args.history_depth,
        player_name=args.player,
        player_only=not args.all_positions,
    )

    if len(dataset) == 0:
        logger.error("No training samples loaded. Check PGN file.")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Freeze strategy (D3: configurable, was hardcoded to 2)
    freeze_all_but_last_blocks(model, num_blocks_to_train=args.num_blocks_to_train)
    logger.info(f"Training last {args.num_blocks_to_train} transformer blocks")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Output path
    if args.output is None:
        args.output = f"checkpoints/{args.player}_maia3-5m.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Provenance computed once (stamped into every saved checkpoint)
    pgn_sha = _file_sha256(args.pgn)

    logger.info(f"Training {args.base_model} on {len(dataset)} positions")
    logger.info(f"Train: {train_size}, Val: {val_size}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Player ELO: {dataset.player_elo}")
    logger.info(f"Provenance: git={_git_sha()} pgn_sha256={pgn_sha[:12]}...")

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_policy_loss, train_value_loss = train_epoch(model, train_loader, optimizer, device, all_moves_dict)
        val_acc = eval_epoch(model, val_loader, device)

        logger.info(f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Train policy loss: {train_policy_loss:.4f}, value loss: {train_value_loss:.4f} | "
                    f"Val accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "player_elo": dataset.player_elo,
                "player_name": args.player,
                "base_model": args.base_model,
                "num_blocks_to_train": args.num_blocks_to_train,
                "player_only": not args.all_positions,
                "epochs": args.epochs,
                "lr": args.lr,
                # Provenance (so a checkpoint records exactly what produced it)
                "train_git_sha": _git_sha(),
                "pgn_sha256": pgn_sha,
                "pgn_path": args.pgn,
                "seed": args.seed,
                "trained_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                "val_accuracy": best_val_acc,
            }, output_path)
            logger.info(f"  Saved best checkpoint to {output_path}")

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
