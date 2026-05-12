"""
Collision classifier C — pretrained MLP that predicts P(self-collision | qpos).

Used by trainer.py as a FROZEN, differentiable proxy for the sapien
self-collision check, so gradients can flow from the GeoRT collision loss

    L_col = -E_xH [ log(1 - sigmoid(C(f(xH)))) ]

back through C and through the FK-via-keypoint pipeline into the IK
(retargeting) network f. See the GeoRT paper, Criterion V.

Conventions:
- Classifier input  : normalized qpos in [-1, 1]  (matches IK model output)
- Classifier output : a single logit per sample (apply sigmoid for prob)

The training set is the dataset produced by collision_data.py
(`data/{hand_name}_collision.npz`), which records raw-joint qpos + binary
labels from the sapien collision checker. We normalize here so that the
classifier learns in the same input space the IK model will eventually
feed it.

Usage:
  # 1) make sure the dataset exists
  python glove_based/geort/collision_data.py --hand v6_right.json
  # 2) train the classifier
  python glove_based/geort/collision_classifier.py --hand v6_right.json
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel
from geort.formatter import HandFormatter
from geort.utils.path import get_data_root, get_checkpoint_root


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CollisionClassifier(nn.Module):
    """Small MLP: normalized qpos -> single logit (collision probability)."""

    def __init__(self, dof: int, hidden: int = 128):
        super().__init__()
        self.dof = dof
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(dof, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, qpos_normalized: torch.Tensor) -> torch.Tensor:
        # qpos_normalized: (B, dof) in [-1, 1]
        # returns: (B, 1) logits
        return self.net(qpos_normalized)


def classifier_ckpt_path(hand_name: str) -> Path:
    return Path(get_checkpoint_root()) / f"collision_classifier_{hand_name}.pth"


def load_classifier(hand_name: str, device: str = "cuda"):
    """Load a previously-trained classifier in eval/frozen mode.

    Returns None if no checkpoint exists. Callers should treat None as
    "collision loss disabled" and fall through gracefully.
    """
    path = classifier_ckpt_path(hand_name)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = CollisionClassifier(dof=cfg["dof"], hidden=cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CollisionDataset(Dataset):
    def __init__(self, qpos_raw: np.ndarray, label: np.ndarray, normalizer: HandFormatter):
        self.qpos = normalizer.normalize(qpos_raw.astype(np.float32)).astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return self.qpos.shape[0]

    def __getitem__(self, i):
        return self.qpos[i], self.label[i]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the collision classifier.")
    parser.add_argument("--hand", type=str, required=True,
                        help="Hand config name (e.g., v6_right.json)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_config(args.hand)
    hand_name = config["name"]

    # Build the hand model only to recover joint limits → normalizer.
    # (The IK model uses the same normalizer at inference, so the classifier
    # must learn in the same normalized input space.)
    hand_model = HandKinematicModel.build_from_config(config, render=False)
    lo, hi = hand_model.get_joint_limit()
    normalizer = HandFormatter(lo, hi)
    dof = hand_model.get_n_dof()

    # Load collision dataset produced by collision_data.py
    data_path = Path(get_data_root()) / f"{hand_name}_collision.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Collision dataset not found at {data_path}. "
            f"Generate it first with:\n"
            f"  python glove_based/geort/collision_data.py --hand {args.hand}"
        )
    data = np.load(data_path)
    qpos_all, label_all = data["qpos"], data["label"]

    n = len(qpos_all)
    pos_rate = label_all.mean()
    print(f"Loaded {n} samples from {data_path.name}")
    print(f"  collision rate: {pos_rate * 100:.2f}%")
    if pos_rate == 0.0 or pos_rate == 1.0:
        raise ValueError("Dataset is single-class (all collision or none). "
                         "Cannot train a meaningful classifier — regenerate data.")

    # Train/val split (shuffled)
    perm = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = CollisionDataset(qpos_all[train_idx], label_all[train_idx], normalizer)
    val_ds = CollisionDataset(qpos_all[val_idx], label_all[val_idx], normalizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class imbalance handling: BCE with pos_weight = neg/pos.
    n_pos = max(int(label_all.sum()), 1)
    n_neg = max(n - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).cuda()
    print(f"  pos_weight (neg/pos): {pos_weight.item():.2f}")

    model = CollisionClassifier(dof=dof, hidden=args.hidden).cuda()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = classifier_ckpt_path(hand_name)
    os.makedirs(get_checkpoint_root(), exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ------- train pass -------
        model.train()
        tr_losses, tr_correct, tr_total = [], 0, 0
        for qpos, label in train_loader:
            qpos = qpos.cuda()
            label = label.cuda()
            logit = model(qpos).squeeze(-1)
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).float()
                tr_correct += (pred == label).sum().item()
                tr_total += label.numel()

        # ------- val pass -------
        model.eval()
        va_losses, va_correct, va_total = [], 0, 0
        tp = fp = fn = 0
        with torch.no_grad():
            for qpos, label in val_loader:
                qpos = qpos.cuda()
                label = label.cuda()
                logit = model(qpos).squeeze(-1)
                loss = criterion(logit, label)
                va_losses.append(loss.item())
                pred = (torch.sigmoid(logit) > 0.5).float()
                va_correct += (pred == label).sum().item()
                va_total += label.numel()
                tp += ((pred == 1) & (label == 1)).sum().item()
                fp += ((pred == 1) & (label == 0)).sum().item()
                fn += ((pred == 0) & (label == 1)).sum().item()

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-7)
        print(
            f"Epoch {epoch:3d}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}  "
            f"train_acc={tr_correct / tr_total:.4f} val_acc={va_correct / va_total:.4f}  "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"dof": dof, "hidden": args.hidden, "hand": hand_name},
                "metrics": {
                    "val_loss": va_loss,
                    "val_acc": va_correct / va_total,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
            }, ckpt_path)

    print(f"\nBest classifier saved to: {ckpt_path}")
    print(f"  best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
