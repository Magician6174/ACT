"""Dataset plumbing for ACT training.

We reuse lerobot's `LeRobotDataset` purely as a *reader* (the recorder already
wrote the data in this format). The only ACT-specific piece is `delta_timestamps`,
which makes the loader return a length-`chunk_size` action window per sample plus
an `action_is_pad` mask for windows that run past the episode end.
"""
import os

# Local-only dataset; never reach the Hub. Set before importing lerobot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Known OpenMP duplicate-runtime quirk on this machine.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from config import ACTConfig


def _action_delta_timestamps(cfg: ACTConfig) -> dict:
    # Action chunk = current frame + (chunk_size-1) future frames, spaced 1/fps.
    return {"action": [i / cfg.fps for i in range(cfg.chunk_size)]}


def episode_split(cfg: ACTConfig) -> tuple[list[int], list[int]]:
    """Split episode indices into train/val (split by *episode*, never by frame,
    so val frames are from demos the policy never trained on)."""
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.data_root)
    n = meta.total_episodes
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * cfg.val_fraction)))
    val = sorted(order[:n_val].tolist())
    train = sorted(order[n_val:].tolist())
    return train, val


def make_datasets(cfg: ACTConfig):
    """Return (train_ds, val_ds, stats). `stats` is the raw lerobot stats dict
    used by the model for normalization."""
    train_eps, val_eps = episode_split(cfg)
    delta = _action_delta_timestamps(cfg)
    common = dict(root=cfg.data_root, delta_timestamps=delta)

    train_ds = LeRobotDataset(cfg.repo_id, episodes=train_eps, **common)
    val_ds = LeRobotDataset(cfg.repo_id, episodes=val_eps, **common)
    stats = train_ds.meta.stats
    print(f"[dataset] train: {len(train_eps)} eps / {len(train_ds)} frames | "
          f"val: {len(val_eps)} eps / {len(val_ds)} frames")
    return train_ds, val_ds, stats


def make_loaders(cfg: ACTConfig):
    train_ds, val_ds, stats = make_datasets(cfg)
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, stats


# --- batch helpers ------------------------------------------------------------
# A collated batch is a dict with:
#   "observation.state"            (B, state_dim)
#   "observation.images.{cam}"     (B, 3, H, W)   float in [0,1]
#   "action"                       (B, chunk_size, action_dim)
#   "action_is_pad"                (B, chunk_size) bool
# (default_collate stacks these; "task" comes through as a list[str] we ignore)

def batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out
