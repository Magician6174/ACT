"""Train ACT on the Panda pick-and-place dataset.

Pure offline imitation: sample (obs, action_chunk) -> L1 + KL -> backprop. The
environment is never touched here; closed-loop success is measured separately by
rollout.py (on the mac). We track validation loss to pick the best checkpoint.

Usage (SageMaker Studio, CUDA):
    python train.py --data_root /home/.../panda_pick_place --n_steps 100000
    python train.py --smoke            # tiny end-to-end shape/sanity check
LD_LIBRARY_PATH=$CONDA_PREFIX/lib KMP_DUPLICATE_LIB_OK=TRUE python train.py -
-data_root data/panda_pick_place --n_steps 100000 --batch_size 16
"""
import argparse
import csv
import json
import time
from dataclasses import fields
from pathlib import Path

import torch

from config import ACTConfig
from dataset import make_loaders, batch_to_device
from model import ACTPolicy


def parse_args() -> ACTConfig:
    p = argparse.ArgumentParser()
    for f in fields(ACTConfig):
        if f.type in (int, float, str):
            p.add_argument(f"--{f.name}", type=eval(f.type) if isinstance(f.type, str) else f.type)
        elif f.type == bool or f.name == "use_amp":
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "smoke" and v is not None}
    cfg = ACTConfig(**overrides)
    if args.smoke:
        cfg.batch_size = 2
        cfg.num_workers = 0
        cfg.n_steps = 4
        cfg.val_freq = 2
        cfg.save_freq = 1000
        cfg.log_freq = 1
    return cfg


def build_optimizer(policy: ACTPolicy, cfg: ACTConfig):
    # Separate LR for the pretrained vision backbone (DETR/ACT convention).
    backbone, rest = [], []
    for n, pm in policy.named_parameters():
        if not pm.requires_grad:
            continue
        (backbone if "model.backbone" in n else rest).append(pm)
    return torch.optim.AdamW(
        [
            {"params": rest, "lr": cfg.optimizer_lr},
            {"params": backbone, "lr": cfg.optimizer_lr_backbone},
        ],
        weight_decay=cfg.optimizer_weight_decay,
    )


def cycle(loader):
    """Turn a finite DataLoader into an INFINITE batch stream.

    Training is counted in steps (100k), not epochs, so we need a batch on every
    step regardless of dataset size. A plain DataLoader stops after one pass;
    this generator restarts it forever (drains one epoch, then loops). With a
    shuffling loader each new pass re-shuffles, so batch order still varies.
    """
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def evaluate(policy, val_loader, cfg, max_batches=50):
    policy.train()  # keep VAE active so val loss is comparable to train loss
    tot, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, cfg.device)
        _, info = policy.compute_loss(batch)
        tot += info["loss"]
        n += 1
    return tot / max(n, 1)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"[train] device={cfg.device} amp={cfg.use_amp} bs={cfg.batch_size} "
          f"steps={cfg.n_steps} chunk={cfg.chunk_size}")

    train_loader, val_loader, stats = make_loaders(cfg)
    policy = ACTPolicy(cfg, stats).to(cfg.device)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params/1e6:.1f}M")

    optimizer = build_optimizer(policy, cfg)
    amp_device = "cuda" if cfg.device == "cuda" else "cpu"
    # GradScaler: loss scaling for mixed precision (fp16). fp16 gradients can be
    # so small they underflow to 0; the scaler multiplies the loss by a large
    # factor before backward (lifting grads out of the underflow zone) and
    # divides them back before the optimizer step. No-op when use_amp is False
    # (auto-disabled off-CUDA), so it passes through untouched on mac.
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as fp:
        csv.writer(fp).writerow(["step", "loss", "l1_loss", "kld_loss", "val_loss", "sec_per_step"])

    best_val = float("inf")
    data_iter = cycle(train_loader)
    policy.train()
    t0 = time.time()

    for step in range(1, cfg.n_steps + 1):
        batch = batch_to_device(next(data_iter), cfg.device)
        optimizer.zero_grad(set_to_none=True)
        # autocast runs the forward pass in fp16 on CUDA (faster, less memory).
        with torch.autocast(device_type=amp_device, enabled=cfg.use_amp):
            loss, info = policy.compute_loss(batch)
        scaler.scale(loss).backward()     # scale loss up -> grads scaled up too
        scaler.unscale_(optimizer)        # un-scale grads BEFORE clipping (clip needs true magnitudes)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)            # applies update; skips it if grads overflowed (inf/NaN)
        scaler.update()                   # auto-tune the scale factor for next step

        val = ""
        if step % cfg.val_freq == 0 or step == cfg.n_steps:
            val = evaluate(policy, val_loader, cfg)
            policy.train()
            if val < best_val:
                best_val = val
                torch.save({"step": step, "model": policy.state_dict(),
                            "config": cfg.to_dict(), "val_loss": val},
                           out / "best.pt")
                print(f"[train] step {step}: new best val_loss={val:.4f} -> best.pt")

        if step % cfg.log_freq == 0 or step == cfg.n_steps:
            sps = (time.time() - t0) / cfg.log_freq
            t0 = time.time()
            print(f"step {step}/{cfg.n_steps} loss={info['loss']:.4f} "
                  f"l1={info['l1_loss']:.4f} kld={info.get('kld_loss', 0):.4f} "
                  f"val={val if val == '' else f'{val:.4f}'} ({sps:.2f}s/it)")
            with open(log_path, "a", newline="") as fp:
                csv.writer(fp).writerow([step, info["loss"], info["l1_loss"],
                                         info.get("kld_loss", ""), val, f"{sps:.3f}"])

        if step % cfg.save_freq == 0:
            torch.save({"step": step, "model": policy.state_dict(),
                        "config": cfg.to_dict()}, out / f"step_{step:06d}.pt")

    torch.save({"step": cfg.n_steps, "model": policy.state_dict(),
                "config": cfg.to_dict()}, out / "last.pt")
    print(f"[train] done. best val_loss={best_val:.4f}. checkpoints in {out}")


if __name__ == "__main__":
    main()
