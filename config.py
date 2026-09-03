"""ACT configuration for the Panda pick-and-place task.

Single source of truth for dims, architecture, and training hyperparameters.
Defaults follow the original ACT paper (Zhao et al., 2304.13705) and lerobot's
proven presets, adapted to our 8-DoF / 3-camera setup.
"""
from dataclasses import dataclass, field, asdict

import torch


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ACTConfig:
    # --- data -------------------------------------------------------------
    repo_id: str = "panda_pick_place"
    # Local path the dataset is synced to (from S3) before training. Random-access
    # video decode over S3 is far too slow, so we always read from local disk.
    data_root: str = "data/panda_pick_place"
    cameras: tuple = ("front", "diag", "wrist")
    state_dim: int = 8                 # 7 arm joints + gripper finger pos
    action_dim: int = 8                # 7 arm joint targets + gripper ctrl
    image_hw: tuple = (480, 640)
    fps: int = 30
    # RGBD toggle. When True, every batch must include `observation.depths.{cam}`
    # (recorder saves depth as a uint8-quantized video; decoded tensor has 3
    # identical channels -- we take the first). The shared vision backbone`s
    # conv1 is expanded from 3 to 4 input channels; the new depth channel is
    # initialized to mean(pretrained RGB filters) so edge/blob detectors get a
    # warm start on depth-as-shape-cue instead of starting from noise.
    use_depth: bool = False

    # --- action chunking --------------------------------------------------
    chunk_size: int = 100              # k: actions predicted per query (~3.3s @30fps)
    n_obs_steps: int = 1               # ACT sees only the current observation

    # --- vision backbone --------------------------------------------------
    vision_backbone: str = "resnet18"
    # One shared backbone across all cameras. IMAGENET weights -> faster convergence.
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"

    # --- transformer ------------------------------------------------------
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # The original ACT had a bug where only the first decoder layer ran; the
    # paper/lerobot intentionally match it with 1 layer. We follow suit.
    n_decoder_layers: int = 1
    pre_norm: bool = True
    dropout: float = 0.1

    # --- CVAE -------------------------------------------------------------
    use_vae: bool = True
    latent_dim: int = 32               # style variable z
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0            # beta on the KL term

    # --- inference (temporal ensembling) ----------------------------------
    # w_i = exp(-coeff * i), i=0 == oldest prediction. Smaller -> trust new obs faster.
    temporal_ensemble_coeff: float = 0.01

    # --- training ---------------------------------------------------------
    batch_size: int = 8
    num_workers: int = 4
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    n_steps: int = 100_000
    grad_clip_norm: float = 10.0
    val_fraction: float = 0.1
    seed: int = 1000

    # mixed precision (safe + fast on CUDA; auto-disabled off-CUDA)
    use_amp: bool = True

    # --- logging / io -----------------------------------------------------
    device: str = field(default_factory=_auto_device)
    output_dir: str = "checkpoints"
    log_freq: int = 100
    val_freq: int = 1000
    save_freq: int = 5000

    def __post_init__(self):
        assert self.n_obs_steps == 1, "ACT only supports a single observation step."
        assert self.chunk_size >= 1
        # AMP only makes sense on CUDA; MPS/CPU autocast is flaky or pointless.
        if self.device != "cuda":
            self.use_amp = False

    def to_dict(self) -> dict:
        return asdict(self)
