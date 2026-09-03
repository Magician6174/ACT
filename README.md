# ACT — Action Chunking with Transformers (Panda pick-and-place)

From-scratch ACT implementation ([Zhao et al. 2023](https://arxiv.org/abs/2304.13705))
for the MuJoCo Panda pick-and-place task. Trained offline on teleop demos
(LeRobot dataset), evaluated closed-loop in the sim.

## What ACT is (1-minute recap)
A **CVAE** that predicts a *chunk* of `k` future actions per observation,
mitigating compounding error.
- **vae_encoder** (training only): sees the ground-truth action chunk + state,
  compresses the demo "style" into a latent `z`. Discarded at test time.
- **encoder** (the policy): fuses `[z, state, image tokens]` into memory.
- **decoder**: `k` learned query slots cross-attend to memory → `k`-step action chunk.
- **Loss**: masked **L1** on actions + `kl_weight · KL(q(z) ‖ N(0,I))`.
- **Inference**: `z = 0` (prior mean, deterministic), query every step,
  **temporal-ensemble** overlapping chunks (`n_action_steps = 1`).

## Files
| file | role |
|------|------|
| `config.py`  | `ACTConfig` dataclass — all dims/hparams, one source of truth |
| `dataset.py` | `LeRobotDataset` reader + action `delta_timestamps` chunking; episode-split |
| `model.py`   | network (vae_encoder/encoder/decoder), `Normalize`, `TemporalEnsembler`, `ACTPolicy` |
| `train.py`   | offline training loop (AdamW, AMP on CUDA, val-loss → `best.pt`) |
| `rollout.py` | closed-loop eval in MuJoCo (mac/glfw) → task success rate |

## Setup notes
- **State** = `[qpos[:7], finger_pos]` (8-d). **Action** = `[ctrl[:7], gripper_ctrl 0-255]` (8-d, absolute joint targets).
- 3 cameras: `front`, `diag` (fixed) + `wrist` (eye-in-hand), 480×640.
- Normalization (mean/std) comes from the dataset's `meta/stats.json`; baked into the model as buffers so it travels with the checkpoint.
- Imports only `lerobot.datasets` (the `lerobot.policies` package is broken in this checkout — a groot/huggingface_hub `StrictDataclass` clash).
- `KMP_DUPLICATE_LIB_OK=TRUE` is required (OpenMP duplicate-runtime quirk on this machine).

## Training (SageMaker Studio, g6 / CUDA)
Run **from inside `ACT/`** (flat imports: `from config import ...`).

```bash
# 1. dataset lives in S3 (durable). Sync to LOCAL EBS once — do NOT stream from
#    S3 during training (random-access video decode over S3 is far too slow).
aws s3 sync s3://<bucket>/panda_pick_place ./local_data/panda_pick_place

# 2. FIRST verify AV1 video decode works in THIS environment (backend differs
#    from the mac — torchcodec/ffmpeg). One frame is enough:
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
ds=D('panda_pick_place', root='./local_data/panda_pick_place'); \
print(ds[0]['observation.images.front'].shape)"

# 3. train
KMP_DUPLICATE_LIB_OK=TRUE python train.py \
  --data_root ./local_data/panda_pick_place \
  --n_steps 100000 --batch_size 16

# smoke test (tiny end-to-end shape/sanity check, any device):
python train.py --smoke --device cpu --data_root data/panda_pick_place
```
Checkpoints land in `checkpoints/`: `best.pt` (lowest val loss), periodic
`step_NNNNNN.pt`, and `last.pt`. `train_log.csv` logs loss/l1/kld/val.

## Rollout / evaluation (mac, glfw)
Closed-loop is the **only** real success metric (offline L1 ≠ task success). Run
**from `Panda/`** (one level above `ACT/`) so `control`/`scene_gen` resolve. Pull
`best.pt` down from Studio first.

```bash
cd Panda
KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
  ~/miniconda3/envs/python_robotics/bin/python ACT/rollout.py \
  --checkpoint ACT/checkpoints/best.pt --episodes 20
# --no_render to skip the live window, --device cpu/mps/cuda, --max_steps N
```
Reports per-episode SUCCESS/fail (object settled inside the bin footprint, below
the rim) and the overall success rate.
