"""Closed-loop evaluation of a trained ACT policy in MuJoCo (mac / glfw).

This is the *only* measure of real performance: training optimizes offline L1+KL,
which correlates only loosely with task success. Here we load a checkpoint, spawn
randomized scenes (scene_gen), drive the arm with the policy, and check whether the
object ends up in the bin.

Bridge details:
  - Reuses control.Viewer.grab() to render the 3 cameras through the offscreen GL
    buffer at 640x480 -- identical to how the dataset was recorded.
  - Queries the policy every frame (n_action_steps=1) and temporally ensembles.
  - Holds each predicted control target for one policy frame (1/fps) = several
    physics substeps, mirroring the 30 fps decimation used during recording.

Run from Panda/ (one level above ACT/) so `import control`/`scene_gen` resolve:
    cd Panda
    KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
      ~/miniconda3/envs/python_robotics/bin/python ACT/rollout.py \
      --checkpoint ACT/checkpoints/best.pt --episodes 20
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import mujoco
import torch

# ACT package lives in ./ACT; add it so the flat imports (config/model) resolve
# whether we're launched from Panda/ or from inside ACT/.
HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PANDA)

from config import ACTConfig
from model import ACTPolicy

import control as C            # global model/data + Viewer + reset_episode + GRIPPER
from scene_gen import SceneManager


def load_policy(ckpt_path: str, device: str, stats_path: str) -> tuple[ACTPolicy, ACTConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ACTConfig(**ckpt["config"])
    cfg.device = device                      # override training device for local rollout
    stats = json.load(open(stats_path))
    policy = ACTPolicy(cfg, stats)
    policy.load_state_dict(ckpt["model"])    # restores norm buffers too
    policy.to(device).eval()
    return policy, cfg


def build_obs(viewer, cams, device, use_depth: bool = True) -> dict:
    """Grab the 3 cameras + proprio state into the policy`s expected obs dict.

    Depth branch: replicate the training-time transform exactly so normalization
    stats line up. The training pipeline was:
        float32 metres  ->  depth_to_uint8 (tile to 3 channels)
                        ->  H.264 encode+decode  ->  CHW float32 in [0,1]
    At inference we skip the codec (no benefit, only loss), but keep the same
    uint8 quantization step so the tensor magnitudes match what the model saw."""
    from recorder import depth_to_uint8, CAMS  # noqa: F401 (import late so recorder deps do not load unnecessarily)
    obs = {}
    state = np.concatenate([C.data.qpos[:7], [C.data.qpos[7]]]).astype(np.float32)
    obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    for cam in cams:
        if use_depth:
            rgb, dep = viewer.grab_rgbd(cam)     # (H,W,3) uint8 + (H,W) float32 metres
        else:
            rgb = viewer.grab(cam)               # (H,W,3) uint8
            dep = None
        t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
        obs[f"observation.images.{cam}"] = t.unsqueeze(0).to(device)
        if use_depth:
            dep_u8 = depth_to_uint8(dep)         # (H,W,3) uint8 -- same quantization as recorder
            td = torch.from_numpy(dep_u8).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
            obs[f"observation.depths.{cam}"] = td.unsqueeze(0).to(device)
    return obs


def is_success(info: dict) -> bool:
    """Object resting inside the bin: centre within the inner footprint and below
    the rim (so it actually dropped in, not hovering above)."""
    bx, by, bin_z = info["bin_pose"]
    half = info["bin"]["half"]
    rim = bin_z + info["bin"]["wh"]
    obj = C.data.body("object").xpos
    in_xy = abs(obj[0] - bx) <= half and abs(obj[1] - by) <= half
    below_rim = obj[2] <= rim + 0.01
    on_floor = obj[2] > 0.0
    return bool(in_xy and below_rim and on_floor)


def run_episode(policy, cfg, sm, rng, viewer, max_steps, render=True) -> bool:
    info = C.reset_episode(sm, rng, viewer)
    policy.reset()                           # fresh temporal-ensemble buffer
    substeps = max(1, round((1.0 / cfg.fps) / C.SIM_DT))  # physics steps per policy frame

    success = False
    for _ in range(max_steps):
        obs = build_obs(viewer, cfg.cameras, cfg.device, use_depth=cfg.use_depth)
        action = policy.select_action(obs)[0].cpu().numpy()  # (8,) absolute targets
        C.data.ctrl[:7] = action[:7]
        C.data.ctrl[C.GRIPPER] = action[7]
        np.clip(C.data.ctrl, C.CTRL_RANGE[:, 0], C.CTRL_RANGE[:, 1], out=C.data.ctrl)
        for _ in range(substeps):
            mujoco.mj_step(C.model, C.data)
        if render:
            viewer.render()
        if is_success(info):                 # early stop once the object settles in
            success = True
            break
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ACT/checkpoints/best.pt")
    ap.add_argument("--stats", default="ACT/data/panda_pick_place/meta/stats.json")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500, help="policy frames per episode (@fps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu/mps/cuda (default: auto)")
    ap.add_argument("--no_render", action="store_true", help="run without the live window")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    policy, cfg = load_policy(args.checkpoint, device, args.stats)
    print(f"[rollout] device={device} episodes={args.episodes} "
          f"chunk={cfg.chunk_size} ensemble_coeff={cfg.temporal_ensemble_coeff}")

    sm = SceneManager()
    rng = np.random.default_rng(args.seed)
    viewer = C.Viewer(title="ACT rollout")

    results = []
    for ep in range(args.episodes):
        ok = run_episode(policy, cfg, sm, rng, viewer,
                         max_steps=args.max_steps, render=not args.no_render)
        results.append(ok)
        print(f"  episode {ep + 1:2d}/{args.episodes}: {'SUCCESS' if ok else 'fail'} "
              f"(running {sum(results)}/{len(results)} = {np.mean(results):.0%})")

    viewer.close()
    print(f"\n[rollout] success rate: {sum(results)}/{len(results)} = {np.mean(results):.1%}")


if __name__ == "__main__":
    main()
