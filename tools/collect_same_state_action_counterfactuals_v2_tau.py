#!/usr/bin/env python3
"""
Collect same-state continuous action counterfactuals with per-step state
sequences and hitting-time / reachability labels.

v2 goal:
  Upgrade v1 progress-only labels to reachability / hitting-time labels.

Core design:
  For the same initial state, generate continuous target points:

      target_point = alpha * bowl2 + (1 - alpha) * bowl1 + noise

  Execute a waypoint controller toward this target point for H steps.

Saved per candidate:
  - actions: [T, 7]
  - rewards: [T]
  - dones: [T]
  - eef_pos_seq: [T+1, 3]
  - bowl1_pos_seq: [T+1, 3]
  - bowl2_pos_seq: [T+1, 3]
  - plate_pos_seq: [T+1, 3]
  - ramekin_pos_seq: [T+1, 3]
  - dist_eef_to_bowl*_seq: [T+1]
  - progress labels
  - tau_source_eps labels for multiple thresholds

Default task:
  LIBERO-spatial task8
  default source:  akita_black_bowl_1
  language source: akita_black_bowl_2
  target object:   plate_1
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


OBJECT_KEYS = {
    "bowl1": "akita_black_bowl_1_pos",
    "bowl2": "akita_black_bowl_2_pos",
    "plate": "plate_1_pos",
    "ramekin": "glazed_rim_porcelain_ramekin_1_pos",
}

SUMMARY_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "akita_black_bowl_1_pos",
    "akita_black_bowl_2_pos",
    "plate_1_pos",
    "glazed_rim_porcelain_ramekin_1_pos",
]

IMAGE_KEY_HINTS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "frontview_image",
    "sideview_image",
    "image",
]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_name(x: str) -> str:
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    return x


def threshold_key(eps: float) -> str:
    return str(eps).replace(".", "p").replace("-", "m")


def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def as_vec(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    if arr is None:
        raise ValueError(f"Cannot convert to numpy: {type(x)}")
    return arr.astype(float).reshape(-1)


def summarize_obs(obs: Dict[str, Any]) -> Dict[str, List[float]]:
    summary: Dict[str, List[float]] = {}
    if not isinstance(obs, dict):
        return summary

    for key in SUMMARY_KEYS:
        if key in obs:
            arr = as_vec(obs[key])
            summary[key] = arr.tolist()

    for key, value in obs.items():
        if key.endswith("_pos") and key not in summary:
            arr = as_vec(value)
            if arr.size <= 10:
                summary[key] = arr.tolist()

    return summary


def get_pos(summary: Dict[str, List[float]], key: str) -> np.ndarray:
    if key not in summary:
        raise KeyError(f"Missing key in summary: {key}. Available keys: {sorted(summary.keys())}")
    return np.asarray(summary[key], dtype=float).reshape(-1)[:3]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def compute_progress(start_summary: Dict[str, List[float]], final_summary: Dict[str, List[float]]) -> Dict[str, float]:
    """Positive progress means EEF got closer to the object."""
    eef_start = get_pos(start_summary, "robot0_eef_pos")
    eef_final = get_pos(final_summary, "robot0_eef_pos")

    out: Dict[str, float] = {}
    for short_name, obs_key in OBJECT_KEYS.items():
        if obs_key in start_summary and obs_key in final_summary:
            obj_start = get_pos(start_summary, obs_key)
            obj_final = get_pos(final_summary, obs_key)

            d0 = dist(eef_start, obj_start)
            d1 = dist(eef_final, obj_final)

            out[f"start_dist_eef_to_{short_name}"] = d0
            out[f"final_dist_eef_to_{short_name}"] = d1
            out[f"progress_to_{short_name}"] = d0 - d1

    if "progress_to_bowl2" in out and "progress_to_bowl1" in out:
        out["source_advantage_bowl2_over_bowl1"] = out["progress_to_bowl2"] - out["progress_to_bowl1"]
        out["bowl2_over_bowl1"] = float(out["source_advantage_bowl2_over_bowl1"] > 0.0)

    if "progress_to_bowl2" in out and "progress_to_plate" in out:
        out["source_advantage_bowl2_over_plate"] = out["progress_to_bowl2"] - out["progress_to_plate"]
        out["bowl2_over_plate"] = float(out["source_advantage_bowl2_over_plate"] > 0.0)

    if "progress_to_bowl1" in out and "progress_to_plate" in out:
        out["default_source_over_plate"] = out["progress_to_bowl1"] - out["progress_to_plate"]

    return out


def extract_state_for_sequence(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    summary = summarize_obs(obs)

    state = {
        "eef": get_pos(summary, "robot0_eef_pos").astype(np.float32),
        "bowl1": get_pos(summary, OBJECT_KEYS["bowl1"]).astype(np.float32),
        "bowl2": get_pos(summary, OBJECT_KEYS["bowl2"]).astype(np.float32),
        "plate": get_pos(summary, OBJECT_KEYS["plate"]).astype(np.float32),
        "ramekin": get_pos(summary, OBJECT_KEYS["ramekin"]).astype(np.float32),
    }
    return state


def append_sequence(seq: Dict[str, List[np.ndarray]], obs: Dict[str, Any]) -> None:
    state = extract_state_for_sequence(obs)

    for key in ["eef", "bowl1", "bowl2", "plate", "ramekin"]:
        seq[f"{key}_pos"].append(state[key])

    eef = state["eef"]
    for obj in ["bowl1", "bowl2", "plate", "ramekin"]:
        seq[f"dist_eef_to_{obj}"].append(np.array(dist(eef, state[obj]), dtype=np.float32))


def sequence_to_arrays(seq: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    for key, values in seq.items():
        arr = np.asarray(values, dtype=np.float32)
        out[f"{key}_seq"] = arr

    return out


def compute_tau_labels(seq_arrays: Dict[str, np.ndarray], thresholds: List[float]) -> Dict[str, float]:
    labels: Dict[str, float] = {}

    d_bowl2 = np.asarray(seq_arrays["dist_eef_to_bowl2_seq"], dtype=np.float32).reshape(-1)
    d_bowl1 = np.asarray(seq_arrays["dist_eef_to_bowl1_seq"], dtype=np.float32).reshape(-1)

    H = int(len(d_bowl2) - 1)

    labels["min_dist_to_bowl2"] = float(np.min(d_bowl2))
    labels["final_dist_to_bowl2"] = float(d_bowl2[-1])
    labels["min_dist_to_bowl1"] = float(np.min(d_bowl1))
    labels["final_dist_to_bowl1"] = float(d_bowl1[-1])

    labels["min_dist_advantage_bowl1_minus_bowl2"] = labels["min_dist_to_bowl1"] - labels["min_dist_to_bowl2"]
    labels["final_dist_advantage_bowl1_minus_bowl2"] = labels["final_dist_to_bowl1"] - labels["final_dist_to_bowl2"]

    # tau_adv: first time EEF is closer to bowl2 than bowl1.
    adv_hits = np.where(d_bowl2 < d_bowl1)[0]
    if len(adv_hits) > 0:
        labels["reached_adv_bowl2_over_bowl1"] = 1.0
        labels["tau_adv_bowl2_over_bowl1"] = float(int(adv_hits[0]))
    else:
        labels["reached_adv_bowl2_over_bowl1"] = 0.0
        labels["tau_adv_bowl2_over_bowl1"] = float(H + 1)

    for eps in thresholds:
        key = threshold_key(eps)
        hits = np.where(d_bowl2 <= eps)[0]

        if len(hits) > 0:
            labels[f"reached_source_eps_{key}"] = 1.0
            labels[f"tau_source_eps_{key}"] = float(int(hits[0]))
        else:
            labels[f"reached_source_eps_{key}"] = 0.0
            labels[f"tau_source_eps_{key}"] = float(H + 1)

    return labels


def extract_frame(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict):
        return None

    candidate_keys: List[str] = []
    for hint in IMAGE_KEY_HINTS:
        for k in obs.keys():
            if hint in k and k not in candidate_keys:
                candidate_keys.append(k)

    for key in candidate_keys:
        arr = to_numpy(obs.get(key))
        if arr is None:
            continue

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3:
            continue

        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        if arr.shape[-1] != 3:
            continue

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    return None


def make_xyz_action(
    current_eef: np.ndarray,
    waypoint: np.ndarray,
    kp: float,
    max_action: float,
    gripper_value: float,
) -> np.ndarray:
    delta = waypoint - current_eef
    xyz = kp * delta
    xyz = np.clip(xyz, -max_action, max_action)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = xyz.astype(np.float32)
    action[3:6] = 0.0
    action[6] = float(gripper_value)
    return action


def make_interp_target_point(
    summary: Dict[str, List[float]],
    alpha: float,
    target_noise_std: float,
    rng: np.random.Generator,
    noise_xy_only: bool = True,
) -> np.ndarray:
    bowl1 = get_pos(summary, OBJECT_KEYS["bowl1"])
    bowl2 = get_pos(summary, OBJECT_KEYS["bowl2"])

    target = alpha * bowl2 + (1.0 - alpha) * bowl1

    noise = rng.normal(loc=0.0, scale=target_noise_std, size=3)
    if noise_xy_only:
        noise[2] = 0.0

    return target + noise


def action_toward_target_point(
    obs: Dict[str, Any],
    target_point: np.ndarray,
    z_offset: float,
    kp: float,
    max_action: float,
    gripper_value: float,
    action_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    summary = summarize_obs(obs)
    eef = get_pos(summary, "robot0_eef_pos")
    waypoint = np.asarray(target_point, dtype=float).copy()
    waypoint[2] += z_offset

    action = make_xyz_action(eef, waypoint, kp, max_action, gripper_value)

    if action_noise_std > 0:
        action[:3] += rng.normal(loc=0.0, scale=action_noise_std, size=3)
        action[:3] = np.clip(action[:3], -max_action, max_action)

    return action.astype(np.float32)


def action_toward_object(
    obs: Dict[str, Any],
    object_key: str,
    z_offset: float,
    kp: float,
    max_action: float,
    gripper_value: float,
) -> np.ndarray:
    summary = summarize_obs(obs)
    eef = get_pos(summary, "robot0_eef_pos")
    obj = get_pos(summary, object_key)
    waypoint = obj.copy()
    waypoint[2] += z_offset
    return make_xyz_action(eef, waypoint, kp, max_action, gripper_value)


def action_stall(gripper_value: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[6] = float(gripper_value)
    return action


def action_random_local(
    rng: np.random.Generator,
    random_std: float,
    max_action: float,
    gripper_value: float,
) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = rng.normal(loc=0.0, scale=random_std, size=3)
    action[:3] = np.clip(action[:3], -max_action, max_action)
    action[6] = float(gripper_value)
    return action.astype(np.float32)


def step_env(env: Any, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    raise RuntimeError(f"Unexpected env.step output type/length: {type(out)}")


def reset_to_init(env: Any, init_state: Any) -> Dict[str, Any]:
    env.reset()
    obs = env.set_init_state(init_state)
    return obs


def make_env(args: argparse.Namespace, task: Any) -> Any:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": args.resolution,
        "camera_widths": args.resolution,
    }
    return OffScreenRenderEnv(**env_args)


def save_video(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), frames, fps=fps)
    except Exception as e:
        print(f"[video warning] failed to save {path}: {repr(e)}")


def build_candidate_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    alphas = parse_float_list(args.alphas)
    target_noise_stds = parse_float_list(args.target_noise_stds)

    for alpha in alphas:
        for noise_std in target_noise_stds:
            name = f"interp_alpha_{alpha:+.2f}_noise_{noise_std:.2f}"
            specs.append(
                {
                    "candidate_name": safe_name(name),
                    "candidate_family": "interp_bowl1_bowl2",
                    "alpha": float(alpha),
                    "target_noise_std": float(noise_std),
                    "z_offset": float(args.z_offset),
                    "action_noise_std": float(args.action_noise_std),
                }
            )

    if not args.no_plate:
        specs.append(
            {
                "candidate_name": "move_to_plate",
                "candidate_family": "move_to_plate",
                "alpha": float("nan"),
                "target_noise_std": 0.0,
                "z_offset": float(args.z_offset),
                "action_noise_std": 0.0,
            }
        )

    if not args.no_stall:
        specs.append(
            {
                "candidate_name": "stall_zero",
                "candidate_family": "stall",
                "alpha": float("nan"),
                "target_noise_std": 0.0,
                "z_offset": float("nan"),
                "action_noise_std": 0.0,
            }
        )

    if not args.no_random:
        specs.append(
            {
                "candidate_name": "random_local",
                "candidate_family": "random",
                "alpha": float("nan"),
                "target_noise_std": float(args.random_std),
                "z_offset": float("nan"),
                "action_noise_std": float(args.random_std),
            }
        )

    return specs


def run_candidate(
    env: Any,
    init_state: Any,
    init_state_idx: int,
    spec: Dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    thresholds: List[float],
    candidate_dir: Path,
) -> Dict[str, Any]:
    obs = reset_to_init(env, init_state)
    start_summary = summarize_obs(obs)

    target_point = None
    if spec["candidate_family"] == "interp_bowl1_bowl2":
        target_point = make_interp_target_point(
            start_summary,
            alpha=float(spec["alpha"]),
            target_noise_std=float(spec["target_noise_std"]),
            rng=rng,
            noise_xy_only=not args.noise_xyz,
        )

    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    frames: List[np.ndarray] = []

    seq: Dict[str, List[np.ndarray]] = {
        "eef_pos": [],
        "bowl1_pos": [],
        "bowl2_pos": [],
        "plate_pos": [],
        "ramekin_pos": [],
        "dist_eef_to_bowl1": [],
        "dist_eef_to_bowl2": [],
        "dist_eef_to_plate": [],
        "dist_eef_to_ramekin": [],
    }

    append_sequence(seq, obs)

    first_frame = extract_frame(obs)
    if args.save_video and first_frame is not None:
        frames.append(first_frame)

    for _t in range(args.horizon):
        family = spec["candidate_family"]

        if family == "interp_bowl1_bowl2":
            assert target_point is not None
            action = action_toward_target_point(
                obs=obs,
                target_point=target_point,
                z_offset=float(spec["z_offset"]),
                kp=args.kp,
                max_action=args.max_action,
                gripper_value=args.gripper_value,
                action_noise_std=float(spec["action_noise_std"]),
                rng=rng,
            )
        elif family == "move_to_plate":
            action = action_toward_object(
                obs=obs,
                object_key=OBJECT_KEYS["plate"],
                z_offset=args.z_offset,
                kp=args.kp,
                max_action=args.max_action,
                gripper_value=args.gripper_value,
            )
        elif family == "stall":
            action = action_stall(args.gripper_value)
        elif family == "random":
            action = action_random_local(
                rng=rng,
                random_std=args.random_std,
                max_action=args.max_action,
                gripper_value=args.gripper_value,
            )
        else:
            raise ValueError(f"Unknown candidate_family: {family}")

        obs, reward, done, _info = step_env(env, action)

        actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))

        append_sequence(seq, obs)

        frame = extract_frame(obs)
        if args.save_video and frame is not None:
            frames.append(frame)

        if done and not args.continue_after_done:
            break

    final_summary = summarize_obs(obs)
    progress = compute_progress(start_summary, final_summary)

    seq_arrays = sequence_to_arrays(seq)
    tau_labels = compute_tau_labels(seq_arrays, thresholds=thresholds)

    candidate_dir.mkdir(parents=True, exist_ok=True)

    actions_arr = np.asarray(actions, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)

    npz_payload = {
        "actions": actions_arr,
        "rewards": rewards_arr,
        "dones": dones_arr,
        "init_state_idx": np.asarray(init_state_idx, dtype=np.int32),
    }
    npz_payload.update(seq_arrays)

    np.savez_compressed(candidate_dir / "traj.npz", **npz_payload)

    target_point_list = None if target_point is None else np.asarray(target_point, dtype=float).tolist()

    meta = {
        "init_state_idx": init_state_idx,
        "candidate_name": spec["candidate_name"],
        "candidate_family": spec["candidate_family"],
        "candidate_spec": spec,
        "target_point": target_point_list,
        "horizon_requested": args.horizon,
        "horizon_executed": int(len(actions)),
        "start_summary": start_summary,
        "final_summary": final_summary,
        "progress": progress,
        "tau_labels": tau_labels,
        "sum_reward": float(np.sum(rewards_arr)) if len(rewards_arr) else 0.0,
        "any_done": bool(np.any(dones_arr)) if len(dones_arr) else False,
        "controller_args": {
            "kp": args.kp,
            "max_action": args.max_action,
            "z_offset": args.z_offset,
            "gripper_value": args.gripper_value,
            "random_std": args.random_std,
            "noise_xyz": bool(args.noise_xyz),
            "thresholds": thresholds,
        },
    }

    with open(candidate_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_video:
        save_video(frames, candidate_dir / "video.mp4", args.video_fps)

    row: Dict[str, Any] = {
        "init_state_idx": init_state_idx,
        "candidate_name": spec["candidate_name"],
        "candidate_family": spec["candidate_family"],
        "alpha": spec.get("alpha", float("nan")),
        "target_noise_std": spec.get("target_noise_std", float("nan")),
        "z_offset": spec.get("z_offset", float("nan")),
        "action_noise_std": spec.get("action_noise_std", float("nan")),
        "target_point_x": float(target_point[0]) if target_point is not None else float("nan"),
        "target_point_y": float(target_point[1]) if target_point is not None else float("nan"),
        "target_point_z": float(target_point[2]) if target_point is not None else float("nan"),
        "horizon_executed": int(len(actions)),
        "sum_reward": meta["sum_reward"],
        "any_done": int(meta["any_done"]),
    }
    row.update(progress)
    row.update(tau_labels)
    return row


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    base_fields = [
        "init_state_idx",
        "candidate_name",
        "candidate_family",
        "alpha",
        "target_noise_std",
        "z_offset",
        "action_noise_std",
        "target_point_x",
        "target_point_y",
        "target_point_z",
        "horizon_executed",
        "sum_reward",
        "any_done",
    ]
    extra_fields = sorted({k for r in rows for k in r.keys()} - set(base_fields))
    fields = base_fields + extra_fields

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_tau_preview(rows: List[Dict[str, Any]], thresholds: List[float], max_seeds: int = 3) -> None:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["init_state_idx"]), []).append(row)

    print("\n" + "=" * 120)
    print("[within-seed v2 tau preview]")

    first_eps_key = threshold_key(thresholds[0])

    for init_idx in sorted(grouped.keys())[:max_seeds]:
        print(f"\ninit_state_idx={init_idx}")

        rows_i = grouped[init_idx]

        def sort_key(r: Dict[str, Any]):
            fam = str(r.get("candidate_family"))
            alpha = r.get("alpha", float("nan"))
            noise = r.get("target_noise_std", float("nan"))
            alpha_key = 999.0 if isinstance(alpha, float) and math.isnan(alpha) else float(alpha)
            noise_key = 999.0 if isinstance(noise, float) and math.isnan(noise) else float(noise)
            return (fam, alpha_key, noise_key, str(r["candidate_name"]))

        for row in sorted(rows_i, key=sort_key):
            adv = row.get("source_advantage_bowl2_over_bowl1", float("nan"))
            min_d = row.get("min_dist_to_bowl2", float("nan"))
            tau = row.get(f"tau_source_eps_{first_eps_key}", float("nan"))
            reached = row.get(f"reached_source_eps_{first_eps_key}", float("nan"))
            alpha = row.get("alpha", float("nan"))
            print(
                f"  {row['candidate_name']:<32s} "
                f"fam={row['candidate_family']:<18s} "
                f"alpha={alpha!s:<6s} "
                f"adv={adv:+.4f} "
                f"min_d_bowl2={min_d:.4f} "
                f"tau_eps_{first_eps_key}={tau:>5} "
                f"reached={reached}"
            )

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)
    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--alphas", type=str, default="-0.25,0,0.25,0.5,0.75,0.85,0.95,1.0,1.05,1.15,1.25")
    parser.add_argument("--target_noise_stds", type=str, default="0,0.02,0.05")
    parser.add_argument("--thresholds", type=str, default="0.12,0.10,0.08")

    parser.add_argument("--z_offset", type=float, default=0.08)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--max_action", type=float, default=0.5)
    parser.add_argument("--gripper_value", type=float, default=0.0)

    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--random_std", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--no_plate", action="store_true")
    parser.add_argument("--no_stall", action="store_true")
    parser.add_argument("--no_random", action="store_true")
    parser.add_argument("--noise_xyz", action="store_true")

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--continue_after_done", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_float_list(args.thresholds)

    print("=" * 120)
    print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[env] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("[env] PYTHONPATH =", os.environ.get("PYTHONPATH", "")[:500])
    print("=" * 120)

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)

    print(f"[task_suite] {args.task_suite_name}")
    print(f"[task_id] {args.task_id}")
    print(f"[task_language] {getattr(task, 'language', '')}")
    print(f"[num_init_states] {len(init_states)}")
    print(f"[thresholds] {thresholds}")

    end_idx = args.start_init_state_idx + args.num_seeds
    if end_idx > len(init_states):
        raise ValueError(
            f"Requested init states [{args.start_init_state_idx}, {end_idx}), "
            f"but only {len(init_states)} init states are available."
        )

    specs = build_candidate_specs(args)
    print(f"[num candidates per init] {len(specs)}")
    for i, spec in enumerate(specs):
        print(f"  {i:03d}: {spec}")

    env = make_env(args, task)
    rng = np.random.default_rng(args.seed)

    all_rows: List[Dict[str, Any]] = []

    try:
        for local_i, init_state_idx in enumerate(range(args.start_init_state_idx, end_idx)):
            print("\n" + "-" * 120)
            print(f"[init_state] {init_state_idx} ({local_i + 1}/{args.num_seeds})")

            seed_dir = out_dir / f"init_{init_state_idx:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            init_obs = reset_to_init(env, init_states[init_state_idx])
            init_summary = summarize_obs(init_obs)

            with open(seed_dir / "initial_state.json", "w") as f:
                json.dump(
                    {
                        "init_state_idx": init_state_idx,
                        "task_suite_name": args.task_suite_name,
                        "task_id": args.task_id,
                        "task_language": getattr(task, "language", ""),
                        "summary": init_summary,
                    },
                    f,
                    indent=2,
                )

            for cand_idx, spec in enumerate(specs):
                cand_name = str(spec["candidate_name"])
                print(f"[candidate] {cand_idx:03d} {cand_name}")
                cand_dir = seed_dir / f"candidate_{cand_idx:03d}_{cand_name}"

                row = run_candidate(
                    env=env,
                    init_state=init_states[init_state_idx],
                    init_state_idx=init_state_idx,
                    spec=spec,
                    args=args,
                    rng=rng,
                    thresholds=thresholds,
                    candidate_dir=cand_dir,
                )
                all_rows.append(row)

                adv = row.get("source_advantage_bowl2_over_bowl1", float("nan"))
                min_d = row.get("min_dist_to_bowl2", float("nan"))
                tau_key = threshold_key(thresholds[0])
                tau = row.get(f"tau_source_eps_{tau_key}", float("nan"))
                reached = row.get(f"reached_source_eps_{tau_key}", float("nan"))

                print(
                    f"    source_adv={adv:+.4f}, "
                    f"min_dist_bowl2={min_d:.4f}, "
                    f"tau_eps_{tau_key}={tau}, "
                    f"reached={reached}"
                )

            write_summary_csv(all_rows, out_dir / "summary.csv")

    finally:
        if hasattr(env, "close"):
            env.close()

    write_summary_csv(all_rows, out_dir / "summary.csv")
    print_tau_preview(all_rows, thresholds=thresholds)

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[done] saved to: {out_dir}")
    print(f"[done] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()