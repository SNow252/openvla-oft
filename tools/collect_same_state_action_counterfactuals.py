#!/usr/bin/env python3
"""
Collect same-state action counterfactuals for LIBERO.

v0 goal:
  For the same initial state, execute multiple candidate action chunks and
  measure different source-approach progress labels.

This avoids the observational confounding in normal VLA rollouts:
  state -> deterministic/overfit policy action -> outcome

Instead:
  same state -> different action chunks -> different progress/outcomes

First target:
  LIBERO-spatial task8
  language source = akita_black_bowl_2
  default source  = akita_black_bowl_1
  target          = plate_1
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def to_numpy(x: Any) -> np.ndarray | None:
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
    """Extract compact robot/object state from LIBERO obs dict."""
    summary: Dict[str, List[float]] = {}
    if not isinstance(obs, dict):
        return summary

    for key in SUMMARY_KEYS:
        if key in obs:
            arr = as_vec(obs[key])
            summary[key] = arr.tolist()

    # Also keep any small *_pos keys to help debug object naming.
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
    """Positive progress means EEF got closer to that object."""
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

    return out


def extract_frame(obs: Dict[str, Any]) -> np.ndarray | None:
    """Best-effort image extraction for video saving."""
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

        # Handle CHW -> HWC.
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


def action_move_to_object(
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


def action_random_local(rng: np.random.Generator, random_std: float, max_action: float, gripper_value: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = rng.normal(loc=0.0, scale=random_std, size=3)
    action[:3] = np.clip(action[:3], -max_action, max_action)
    action[6] = float(gripper_value)
    return action


def action_noisy_move_to_object(
    obs: Dict[str, Any],
    object_key: str,
    z_offset: float,
    kp: float,
    max_action: float,
    gripper_value: float,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    action = action_move_to_object(obs, object_key, z_offset, kp, max_action, gripper_value)
    action[:3] += rng.normal(loc=0.0, scale=noise_std, size=3)
    action[:3] = np.clip(action[:3], -max_action, max_action)
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


def run_candidate(
    env: Any,
    init_state: Any,
    init_state_idx: int,
    candidate_name: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
    candidate_dir: Path,
) -> Dict[str, Any]:
    obs = reset_to_init(env, init_state)
    start_summary = summarize_obs(obs)

    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    frames: List[np.ndarray] = []

    first_frame = extract_frame(obs)
    if args.save_video and first_frame is not None:
        frames.append(first_frame)

    for t in range(args.horizon):
        if candidate_name == "move_to_bowl2":
            action = action_move_to_object(
                obs,
                OBJECT_KEYS["bowl2"],
                args.z_offset,
                args.kp,
                args.max_action,
                args.gripper_value,
            )
        elif candidate_name == "move_to_bowl1":
            action = action_move_to_object(
                obs,
                OBJECT_KEYS["bowl1"],
                args.z_offset,
                args.kp,
                args.max_action,
                args.gripper_value,
            )
        elif candidate_name == "move_to_plate":
            action = action_move_to_object(
                obs,
                OBJECT_KEYS["plate"],
                args.z_offset,
                args.kp,
                args.max_action,
                args.gripper_value,
            )
        elif candidate_name == "stall_zero":
            action = action_stall(args.gripper_value)
        elif candidate_name == "random_local":
            action = action_random_local(rng, args.random_std, args.max_action, args.gripper_value)
        elif candidate_name == "noisy_bowl2":
            action = action_noisy_move_to_object(
                obs,
                OBJECT_KEYS["bowl2"],
                args.z_offset,
                args.kp,
                args.max_action,
                args.gripper_value,
                rng,
                args.noise_std,
            )
        else:
            raise ValueError(f"Unknown candidate_name: {candidate_name}")

        obs, reward, done, info = step_env(env, action)

        actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))

        frame = extract_frame(obs)
        if args.save_video and frame is not None:
            frames.append(frame)

        if done and not args.continue_after_done:
            break

    final_summary = summarize_obs(obs)
    progress = compute_progress(start_summary, final_summary)

    if "progress_to_bowl2" in progress and "progress_to_bowl1" in progress:
        progress["source_advantage_bowl2_over_bowl1"] = (
            progress["progress_to_bowl2"] - progress["progress_to_bowl1"]
        )
        progress["bowl2_over_bowl1"] = float(
            progress["source_advantage_bowl2_over_bowl1"] > 0.0
        )

    if "progress_to_bowl2" in progress and "progress_to_plate" in progress:
        progress["source_advantage_bowl2_over_plate"] = (
            progress["progress_to_bowl2"] - progress["progress_to_plate"]
        )
        progress["bowl2_over_plate"] = float(
            progress["source_advantage_bowl2_over_plate"] > 0.0
        )

    if "progress_to_bowl1" in progress and "progress_to_plate" in progress:
        progress["default_source_over_plate"] = (
            progress["progress_to_bowl1"] - progress["progress_to_plate"]
        )
    
    candidate_dir.mkdir(parents=True, exist_ok=True)

    actions_arr = np.asarray(actions, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)

    np.savez_compressed(
        candidate_dir / "traj.npz",
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        init_state_idx=np.asarray(init_state_idx, dtype=np.int32),
    )

    meta = {
        "init_state_idx": init_state_idx,
        "candidate_name": candidate_name,
        "horizon_requested": args.horizon,
        "horizon_executed": int(len(actions)),
        "start_summary": start_summary,
        "final_summary": final_summary,
        "progress": progress,
        "sum_reward": float(np.sum(rewards_arr)) if len(rewards_arr) else 0.0,
        "any_done": bool(np.any(dones_arr)) if len(dones_arr) else False,
        "args": {
            "kp": args.kp,
            "max_action": args.max_action,
            "z_offset": args.z_offset,
            "random_std": args.random_std,
            "noise_std": args.noise_std,
            "gripper_value": args.gripper_value,
        },
    }

    with open(candidate_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_video:
        save_video(frames, candidate_dir / "video.mp4", args.video_fps)

    row = {
        "init_state_idx": init_state_idx,
        "candidate_name": candidate_name,
        "horizon_executed": int(len(actions)),
        "sum_reward": meta["sum_reward"],
        "any_done": int(meta["any_done"]),
    }
    row.update(progress)
    return row


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    # Stable field order.
    base_fields = ["init_state_idx", "candidate_name", "horizon_executed", "sum_reward", "any_done"]
    extra_fields = sorted({k for r in rows for k in r.keys()} - set(base_fields))
    fields = base_fields + extra_fields

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_within_seed_progress(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["init_state_idx"]), []).append(row)

    print("\n" + "=" * 80)
    print("[within-seed progress preview]")
    for init_idx in sorted(grouped.keys())[:10]:
        print(f"\ninit_state_idx={init_idx}")
        for row in grouped[init_idx]:
            p2 = row.get("progress_to_bowl2", float("nan"))
            p1 = row.get("progress_to_bowl1", float("nan"))
            pp = row.get("progress_to_plate", float("nan"))
            print(
                f"  {row['candidate_name']:<16s} "
                f"prog_bowl2={p2:+.4f}  "
                f"prog_bowl1={p1:+.4f}  "
                f"prog_plate={pp:+.4f}"
            )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)

    # Use LIBERO init state index as "same-state seed".
    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--out_dir", type=str, required=True)

    # Simple controller params.
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--max_action", type=float, default=0.5)
    parser.add_argument("--z_offset", type=float, default=0.12)
    parser.add_argument("--gripper_value", type=float, default=0.0)

    # Noise/random params.
    parser.add_argument("--random_std", type=float, default=0.20)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--continue_after_done", action="store_true")

    parser.add_argument(
        "--candidates",
        type=str,
        default="move_to_bowl2,move_to_bowl1,move_to_plate,stall_zero,random_local,noisy_bowl2",
        help="Comma-separated candidate names.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[env] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("[env] PYTHONPATH =", os.environ.get("PYTHONPATH", "")[:500])
    print("=" * 80)

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)

    print(f"[task_suite] {args.task_suite_name}")
    print(f"[task_id] {args.task_id}")
    print(f"[task_language] {getattr(task, 'language', '')}")
    print(f"[num_init_states] {len(init_states)}")

    end_idx = args.start_init_state_idx + args.num_seeds
    if end_idx > len(init_states):
        raise ValueError(
            f"Requested init states [{args.start_init_state_idx}, {end_idx}), "
            f"but only {len(init_states)} init states are available."
        )

    env = make_env(args, task)
    rng = np.random.default_rng(args.seed)

    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    print(f"[candidates] {candidates}")

    all_rows: List[Dict[str, Any]] = []

    try:
        for local_i, init_state_idx in enumerate(range(args.start_init_state_idx, end_idx)):
            print("\n" + "-" * 80)
            print(f"[init_state] {init_state_idx} ({local_i + 1}/{args.num_seeds})")
            seed_dir = out_dir / f"init_{init_state_idx:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Save one initial state summary.
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

            for cand_idx, cand_name in enumerate(candidates):
                print(f"[candidate] {cand_idx:02d} {cand_name}")
                cand_dir = seed_dir / f"candidate_{cand_idx:02d}_{cand_name}"
                row = run_candidate(
                    env=env,
                    init_state=init_states[init_state_idx],
                    init_state_idx=init_state_idx,
                    candidate_name=cand_name,
                    args=args,
                    rng=rng,
                    candidate_dir=cand_dir,
                )
                all_rows.append(row)

                p2 = row.get("progress_to_bowl2", float("nan"))
                p1 = row.get("progress_to_bowl1", float("nan"))
                pp = row.get("progress_to_plate", float("nan"))
                print(
                    f"    progress_to_bowl2={p2:+.4f}, "
                    f"progress_to_bowl1={p1:+.4f}, "
                    f"progress_to_plate={pp:+.4f}"
                )

            # Update summary after each init state, so partial results survive interruption.
            write_summary_csv(all_rows, out_dir / "summary.csv")

    finally:
        if hasattr(env, "close"):
            env.close()

    write_summary_csv(all_rows, out_dir / "summary.csv")
    print_within_seed_progress(all_rows)

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[done] saved to: {out_dir}")
    print(f"[done] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()