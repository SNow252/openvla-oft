#!/usr/bin/env python3
"""
Replay saved policy action chunks under same LIBERO init states and compute v3
object-level consequence labels.

Purpose:
  Move from scripted grasp macro consequence prediction to learned policy action
  evaluation.

Input action chunk format:
  Each .npz should contain:
    actions: [T, 7]

Supported chunk layouts:

  Layout A:
    CHUNK_DIR/init_000/openvla_original.npz
    CHUNK_DIR/init_000/smolvla_original.npz
    CHUNK_DIR/init_001/openvla_original.npz
    ...

    Use:
      --chunk_dir CHUNK_DIR
      --chunk_glob "*.npz"

  Layout B, replay existing v3 scripted chunks:
    DATA_DIR/init_000/candidate_000_xxx/traj.npz
    DATA_DIR/init_001/candidate_000_xxx/traj.npz
    ...

    Use:
      --chunk_dir DATA_DIR
      --chunk_glob "init_{init:03d}/candidate_*/traj.npz"

Outputs:
  OUT_DIR/summary.csv
  OUT_DIR/init_XXX/candidate_YYY_<chunk_name>/traj.npz
  OUT_DIR/init_XXX/candidate_YYY_<chunk_name>/meta.json

Labels:
  Same v3 labels:
    bowl2_grasp_success_proxy
    bowl2_source_lifted
    bowl2_source_z_delta_max
    bowl2_source_displacement_max
    bowl2_grasp_quality_score
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.collect_same_state_grasp_counterfactuals_v3 import (  # noqa: E402
    append_sequence,
    compute_grasp_labels,
    extract_frame,
    make_env,
    reset_to_init,
    save_video,
    sequence_to_arrays,
    settle_after_reset,
    step_env,
    summarize_obs,
    write_summary_csv,
)


def safe_name(x: str) -> str:
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    return x


def read_npz_scalar(data: Any, key: str, default: str = "") -> str:
    if key not in data:
        return default
    try:
        v = data[key]
        if isinstance(v, np.ndarray):
            if v.shape == ():
                return str(v.item())
            if v.size == 1:
                return str(v.reshape(-1)[0])
        return str(v)
    except Exception:
        return default


def infer_chunk_name(path: Path) -> str:
    if path.name == "traj.npz":
        # Existing v3 dataset layout: candidate_000_xxx/traj.npz
        return path.parent.name
    return path.stem


def infer_policy_name(chunk_name: str) -> str:
    lower = chunk_name.lower()
    if "openvla" in lower:
        return "openvla"
    if "smolvla" in lower:
        return "smolvla"
    if "diffusion" in lower:
        return "diffusion_policy"
    if "act" in lower:
        return "act"
    if "scripted" in lower:
        return "scripted"
    if "candidate" in lower:
        return "scripted_or_existing_candidate"
    return "unknown"


def find_chunks_for_init(chunk_dir: Path, chunk_glob: str, init_state_idx: int) -> List[Path]:
    if "{init" in chunk_glob:
        rel_pattern = chunk_glob.format(init=init_state_idx)
        pattern = str(chunk_dir / rel_pattern)
    else:
        pattern = str(chunk_dir / f"init_{init_state_idx:03d}" / chunk_glob)

    return sorted(Path(p) for p in Path().glob(pattern) if Path(p).is_file())


def find_chunks_for_init_globlib(chunk_dir: Path, chunk_glob: str, init_state_idx: int) -> List[Path]:
    """
    pathlib.Path.glob does not accept absolute patterns. Use stdlib glob.
    """
    import glob

    if "{init" in chunk_glob:
        rel_pattern = chunk_glob.format(init=init_state_idx)
        pattern = str(chunk_dir / rel_pattern)
    else:
        pattern = str(chunk_dir / f"init_{init_state_idx:03d}" / chunk_glob)

    return sorted(Path(p) for p in glob.glob(pattern))


def load_actions_from_chunk(path: Path, max_steps: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)

    if "actions" not in data:
        raise KeyError(f"{path} missing key 'actions'")

    actions = np.asarray(data["actions"], dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"{path} actions should be [T, A], got shape={actions.shape}")

    if actions.shape[1] < 7:
        raise ValueError(f"{path} actions should have at least 7 dims, got shape={actions.shape}")

    actions = actions[:, :7]

    if max_steps > 0:
        actions = actions[:max_steps]

    meta = {
        "chunk_path": str(path),
        "chunk_name": read_npz_scalar(data, "chunk_name", infer_chunk_name(path)),
        "policy_name": read_npz_scalar(data, "policy_name", infer_policy_name(infer_chunk_name(path))),
        "language": read_npz_scalar(data, "language", ""),
        "raw_num_steps": int(np.asarray(data["actions"]).shape[0]),
        "used_num_steps": int(actions.shape[0]),
    }

    return actions.astype(np.float32), meta


def init_empty_sequence() -> Dict[str, List[np.ndarray]]:
    return {
        "eef_pos": [],
        "gripper_qpos": [],
        "bowl1_pos": [],
        "bowl2_pos": [],
        "plate_pos": [],
        "ramekin_pos": [],
        "dist_eef_to_bowl1": [],
        "dist_eef_to_bowl2": [],
        "dist_eef_to_plate": [],
        "dist_eef_to_ramekin": [],
    }


def replay_chunk(
    env: Any,
    init_state: Any,
    init_state_idx: int,
    actions: np.ndarray,
    chunk_meta: Dict[str, Any],
    args: argparse.Namespace,
    candidate_dir: Path,
) -> Dict[str, Any]:
    obs_raw = reset_to_init(env, init_state)
    raw_start_summary = summarize_obs(obs_raw)

    obs = settle_after_reset(
        env=env,
        obs=obs_raw,
        settle_steps=args.settle_steps,
        gripper_open_value=args.gripper_open_value,
    )
    start_summary = summarize_obs(obs)

    seq = init_empty_sequence()
    append_sequence(seq, obs)

    frames: List[np.ndarray] = []
    frame = extract_frame(obs)
    if args.save_video and frame is not None:
        frames.append(frame)

    executed_actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []

    for t in range(actions.shape[0]):
        action = actions[t].astype(np.float32)

        if args.action_clip > 0:
            action = np.clip(action, -args.action_clip, args.action_clip)

        obs, reward, done, _info = step_env(env, action)

        executed_actions.append(action.copy())
        rewards.append(float(reward))
        dones.append(bool(done))

        append_sequence(seq, obs)

        frame = extract_frame(obs)
        if args.save_video and frame is not None:
            frames.append(frame)

        if done and not args.continue_after_done:
            break

    final_summary = summarize_obs(obs)
    seq_arrays = sequence_to_arrays(seq)

    labels_bowl2 = compute_grasp_labels(
        seq_arrays=seq_arrays,
        source="bowl2",
        distractor="bowl1",
        contact_threshold=args.contact_threshold,
        move_threshold=args.move_threshold,
        lift_threshold=args.lift_threshold,
    )

    labels_bowl2_prefixed = {
        f"bowl2_{k}": v
        for k, v in labels_bowl2.items()
        if isinstance(v, (int, float, np.floating, str))
    }

    candidate_dir.mkdir(parents=True, exist_ok=True)

    executed_actions_arr = np.asarray(executed_actions, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)

    npz_payload: Dict[str, Any] = {
        "actions": executed_actions_arr,
        "input_actions": actions.astype(np.float32),
        "rewards": rewards_arr,
        "dones": dones_arr,
        "init_state_idx": np.asarray(init_state_idx, dtype=np.int32),
    }
    npz_payload.update(seq_arrays)

    np.savez_compressed(candidate_dir / "traj.npz", **npz_payload)

    meta = {
        "init_state_idx": init_state_idx,
        "candidate_name": chunk_meta["chunk_name"],
        "candidate_family": "policy_chunk",
        "policy_name": chunk_meta["policy_name"],
        "chunk_meta": chunk_meta,
        "horizon_executed": int(len(executed_actions)),
        "raw_start_summary": raw_start_summary,
        "start_summary": start_summary,
        "settle_steps": int(args.settle_steps),
        "final_summary": final_summary,
        "labels": labels_bowl2_prefixed,
        "sum_reward": float(np.sum(rewards_arr)) if len(rewards_arr) else 0.0,
        "any_done": bool(np.any(dones_arr)) if len(dones_arr) else False,
        "controller_args": vars(args),
    }

    with open(candidate_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_video:
        save_video(frames, candidate_dir / "video.mp4", args.video_fps)

    row: Dict[str, Any] = {
        "init_state_idx": init_state_idx,
        "candidate_name": chunk_meta["chunk_name"],
        "candidate_family": "policy_chunk",
        "policy_name": chunk_meta["policy_name"],
        "chunk_path": chunk_meta["chunk_path"],
        "language": chunk_meta.get("language", ""),
        "settle_steps": int(args.settle_steps),
        "horizon_executed": int(len(executed_actions)),
        "sum_reward": meta["sum_reward"],
        "any_done": int(meta["any_done"]),
    }
    row.update(labels_bowl2_prefixed)

    return row


def print_compact_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    keys = [
        "bowl2_source_displacement_max",
        "bowl2_source_z_delta_max",
        "bowl2_source_moved",
        "bowl2_source_lifted",
        "bowl2_grasp_success_proxy",
        "bowl2_clean_grasp_proxy",
        "bowl2_grasp_quality_score",
        "bowl2_min_dist_to_source",
    ]

    print("\n" + "=" * 120)
    print("[compact summary]")
    print("rows:", len(rows))

    for k in keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if vals:
            print(f"{k:36s} mean={np.mean(vals):+.5f} min={np.min(vals):+.5f} max={np.max(vals):+.5f}")

    print("\n[by policy_name]")
    policies = sorted(set(str(r.get("policy_name", "unknown")) for r in rows))
    for p in policies:
        rs = [r for r in rows if str(r.get("policy_name", "unknown")) == p]
        print(f"\npolicy={p}, n={len(rs)}")
        for k in [
            "bowl2_grasp_success_proxy",
            "bowl2_source_lifted",
            "bowl2_source_z_delta_max",
            "bowl2_grasp_quality_score",
        ]:
            vals = []
            for r in rs:
                try:
                    vals.append(float(r[k]))
                except Exception:
                    pass
            if vals:
                print(f"  {k:34s} mean={np.mean(vals):+.5f} max={np.max(vals):+.5f}")

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)

    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument("--chunk_dir", type=str, required=True)
    parser.add_argument(
        "--chunk_glob",
        type=str,
        default="*.npz",
        help=(
            "Either pattern under chunk_dir/init_XXX, e.g. '*.npz', "
            "or template with {init:03d}, e.g. 'init_{init:03d}/candidate_*/traj.npz'."
        ),
    )

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--gripper_open_value", type=float, default=-1.0)

    parser.add_argument("--contact_threshold", type=float, default=0.10)
    parser.add_argument("--move_threshold", type=float, default=0.020)
    parser.add_argument("--lift_threshold", type=float, default=0.010)

    parser.add_argument("--max_steps", type=int, default=140)
    parser.add_argument("--action_clip", type=float, default=1.0)

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--continue_after_done", action="store_true")

    parser.add_argument("--write_summary_every", type=int, default=1)

    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("[env] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[env] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("[chunk_dir]", chunk_dir)
    print("[chunk_glob]", args.chunk_glob)
    print("[out_dir]", out_dir)
    print("[max_steps]", args.max_steps)
    print("[settle_steps]", args.settle_steps)
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

    end_idx = args.start_init_state_idx + args.num_seeds
    if end_idx > len(init_states):
        raise ValueError(
            f"Requested init states [{args.start_init_state_idx}, {end_idx}), "
            f"but only {len(init_states)} init states are available."
        )

    env = make_env(args, task)

    all_rows: List[Dict[str, Any]] = []

    try:
        for local_i, init_state_idx in enumerate(range(args.start_init_state_idx, end_idx)):
            print("\n" + "-" * 120)
            print(f"[init_state] {init_state_idx} ({local_i + 1}/{args.num_seeds})")

            chunks = find_chunks_for_init_globlib(
                chunk_dir=chunk_dir,
                chunk_glob=args.chunk_glob,
                init_state_idx=init_state_idx,
            )

            if not chunks:
                print(f"[warning] no chunks found for init_state={init_state_idx}")
                continue

            print(f"[chunks found] {len(chunks)}")

            seed_dir = out_dir / f"init_{init_state_idx:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            for chunk_idx, chunk_path in enumerate(chunks):
                actions, chunk_meta = load_actions_from_chunk(chunk_path, max_steps=args.max_steps)

                chunk_name = safe_name(str(chunk_meta["chunk_name"]))
                policy_name = safe_name(str(chunk_meta["policy_name"]))

                candidate_name = safe_name(f"{policy_name}_{chunk_name}")
                chunk_meta["chunk_name"] = candidate_name

                print(f"[chunk] {chunk_idx:03d} {candidate_name} steps={actions.shape[0]}")

                cand_dir = seed_dir / f"candidate_{chunk_idx:03d}_{candidate_name}"

                row = replay_chunk(
                    env=env,
                    init_state=init_states[init_state_idx],
                    init_state_idx=init_state_idx,
                    actions=actions,
                    chunk_meta=chunk_meta,
                    args=args,
                    candidate_dir=cand_dir,
                )

                all_rows.append(row)

                if args.write_summary_every > 0 and len(all_rows) % args.write_summary_every == 0:
                    write_summary_csv(all_rows, out_dir / "summary.csv")

                print(
                    f"    policy={row.get('policy_name')} "
                    f"grasp={float(row.get('bowl2_grasp_success_proxy', float('nan'))):.1f} "
                    f"lifted={float(row.get('bowl2_source_lifted', float('nan'))):.1f} "
                    f"zmax={float(row.get('bowl2_source_z_delta_max', float('nan'))):.4f} "
                    f"disp={float(row.get('bowl2_source_displacement_max', float('nan'))):.4f} "
                    f"quality={float(row.get('bowl2_grasp_quality_score', float('nan'))):+.4f}"
                )

            write_summary_csv(all_rows, out_dir / "summary.csv")

    finally:
        if hasattr(env, "close"):
            env.close()

    write_summary_csv(all_rows, out_dir / "summary.csv")

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print_compact_summary(all_rows)

    print(f"\n[done] saved to: {out_dir}")
    print(f"[done] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()