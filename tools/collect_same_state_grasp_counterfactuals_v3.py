#!/usr/bin/env python3
"""
Collect same-state grasp/contact counterfactuals for LIBERO.

v3 goal:
  Move beyond EEF reachability, which can be largely explained by simple
  kinematics. This collector tests contact/object-dynamics consequences:
    - Does the action move the source object?
    - Does the object get lifted?
    - Does gripper timing / xy offset / grasp height matter?
    - Can kinematic EEF baselines still explain object movement?

Default task:
  LIBERO-spatial task8
  source object: akita_black_bowl_2
  distractor/default source: akita_black_bowl_1

Macro action:
  approach above source
  descend near source
  close gripper
  hold close
  lift

Candidate variations:
  xy offset
  grasp z offset
  lift height
  gripper close value
  optional source=bowl1 wrong-source variants

Important:
  Gripper sign may differ by environment/controller. Therefore this script
  supports multiple --gripper_close_values, e.g. "-1,1".
  The labels will reveal which close sign actually grasps/lifts.
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


def parse_xy_offsets(s: str) -> List[Tuple[float, float]]:
    """
    Format:
      "0,0;0.02,0;-0.02,0;0,0.02;0,-0.02"
    """
    out = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        out.append((float(x.strip()), float(y.strip())))
    return out


def safe_name(x: str) -> str:
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    return x


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


def get_vec(summary: Dict[str, List[float]], key: str, dim: int) -> np.ndarray:
    if key not in summary:
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(summary[key], dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    out[: min(dim, arr.size)] = arr[:dim]
    return out


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def extract_state_for_sequence(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    summary = summarize_obs(obs)

    state = {
        "eef": get_pos(summary, "robot0_eef_pos").astype(np.float32),
        "gripper_qpos": get_vec(summary, "robot0_gripper_qpos", 2).astype(np.float32),
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

    seq["gripper_qpos"].append(state["gripper_qpos"])

    eef = state["eef"]
    for obj in ["bowl1", "bowl2", "plate", "ramekin"]:
        seq[f"dist_eef_to_{obj}"].append(np.array(dist(eef, state[obj]), dtype=np.float32))


def sequence_to_arrays(seq: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, values in seq.items():
        out[f"{key}_seq"] = np.asarray(values, dtype=np.float32)
    return out


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


def action_to_waypoint(
    obs: Dict[str, Any],
    waypoint: np.ndarray,
    kp: float,
    max_action: float,
    gripper_value: float,
    action_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    summary = summarize_obs(obs)
    eef = get_pos(summary, "robot0_eef_pos")
    action = make_xyz_action(eef, waypoint, kp, max_action, gripper_value)

    if action_noise_std > 0:
        action[:3] += rng.normal(0.0, action_noise_std, size=3)
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


def build_candidate_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    xy_offsets = parse_xy_offsets(args.xy_offsets)
    z_grasp_offsets = parse_float_list(args.z_grasp_offsets)
    lift_heights = parse_float_list(args.lift_heights)
    close_values = parse_float_list(args.gripper_close_values)

    sources = ["bowl2"]
    if args.include_wrong_source:
        sources.append("bowl1")

    specs: List[Dict[str, Any]] = []
    for source in sources:
        for xy in xy_offsets:
            for zgo in z_grasp_offsets:
                for lh in lift_heights:
                    for close_value in close_values:
                        name = (
                            f"src_{source}_xy_{xy[0]:+.2f}_{xy[1]:+.2f}"
                            f"_zg_{zgo:.2f}_lift_{lh:.2f}_close_{close_value:+.1f}"
                        )
                        specs.append(
                            {
                                "candidate_name": safe_name(name),
                                "candidate_family": "grasp_macro",
                                "source": source,
                                "xy_offset_x": float(xy[0]),
                                "xy_offset_y": float(xy[1]),
                                "z_grasp_offset": float(zgo),
                                "lift_height": float(lh),
                                "gripper_close_value": float(close_value),
                            }
                        )

    if args.include_stall:
        specs.append(
            {
                "candidate_name": "stall_zero",
                "candidate_family": "stall",
                "source": "none",
                "xy_offset_x": float("nan"),
                "xy_offset_y": float("nan"),
                "z_grasp_offset": float("nan"),
                "lift_height": float("nan"),
                "gripper_close_value": float("nan"),
            }
        )

    if args.include_random:
        specs.append(
            {
                "candidate_name": "random_local",
                "candidate_family": "random",
                "source": "none",
                "xy_offset_x": float("nan"),
                "xy_offset_y": float("nan"),
                "z_grasp_offset": float("nan"),
                "lift_height": float("nan"),
                "gripper_close_value": float("nan"),
            }
        )

    return specs


def compute_grasp_labels(
    seq_arrays: Dict[str, np.ndarray],
    source: str,
    distractor: str,
    contact_threshold: float,
    move_threshold: float,
    lift_threshold: float,
) -> Dict[str, float]:
    src_pos = seq_arrays[f"{source}_pos_seq"]
    dis_pos = seq_arrays[f"{distractor}_pos_seq"]
    eef_pos = seq_arrays["eef_pos_seq"]
    gripper_qpos = seq_arrays["gripper_qpos_seq"]

    d_src = np.linalg.norm(eef_pos - src_pos, axis=1)
    d_dis = np.linalg.norm(eef_pos - dis_pos, axis=1)

    src_start = src_pos[0]
    src_final = src_pos[-1]
    dis_start = dis_pos[0]
    dis_final = dis_pos[-1]

    src_disp_seq = np.linalg.norm(src_pos - src_start.reshape(1, 3), axis=1)
    dis_disp_seq = np.linalg.norm(dis_pos - dis_start.reshape(1, 3), axis=1)

    src_disp_final = float(np.linalg.norm(src_final - src_start))
    src_disp_max = float(np.max(src_disp_seq))
    src_xy_disp_final = float(np.linalg.norm(src_final[:2] - src_start[:2]))
    src_z_delta_final = float(src_final[2] - src_start[2])
    src_z_delta_max = float(np.max(src_pos[:, 2]) - src_start[2])

    dis_disp_final = float(np.linalg.norm(dis_final - dis_start))
    dis_disp_max = float(np.max(dis_disp_seq))
    dis_z_delta_max = float(np.max(dis_pos[:, 2]) - dis_start[2])

    min_d_src = float(np.min(d_src))
    final_d_src = float(d_src[-1])
    min_d_dis = float(np.min(d_dis))
    final_d_dis = float(d_dis[-1])

    contact_hits = np.where(d_src <= contact_threshold)[0]
    if len(contact_hits) > 0:
        reached_contact = 1.0
        tau_contact = float(int(contact_hits[0]))
    else:
        reached_contact = 0.0
        tau_contact = float(len(d_src))

    moved = float(src_disp_max >= move_threshold)
    lifted = float(src_z_delta_max >= lift_threshold)
    wrong_moved = float(dis_disp_max >= move_threshold)

    grasp_success_proxy = float((moved > 0.5) and (lifted > 0.5))
    clean_grasp_proxy = float((grasp_success_proxy > 0.5) and (wrong_moved < 0.5))

    labels = {
        "source_object": source,
        "distractor_object": distractor,

        "min_dist_to_source": min_d_src,
        "final_dist_to_source": final_d_src,
        "min_dist_to_distractor": min_d_dis,
        "final_dist_to_distractor": final_d_dis,

        "reached_contact_region": reached_contact,
        "tau_contact_region": tau_contact,

        "source_displacement_final": src_disp_final,
        "source_displacement_max": src_disp_max,
        "source_xy_displacement_final": src_xy_disp_final,
        "source_z_delta_final": src_z_delta_final,
        "source_z_delta_max": src_z_delta_max,

        "distractor_displacement_final": dis_disp_final,
        "distractor_displacement_max": dis_disp_max,
        "distractor_z_delta_max": dis_z_delta_max,

        "source_moved": moved,
        "source_lifted": lifted,
        "wrong_object_moved": wrong_moved,
        "grasp_success_proxy": grasp_success_proxy,
        "clean_grasp_proxy": clean_grasp_proxy,

        # Higher-is-better continuous score.
        "grasp_quality_score": (
            src_z_delta_max
            + 0.5 * src_disp_max
            - 0.25 * dis_disp_max
            - 0.1 * min_d_src
        ),

        # Gripper diagnostics.
        "gripper_qpos_start_0": float(gripper_qpos[0, 0]) if gripper_qpos.ndim == 2 and gripper_qpos.shape[1] > 0 else float("nan"),
        "gripper_qpos_final_0": float(gripper_qpos[-1, 0]) if gripper_qpos.ndim == 2 and gripper_qpos.shape[1] > 0 else float("nan"),
        "gripper_qpos_delta_0": float(gripper_qpos[-1, 0] - gripper_qpos[0, 0]) if gripper_qpos.ndim == 2 and gripper_qpos.shape[1] > 0 else float("nan"),
    }

    return labels


def run_candidate(
    env: Any,
    init_state: Any,
    init_state_idx: int,
    spec: Dict[str, Any],
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

    seq: Dict[str, List[np.ndarray]] = {
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

    append_sequence(seq, obs)

    first_frame = extract_frame(obs)
    if args.save_video and first_frame is not None:
        frames.append(first_frame)

    family = spec["candidate_family"]

    # Build an explicit phase plan: list of (phase_name, steps, waypoint, gripper_value).
    phase_plan: List[Tuple[str, int, Optional[np.ndarray], float]] = []

    if family == "grasp_macro":
        source = str(spec["source"])
        source_key = OBJECT_KEYS[source]
        source_pos = get_pos(start_summary, source_key)

        xy_offset = np.array([spec["xy_offset_x"], spec["xy_offset_y"], 0.0], dtype=float)
        z_grasp_offset = float(spec["z_grasp_offset"])
        lift_height = float(spec["lift_height"])
        close_value = float(spec["gripper_close_value"])

        approach_wp = source_pos + xy_offset + np.array([0.0, 0.0, args.approach_height], dtype=float)
        grasp_wp = source_pos + xy_offset + np.array([0.0, 0.0, z_grasp_offset], dtype=float)
        lift_wp = source_pos + xy_offset + np.array([0.0, 0.0, lift_height], dtype=float)

        phase_plan = [
            ("approach", args.approach_steps, approach_wp, args.gripper_open_value),
            ("descend", args.descend_steps, grasp_wp, args.gripper_open_value),
            ("preclose_hold", args.preclose_hold_steps, grasp_wp, args.gripper_open_value),
            ("close_hold", args.close_hold_steps, grasp_wp, close_value),
            ("lift", args.lift_steps, lift_wp, close_value),
        ]

    elif family == "stall":
        phase_plan = [("stall", args.total_stall_steps, None, args.gripper_open_value)]

    elif family == "random":
        phase_plan = [("random", args.total_random_steps, None, args.gripper_open_value)]

    else:
        raise ValueError(f"Unknown candidate_family: {family}")

    phase_names_executed: List[str] = []

    for phase_name, steps, waypoint, gripper_value in phase_plan:
        for _ in range(steps):
            if family == "random":
                action = np.zeros(7, dtype=np.float32)
                action[:3] = rng.normal(0.0, args.random_std, size=3)
                action[:3] = np.clip(action[:3], -args.max_action, args.max_action)
                action[6] = float(gripper_value)
            elif waypoint is None:
                action = np.zeros(7, dtype=np.float32)
                action[6] = float(gripper_value)
            else:
                action = action_to_waypoint(
                    obs=obs,
                    waypoint=waypoint,
                    kp=args.kp,
                    max_action=args.max_action,
                    gripper_value=gripper_value,
                    action_noise_std=args.action_noise_std,
                    rng=rng,
                )

            obs, reward, done, _info = step_env(env, action)

            actions.append(action.copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            phase_names_executed.append(phase_name)

            append_sequence(seq, obs)

            frame = extract_frame(obs)
            if args.save_video and frame is not None:
                frames.append(frame)

            if done and not args.continue_after_done:
                break

        if dones and dones[-1] and not args.continue_after_done:
            break

    final_summary = summarize_obs(obs)
    seq_arrays = sequence_to_arrays(seq)

    # Compute labels for source=bowl2 by default; if wrong-source candidate, use its chosen source too.
    source = str(spec.get("source", "bowl2"))
    if source not in {"bowl1", "bowl2"}:
        source = "bowl2"
    distractor = "bowl1" if source == "bowl2" else "bowl2"

    labels = compute_grasp_labels(
        seq_arrays=seq_arrays,
        source=source,
        distractor=distractor,
        contact_threshold=args.contact_threshold,
        move_threshold=args.move_threshold,
        lift_threshold=args.lift_threshold,
    )

    # Also compute bowl2-centric labels, regardless of candidate source.
    labels_bowl2 = compute_grasp_labels(
        seq_arrays=seq_arrays,
        source="bowl2",
        distractor="bowl1",
        contact_threshold=args.contact_threshold,
        move_threshold=args.move_threshold,
        lift_threshold=args.lift_threshold,
    )
    labels_bowl2 = {f"bowl2_{k}": v for k, v in labels_bowl2.items() if isinstance(v, (int, float, np.floating))}
    labels.update(labels_bowl2)

    candidate_dir.mkdir(parents=True, exist_ok=True)

    actions_arr = np.asarray(actions, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)

    npz_payload = {
        "actions": actions_arr,
        "rewards": rewards_arr,
        "dones": dones_arr,
        "init_state_idx": np.asarray(init_state_idx, dtype=np.int32),
        "phase_names": np.asarray(phase_names_executed, dtype=object),
    }
    npz_payload.update(seq_arrays)

    np.savez_compressed(candidate_dir / "traj.npz", **npz_payload)

    meta = {
        "init_state_idx": init_state_idx,
        "candidate_name": spec["candidate_name"],
        "candidate_family": spec["candidate_family"],
        "candidate_spec": spec,
        "horizon_executed": int(len(actions)),
        "start_summary": start_summary,
        "final_summary": final_summary,
        "labels": labels,
        "sum_reward": float(np.sum(rewards_arr)) if len(rewards_arr) else 0.0,
        "any_done": bool(np.any(dones_arr)) if len(dones_arr) else False,
        "phase_plan": [
            {
                "phase_name": p[0],
                "steps": p[1],
                "waypoint": None if p[2] is None else p[2].tolist(),
                "gripper_value": p[3],
            }
            for p in phase_plan
        ],
        "controller_args": vars(args),
    }

    with open(candidate_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_video:
        save_video(frames, candidate_dir / "video.mp4", args.video_fps)

    row: Dict[str, Any] = {
        "init_state_idx": init_state_idx,
        "candidate_name": spec["candidate_name"],
        "candidate_family": spec["candidate_family"],
        "source": spec.get("source", ""),
        "xy_offset_x": spec.get("xy_offset_x", float("nan")),
        "xy_offset_y": spec.get("xy_offset_y", float("nan")),
        "z_grasp_offset": spec.get("z_grasp_offset", float("nan")),
        "lift_height": spec.get("lift_height", float("nan")),
        "gripper_close_value": spec.get("gripper_close_value", float("nan")),
        "horizon_executed": int(len(actions)),
        "sum_reward": meta["sum_reward"],
        "any_done": int(meta["any_done"]),
    }
    row.update(labels)
    return row


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    base_fields = [
        "init_state_idx",
        "candidate_name",
        "candidate_family",
        "source",
        "xy_offset_x",
        "xy_offset_y",
        "z_grasp_offset",
        "lift_height",
        "gripper_close_value",
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


def print_preview(rows: List[Dict[str, Any]], max_rows: int = 30) -> None:
    print("\n" + "=" * 120)
    print("[v3 grasp preview]")
    for r in rows[:max_rows]:
        print(
            f"init={r['init_state_idx']:03d} "
            f"{r['candidate_name'][:50]:50s} "
            f"src={r.get('source', ''):5s} "
            f"xy=({r.get('xy_offset_x', float('nan')):+.2f},{r.get('xy_offset_y', float('nan')):+.2f}) "
            f"zg={r.get('z_grasp_offset', float('nan')):.2f} "
            f"close={r.get('gripper_close_value', float('nan')):+.1f} "
            f"disp={r.get('bowl2_source_displacement_max', float('nan')):.4f} "
            f"lift={r.get('bowl2_source_z_delta_max', float('nan')):.4f} "
            f"moved={r.get('bowl2_source_moved', float('nan'))} "
            f"lifted={r.get('bowl2_source_lifted', float('nan'))} "
            f"grasp={r.get('bowl2_grasp_success_proxy', float('nan'))}"
        )
    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)
    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=2)

    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--xy_offsets", type=str, default="0,0;0.02,0;-0.02,0;0,0.02;0,-0.02;0.04,0;-0.04,0")
    parser.add_argument("--z_grasp_offsets", type=str, default="0.00,0.02,0.04,0.06")
    parser.add_argument("--lift_heights", type=str, default="0.08,0.12")
    parser.add_argument("--gripper_close_values", type=str, default="-1,1")
    parser.add_argument("--include_wrong_source", action="store_true")
    parser.add_argument("--include_stall", action="store_true")
    parser.add_argument("--include_random", action="store_true")

    parser.add_argument("--approach_height", type=float, default=0.18)
    parser.add_argument("--approach_steps", type=int, default=24)
    parser.add_argument("--descend_steps", type=int, default=20)
    parser.add_argument("--preclose_hold_steps", type=int, default=4)
    parser.add_argument("--close_hold_steps", type=int, default=10)
    parser.add_argument("--lift_steps", type=int, default=24)

    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--max_action", type=float, default=0.5)
    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--random_std", type=float, default=0.20)

    parser.add_argument("--gripper_open_value", type=float, default=0.0)

    parser.add_argument("--contact_threshold", type=float, default=0.08)
    parser.add_argument("--move_threshold", type=float, default=0.015)
    parser.add_argument("--lift_threshold", type=float, default=0.020)

    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=10)
    parser.add_argument("--continue_after_done", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    end_idx = args.start_init_state_idx + args.num_seeds
    if end_idx > len(init_states):
        raise ValueError(
            f"Requested init states [{args.start_init_state_idx}, {end_idx}), "
            f"but only {len(init_states)} init states are available."
        )

    specs = build_candidate_specs(args)
    print(f"[num candidates per init] {len(specs)}")
    for i, spec in enumerate(specs[:20]):
        print(f"  {i:03d}: {spec}")
    if len(specs) > 20:
        print(f"  ... {len(specs) - 20} more specs")

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
                    candidate_dir=cand_dir,
                )
                all_rows.append(row)

                print(
                    f"    bowl2_disp={row.get('bowl2_source_displacement_max', float('nan')):.4f}, "
                    f"bowl2_lift={row.get('bowl2_source_z_delta_max', float('nan')):.4f}, "
                    f"bowl2_moved={row.get('bowl2_source_moved', float('nan'))}, "
                    f"bowl2_lifted={row.get('bowl2_source_lifted', float('nan'))}, "
                    f"bowl2_grasp={row.get('bowl2_grasp_success_proxy', float('nan'))}"
                )

            write_summary_csv(all_rows, out_dir / "summary.csv")

    finally:
        if hasattr(env, "close"):
            env.close()

    write_summary_csv(all_rows, out_dir / "summary.csv")
    print_preview(all_rows)

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[done] saved to: {out_dir}")
    print(f"[done] summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()