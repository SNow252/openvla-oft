#!/usr/bin/env python3
"""
Export JSONL policy action dumps into standard v3 policy chunk format,
inferring init_state_idx by matching the JSONL initial obs_summary to LIBERO
reference init states.

Why this is needed:
  Some old OpenVLA / SmolVLA dumps do not save init_state_idx explicitly.
  We infer it by comparing object/robot state summaries.

Input:
  DUMP_DIR/*.jsonl

Uses:
  processed_action: [7]  as the executed action sequence by default.

Output:
  OUT_DIR/init_000/openvla_oft_<chunk_name>_xxx.npz
  OUT_DIR/init_001/openvla_oft_<chunk_name>_xxx.npz
  ...

Each output .npz contains:
  actions: [T, 7]
  policy_name
  chunk_name
  language
  source_path
  inferred_init_state_idx
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.collect_same_state_grasp_counterfactuals_v3 import (  # noqa: E402
    make_env,
    reset_to_init,
    settle_after_reset,
    summarize_obs,
)
from tools.export_policy_chunks_from_jsonl_dump_v3 import (  # noqa: E402
    extract_step_actions,
    extract_first_chunk,
    find_condition,
    find_language,
    infer_file_short_id,
    read_jsonl,
    safe_name,
)


MATCH_KEYS_PRIORITY = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "akita_black_bowl_1_pos",
    "akita_black_bowl_2_pos",
    "plate_1_pos",
    "glazed_rim_porcelain_ramekin_1_pos",
]

SUMMARY_CONTAINER_KEYS = [
    "query_obs_summary",
    "obs_before_summary",
    "obs_summary",
    "initial_obs_summary",
    "start_obs_summary",
    "obs_after_summary",
]


def try_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size > 20:
            return None
        return arr
    except Exception:
        return None


def nested_items(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from nested_items(v, p)
    elif isinstance(obj, list):
        # Avoid expanding large numeric arrays.
        if len(obj) > 0 and all(isinstance(x, (int, float, bool)) for x in obj[: min(20, len(obj))]):
            return
        for i, v in enumerate(obj[:20]):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from nested_items(v, p)


def get_direct_nested(obj: Dict[str, Any], key: str) -> Optional[Any]:
    """
    Search any nested dictionary for exact key.
    """
    for _path, value in nested_items(obj):
        if _path.split(".")[-1] == key:
            return value
    return None


def flatten_summary_like(obj: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Convert any summary-like dict into a flat numeric summary:
      key -> list[float]
    """
    out: Dict[str, List[float]] = {}

    for path, value in nested_items(obj):
        last = path.split(".")[-1]
        arr = try_array(value)
        if arr is None:
            continue

        # Prefer exact final key name, e.g. robot0_eef_pos.
        out[last] = arr.astype(float).tolist()

    return out


def find_first_summary(records: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], str]:
    """
    Find the first obs summary in JSONL records.

    Returns:
      summary dict, source path description.
    """
    for rec in records:
        line_idx = rec.get("_line_idx", -1)

        # First try known summary container keys.
        for key in SUMMARY_CONTAINER_KEYS:
            value = get_direct_nested(rec, key)
            if isinstance(value, dict):
                flat = flatten_summary_like(value)
                if flat:
                    return flat, f"line[{line_idx}].{key}"

        # Fallback: scan the whole record and look for summary-like keys.
        flat = {}
        for path, value in nested_items(rec):
            last = path.split(".")[-1]
            if last in MATCH_KEYS_PRIORITY:
                arr = try_array(value)
                if arr is not None:
                    flat[last] = arr.astype(float).tolist()

        if flat:
            return flat, f"line[{line_idx}].<scanned_record>"

    return {}, ""


def summary_distance(
    a: Dict[str, List[float]],
    b: Dict[str, List[float]],
) -> Tuple[float, int, List[str]]:
    """
    Lower is better. Only compares overlapping keys.
    """
    keys = [k for k in MATCH_KEYS_PRIORITY if k in a and k in b]

    if not keys:
        # fallback to all shared numeric keys
        keys = sorted(set(a.keys()) & set(b.keys()))

    total = 0.0
    used = 0
    used_keys = []

    for k in keys:
        av = try_array(a[k])
        bv = try_array(b[k])
        if av is None or bv is None:
            continue

        n = min(av.size, bv.size)
        if n == 0:
            continue

        diff = av[:n] - bv[:n]
        total += float(np.sum(diff * diff))
        used += n
        used_keys.append(k)

    if used == 0:
        return float("inf"), 0, []

    return math.sqrt(total / used), used, used_keys


def build_reference_summaries(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)

    env = make_env(args, task)

    refs = []

    try:
        for init_idx in range(args.start_init_state_idx, args.start_init_state_idx + args.num_seeds):
            obs = reset_to_init(env, init_states[init_idx])

            if args.match_settle_steps > 0:
                obs = settle_after_reset(
                    env=env,
                    obs=obs,
                    settle_steps=args.match_settle_steps,
                    gripper_open_value=args.gripper_open_value,
                )

            summary = summarize_obs(obs)

            refs.append(
                {
                    "init_state_idx": init_idx,
                    "summary": summary,
                }
            )

    finally:
        if hasattr(env, "close"):
            env.close()

    return refs


def infer_init_from_summary(
    summary: Dict[str, List[float]],
    refs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = []

    for ref in refs:
        d, used, keys = summary_distance(summary, ref["summary"])
        candidates.append(
            {
                "init_state_idx": int(ref["init_state_idx"]),
                "distance": float(d),
                "used_dims": int(used),
                "used_keys": keys,
            }
        )

    candidates = sorted(candidates, key=lambda x: x["distance"])

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    return {
        "best_init_state_idx": int(best["init_state_idx"]),
        "best_distance": float(best["distance"]),
        "best_used_dims": int(best["used_dims"]),
        "best_used_keys": ",".join(best["used_keys"]),
        "second_init_state_idx": int(second["init_state_idx"]) if second else -1,
        "second_distance": float(second["distance"]) if second else float("nan"),
        "margin": float(second["distance"] - best["distance"]) if second else float("nan"),
    }


def maybe_clip_steps(actions: np.ndarray, start_step: int, max_steps: int) -> np.ndarray:
    start = max(0, int(start_step))
    end = actions.shape[0]

    if max_steps > 0:
        end = min(end, start + int(max_steps))

    return actions[start:end].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument("--policy_name", default="openvla_oft")
    parser.add_argument("--chunk_name", default="jsonl_dump")

    parser.add_argument(
        "--mode",
        choices=["step_actions", "first_chunk"],
        default="step_actions",
    )
    parser.add_argument("--action_field", default="processed_action")
    parser.add_argument("--fallback_action_field", default="raw_action_before_process")
    parser.add_argument("--chunk_field", default="raw_actions_chunk")

    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=140)

    parser.add_argument(
        "--match_settle_steps",
        type=int,
        default=0,
        help="Use 0 first. If matching distances are bad, try 30.",
    )
    parser.add_argument("--gripper_open_value", type=float, default=-1.0)

    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--filter_contains", default="")
    parser.add_argument(
        "--max_match_distance_warn",
        type=float,
        default=1e-3,
        help="Warn if best observation matching distance is larger than this.",
    )

    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dump_dir.glob("*.jsonl"))

    if args.filter_contains:
        files = [p for p in files if args.filter_contains in str(p)]

    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise RuntimeError(f"No jsonl files found in {dump_dir}")

    policy_name = safe_name(args.policy_name)
    base_chunk_name = safe_name(args.chunk_name)

    print("=" * 120)
    print("[dump_dir]", dump_dir)
    print("[out_dir]", out_dir)
    print("[files]", len(files))
    print("[mode]", args.mode)
    print("[match_settle_steps]", args.match_settle_steps)
    print("[policy_name]", policy_name)
    print("[chunk_name]", base_chunk_name)
    print("=" * 120)

    refs = build_reference_summaries(args)
    print("[reference summaries]", len(refs))

    exported = []
    empty_actions = []
    missing_summary = []
    high_distance = []

    for file_i, path in enumerate(files):
        records = read_jsonl(path)

        if not records:
            empty_actions.append(str(path))
            continue

        summary, summary_source = find_first_summary(records)

        if not summary:
            missing_summary.append(str(path))
            continue

        infer = infer_init_from_summary(summary, refs)
        init_idx = int(infer["best_init_state_idx"])

        if infer["best_distance"] > args.max_match_distance_warn:
            high_distance.append((str(path), infer))

        if args.mode == "step_actions":
            actions = extract_step_actions(
                records=records,
                action_field=args.action_field,
                fallback_action_field=args.fallback_action_field,
            )
        else:
            actions = extract_first_chunk(records=records, chunk_field=args.chunk_field)

        actions = maybe_clip_steps(actions, start_step=args.start_step, max_steps=args.max_steps)

        if actions.shape[0] == 0:
            empty_actions.append(str(path))
            continue

        language = find_language(records)
        condition = find_condition(records)
        condition_name = safe_name(condition) if condition else "unknown_condition"
        short_id = infer_file_short_id(path)

        final_chunk_name = safe_name(f"{policy_name}_{base_chunk_name}_{condition_name}_{short_id}")

        seed_dir = out_dir / f"init_{init_idx:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        out_path = seed_dir / f"{final_chunk_name}.npz"

        np.savez_compressed(
            out_path,
            actions=actions.astype(np.float32),
            policy_name=policy_name,
            chunk_name=final_chunk_name,
            language=language,
            condition=condition,
            source_path=str(path),
            inferred_init_state_idx=np.asarray(init_idx, dtype=np.int32),
            init_state_idx=np.asarray(init_idx, dtype=np.int32),
            start_step=np.asarray(args.start_step, dtype=np.int32),
            used_num_steps=np.asarray(actions.shape[0], dtype=np.int32),
            mode=args.mode,
            action_field=args.action_field,
            fallback_action_field=args.fallback_action_field,
            chunk_field=args.chunk_field,
            summary_source=summary_source,
            match_distance=np.asarray(infer["best_distance"], dtype=np.float32),
            match_second_distance=np.asarray(infer["second_distance"], dtype=np.float32),
            match_margin=np.asarray(infer["margin"], dtype=np.float32),
            match_used_dims=np.asarray(infer["best_used_dims"], dtype=np.int32),
        )

        row = {
            "file_i": file_i,
            "init_state_idx": init_idx,
            "source_path": str(path),
            "out_path": str(out_path),
            "steps": int(actions.shape[0]),
            "language": language,
            "condition": condition,
            "chunk_name": final_chunk_name,
            "summary_source": summary_source,
            **infer,
        }
        exported.append(row)

        print(
            f"[export] file={file_i:04d} init={init_idx:03d} "
            f"dist={infer['best_distance']:.8f} "
            f"margin={infer['margin']:.8f} "
            f"dims={infer['best_used_dims']:03d} "
            f"steps={actions.shape[0]:03d} "
            f"src={summary_source} -> {out_path.name}"
        )

    manifest = {
        "dump_dir": str(dump_dir),
        "out_dir": str(out_dir),
        "args": vars(args),
        "exported": exported,
        "empty_actions": empty_actions,
        "missing_summary": missing_summary,
        "high_distance": high_distance,
    }

    manifest_path = out_dir / f"manifest_{policy_name}_{base_chunk_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    csv_path = out_dir / f"manifest_{policy_name}_{base_chunk_name}.csv"
    fields = [
        "file_i",
        "init_state_idx",
        "steps",
        "condition",
        "language",
        "chunk_name",
        "summary_source",
        "best_distance",
        "second_distance",
        "margin",
        "best_used_dims",
        "best_used_keys",
        "source_path",
        "out_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in exported:
            writer.writerow({k: r.get(k, "") for k in fields})

    print("\n" + "=" * 120)
    print("Exported:", len(exported))
    print("Empty actions:", len(empty_actions))
    print("Missing summary:", len(missing_summary))
    print("High-distance matches:", len(high_distance))
    print("Manifest:", manifest_path)
    print("CSV:", csv_path)
    print("=" * 120)

    if high_distance:
        print("\nHigh-distance examples:")
        for path, infer in high_distance[:20]:
            print(
                f"  init={infer['best_init_state_idx']:03d} "
                f"dist={infer['best_distance']:.8f} "
                f"margin={infer['margin']:.8f} "
                f"path={path}"
            )

    if missing_summary:
        print("\nMissing summary examples:")
        for x in missing_summary[:20]:
            print(" ", x)

    if empty_actions:
        print("\nEmpty action examples:")
        for x in empty_actions[:20]:
            print(" ", x)


if __name__ == "__main__":
    main()