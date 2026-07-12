#!/usr/bin/env python3
"""
Configurable object-pair replay labels.

Compared with the older task8-specific dual-source label script, this version
uses a JSON config to define:

  primary_object
  alternative_object
  target_object

It produces object-level consequence labels for both objects and their contrast:

  primary_grasp_success_proxy
  alternative_grasp_success_proxy
  alternative_minus_primary_quality
  ...

The script expects replay output layout:
  replay_dir/
    summary.csv
    init_XXX/candidate_*/traj.npz

The trajectory npz should contain:
  eef_pos_seq
  gripper_qpos_seq
  <object>_pos_seq for primary/alternative/target

Example object names:
  akita_black_bowl_1
  akita_black_bowl_2
  plate_1
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    extras = sorted({k for r in rows for k in r.keys()} - set(fields))
    fields += extras
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_mapping(config_path: Path, suite: str, task_key: str):
    cfg = json.loads(config_path.read_text())
    return cfg[suite][task_key]


def find_traj_npz(data_dir: Path, row: Dict[str, Any]):
    init_idx = int(safe_float(row["init_state_idx"]))
    cand_name = row.get("candidate_name", "")

    # Preferred direct pattern.
    init_dir = data_dir / f"init_{init_idx:03d}"
    if cand_name:
        hits = sorted(init_dir.glob(f"candidate_*_{cand_name}/traj.npz"))
        if hits:
            return hits[0]

    # Fallback: use chunk_name if present.
    for key in ["chunk_name", "candidate", "name"]:
        val = row.get(key, "")
        if val:
            hits = sorted(init_dir.glob(f"candidate_*_{val}/traj.npz"))
            if hits:
                return hits[0]

    # Last fallback: if only one candidate dir contains matching candidate substring.
    hits = sorted(init_dir.glob("candidate_*/traj.npz"))
    if cand_name:
        filtered = [p for p in hits if cand_name in str(p.parent.name)]
        if filtered:
            return filtered[0]

    raise FileNotFoundError(f"Cannot find traj.npz for init={init_idx} candidate={cand_name}")


def key_for_object(obj_name: str):
    return f"{obj_name}_pos_seq"


def first_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return x[:, 0]


def object_consequence(seq, obj_name, eef_name="eef_pos_seq", gripper_name="gripper_qpos_seq",
                       contact_threshold=0.10, move_threshold=0.020, lift_threshold=0.010):
    obj_key = key_for_object(obj_name)
    if obj_key not in seq:
        raise KeyError(f"Missing {obj_key} in traj.npz. Available keys: {list(seq.keys())}")

    eef = np.asarray(seq[eef_name], dtype=float)
    obj = np.asarray(seq[obj_key], dtype=float)
    grip = np.asarray(seq[gripper_name], dtype=float)

    T = min(len(eef), len(obj), len(grip))
    eef = eef[:T]
    obj = obj[:T]
    grip = grip[:T]

    dist = np.linalg.norm(eef - obj, axis=1)
    min_dist = float(np.min(dist))

    obj0 = obj[0]
    disp = np.linalg.norm(obj - obj0[None, :], axis=1)
    disp_max = float(np.max(disp))

    z_delta = obj[:, 2] - obj0[2]
    z_delta_max = float(np.max(z_delta))

    q = first_col(grip).astype(float)
    # In LIBERO/Franka, q tends to decrease when gripper closes.
    gripper_qpos_delta_0 = float(np.min(q) - q[0])

    moved = float(disp_max >= move_threshold)
    lifted = float(z_delta_max >= lift_threshold)

    contact_like = float(min_dist <= contact_threshold)

    grasp_success_proxy = float((lifted > 0.5) and (contact_like > 0.5))
    # Keep quality continuous and simple.
    quality = (
        +1.0 * z_delta_max
        +0.5 * disp_max
        -0.5 * min_dist
    )

    return {
        "min_dist": min_dist,
        "source_displacement_max": disp_max,
        "source_z_delta_max": z_delta_max,
        "source_moved": moved,
        "source_lifted": lifted,
        "contact_like": contact_like,
        "gripper_qpos_delta_0": gripper_qpos_delta_0,
        "grasp_success_proxy": grasp_success_proxy,
        "grasp_quality_score": float(quality),
    }


def add_prefixed(out, prefix, d):
    for k, v in d.items():
        out[f"{prefix}_{k}"] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--input_csv", default="summary.csv")
    ap.add_argument("--output_csv", default="summary_object_pair.csv")
    ap.add_argument("--config_json", required=True)
    ap.add_argument("--task_suite_name", default="libero_spatial")
    ap.add_argument("--task_key", required=True, help="e.g., task8, task3, task9")
    ap.add_argument("--contact_threshold", type=float, default=0.10)
    ap.add_argument("--move_threshold", type=float, default=0.020)
    ap.add_argument("--lift_threshold", type=float, default=0.010)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mapping = load_mapping(Path(args.config_json), args.task_suite_name, args.task_key)

    primary = mapping["primary_object"]
    alternative = mapping["alternative_object"]

    rows = read_csv(data_dir / args.input_csv)
    out_rows = []

    for r in rows:
        traj_path = find_traj_npz(data_dir, r)
        seq = np.load(traj_path, allow_pickle=True)

        primary_d = object_consequence(
            seq, primary,
            contact_threshold=args.contact_threshold,
            move_threshold=args.move_threshold,
            lift_threshold=args.lift_threshold,
        )
        alt_d = object_consequence(
            seq, alternative,
            contact_threshold=args.contact_threshold,
            move_threshold=args.move_threshold,
            lift_threshold=args.lift_threshold,
        )

        out = dict(r)
        out["task_key_for_labels"] = args.task_key
        out["primary_object"] = primary
        out["alternative_object"] = alternative
        out["target_object"] = mapping.get("target_object", "")
        out["primary_role"] = mapping.get("primary_role", "")
        out["alternative_role"] = mapping.get("alternative_role", "")

        add_prefixed(out, "primary", primary_d)
        add_prefixed(out, "alternative", alt_d)

        out["alternative_minus_primary_grasp_success_proxy"] = (
            alt_d["grasp_success_proxy"] - primary_d["grasp_success_proxy"]
        )
        out["alternative_minus_primary_lifted"] = (
            alt_d["source_lifted"] - primary_d["source_lifted"]
        )
        out["alternative_minus_primary_quality"] = (
            alt_d["grasp_quality_score"] - primary_d["grasp_quality_score"]
        )

        # Backward-compatible aliases when primary/alternative are bowl1/bowl2.
        if primary == "akita_black_bowl_1":
            add_prefixed(out, "bowl1", primary_d)
        if alternative == "akita_black_bowl_2":
            add_prefixed(out, "bowl2", alt_d)
        if primary == "akita_black_bowl_1" and alternative == "akita_black_bowl_2":
            out["bowl2_minus_bowl1_grasp_success_proxy"] = out["alternative_minus_primary_grasp_success_proxy"]
            out["bowl2_minus_bowl1_lifted"] = out["alternative_minus_primary_lifted"]
            out["bowl2_minus_bowl1_quality"] = out["alternative_minus_primary_quality"]

        out_rows.append(out)

    out_path = data_dir / args.output_csv
    write_csv(out_rows, out_path)

    print("[saved]", out_path)
    print("=" * 100)
    print(f"Rows: {len(out_rows)}")
    print(f"task_key: {args.task_key}")
    print(f"primary: {primary}")
    print(f"alternative: {alternative}")
    print()

    # Compact by condition.
    import collections
    by_cond = collections.defaultdict(list)
    for r in out_rows:
        by_cond[r.get("condition", "unknown")].append(r)

    def mean(rs, k):
        vals = []
        for x in rs:
            v = safe_float(x.get(k))
            if math.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    print("By condition:")
    for cond, rs in sorted(by_cond.items()):
        print(f"\n{cond}, n={len(rs)}")
        for k in [
            "primary_grasp_success_proxy",
            "primary_source_lifted",
            "primary_grasp_quality_score",
            "alternative_grasp_success_proxy",
            "alternative_source_lifted",
            "alternative_grasp_quality_score",
            "alternative_minus_primary_grasp_success_proxy",
            "alternative_minus_primary_quality",
        ]:
            print(f"  {k:<50} mean={mean(rs,k):+.5f}")


if __name__ == "__main__":
    main()