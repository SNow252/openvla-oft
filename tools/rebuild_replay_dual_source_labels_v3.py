#!/usr/bin/env python3
"""
Rebuild dual-source object-level labels for replayed policy chunks.

Current replay_policy_action_chunks_v3.py computes only bowl2-centric labels.
This script recomputes both:
  bowl1_* labels
  bowl2_* labels

Then adds contrast labels:
  bowl2_minus_bowl1_grasp_success_proxy
  bowl2_minus_bowl1_lifted
  bowl2_minus_bowl1_z_delta_max
  bowl2_minus_bowl1_displacement_max
  bowl2_minus_bowl1_quality
  bowl2_closer_than_bowl1_score

This is important because task8 original/default may manipulate bowl1,
while task1 native may manipulate bowl2.
"""

import argparse
import csv
import glob
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.collect_same_state_grasp_counterfactuals_v3 import compute_grasp_labels


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                try:
                    fv = float(v)
                    if k == "init_state_idx":
                        out[k] = int(fv)
                    else:
                        out[k] = fv
                except Exception:
                    out[k] = v
            rows.append(out)
    return rows


def find_candidate_dir(data_dir: Path, init_state_idx: int, candidate_name: str) -> Path:
    seed_dir = data_dir / f"init_{init_state_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No candidate dir for init={init_state_idx}, candidate={candidate_name}, pattern={pattern}"
        )

    if len(matches) > 1:
        print(f"[warning] multiple candidate dirs for {candidate_name}; using first")

    return Path(matches[0])


def load_seq_arrays(traj_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(traj_path, allow_pickle=True)

    required = [
        "eef_pos_seq",
        "gripper_qpos_seq",
        "bowl1_pos_seq",
        "bowl2_pos_seq",
    ]

    for k in required:
        if k not in data:
            raise KeyError(f"{traj_path} missing key: {k}")

    return {
        "eef_pos_seq": np.asarray(data["eef_pos_seq"], dtype=np.float32),
        "gripper_qpos_seq": np.asarray(data["gripper_qpos_seq"], dtype=np.float32),
        "bowl1_pos_seq": np.asarray(data["bowl1_pos_seq"], dtype=np.float32),
        "bowl2_pos_seq": np.asarray(data["bowl2_pos_seq"], dtype=np.float32),
    }


def prefix_numeric_labels(labels: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = {}
    for k, v in labels.items():
        if isinstance(v, (int, float, np.floating, str)):
            out[f"{prefix}_{k}"] = v
    return out


def parse_condition(candidate_name: str) -> str:
    name = str(candidate_name)

    # Fresh dump candidate examples:
    # openvla_oft_openvla_oft_task8_original_task8_init000_178...
    # openvla_oft_openvla_oft_task8_empty_language_task8_init000_178...
    # smolvla_smolvla_task8_original_task8_init000_178...

    # Old dump candidate examples:
    # openvla_oft_openvla_oft_mismatch_v2_processed_task8_original_xxxxx
    # openvla_oft_mismatch_v2_processed_task8_empty_language_xxxxx

    prefixes = [
        "openvla_oft_openvla_oft_mismatch_v2_processed_",
        "openvla_oft_mismatch_v2_processed_",
        "openvla_oft_openvla_oft_",
        "openvla_oft_",
        "smolvla_smolvla_",
        "smolvla_",
    ]

    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break

    # Fresh format: <condition>_task<id>_init<idx>_<timestamp>_<pid>_<hash>
    # Keep only <condition>.
    m = re.match(r"^(.*)_task\d+_init\d+_", name)
    if m:
        return m.group(1)

    # Old format: <condition>_<hash>
    # Remove final short hash token if present.
    parts = name.split("_")
    if len(parts) > 1 and re.match(r"^[0-9a-fA-F]{6,12}$", parts[-1]):
        name = "_".join(parts[:-1])

    return name


def add_contrast_labels(row: Dict[str, Any]) -> None:
    pairs = [
        ("grasp_success_proxy", "bowl2_minus_bowl1_grasp_success_proxy"),
        ("clean_grasp_proxy", "bowl2_minus_bowl1_clean_grasp_proxy"),
        ("source_lifted", "bowl2_minus_bowl1_lifted"),
        ("source_moved", "bowl2_minus_bowl1_moved"),
        ("source_z_delta_max", "bowl2_minus_bowl1_z_delta_max"),
        ("source_displacement_max", "bowl2_minus_bowl1_displacement_max"),
        ("grasp_quality_score", "bowl2_minus_bowl1_quality"),
    ]

    for base, out_key in pairs:
        b2 = safe_float(row.get(f"bowl2_{base}"))
        b1 = safe_float(row.get(f"bowl1_{base}"))
        row[out_key] = b2 - b1

    # Positive means closer to bowl2 than bowl1.
    d2 = safe_float(row.get("bowl2_min_dist_to_source"))
    d1 = safe_float(row.get("bowl1_min_dist_to_source"))
    row["bowl2_closer_than_bowl1_score"] = d1 - d2


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    base_fields = [
        "init_state_idx",
        "candidate_name",
        "condition",
        "candidate_family",
        "policy_name",
        "horizon_executed",
        "sum_reward",
        "any_done",
    ]

    extra = sorted({k for r in rows for k in r.keys()} - set(base_fields))
    fields = base_fields + extra

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def vals(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in rows:
        v = safe_float(r.get(key))
        if not math.isnan(v):
            out.append(v)
    return out


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("=" * 120)
    print("Rows:", len(rows))

    keys = [
        "bowl1_grasp_success_proxy",
        "bowl1_source_lifted",
        "bowl1_source_z_delta_max",
        "bowl1_source_displacement_max",
        "bowl1_grasp_quality_score",
        "bowl2_grasp_success_proxy",
        "bowl2_source_lifted",
        "bowl2_source_z_delta_max",
        "bowl2_source_displacement_max",
        "bowl2_grasp_quality_score",
        "bowl2_minus_bowl1_grasp_success_proxy",
        "bowl2_minus_bowl1_lifted",
        "bowl2_minus_bowl1_quality",
    ]

    print("\nOverall:")
    for k in keys:
        vs = vals(rows, k)
        if vs:
            print(f"{k:42s} mean={np.mean(vs):+.5f} min={np.min(vs):+.5f} max={np.max(vs):+.5f}")

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r.get("condition", "unknown")].append(r)

    print("\nBy condition:")
    for cond, rs in sorted(by_cond.items()):
        print(f"\n{cond}, n={len(rs)}")
        for k in [
            "bowl1_grasp_success_proxy",
            "bowl1_source_lifted",
            "bowl1_grasp_quality_score",
            "bowl2_grasp_success_proxy",
            "bowl2_source_lifted",
            "bowl2_grasp_quality_score",
            "bowl2_minus_bowl1_grasp_success_proxy",
            "bowl2_minus_bowl1_quality",
        ]:
            vs = vals(rs, k)
            if vs:
                print(f"  {k:40s} mean={np.mean(vs):+.5f} max={np.max(vs):+.5f}")

    print("\nTop 20 by bowl1 quality:")
    top1 = sorted(rows, key=lambda r: safe_float(r.get("bowl1_grasp_quality_score"), -999), reverse=True)
    for r in top1[:20]:
        print(
            f"init={int(r['init_state_idx']):02d} "
            f"cond={r.get('condition')} "
            f"b1_grasp={r.get('bowl1_grasp_success_proxy')} "
            f"b2_grasp={r.get('bowl2_grasp_success_proxy')} "
            f"b1_q={safe_float(r.get('bowl1_grasp_quality_score')):+.4f} "
            f"b2_q={safe_float(r.get('bowl2_grasp_quality_score')):+.4f} "
            f"name={str(r.get('candidate_name'))[:80]}"
        )

    print("\nTop 20 by bowl2 quality:")
    top2 = sorted(rows, key=lambda r: safe_float(r.get("bowl2_grasp_quality_score"), -999), reverse=True)
    for r in top2[:20]:
        print(
            f"init={int(r['init_state_idx']):02d} "
            f"cond={r.get('condition')} "
            f"b1_grasp={r.get('bowl1_grasp_success_proxy')} "
            f"b2_grasp={r.get('bowl2_grasp_success_proxy')} "
            f"b1_q={safe_float(r.get('bowl1_grasp_quality_score')):+.4f} "
            f"b2_q={safe_float(r.get('bowl2_grasp_quality_score')):+.4f} "
            f"name={str(r.get('candidate_name'))[:80]}"
        )

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--input_csv", default="summary.csv")
    parser.add_argument("--output_csv", default="summary_dual_source.csv")
    parser.add_argument("--contact_threshold", type=float, default=0.10)
    parser.add_argument("--move_threshold", type=float, default=0.020)
    parser.add_argument("--lift_threshold", type=float, default=0.010)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows_in = read_csv(data_dir / args.input_csv)

    rows_out = []

    for r in rows_in:
        init_idx = int(r["init_state_idx"])
        cand_name = str(r["candidate_name"])

        cand_dir = find_candidate_dir(data_dir, init_idx, cand_name)
        traj_path = cand_dir / "traj.npz"
        seq_arrays = load_seq_arrays(traj_path)

        labels_bowl1 = compute_grasp_labels(
            seq_arrays=seq_arrays,
            source="bowl1",
            distractor="bowl2",
            contact_threshold=args.contact_threshold,
            move_threshold=args.move_threshold,
            lift_threshold=args.lift_threshold,
        )

        labels_bowl2 = compute_grasp_labels(
            seq_arrays=seq_arrays,
            source="bowl2",
            distractor="bowl1",
            contact_threshold=args.contact_threshold,
            move_threshold=args.move_threshold,
            lift_threshold=args.lift_threshold,
        )

        out = dict(r)
        out["condition"] = parse_condition(cand_name)

        out.update(prefix_numeric_labels(labels_bowl1, "bowl1"))
        out.update(prefix_numeric_labels(labels_bowl2, "bowl2"))

        add_contrast_labels(out)
        rows_out.append(out)

    out_path = data_dir / args.output_csv
    write_csv(rows_out, out_path)

    print(f"[saved] {out_path}")
    print_summary(rows_out)


if __name__ == "__main__":
    main()