#!/usr/bin/env python3
"""
Compare replayed v3 action chunks against the original v3 source summary.

Purpose:
  Sanity-check that replay_policy_action_chunks_v3.py reproduces the same
  object-level consequences when replaying saved action chunks.

Inputs:
  --source_dir: original v3 dataset, e.g. same_state_grasp_cf_v3p2_settle_20seeds_...
  --replay_dir: replay output, e.g. policy_chunk_replay_sanity_v3_...
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_LABELS = [
    "bowl2_source_displacement_max",
    "bowl2_source_z_delta_max",
    "bowl2_source_moved",
    "bowl2_source_lifted",
    "bowl2_grasp_success_proxy",
    "bowl2_clean_grasp_proxy",
    "bowl2_grasp_quality_score",
    "bowl2_min_dist_to_source",
]


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


def parse_original_candidate_from_chunk_path(chunk_path: str) -> str:
    """
    replay summary contains original chunk_path, e.g.
      .../init_000/candidate_012_src_bowl2_xy_.../traj.npz

    source summary candidate_name is:
      src_bowl2_xy_...
    """
    p = Path(str(chunk_path))
    parent = p.parent.name

    m = re.match(r"candidate_\d+_(.*)", parent)
    if m:
        return m.group(1)

    # Fallback for non-candidate layout.
    return p.stem


def build_source_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    idx = {}
    for r in rows:
        key = (int(r["init_state_idx"]), str(r["candidate_name"]))
        idx[key] = r
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--source_csv", default="summary_recovered.csv")
    parser.add_argument("--replay_csv", default="summary.csv")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--out_csv", default="replay_vs_source_diff.csv")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    replay_dir = Path(args.replay_dir)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    source_rows = read_csv(source_dir / args.source_csv)
    replay_rows = read_csv(replay_dir / args.replay_csv)

    source_idx = build_source_index(source_rows)

    diffs = []
    missing = []

    for rr in replay_rows:
        init_idx = int(rr["init_state_idx"])

        if "chunk_path" not in rr:
            raise KeyError("Replay summary missing chunk_path. Cannot map to original candidate.")

        orig_cand = parse_original_candidate_from_chunk_path(str(rr["chunk_path"]))
        key = (init_idx, orig_cand)

        if key not in source_idx:
            missing.append(key)
            continue

        sr = source_idx[key]

        out = {
            "init_state_idx": init_idx,
            "source_candidate_name": orig_cand,
            "replay_candidate_name": rr.get("candidate_name", ""),
            "chunk_path": rr.get("chunk_path", ""),
        }

        for lab in labels:
            s = safe_float(sr.get(lab))
            r = safe_float(rr.get(lab))
            d = abs(s - r) if not (math.isnan(s) or math.isnan(r)) else float("nan")

            out[f"source_{lab}"] = s
            out[f"replay_{lab}"] = r
            out[f"absdiff_{lab}"] = d

        diffs.append(out)

    print("=" * 100)
    print("source_dir:", source_dir)
    print("replay_dir:", replay_dir)
    print("source rows:", len(source_rows))
    print("replay rows:", len(replay_rows))
    print("matched rows:", len(diffs))
    print("missing rows:", len(missing))
    print("=" * 100)

    if missing:
        print("\nMissing examples:")
        for x in missing[:20]:
            print(" ", x)

    print("\nLabel differences:")
    for lab in labels:
        vals = [
            safe_float(d.get(f"absdiff_{lab}"))
            for d in diffs
            if not math.isnan(safe_float(d.get(f"absdiff_{lab}")))
        ]

        if not vals:
            continue

        max_diff = max(vals)
        mean_diff = sum(vals) / len(vals)

        # For binary labels, count mismatches.
        mismatch = sum(v > 1e-6 for v in vals)

        print(
            f"{lab:36s} "
            f"mean_abs={mean_diff:.8f} "
            f"max_abs={max_diff:.8f} "
            f"mismatch={mismatch}/{len(vals)}"
        )

    out_path = replay_dir / args.out_csv

    if diffs:
        fields = list(diffs[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in diffs:
                writer.writerow(r)

        print("\nSaved:", out_path)


if __name__ == "__main__":
    main()