#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np


def key_from_eps(eps: float) -> str:
    # Preserve trailing zeros: 0.10 -> 0p10
    return f"{eps:.2f}".replace(".", "p").replace("-", "m")


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def find_candidate_dir(data_dir: Path, init_idx: int, candidate_name: str) -> Path:
    seed_dir = data_dir / f"init_{init_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No candidate dir: {pattern}")
    return Path(matches[0])


def load_rows(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(r)
    return fieldnames, rows


def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def write_rows(path: Path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--thresholds", default="0.20,0.18,0.16")
    parser.add_argument("--input_csv", default="summary.csv")
    parser.add_argument("--output_csv", default="summary_tau_rebuilt.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    thresholds = parse_float_list(args.thresholds)

    fieldnames, rows = load_rows(data_dir / args.input_csv)

    new_fields = []
    for eps in thresholds:
        k = key_from_eps(eps)
        new_fields += [
            f"reached_source_eps_{k}",
            f"tau_source_eps_{k}",
        ]

    new_fields += [
        "min_dist_to_bowl2_rebuilt",
        "final_dist_to_bowl2_rebuilt",
        "min_dist_to_bowl1_rebuilt",
        "final_dist_to_bowl1_rebuilt",
        "min_dist_advantage_bowl1_minus_bowl2_rebuilt",
        "final_dist_advantage_bowl1_minus_bowl2_rebuilt",
        "reached_adv_bowl2_over_bowl1_rebuilt",
        "tau_adv_bowl2_over_bowl1_rebuilt",
    ]

    for nf in new_fields:
        if nf not in fieldnames:
            fieldnames.append(nf)

    for r in rows:
        init_idx = int(float(r["init_state_idx"]))
        cand = r["candidate_name"]

        cand_dir = find_candidate_dir(data_dir, init_idx, cand)
        traj_path = cand_dir / "traj.npz"

        data = np.load(traj_path)

        d2 = np.asarray(data["dist_eef_to_bowl2_seq"], dtype=np.float32).reshape(-1)
        d1 = np.asarray(data["dist_eef_to_bowl1_seq"], dtype=np.float32).reshape(-1)

        H = len(d2) - 1

        for eps in thresholds:
            k = key_from_eps(eps)
            hit = np.where(d2 <= eps)[0]
            if len(hit) > 0:
                r[f"reached_source_eps_{k}"] = 1
                r[f"tau_source_eps_{k}"] = int(hit[0])
            else:
                r[f"reached_source_eps_{k}"] = 0
                r[f"tau_source_eps_{k}"] = H + 1

        r["min_dist_to_bowl2_rebuilt"] = float(np.min(d2))
        r["final_dist_to_bowl2_rebuilt"] = float(d2[-1])
        r["min_dist_to_bowl1_rebuilt"] = float(np.min(d1))
        r["final_dist_to_bowl1_rebuilt"] = float(d1[-1])

        r["min_dist_advantage_bowl1_minus_bowl2_rebuilt"] = (
            float(np.min(d1)) - float(np.min(d2))
        )
        r["final_dist_advantage_bowl1_minus_bowl2_rebuilt"] = (
            float(d1[-1]) - float(d2[-1])
        )

        adv_hit = np.where(d2 < d1)[0]
        if len(adv_hit) > 0:
            r["reached_adv_bowl2_over_bowl1_rebuilt"] = 1
            r["tau_adv_bowl2_over_bowl1_rebuilt"] = int(adv_hit[0])
        else:
            r["reached_adv_bowl2_over_bowl1_rebuilt"] = 0
            r["tau_adv_bowl2_over_bowl1_rebuilt"] = H + 1

    out_path = data_dir / args.output_csv
    write_rows(out_path, rows, fieldnames)

    print(f"[done] wrote {out_path}")
    print(f"[thresholds] {thresholds}")
    print(f"[rows] {len(rows)}")


if __name__ == "__main__":
    main()