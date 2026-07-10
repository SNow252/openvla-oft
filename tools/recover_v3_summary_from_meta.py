#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def flatten_meta(meta):
    row = {
        "init_state_idx": meta.get("init_state_idx"),
        "candidate_name": meta.get("candidate_name"),
        "candidate_family": meta.get("candidate_family"),
        "horizon_executed": meta.get("horizon_executed"),
        "sum_reward": meta.get("sum_reward"),
        "any_done": int(bool(meta.get("any_done", False))),
    }

    spec = meta.get("candidate_spec", {})
    for k, v in spec.items():
        row[k] = v

    labels = meta.get("labels", {})
    for k, v in labels.items():
        row[k] = v

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_csv", default="summary_recovered.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metas = sorted(data_dir.glob("init_*/candidate_*/meta.json"))

    rows = []
    for p in metas:
        with open(p) as f:
            meta = json.load(f)
        row = flatten_meta(meta)
        row["meta_path"] = str(p)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No meta.json found under {data_dir}")

    fields = sorted({k for r in rows for k in r.keys()})
    out_path = data_dir / args.output_csv

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Recovered rows:", len(rows))
    print("Saved:", out_path)

    keys = [
        "bowl2_source_displacement_max",
        "bowl2_source_z_delta_max",
        "bowl2_source_moved",
        "bowl2_source_lifted",
        "bowl2_grasp_success_proxy",
        "bowl2_gripper_qpos_delta_0",
        "bowl2_min_dist_to_source",
    ]

    for k in keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        if vals:
            print(f"{k:36s} mean={sum(vals)/len(vals):+.5f} min={min(vals):+.5f} max={max(vals):+.5f}")


if __name__ == "__main__":
    main()