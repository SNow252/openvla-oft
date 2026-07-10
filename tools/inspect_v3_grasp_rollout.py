#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
            rows.append(out)
    return rows


def find_candidate_dirs(data_dir):
    return sorted(Path(data_dir).glob("init_*/candidate_*"))


def summarize_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    actions = data["actions"]
    eef = data["eef_pos_seq"]
    bowl2 = data["bowl2_pos_seq"]
    gq = data["gripper_qpos_seq"]
    phases = data["phase_names"] if "phase_names" in data else None

    d = np.linalg.norm(eef - bowl2, axis=1)
    bowl2_disp = np.linalg.norm(bowl2 - bowl2[0:1], axis=1)
    bowl2_z = bowl2[:, 2] - bowl2[0, 2]

    out = {
        "T": int(actions.shape[0]),
        "min_d": float(d.min()),
        "argmin_d": int(d.argmin()),
        "final_d": float(d[-1]),
        "bowl2_disp_max": float(bowl2_disp.max()),
        "bowl2_disp_argmax": int(bowl2_disp.argmax()),
        "bowl2_z_max": float(bowl2_z.max()),
        "bowl2_z_argmax": int(bowl2_z.argmax()),
        "eef_z_min": float(eef[:, 2].min()),
        "eef_z_max": float(eef[:, 2].max()),
        "bowl2_z0": float(bowl2[0, 2]),
        "eef_z0": float(eef[0, 2]),
        "eef_z_final": float(eef[-1, 2]),
        "gripper_qpos0": float(gq[0, 0]) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
        "gripper_qpos_final": float(gq[-1, 0]) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
        "gripper_qpos_min": float(gq[:, 0].min()) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
        "gripper_qpos_max": float(gq[:, 0].max()) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
        "action_gripper_min": float(actions[:, 6].min()),
        "action_gripper_max": float(actions[:, 6].max()),
    }

    if phases is not None:
        phase_report = {}
        phases = np.asarray(phases).astype(str)

        for ph in sorted(set(phases.tolist())):
            idx_action = np.where(phases == ph)[0]
            if len(idx_action) == 0:
                continue

            # state sequence has T+1 states; use action indices plus next state.
            idx_state = np.clip(idx_action + 1, 0, len(d) - 1)

            phase_report[ph] = {
                "n": int(len(idx_action)),
                "min_d": float(d[idx_state].min()),
                "bowl2_disp_max": float(bowl2_disp[idx_state].max()),
                "bowl2_z_max": float(bowl2_z[idx_state].max()),
                "eef_z_min": float(eef[idx_state, 2].min()),
                "eef_z_max": float(eef[idx_state, 2].max()),
                "gripper_qpos_min": float(gq[idx_state, 0].min()) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
                "gripper_qpos_max": float(gq[idx_state, 0].max()) if gq.ndim == 2 and gq.shape[1] > 0 else float("nan"),
                "action_gripper_min": float(actions[idx_action, 6].min()),
                "action_gripper_max": float(actions[idx_action, 6].max()),
            }

        out["phase_report"] = phase_report

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--summary_csv", default="summary_recovered.csv")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = load_rows(data_dir / args.summary_csv)

    print("=" * 120)
    print("DATA_DIR:", data_dir)
    print("Rows:", len(rows))

    keys = [
        "bowl2_source_displacement_max",
        "bowl2_source_z_delta_max",
        "bowl2_source_moved",
        "bowl2_source_lifted",
        "bowl2_grasp_success_proxy",
        "bowl2_gripper_qpos_delta_0",
        "bowl2_min_dist_to_source",
    ]

    print("\nOverall:")
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), float)]
        if vals:
            print(f"{k:36s} mean={np.mean(vals):+.5f} min={np.min(vals):+.5f} max={np.max(vals):+.5f}")

    print("\nBy close value:")
    by_close = defaultdict(list)
    for r in rows:
        by_close[r.get("gripper_close_value")].append(r)

    for close, rs in sorted(by_close.items(), key=lambda x: str(x[0])):
        print(f"\nclose={close}, n={len(rs)}")
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r.get(k), float)]
            if vals:
                print(f"  {k:34s} mean={np.mean(vals):+.5f} min={np.min(vals):+.5f} max={np.max(vals):+.5f}")

    print("\nTop by min distance:")
    rows_min = sorted(rows, key=lambda r: r.get("bowl2_min_dist_to_source", 999))
    for r in rows_min[:args.top_k]:
        print(
            f"{str(r['candidate_name'])[:80]:80s} "
            f"close={r.get('gripper_close_value')} "
            f"xy=({r.get('xy_offset_x'):+.2f},{r.get('xy_offset_y'):+.2f}) "
            f"zg={r.get('z_grasp_offset'):.2f} "
            f"min_d={r.get('bowl2_min_dist_to_source'):.4f} "
            f"disp={r.get('bowl2_source_displacement_max'):.4f} "
            f"lift={r.get('bowl2_source_z_delta_max'):.4f} "
            f"gq_delta={r.get('bowl2_gripper_qpos_delta_0'):.4f}"
        )

    print("\nTop by lift:")
    rows_lift = sorted(rows, key=lambda r: r.get("bowl2_source_z_delta_max", -999), reverse=True)
    for r in rows_lift[:args.top_k]:
        print(
            f"{str(r['candidate_name'])[:80]:80s} "
            f"close={r.get('gripper_close_value')} "
            f"xy=({r.get('xy_offset_x'):+.2f},{r.get('xy_offset_y'):+.2f}) "
            f"zg={r.get('z_grasp_offset'):.2f} "
            f"min_d={r.get('bowl2_min_dist_to_source'):.4f} "
            f"disp={r.get('bowl2_source_displacement_max'):.4f} "
            f"lift={r.get('bowl2_source_z_delta_max'):.4f} "
            f"gq_delta={r.get('bowl2_gripper_qpos_delta_0'):.4f}"
        )

    print("\nDetailed phase report for closest candidates:")
    candidate_dirs = find_candidate_dirs(data_dir)
    cand_map = {p.name.split("_", 2)[-1]: p for p in candidate_dirs}

    for r in rows_min[:min(args.top_k, 5)]:
        cname = str(r["candidate_name"])
        cdir = None
        for p in candidate_dirs:
            if p.name.endswith(cname):
                cdir = p
                break

        if cdir is None:
            print("Missing candidate dir:", cname)
            continue

        print("\n" + "-" * 120)
        print(cname)
        rep = summarize_npz(cdir / "traj.npz")
        for k, v in rep.items():
            if k == "phase_report":
                continue
            print(f"{k:24s}: {v}")

        if "phase_report" in rep:
            print("phase_report:")
            for ph, pr in rep["phase_report"].items():
                print(f"  [{ph}]")
                for k, v in pr.items():
                    print(f"    {k:22s}: {v}")


if __name__ == "__main__":
    main()