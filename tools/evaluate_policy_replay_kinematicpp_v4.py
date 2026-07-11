#!/usr/bin/env python3
"""
Kinematic++ baselines for policy replay consequence analysis.

This script evaluates whether object-level consequences can be explained by
simple EEF / gripper / object-relative geometry.

Input:
  replay_dir/
    summary_dual_source.csv
    init_000/candidate_*/traj.npz

Output:
  kinematicpp_object_scores.csv
  kinematicpp_metrics.csv
  kinematicpp_metrics.md

Baselines:
  1. eef_min_dist:
       score = - min_t ||eef_t - object_t||

  2. close_min_dist:
       score = - min_t_closed ||eef_t - object_t||

  3. relative_margin:
       score = min_dist_to_distractor - min_dist_to_source

  4. grasp_heuristic:
       combines source proximity, closed-gripper proximity, lift-after-close,
       and penalty for being closer to the distractor.

These are not learned models. They are strong analytical controls.
"""

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def to_float(x: Any):
    try:
        return float(x)
    except Exception:
        return x


def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: to_float(v) for k, v in r.items()})
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def find_candidate_dir(replay_dir: Path, init_state_idx: int, candidate_name: str) -> Path:
    seed_dir = replay_dir / f"init_{init_state_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No candidate dir found: {pattern}")
    return Path(matches[0])


def load_traj(traj_path: Path):
    d = np.load(traj_path, allow_pickle=True)
    needed = ["eef_pos_seq", "gripper_qpos_seq", "bowl1_pos_seq", "bowl2_pos_seq"]
    for k in needed:
        if k not in d:
            raise KeyError(f"{traj_path} missing {k}")
    return {
        "eef": np.asarray(d["eef_pos_seq"], dtype=np.float32),
        "gripper": np.asarray(d["gripper_qpos_seq"], dtype=np.float32),
        "bowl1": np.asarray(d["bowl1_pos_seq"], dtype=np.float32),
        "bowl2": np.asarray(d["bowl2_pos_seq"], dtype=np.float32),
    }


def first_col(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    return x[:, 0]


def rankdata_average(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    rx = rankdata_average(x[mask])
    ry = rankdata_average(y[mask])
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_score(y_true, score):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = y > 0.5
    neg = y <= 0.5
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(s)
    sum_pos = float(ranks[pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cosine2d(a, b):
    a = np.asarray(a[:2], dtype=float)
    b = np.asarray(b[:2], dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_object_kinematic_scores(seq, source_name: str, distractor_name: str):
    eef = seq["eef"]
    src = seq[source_name]
    dst = seq[distractor_name]
    q = first_col(seq["gripper"]).astype(float)

    T = min(len(eef), len(src), len(dst), len(q))
    eef = eef[:T]
    src = src[:T]
    dst = dst[:T]
    q = q[:T]

    dist_src = np.linalg.norm(eef - src, axis=1)
    dist_dst = np.linalg.norm(eef - dst, axis=1)

    min_idx = int(np.argmin(dist_src))
    min_d = float(dist_src[min_idx])
    min_distr = float(np.min(dist_dst))

    q0 = float(q[0])
    q_min = float(np.min(q))
    q_max = float(np.max(q))

    # In LIBERO/Franka qpos usually decreases when closing.
    close_amount = q0 - q
    close_thr = max(0.005, 0.25 * max(1e-6, q0 - q_min))
    closed = close_amount >= close_thr

    if np.any(closed):
        closed_min_d = float(np.min(dist_src[closed]))
        first_close = int(np.argmax(closed))
    else:
        closed_min_d = min_d
        first_close = min_idx

    eef_lift_after_close = float(np.max(eef[first_close:, 2]) - eef[first_close, 2])
    eef_down_before_close = float(eef[0, 2] - np.min(eef[: first_close + 1, 2]))

    desired_xy = src[0, :2] - eef[0, :2]
    actual_xy = eef[min_idx, :2] - eef[0, :2]
    approach_cos = cosine2d(desired_xy, actual_xy)

    # Unsupervised analytical scores.
    eef_min_dist_score = -min_d
    close_min_dist_score = -closed_min_d
    relative_margin_score = min_distr - min_d

    # Stronger hand-built grasp heuristic.
    # High when: source is approached, gripper closes near source, some lift after close,
    # and distractor is not the closer object.
    grasp_heuristic = (
        1.0 * (0.10 - min_d)
        + 1.5 * (0.10 - closed_min_d)
        + 0.4 * eef_lift_after_close
        + 0.05 * approach_cos
        + 0.5 * (min_distr - min_d)
    )

    return {
        "min_dist": min_d,
        "closed_min_dist": closed_min_d,
        "min_distractor_dist": min_distr,
        "approach_cos": approach_cos,
        "eef_lift_after_close": eef_lift_after_close,
        "eef_down_before_close": eef_down_before_close,
        "gripper_q0": q0,
        "gripper_q_min": q_min,
        "gripper_q_max": q_max,
        "eef_min_dist": eef_min_dist_score,
        "close_min_dist": close_min_dist_score,
        "relative_margin": relative_margin_score,
        "grasp_heuristic": grasp_heuristic,
    }


def parse_condition(row):
    return str(row.get("condition", "unknown"))


def build_object_score_rows(replay_dir: Path):
    summary_path = replay_dir / "summary_dual_source.csv"
    rows = read_csv(summary_path)

    out = []

    for r in rows:
        init_idx = int(safe_float(r["init_state_idx"]))
        cand = str(r["candidate_name"])
        cand_dir = find_candidate_dir(replay_dir, init_idx, cand)
        traj = load_traj(cand_dir / "traj.npz")

        policy = str(r.get("policy_name", "unknown"))
        cond = parse_condition(r)

        for obj, distractor in [("bowl1", "bowl2"), ("bowl2", "bowl1")]:
            scores = compute_object_kinematic_scores(traj, obj, distractor)

            prefix = obj
            out_row = {
                "policy": policy,
                "condition": cond,
                "init_state_idx": init_idx,
                "candidate_name": cand,
                "object": obj,
                "object_is_bowl2": 1.0 if obj == "bowl2" else 0.0,
                "grasp_label": safe_float(r.get(f"{prefix}_grasp_success_proxy")),
                "lift_label": safe_float(r.get(f"{prefix}_source_lifted")),
                "quality_label": safe_float(r.get(f"{prefix}_grasp_quality_score")),
            }
            out_row.update(scores)
            out.append(out_row)

    return out


def pairwise_rows(object_rows):
    by_candidate = {}
    for r in object_rows:
        key = (r["policy"], r["condition"], r["init_state_idx"], r["candidate_name"])
        by_candidate.setdefault(key, {})[r["object"]] = r

    pairs = []
    methods = ["eef_min_dist", "close_min_dist", "relative_margin", "grasp_heuristic"]

    for key, d in by_candidate.items():
        if "bowl1" not in d or "bowl2" not in d:
            continue

        b1 = d["bowl1"]
        b2 = d["bowl2"]

        row = {
            "policy": key[0],
            "condition": key[1],
            "init_state_idx": key[2],
            "candidate_name": key[3],
            "label_delta_quality_b2_minus_b1": b2["quality_label"] - b1["quality_label"],
            "label_delta_grasp_b2_minus_b1": b2["grasp_label"] - b1["grasp_label"],
        }

        for m in methods:
            row[f"{m}_delta_b2_minus_b1"] = b2[m] - b1[m]

        pairs.append(row)

    return pairs


def sign_acc(label_delta, score_delta):
    lab = np.asarray(label_delta, dtype=float)
    pred = np.asarray(score_delta, dtype=float)
    mask = np.isfinite(lab) & np.isfinite(pred) & (np.abs(lab) > 1e-9)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(lab[mask]) == np.sign(pred[mask])).mean())


def summarize_metrics(object_rows, pair_rows):
    methods = ["eef_min_dist", "close_min_dist", "relative_margin", "grasp_heuristic"]

    metric_rows = []

    policies = sorted({r["policy"] for r in object_rows})
    groups = [("ALL", object_rows, pair_rows)]

    for p in policies:
        obj_p = [r for r in object_rows if r["policy"] == p]
        pair_p = [r for r in pair_rows if r["policy"] == p]
        groups.append((p, obj_p, pair_p))

    for group_name, obj_rs, pair_rs in groups:
        for m in methods:
            y_grasp = [r["grasp_label"] for r in obj_rs]
            y_lift = [r["lift_label"] for r in obj_rs]
            y_q = [r["quality_label"] for r in obj_rs]
            score = [r[m] for r in obj_rs]

            q_delta = [r["label_delta_quality_b2_minus_b1"] for r in pair_rs]
            g_delta = [r["label_delta_grasp_b2_minus_b1"] for r in pair_rs]
            s_delta = [r[f"{m}_delta_b2_minus_b1"] for r in pair_rs]

            metric_rows.append(
                {
                    "group": group_name,
                    "method": m,
                    "object_auc_grasp": auc_score(y_grasp, score),
                    "object_auc_lift": auc_score(y_lift, score),
                    "object_spearman_quality": spearman(y_q, score),
                    "pair_acc_quality_preference": sign_acc(q_delta, s_delta),
                    "pair_acc_grasp_preference": sign_acc(g_delta, s_delta),
                    "n_object_rows": len(obj_rs),
                    "n_pair_rows": len(pair_rs),
                }
            )

    return metric_rows


def write_markdown(metric_rows, path: Path):
    lines = []
    lines.append("| Group | Method | Grasp AUC | Lift AUC | Quality ρ | PairAcc-Q | PairAcc-Grasp |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in metric_rows:
        lines.append(
            f"| {r['group']} | {r['method']} | "
            f"{r['object_auc_grasp']:.3f} | {r['object_auc_lift']:.3f} | "
            f"{r['object_spearman_quality']:.3f} | "
            f"{r['pair_acc_quality_preference']:.3f} | "
            f"{r['pair_acc_grasp_preference']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    replay_dir = Path(args.replay_dir)
    out_dir = Path(args.out_dir) if args.out_dir else replay_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    object_rows = build_object_score_rows(replay_dir)
    pair_rows = pairwise_rows(object_rows)
    metric_rows = summarize_metrics(object_rows, pair_rows)

    object_csv = out_dir / "kinematicpp_object_scores.csv"
    pair_csv = out_dir / "kinematicpp_pair_predictions.csv"
    metrics_csv = out_dir / "kinematicpp_metrics.csv"
    metrics_md = out_dir / "kinematicpp_metrics.md"

    write_csv(object_rows, object_csv)
    write_csv(pair_rows, pair_csv)
    write_csv(metric_rows, metrics_csv)
    write_markdown(metric_rows, metrics_md)

    print("[saved]", object_csv)
    print("[saved]", pair_csv)
    print("[saved]", metrics_csv)
    print("[saved]", metrics_md)
    print()
    print(metrics_md.read_text())


if __name__ == "__main__":
    main()