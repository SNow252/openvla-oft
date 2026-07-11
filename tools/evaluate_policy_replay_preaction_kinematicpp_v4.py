#!/usr/bin/env python3
"""
Pre-action Kinematic++ baselines for policy action consequence evaluation.

This script is different from post-hoc kinematic oracle.

Allowed inputs:
  - initial eef position
  - initial gripper qpos
  - initial bowl1 / bowl2 positions
  - action chunk

Forbidden inputs:
  - future eef_pos_seq[1:]
  - future bowl*_pos_seq[1:]
  - future gripper_qpos_seq[1:]

It predicts a pseudo EEF trajectory by integrating action_xyz:
  pred_eef[t+1] = pred_eef[t] + scale * action[t, :3]

Then it computes object-relative kinematic scores before execution:
  - eef_min_dist
  - close_min_dist
  - relative_margin
  - grasp_heuristic

Outputs:
  preaction_kinematicpp_object_scores.csv
  preaction_kinematicpp_pair_predictions.csv
  preaction_kinematicpp_metrics.csv
  preaction_kinematicpp_metrics.md
  preaction_kinematicpp_best_by_method.csv
"""

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


METHODS = [
    "eef_min_dist",
    "close_min_dist",
    "relative_margin",
    "grasp_heuristic",
]

DEFAULT_SCALES = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]


def to_float(x: Any):
    try:
        return float(x)
    except Exception:
        return x


def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


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


def find_candidate_dir(replay_dir: Path, init_state_idx: int, candidate_name: str) -> Path:
    seed_dir = replay_dir / f"init_{init_state_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No candidate dir found: {pattern}")
    return Path(matches[0])


def load_replay_initial_and_actions(traj_path: Path, chunk_path: str | None = None):
    d = np.load(traj_path, allow_pickle=True)

    # Only use t=0 from replay state sequences.
    needed = ["eef_pos_seq", "gripper_qpos_seq", "bowl1_pos_seq", "bowl2_pos_seq"]
    for k in needed:
        if k not in d:
            raise KeyError(f"{traj_path} missing key: {k}")

    eef0 = np.asarray(d["eef_pos_seq"], dtype=np.float32)[0]
    gripper0 = np.asarray(d["gripper_qpos_seq"], dtype=np.float32)[0]
    bowl1_0 = np.asarray(d["bowl1_pos_seq"], dtype=np.float32)[0]
    bowl2_0 = np.asarray(d["bowl2_pos_seq"], dtype=np.float32)[0]

    actions = None
    if "actions" in d:
        actions = np.asarray(d["actions"], dtype=np.float32)

    if actions is None or actions.size == 0:
        if chunk_path:
            cd = np.load(chunk_path, allow_pickle=True)
            if "actions" in cd:
                actions = np.asarray(cd["actions"], dtype=np.float32)
            elif "processed_actions" in cd:
                actions = np.asarray(cd["processed_actions"], dtype=np.float32)

    if actions is None or actions.size == 0:
        raise KeyError(f"No actions found in {traj_path} or chunk_path={chunk_path}")

    actions = actions.reshape(-1, actions.shape[-1]).astype(np.float32)
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected action_dim=7, got {actions.shape}")

    return {
        "eef0": eef0.astype(np.float32),
        "gripper0": gripper0.astype(np.float32),
        "bowl1": bowl1_0.astype(np.float32),
        "bowl2": bowl2_0.astype(np.float32),
        "actions": actions,
    }


def first_gripper_col(g):
    g = np.asarray(g, dtype=float)
    if g.ndim == 0:
        return float(g)
    return float(g.reshape(-1)[0])


def predict_eef_path(eef0: np.ndarray, actions: np.ndarray, scale: float):
    xyz = np.asarray(actions[:, :3], dtype=np.float32)
    pred = [np.asarray(eef0, dtype=np.float32)]
    cur = pred[0].copy()

    for a in xyz:
        cur = cur + float(scale) * a
        pred.append(cur.copy())

    return np.stack(pred, axis=0)


def action_close_mask(actions: np.ndarray):
    """
    For LIBERO env action convention, +1 usually means close, -1 open.
    We use command sign only, not future gripper state.
    """
    if actions.shape[1] < 7:
        return np.zeros((len(actions),), dtype=bool)

    g = np.asarray(actions[:, 6], dtype=float)
    return g > 0


def cosine2d(a, b):
    a = np.asarray(a[:2], dtype=float)
    b = np.asarray(b[:2], dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_preaction_scores(init, source_name: str, distractor_name: str, scale: float):
    eef0 = init["eef0"]
    actions = init["actions"]
    src = init[source_name]
    dst = init[distractor_name]

    eef_pred = predict_eef_path(eef0, actions, scale=scale)

    # Align close mask with predicted eef path after each action.
    close_mask_action = action_close_mask(actions)
    close_mask_path = np.concatenate([[False], close_mask_action], axis=0)

    dist_src = np.linalg.norm(eef_pred - src[None, :], axis=1)
    dist_dst = np.linalg.norm(eef_pred - dst[None, :], axis=1)

    min_idx = int(np.argmin(dist_src))
    min_d = float(dist_src[min_idx])
    min_distr = float(np.min(dist_dst))

    if np.any(close_mask_path):
        closed_min_d = float(np.min(dist_src[close_mask_path]))
        first_close = int(np.argmax(close_mask_path))
    else:
        closed_min_d = min_d
        first_close = min_idx

    z_after_close = eef_pred[first_close:, 2]
    eef_lift_after_close = float(np.max(z_after_close) - eef_pred[first_close, 2]) if len(z_after_close) else 0.0

    desired_xy = src[:2] - eef0[:2]
    actual_xy = eef_pred[min_idx, :2] - eef0[:2]
    approach_cos = cosine2d(desired_xy, actual_xy)

    # Scores: higher means more likely / better consequence.
    eef_min_dist = -min_d
    close_min_dist = -closed_min_d
    relative_margin = min_distr - min_d

    grasp_heuristic = (
        1.0 * (0.10 - min_d)
        + 1.5 * (0.10 - closed_min_d)
        + 0.4 * eef_lift_after_close
        + 0.05 * approach_cos
        + 0.5 * (min_distr - min_d)
    )

    return {
        "scale": float(scale),
        "min_dist": min_d,
        "closed_min_dist": closed_min_d,
        "min_distractor_dist": min_distr,
        "approach_cos": approach_cos,
        "eef_lift_after_close": eef_lift_after_close,
        "eef_min_dist": eef_min_dist,
        "close_min_dist": close_min_dist,
        "relative_margin": relative_margin,
        "grasp_heuristic": grasp_heuristic,
    }


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


def sign_acc(label_delta, score_delta):
    lab = np.asarray(label_delta, dtype=float)
    pred = np.asarray(score_delta, dtype=float)

    mask = np.isfinite(lab) & np.isfinite(pred) & (np.abs(lab) > 1e-9)
    if mask.sum() == 0:
        return float("nan")

    return float((np.sign(lab[mask]) == np.sign(pred[mask])).mean())


def vals(rows, key):
    out = []
    for r in rows:
        v = safe_float(r.get(key))
        if np.isfinite(v):
            out.append(v)
    return out


def build_object_score_rows(replay_dir: Path, scales: List[float]):
    summary_path = replay_dir / "summary_dual_source.csv"
    rows = read_csv(summary_path)

    out = []

    for r in rows:
        init_idx = int(safe_float(r["init_state_idx"]))
        cand = str(r["candidate_name"])
        cand_dir = find_candidate_dir(replay_dir, init_idx, cand)
        traj_path = cand_dir / "traj.npz"
        chunk_path = str(r.get("chunk_path", "")) if r.get("chunk_path", "") else None

        init = load_replay_initial_and_actions(traj_path, chunk_path=chunk_path)

        policy = str(r.get("policy_name", "unknown"))
        cond = str(r.get("condition", "unknown"))

        for scale in scales:
            for obj, distractor in [("bowl1", "bowl2"), ("bowl2", "bowl1")]:
                scores = compute_preaction_scores(init, obj, distractor, scale)

                out_row = {
                    "policy": policy,
                    "condition": cond,
                    "init_state_idx": init_idx,
                    "candidate_name": cand,
                    "object": obj,
                    "scale": float(scale),
                    "grasp_label": safe_float(r.get(f"{obj}_grasp_success_proxy")),
                    "lift_label": safe_float(r.get(f"{obj}_source_lifted")),
                    "quality_label": safe_float(r.get(f"{obj}_grasp_quality_score")),
                }
                out_row.update(scores)
                out.append(out_row)

    return out


def pairwise_rows(object_rows):
    by_candidate = {}
    for r in object_rows:
        key = (
            r["policy"],
            r["condition"],
            r["init_state_idx"],
            r["candidate_name"],
            r["scale"],
        )
        by_candidate.setdefault(key, {})[r["object"]] = r

    pairs = []

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
            "scale": key[4],
            "label_delta_quality_b2_minus_b1": b2["quality_label"] - b1["quality_label"],
            "label_delta_grasp_b2_minus_b1": b2["grasp_label"] - b1["grasp_label"],
        }

        for m in METHODS:
            row[f"{m}_delta_b2_minus_b1"] = b2[m] - b1[m]

        pairs.append(row)

    return pairs


def summarize_metrics(object_rows, pair_rows):
    policies = sorted({r["policy"] for r in object_rows})
    scales = sorted({float(r["scale"]) for r in object_rows})

    groups = [("ALL", object_rows, pair_rows)]
    for p in policies:
        groups.append(
            (
                p,
                [r for r in object_rows if r["policy"] == p],
                [r for r in pair_rows if r["policy"] == p],
            )
        )

    metric_rows = []

    for group_name, obj_group, pair_group in groups:
        for scale in scales:
            obj_rs = [r for r in obj_group if float(r["scale"]) == scale]
            pair_rs = [r for r in pair_group if float(r["scale"]) == scale]

            for m in METHODS:
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
                        "scale": scale,
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


def best_by_method(metric_rows):
    """
    Pick best scale per group/method using:
      primary: pair_acc_quality_preference
      secondary: object_spearman_quality
    This is a strong tuned kinematic control, not a purely fixed-scale baseline.
    """
    best = {}
    for r in metric_rows:
        key = (r["group"], r["method"])

        score1 = safe_float(r.get("pair_acc_quality_preference"), -999)
        score2 = safe_float(r.get("object_spearman_quality"), -999)
        score3 = safe_float(r.get("object_auc_grasp"), -999)
        rank = (score1, score2, score3)

        if key not in best or rank > best[key][0]:
            best[key] = (rank, r)

    return [v[1] for v in best.values()]


def write_markdown(metric_rows, path: Path, title: str):
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| Group | Scale | Method | Grasp AUC | Lift AUC | Quality ρ | PairAcc-Q | PairAcc-Grasp |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")

    for r in metric_rows:
        lines.append(
            f"| {r['group']} | {float(r['scale']):.4g} | {r['method']} | "
            f"{safe_float(r['object_auc_grasp']):.3f} | "
            f"{safe_float(r['object_auc_lift']):.3f} | "
            f"{safe_float(r['object_spearman_quality']):.3f} | "
            f"{safe_float(r['pair_acc_quality_preference']):.3f} | "
            f"{safe_float(r['pair_acc_grasp_preference']):.3f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument(
        "--scales",
        default=",".join(str(x) for x in DEFAULT_SCALES),
        help="Comma-separated integration scales for action xyz.",
    )
    args = ap.parse_args()

    replay_dir = Path(args.replay_dir)
    out_dir = Path(args.out_dir) if args.out_dir else replay_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    object_rows = build_object_score_rows(replay_dir, scales)
    pair_rows = pairwise_rows(object_rows)
    metric_rows = summarize_metrics(object_rows, pair_rows)
    best_rows = best_by_method(metric_rows)

    object_csv = out_dir / "preaction_kinematicpp_object_scores.csv"
    pair_csv = out_dir / "preaction_kinematicpp_pair_predictions.csv"
    metrics_csv = out_dir / "preaction_kinematicpp_metrics.csv"
    metrics_md = out_dir / "preaction_kinematicpp_metrics.md"
    best_csv = out_dir / "preaction_kinematicpp_best_by_method.csv"
    best_md = out_dir / "preaction_kinematicpp_best_by_method.md"

    write_csv(object_rows, object_csv)
    write_csv(pair_rows, pair_csv)
    write_csv(metric_rows, metrics_csv)
    write_csv(best_rows, best_csv)

    write_markdown(metric_rows, metrics_md, "Pre-action Kinematic++ all scales")
    write_markdown(best_rows, best_md, "Pre-action Kinematic++ best scale per method")

    print("[saved]", object_csv)
    print("[saved]", pair_csv)
    print("[saved]", metrics_csv)
    print("[saved]", metrics_md)
    print("[saved]", best_csv)
    print("[saved]", best_md)
    print()
    print(best_md.read_text())


if __name__ == "__main__":
    main()