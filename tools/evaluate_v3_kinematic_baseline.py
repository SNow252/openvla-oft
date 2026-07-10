#!/usr/bin/env python3
"""
Evaluate analytical EEF-kinematic baselines for v3 grasp/contact counterfactuals.

Purpose:
  Test whether object-level grasp/lift outcomes can be explained by simple
  EEF kinematics instead of learned action consequence modeling.

Dataset:
  same_state_grasp_cf_v3p2_settle_20seeds_*
  summary_recovered.csv
  init_XXX/candidate_YYY_*/traj.npz

Kinematic trajectory modes:
  raw_scale1:
    eef_pred[t] = eef0 + cumsum(action_xyz)

  fit_scalar:
    fit one global scalar scale on train split:
      final_eef_disp ~= scale * sum(action_xyz)

  fit_xyz:
    fit one scale per xyz dimension on train split.

  actual_eef_oracle:
    use logged actual EEF trajectory, but still only compare to initial bowl2
    position and never use future object positions.
    This is an upper-bound EEF-trajectory-only baseline.

Feature groups:
  min_dist:
    only min distance to bowl2.

  close_min_dist:
    min distance while gripper command is close.

  eef_rule:
    contact / close / lift heuristic features.

  eef_full:
    broader EEF-only kinematic feature set.

Targets:
  bowl2_grasp_success_proxy
  bowl2_clean_grasp_proxy
  bowl2_source_lifted
  bowl2_source_z_delta_max
  bowl2_source_displacement_max
  bowl2_grasp_quality_score

Default split:
  train init_state: 0-13
  test init_state: 14-19
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


TARGETS_DEFAULT = [
    "bowl2_grasp_success_proxy",
    "bowl2_clean_grasp_proxy",
    "bowl2_source_lifted",
    "bowl2_source_z_delta_max",
    "bowl2_source_displacement_max",
    "bowl2_grasp_quality_score",
]


BINARY_TARGETS = {
    "bowl2_grasp_success_proxy",
    "bowl2_clean_grasp_proxy",
    "bowl2_source_lifted",
    "bowl2_source_moved",
}


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_range_or_list(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return parse_int_list(s)


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


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
            out: Dict[str, Any] = {}
            for k, v in r.items():
                if k in {
                    "candidate_name",
                    "candidate_family",
                    "source",
                    "source_object",
                    "distractor_object",
                    "meta_path",
                }:
                    out[k] = v
                else:
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
        matches = sorted(
            [
                str(p.parent)
                for p in seed_dir.glob("candidate_*/meta.json")
                if p.parent.name.endswith(candidate_name)
            ]
        )

    if not matches:
        raise FileNotFoundError(
            f"No candidate dir for init={init_state_idx}, candidate={candidate_name}, pattern={pattern}"
        )

    if len(matches) > 1:
        print(f"[warning] multiple candidate dirs for {candidate_name}; using first")

    return Path(matches[0])


def make_state_holdout_split(
    seeds: np.ndarray,
    train_init_states: List[int],
    test_init_states: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.asarray([int(s) in train_init_states for s in seeds], dtype=bool)
    test_mask = np.asarray([int(s) in test_init_states for s in seeds], dtype=bool)
    return train_mask, test_mask


def pad_or_truncate_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions should be [T, A], got {actions.shape}")

    act_dim = actions.shape[1]
    out = np.zeros((horizon, act_dim), dtype=np.float32)
    n = min(horizon, actions.shape[0])
    out[:n] = actions[:n]
    return out


def load_dataset(
    data_dir: Path,
    csv_name: str,
    horizon: int,
    include_families: List[str],
    exclude_controls: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    csv_path = data_dir / csv_name
    if not csv_path.exists():
        fallback = data_dir / "summary.csv"
        if fallback.exists():
            print(f"[warning] {csv_path} not found; using {fallback}")
            csv_path = fallback
        else:
            raise FileNotFoundError(csv_path)

    rows_all = read_csv(csv_path)

    rows: List[Dict[str, Any]] = []
    actions_list: List[np.ndarray] = []
    actual_eef_list: List[np.ndarray] = []
    bowl2_start_list: List[np.ndarray] = []
    seeds: List[int] = []
    families: List[str] = []

    for r in rows_all:
        fam = str(r.get("candidate_family", ""))

        if include_families and fam not in include_families:
            continue

        if exclude_controls and fam in {"stall", "random"}:
            continue

        init_idx = int(r["init_state_idx"])
        cand_name = str(r["candidate_name"])

        cand_dir = find_candidate_dir(data_dir, init_idx, cand_name)
        traj_path = cand_dir / "traj.npz"
        data = np.load(traj_path, allow_pickle=True)

        required = ["actions", "eef_pos_seq", "bowl2_pos_seq"]
        for k in required:
            if k not in data:
                raise KeyError(f"{traj_path} missing key: {k}")

        actions = pad_or_truncate_actions(np.asarray(data["actions"], dtype=np.float32), horizon)

        eef_seq_raw = np.asarray(data["eef_pos_seq"], dtype=np.float32)
        bowl2_seq_raw = np.asarray(data["bowl2_pos_seq"], dtype=np.float32)

        # Ensure actual_eef has horizon+1 states.
        actual_eef = np.zeros((horizon + 1, 3), dtype=np.float32)
        n_state = min(horizon + 1, eef_seq_raw.shape[0])
        actual_eef[:n_state] = eef_seq_raw[:n_state]
        if n_state < horizon + 1:
            actual_eef[n_state:] = actual_eef[n_state - 1]

        bowl2_start = bowl2_seq_raw[0].astype(np.float32)

        rows.append(dict(r))
        actions_list.append(actions)
        actual_eef_list.append(actual_eef)
        bowl2_start_list.append(bowl2_start)
        seeds.append(init_idx)
        families.append(fam)

    if not rows:
        raise RuntimeError("No rows selected. Check include_families / exclude_controls.")

    arrays = {
        "actions": actions_list,
        "actual_eef": actual_eef_list,
        "bowl2_start": bowl2_start_list,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "families": np.asarray(families, dtype=object),
    }
    return rows, arrays


def fit_scalar_scale(
    actions_list: List[np.ndarray],
    actual_eef_list: List[np.ndarray],
    train_mask: np.ndarray,
) -> float:
    num = 0.0
    den = 0.0

    for actions, eef_seq, is_train in zip(actions_list, actual_eef_list, train_mask):
        if not is_train:
            continue

        u = actions[:, :3].sum(axis=0)
        dx = eef_seq[-1] - eef_seq[0]

        num += float(np.dot(u, dx))
        den += float(np.dot(u, u))

    if den < 1e-12:
        return 1.0
    return num / den


def fit_xyz_scale(
    actions_list: List[np.ndarray],
    actual_eef_list: List[np.ndarray],
    train_mask: np.ndarray,
) -> np.ndarray:
    num = np.zeros(3, dtype=np.float64)
    den = np.zeros(3, dtype=np.float64)

    for actions, eef_seq, is_train in zip(actions_list, actual_eef_list, train_mask):
        if not is_train:
            continue

        u = actions[:, :3].sum(axis=0).astype(np.float64)
        dx = (eef_seq[-1] - eef_seq[0]).astype(np.float64)

        num += u * dx
        den += u * u

    scale = np.ones(3, dtype=np.float32)
    for i in range(3):
        if den[i] >= 1e-12:
            scale[i] = float(num[i] / den[i])

    return scale


def predict_eef_seq_from_actions(
    actions: np.ndarray,
    eef0: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    xyz = actions[:, :3].astype(np.float32)

    if isinstance(scale, float):
        step_disp = xyz * float(scale)
    else:
        step_disp = xyz * scale.reshape(1, 3)

    cum = np.cumsum(step_disp, axis=0)

    pred = np.zeros((actions.shape[0] + 1, 3), dtype=np.float32)
    pred[0] = eef0.astype(np.float32)
    pred[1:] = eef0.reshape(1, 3) + cum
    return pred


def compute_kinematic_features(
    eef_seq: np.ndarray,
    actions: np.ndarray,
    bowl2_start: np.ndarray,
    contact_threshold: float,
    close_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Compute EEF-only features. Use only initial bowl2 position, never future object trajectory.
    """
    T = actions.shape[0]
    eef_seq = eef_seq[: T + 1]
    bowl2 = bowl2_start.reshape(1, 3)

    d = np.linalg.norm(eef_seq - bowl2, axis=1)

    grip_action = actions[:, 6] if actions.shape[1] > 6 else np.zeros(T, dtype=np.float32)
    # Map action gripper to next state; state 0 uses first action command for coarse alignment.
    grip_state = np.zeros(T + 1, dtype=np.float32)
    grip_state[0] = grip_action[0] if T > 0 else 0.0
    grip_state[1:] = grip_action

    close_mask = grip_state >= close_threshold

    min_d = float(np.min(d))
    argmin = int(np.argmin(d))
    final_d = float(d[-1])

    contact_hits = np.where(d <= contact_threshold)[0]
    contact = float(len(contact_hits) > 0)
    tau_contact = float(int(contact_hits[0])) if len(contact_hits) else float(T + 1)
    tau_norm = tau_contact / float(T + 1)

    if np.any(close_mask):
        d_close = d[close_mask]
        min_d_close = float(np.min(d_close))
        first_close = int(np.where(close_mask)[0][0])
        close_frac = float(np.mean(close_mask))
        z_at_close = float(eef_seq[first_close, 2])
        z_max_after_close = float(np.max(eef_seq[first_close:, 2]))
        z_lift_after_close = z_max_after_close - z_at_close
        d_at_close = float(d[first_close])
    else:
        min_d_close = float(np.max(d) + 1.0)
        first_close = T
        close_frac = 0.0
        z_at_close = float(eef_seq[0, 2])
        z_max_after_close = float(np.max(eef_seq[:, 2]))
        z_lift_after_close = 0.0
        d_at_close = float(d[0])

    diffs = np.diff(eef_seq, axis=0)
    step_norm = np.linalg.norm(diffs, axis=1) if len(diffs) else np.zeros(1, dtype=np.float32)

    path_len = float(np.sum(step_norm))
    max_step = float(np.max(step_norm))
    mean_step = float(np.mean(step_norm))

    eef_z_min = float(np.min(eef_seq[:, 2]))
    eef_z_max = float(np.max(eef_seq[:, 2]))
    eef_z_range = eef_z_max - eef_z_min
    final_z_delta = float(eef_seq[-1, 2] - eef_seq[0, 2])

    final_disp = eef_seq[-1] - eef_seq[0]
    final_disp_norm = float(np.linalg.norm(final_disp))

    # Rule-like scores where higher should generally be better.
    score_min_dist = -min_d
    score_close_min_dist = -min_d_close
    score_contact_tau = contact - tau_norm
    score_close_lift = -min_d_close + 0.5 * z_lift_after_close + 0.1 * close_frac
    score_full_rule = (
        -min_d_close
        + 0.5 * z_lift_after_close
        + 0.1 * close_frac
        + 0.1 * contact
        - 0.05 * tau_norm
    )

    min_dist_feat = np.asarray([score_min_dist], dtype=np.float32)

    close_min_dist_feat = np.asarray(
        [
            score_close_min_dist,
            close_frac,
            -d_at_close,
        ],
        dtype=np.float32,
    )

    eef_rule_feat = np.asarray(
        [
            score_min_dist,
            score_close_min_dist,
            score_contact_tau,
            score_close_lift,
            score_full_rule,
            contact,
            -tau_norm,
            close_frac,
            z_lift_after_close,
            final_z_delta,
        ],
        dtype=np.float32,
    )

    eef_full_feat = np.asarray(
        [
            score_min_dist,
            score_close_min_dist,
            score_contact_tau,
            score_close_lift,
            score_full_rule,
            min_d,
            min_d_close,
            final_d,
            d_at_close,
            contact,
            tau_norm,
            close_frac,
            z_lift_after_close,
            z_at_close,
            z_max_after_close,
            eef_z_min,
            eef_z_max,
            eef_z_range,
            final_z_delta,
            path_len,
            max_step,
            mean_step,
            final_disp[0],
            final_disp[1],
            final_disp[2],
            final_disp_norm,
            float(eef_seq[argmin, 2]),
        ],
        dtype=np.float32,
    )

    return {
        "min_dist": min_dist_feat,
        "close_min_dist": close_min_dist_feat,
        "eef_rule": eef_rule_feat,
        "eef_full": eef_full_feat,
    }


def standardize_train_test(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    return X_train.astype(np.float32), X_test.astype(np.float32)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    reg = alpha * np.eye(Xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0

    A = Xb.T @ Xb + reg
    b = Xb.T @ y

    beta = np.linalg.solve(A, b)
    return beta.astype(np.float32)


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    return (Xb @ beta).astype(np.float32)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def auc_score(y_bin: np.ndarray, score: np.ndarray) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    score = np.asarray(score).astype(float)

    pos = score[y_bin == 1]
    neg = score[y_bin == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    correct = 0.0
    total = 0.0

    for p in pos:
        for n in neg:
            if p > n:
                correct += 1.0
            elif p == n:
                correct += 0.5
            total += 1.0

    return correct / total


def binary_accuracy(y_bin: np.ndarray, pred_score: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin = (pred_score >= threshold).astype(np.float32)
    return float(np.mean(pred_bin == y_bin))


def within_seed_pairwise(
    seeds: np.ndarray,
    y_score: np.ndarray,
    pred_score: np.ndarray,
    y_bin: np.ndarray,
    margin: float,
) -> Dict[str, float]:
    total = 0
    correct = 0.0

    bin_total = 0
    bin_correct = 0.0

    used_seeds = 0

    for seed in sorted(np.unique(seeds).tolist()):
        idx = np.where(seeds == seed)[0]
        if len(idx) < 2:
            continue

        seed_pairs = 0

        for ai in range(len(idx)):
            for bi in range(ai + 1, len(idx)):
                i = idx[ai]
                j = idx[bi]

                dy = float(y_score[i] - y_score[j])
                if abs(dy) > margin:
                    dp = float(pred_score[i] - pred_score[j])
                    if abs(dp) <= 1e-12:
                        score = 0.5
                    else:
                        score = 1.0 if math.copysign(1.0, dy) == math.copysign(1.0, dp) else 0.0

                    correct += score
                    total += 1
                    seed_pairs += 1

                if y_bin[i] != y_bin[j]:
                    desired = float(y_bin[i] - y_bin[j])
                    dp = float(pred_score[i] - pred_score[j])
                    if abs(dp) <= 1e-12:
                        score = 0.5
                    else:
                        score = 1.0 if math.copysign(1.0, desired) == math.copysign(1.0, dp) else 0.0

                    bin_correct += score
                    bin_total += 1

        if seed_pairs > 0:
            used_seeds += 1

    return {
        "within_seed_pairwise_acc": float(correct / total) if total else float("nan"),
        "within_seed_pairwise_pairs": int(total),
        "within_seed_pairwise_used_seeds": int(used_seeds),
        "within_seed_binpair_acc": float(bin_correct / bin_total) if bin_total else float("nan"),
        "within_seed_binpair_pairs": int(bin_total),
    }


def evaluate_feature_group(
    X: np.ndarray,
    y: np.ndarray,
    y_bin: np.ndarray,
    seeds: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alpha: float,
    rank_margin: float,
) -> Dict[str, Any]:
    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

    beta = fit_ridge(X_train, y_train, alpha=ridge_alpha)
    pred_train = predict_ridge(X_train, beta)
    pred_test = predict_ridge(X_test, beta)

    metrics: Dict[str, Any] = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "dim": int(X.shape[1]),
        "r2": r2_score(y_test, pred_test),
        "mae": mae(y_test, pred_test),
        "auc": auc_score(y_bin[test_mask], pred_test),
        "binary_acc": binary_accuracy(y_bin[test_mask], pred_test),
    }

    metrics.update(
        within_seed_pairwise(
            seeds=seeds[test_mask],
            y_score=y[test_mask],
            pred_score=pred_test,
            y_bin=y_bin[test_mask],
            margin=rank_margin,
        )
    )

    return metrics


def format_metric(m: Dict[str, Any]) -> str:
    return (
        f"R2={m['r2']:+.4f} "
        f"MAE={m['mae']:.5f} "
        f"AUC={m['auc']:.4f} "
        f"pair={m['within_seed_pairwise_acc']:.4f} "
        f"binpair={m['within_seed_binpair_acc']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_name", type=str, default="summary_recovered.csv")
    parser.add_argument("--horizon", type=int, default=140)

    parser.add_argument("--train_init_states", type=str, default="0-13")
    parser.add_argument("--test_init_states", type=str, default="14-19")

    parser.add_argument("--contact_threshold", type=float, default=0.10)
    parser.add_argument("--close_threshold", type=float, default=0.50)

    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(TARGETS_DEFAULT),
        help="Comma-separated target columns to evaluate.",
    )

    parser.add_argument(
        "--binary_col",
        type=str,
        default="",
        help="If empty, binary target uses target itself if binary, otherwise bowl2_grasp_success_proxy.",
    )

    parser.add_argument(
        "--include_families",
        type=str,
        default="",
        help="Comma-separated candidate families to include. Empty means all.",
    )
    parser.add_argument("--exclude_controls", action="store_true")

    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--out_prefix", type=str, default="v3_kinematic_baseline")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    targets = parse_str_list(args.targets)

    train_init_states = parse_int_range_or_list(args.train_init_states)
    test_init_states = parse_int_range_or_list(args.test_init_states)

    include_families = parse_str_list(args.include_families) if args.include_families.strip() else []

    print("=" * 120)
    print(f"[data_dir] {data_dir}")
    print(f"[csv_name] {args.csv_name}")
    print(f"[horizon] {args.horizon}")
    print(f"[targets] {targets}")
    print(f"[train_init_states] {train_init_states}")
    print(f"[test_init_states] {test_init_states}")
    print(f"[contact_threshold] {args.contact_threshold}")
    print(f"[close_threshold] {args.close_threshold}")
    print(f"[include_families] {include_families if include_families else 'ALL'}")
    print(f"[exclude_controls] {args.exclude_controls}")
    print("=" * 120)

    rows, arr = load_dataset(
        data_dir=data_dir,
        csv_name=args.csv_name,
        horizon=args.horizon,
        include_families=include_families,
        exclude_controls=args.exclude_controls,
    )

    actions_list: List[np.ndarray] = arr["actions"]
    actual_eef_list: List[np.ndarray] = arr["actual_eef"]
    bowl2_start_list: List[np.ndarray] = arr["bowl2_start"]
    seeds: np.ndarray = arr["seeds"]
    families: np.ndarray = arr["families"]

    train_mask, test_mask = make_state_holdout_split(
        seeds=seeds,
        train_init_states=train_init_states,
        test_init_states=test_init_states,
    )

    if train_mask.sum() == 0:
        raise RuntimeError("No train samples selected.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test samples selected.")

    scalar_scale = fit_scalar_scale(actions_list, actual_eef_list, train_mask)
    xyz_scale = fit_xyz_scale(actions_list, actual_eef_list, train_mask)

    print(f"[num samples] {len(rows)}")
    print(f"[train samples] {int(train_mask.sum())}")
    print(f"[test samples] {int(test_mask.sum())}")
    print(f"[fit_scalar scale] {scalar_scale:.8f}")
    print(f"[fit_xyz scale] {xyz_scale.tolist()}")

    trajectory_modes: Dict[str, Any] = {
        "raw_scale1": 1.0,
        "fit_scalar": float(scalar_scale),
        "fit_xyz": xyz_scale,
        "actual_eef_oracle": "actual",
    }

    # Precompute feature matrices for all trajectory modes and feature groups.
    feature_store: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for mode_name, scale in trajectory_modes.items():
        feature_store[mode_name] = {
            "min_dist": [],
            "close_min_dist": [],
            "eef_rule": [],
            "eef_full": [],
        }

        for actions, actual_eef, bowl2_start in zip(actions_list, actual_eef_list, bowl2_start_list):
            if isinstance(scale, str) and scale == "actual":
                eef_seq = actual_eef
            else:
                eef_seq = predict_eef_seq_from_actions(
                    actions=actions,
                    eef0=actual_eef[0],
                    scale=scale,
                )

            feats = compute_kinematic_features(
                eef_seq=eef_seq,
                actions=actions,
                bowl2_start=bowl2_start,
                contact_threshold=args.contact_threshold,
                close_threshold=args.close_threshold,
            )

            for group_name, feat in feats.items():
                feature_store[mode_name][group_name].append(feat)

    feature_mats: Dict[str, Dict[str, np.ndarray]] = {}
    for mode_name, group_dict in feature_store.items():
        feature_mats[mode_name] = {}
        for group_name, feats in group_dict.items():
            feature_mats[mode_name][group_name] = np.stack(feats, axis=0).astype(np.float32)

    all_results: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "args": vars(args),
        "num_samples": int(len(rows)),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "fit_scalar_scale": float(scalar_scale),
        "fit_xyz_scale": xyz_scale.tolist(),
        "targets": {},
    }

    for target in targets:
        print("\n" + "=" * 120)
        print(f"[target] {target}")

        if target not in rows[0]:
            raise KeyError(f"Missing target={target}. Available keys include: {sorted(rows[0].keys())[:80]}")

        y = np.asarray([safe_float(r[target]) for r in rows], dtype=np.float32)

        if args.binary_col:
            binary_col = args.binary_col
        elif target in BINARY_TARGETS:
            binary_col = target
        else:
            binary_col = "bowl2_grasp_success_proxy"

        if binary_col not in rows[0]:
            raise KeyError(f"Missing binary_col={binary_col}")

        y_bin = np.asarray([float(safe_float(r[binary_col]) > 0.5) for r in rows], dtype=np.float32)

        print(f"[binary_col] {binary_col}")
        print(f"[target mean all] {float(np.mean(y)):+.6f}")
        print(f"[binary positive rate all] {float(np.mean(y_bin)):.4f}")

        target_results: Dict[str, Any] = {
            "binary_col": binary_col,
            "target_mean_all": float(np.mean(y)),
            "binary_positive_rate_all": float(np.mean(y_bin)),
            "modes": {},
        }

        for mode_name in ["raw_scale1", "fit_scalar", "fit_xyz", "actual_eef_oracle"]:
            print("\n" + "-" * 120)
            print(f"[mode] {mode_name}")

            mode_results: Dict[str, Any] = {}

            for group_name in ["min_dist", "close_min_dist", "eef_rule", "eef_full"]:
                X = feature_mats[mode_name][group_name]

                m = evaluate_feature_group(
                    X=X,
                    y=y,
                    y_bin=y_bin,
                    seeds=seeds,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    ridge_alpha=args.ridge_alpha,
                    rank_margin=args.rank_margin,
                )

                mode_results[group_name] = m
                print(f"  {group_name:16s} {format_metric(m)}")

            target_results["modes"][mode_name] = mode_results

        all_results["targets"][target] = target_results

    result_path = data_dir / f"{args.out_prefix}_state_holdout.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 120)
    print(f"[saved] {result_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()