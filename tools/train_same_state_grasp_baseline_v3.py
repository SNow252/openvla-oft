#!/usr/bin/env python3
"""
Train learned baselines for v3 same-state grasp/contact counterfactual data.

Dataset:
  same_state_grasp_cf_v3p2_settle_20seeds_*
  summary_recovered.csv or summary.csv
  init_XXX/initial_state.json
  init_XXX/candidate_YYY_*/traj.npz

Models:
  state_only
  action_only
  state_action

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

Main metrics:
  R2
  MAE
  AUC with binary_col
  within-seed pairwise ranking
  within-seed binary-pair ranking
"""

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


OBJECT_KEYS = {
    "eef": "robot0_eef_pos",
    "gripper": "robot0_gripper_qpos",
    "joint": "robot0_joint_pos",
    "bowl1": "akita_black_bowl_1_pos",
    "bowl2": "akita_black_bowl_2_pos",
    "plate": "plate_1_pos",
    "ramekin": "glazed_rim_porcelain_ramekin_1_pos",
}


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_range_or_list(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return parse_int_list(s)


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
                if k in {"candidate_name", "candidate_family", "source", "source_object", "distractor_object", "meta_path"}:
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


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_vec(summary: Dict[str, Any], key: str, dim: int) -> np.ndarray:
    if key not in summary:
        return np.zeros(dim, dtype=np.float32)

    arr = np.asarray(summary[key], dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    out[: min(dim, arr.size)] = arr[:dim]
    return out


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_state_features(init_json: Dict[str, Any]) -> np.ndarray:
    """
    Use settled initial state saved under initial_state.json["summary"].
    """
    summary = init_json.get("summary", init_json)

    eef = get_vec(summary, OBJECT_KEYS["eef"], 3)
    gripper = get_vec(summary, OBJECT_KEYS["gripper"], 2)
    joint = get_vec(summary, OBJECT_KEYS["joint"], 7)

    bowl1 = get_vec(summary, OBJECT_KEYS["bowl1"], 3)
    bowl2 = get_vec(summary, OBJECT_KEYS["bowl2"], 3)
    plate = get_vec(summary, OBJECT_KEYS["plate"], 3)
    ramekin = get_vec(summary, OBJECT_KEYS["ramekin"], 3)

    objs = [bowl1, bowl2, plate, ramekin]

    feats: List[float] = []

    # Raw robot state.
    feats.extend(eef.tolist())
    feats.extend(gripper.tolist())
    feats.extend(joint.tolist())

    # Raw object positions.
    for obj in objs:
        feats.extend(obj.tolist())

    # EEF-object relative vectors and distances.
    for obj in objs:
        rel = obj - eef
        feats.extend(rel.tolist())
        feats.append(l2(obj, eef))

    # Object-object relative structure.
    pairs = [
        (bowl2, bowl1),
        (bowl2, plate),
        (bowl2, ramekin),
        (bowl1, plate),
        (bowl1, ramekin),
        (plate, ramekin),
    ]
    for a, b in pairs:
        rel = a - b
        feats.extend(rel.tolist())
        feats.append(l2(a, b))

    # Simple height features.
    feats.extend([
        float(eef[2]),
        float(bowl1[2]),
        float(bowl2[2]),
        float(plate[2]),
        float(ramekin[2]),
    ])

    return np.asarray(feats, dtype=np.float32)


def find_candidate_dir(data_dir: Path, init_state_idx: int, candidate_name: str) -> Path:
    seed_dir = data_dir / f"init_{init_state_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))

    if not matches:
        # Fallback: recovered csv sometimes keeps full meta_path.
        matches = sorted([str(p.parent) for p in seed_dir.glob("candidate_*/meta.json") if p.parent.name.endswith(candidate_name)])

    if not matches:
        raise FileNotFoundError(
            f"No candidate dir for init={init_state_idx}, candidate={candidate_name}, pattern={pattern}"
        )

    if len(matches) > 1:
        print(f"[warning] multiple candidate dirs for {candidate_name}; using first")

    return Path(matches[0])


def pad_or_truncate_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"actions should be [T, A], got shape={actions.shape}")

    act_dim = actions.shape[1]
    out = np.zeros((horizon, act_dim), dtype=np.float32)

    n = min(horizon, actions.shape[0])
    out[:n] = actions[:n]
    return out


def build_action_features(traj_path: Path, horizon: int) -> np.ndarray:
    data = np.load(traj_path, allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float32)

    padded = pad_or_truncate_actions(actions, horizon)
    xyz = padded[:, :3]
    grip = padded[:, 6] if padded.shape[1] > 6 else np.zeros(horizon, dtype=np.float32)

    feats: List[float] = []

    # Full chunk. This is intentionally simple and strong, matching earlier ridge baselines.
    feats.extend(padded.reshape(-1).tolist())

    # Action statistics.
    for arr in [padded, xyz]:
        feats.extend(np.mean(arr, axis=0).tolist())
        feats.extend(np.std(arr, axis=0).tolist())
        feats.extend(np.min(arr, axis=0).tolist())
        feats.extend(np.max(arr, axis=0).tolist())
        feats.extend(np.sum(arr, axis=0).tolist())

    # Integrated EEF-like displacement proxy.
    cum_xyz = np.cumsum(xyz, axis=0)
    final_disp = cum_xyz[-1]
    feats.extend(final_disp.tolist())
    feats.append(float(np.linalg.norm(final_disp)))

    # Trajectory shape summaries.
    path_len = float(np.sum(np.linalg.norm(xyz, axis=1)))
    max_step = float(np.max(np.linalg.norm(xyz, axis=1)))
    mean_step = float(np.mean(np.linalg.norm(xyz, axis=1)))
    feats.extend([path_len, max_step, mean_step])

    # Gripper summaries.
    feats.extend([
        float(np.mean(grip)),
        float(np.std(grip)),
        float(np.min(grip)),
        float(np.max(grip)),
        float(np.sum(grip)),
        float(grip[0]),
        float(grip[-1]),
    ])

    # Phase-aware coarse summaries by thirds.
    thirds = np.array_split(np.arange(horizon), 3)
    for idx in thirds:
        if len(idx) == 0:
            continue
        feats.extend(np.mean(padded[idx], axis=0).tolist())
        feats.extend(np.sum(padded[idx], axis=0).tolist())

    return np.asarray(feats, dtype=np.float32)


def transform_target(raw: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return raw.astype(np.float32)
    if transform == "neg":
        return (-raw).astype(np.float32)
    raise ValueError(f"Unknown target_transform: {transform}")


def invert_prediction(pred_score: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return pred_score.astype(np.float32)
    if transform == "neg":
        return (-pred_score).astype(np.float32)
    raise ValueError(f"Unknown target_transform: {transform}")


def standardize_train_test(X_train_raw: np.ndarray, X_test_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    return X_train.astype(np.float32), X_test.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fit ridge with intercept by appending a bias column.
    """
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


def binary_accuracy(y_bin: np.ndarray, pred_score: np.ndarray, threshold: float) -> float:
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


def build_dataset(
    data_dir: Path,
    csv_name: str,
    target_col: str,
    target_transform: str,
    binary_col: str,
    horizon: int,
    include_families: List[str],
    exclude_controls: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
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
    X_state = []
    X_action = []
    y_raw = []
    y_bin = []
    seeds = []
    families = []
    candidate_names = []

    for r in rows_all:
        fam = str(r.get("candidate_family", ""))

        if include_families and fam not in include_families:
            continue

        if exclude_controls and fam in {"stall", "random"}:
            continue

        if target_col not in r:
            raise KeyError(f"Missing target_col={target_col}. Available keys include: {sorted(r.keys())[:80]}")

        if binary_col and binary_col not in r:
            raise KeyError(f"Missing binary_col={binary_col}. Available keys include: {sorted(r.keys())[:80]}")

        init_idx = int(r["init_state_idx"])
        cand_name = str(r["candidate_name"])

        init_json = load_json(data_dir / f"init_{init_idx:03d}" / "initial_state.json")
        cand_dir = find_candidate_dir(data_dir, init_idx, cand_name)
        traj_path = cand_dir / "traj.npz"

        X_state.append(build_state_features(init_json))
        X_action.append(build_action_features(traj_path, horizon=horizon))

        raw = safe_float(r[target_col])
        y_raw.append(raw)

        if binary_col:
            b = safe_float(r[binary_col])
            y_bin.append(float(b > 0.5))
        else:
            y_bin.append(float(raw > 0.0))

        seeds.append(init_idx)
        families.append(fam)
        candidate_names.append(cand_name)

        rr = dict(r)
        rr["target_raw"] = raw
        rows.append(rr)

    if not rows:
        raise RuntimeError("No rows selected. Check include_families / exclude_controls / csv_name.")

    y_raw_arr = np.asarray(y_raw, dtype=np.float32)
    y_score = transform_target(y_raw_arr, target_transform)

    arrays = {
        "X_state": np.stack(X_state, axis=0).astype(np.float32),
        "X_action": np.stack(X_action, axis=0).astype(np.float32),
        "y_raw": y_raw_arr,
        "y_score": y_score.astype(np.float32),
        "y_bin": np.asarray(y_bin, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "families": np.asarray(families, dtype=object),
        "candidate_names": np.asarray(candidate_names, dtype=object),
    }

    return rows, arrays


def make_state_holdout_split(
    seeds: np.ndarray,
    train_init_states: List[int],
    test_init_states: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.asarray([int(s) in train_init_states for s in seeds], dtype=bool)
    test_mask = np.asarray([int(s) in test_init_states for s in seeds], dtype=bool)
    return train_mask, test_mask


def evaluate_model(
    model_name: str,
    X: np.ndarray,
    y_score: np.ndarray,
    y_raw: np.ndarray,
    y_bin: np.ndarray,
    seeds: np.ndarray,
    families: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alpha: float,
    rank_margin: float,
    target_transform: str,
    binary_threshold: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]

    y_train = y_score[train_mask]
    y_test = y_score[test_mask]

    X_train, X_test, _, _ = standardize_train_test(X_train_raw, X_test_raw)

    beta = fit_ridge(X_train, y_train, alpha=ridge_alpha)

    pred_train = predict_ridge(X_train, beta)
    pred_test = predict_ridge(X_test, beta)

    full_pred = np.zeros_like(y_score, dtype=np.float32)
    full_pred[train_mask] = pred_train
    full_pred[test_mask] = pred_test

    pred_raw_test = invert_prediction(pred_test, target_transform)
    y_raw_test = y_raw[test_mask]

    metrics: Dict[str, Any] = {
        "model": model_name,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "dim": int(X.shape[1]),
        "test_r2_score": r2_score(y_test, pred_test),
        "test_mae_score": mae(y_test, pred_test),
        "test_mae_raw": mae(y_raw_test, pred_raw_test),
        "test_auc_binary_col": auc_score(y_bin[test_mask], pred_test),
        "test_binary_acc": binary_accuracy(y_bin[test_mask], pred_test, threshold=binary_threshold),
    }

    metrics.update(
        within_seed_pairwise(
            seeds=seeds[test_mask],
            y_score=y_score[test_mask],
            pred_score=pred_test,
            y_bin=y_bin[test_mask],
            margin=rank_margin,
        )
    )

    # By family on test.
    by_family = {}
    for fam in sorted(set(str(x) for x in families[test_mask].tolist())):
        idx = np.where(test_mask & (families == fam))[0]
        if len(idx) == 0:
            continue

        by_family[fam] = {
            "n": int(len(idx)),
            "true_raw_mean": float(np.mean(y_raw[idx])),
            "pred_raw_mean": float(np.mean(invert_prediction(full_pred[idx], target_transform))),
            "true_score_mean": float(np.mean(y_score[idx])),
            "pred_score_mean": float(np.mean(full_pred[idx])),
            "bin_rate": float(np.mean(y_bin[idx])),
        }

    metrics["by_family_test"] = by_family

    return metrics, full_pred


def print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"  dim: {metrics['dim']}")
    print(f"  test_r2_score: {metrics['test_r2_score']:+.4f}")
    print(f"  test_mae_score: {metrics['test_mae_score']:.6f}")
    print(f"  test_mae_raw: {metrics['test_mae_raw']:.6f}")
    print(f"  test_auc_binary_col: {metrics['test_auc_binary_col']:.4f}")
    print(f"  test_binary_acc: {metrics['test_binary_acc']:.4f}")
    print(
        f"  within_seed_pairwise_acc: {metrics['within_seed_pairwise_acc']:.4f} "
        f"({metrics['within_seed_pairwise_pairs']} pairs)"
    )
    print(
        f"  within_seed_binpair_acc: {metrics['within_seed_binpair_acc']:.4f} "
        f"({metrics['within_seed_binpair_pairs']} pairs)"
    )

    print("  by family on test:")
    print(
        f"    {'family':>16} {'n':>6} "
        f"{'true_raw':>12} {'pred_raw':>12} "
        f"{'bin_rate':>10}"
    )
    for fam, rep in metrics["by_family_test"].items():
        print(
            f"    {fam:>16} {rep['n']:6d} "
            f"{rep['true_raw_mean']:+12.5f} {rep['pred_raw_mean']:+12.5f} "
            f"{rep['bin_rate']:10.3f}"
        )


def save_predictions_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    y_score: np.ndarray,
    y_raw: np.ndarray,
    y_bin: np.ndarray,
    split: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
    target_transform: str,
) -> None:
    fields = [
        "init_state_idx",
        "candidate_name",
        "candidate_family",
        "xy_offset_x",
        "xy_offset_y",
        "z_grasp_offset",
        "lift_height",
        "target_raw",
        "target_score",
        "target_binary",
        "split",
    ]

    for model in pred_by_model:
        fields.append(f"pred_score_{model}")
        fields.append(f"pred_raw_{model}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, r in enumerate(rows):
            row = {
                "init_state_idx": int(r["init_state_idx"]),
                "candidate_name": str(r["candidate_name"]),
                "candidate_family": str(r.get("candidate_family", "")),
                "xy_offset_x": r.get("xy_offset_x", float("nan")),
                "xy_offset_y": r.get("xy_offset_y", float("nan")),
                "z_grasp_offset": r.get("z_grasp_offset", float("nan")),
                "lift_height": r.get("lift_height", float("nan")),
                "target_raw": float(y_raw[i]),
                "target_score": float(y_score[i]),
                "target_binary": int(y_bin[i]),
                "split": str(split[i]),
            }

            for model, pred in pred_by_model.items():
                row[f"pred_score_{model}"] = float(pred[i])
                row[f"pred_raw_{model}"] = float(invert_prediction(np.asarray([pred[i]]), target_transform)[0])

            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_name", type=str, default="summary_recovered.csv")
    parser.add_argument("--horizon", type=int, default=140)

    parser.add_argument("--target_col", type=str, default="bowl2_grasp_success_proxy")
    parser.add_argument("--target_transform", type=str, default="identity", choices=["identity", "neg"])
    parser.add_argument("--binary_col", type=str, default="")
    parser.add_argument("--binary_threshold", type=float, default=0.5)

    parser.add_argument("--train_init_states", type=str, default="0-13")
    parser.add_argument("--test_init_states", type=str, default="14-19")

    parser.add_argument(
        "--include_families",
        type=str,
        default="",
        help="Comma-separated candidate families to include. Empty means include all.",
    )
    parser.add_argument(
        "--exclude_controls",
        action="store_true",
        help="Exclude stall/random controls and train only on grasp_macro.",
    )

    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--out_prefix", type=str, default="grasp_baseline_v3")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_init_states = parse_int_range_or_list(args.train_init_states)
    test_init_states = parse_int_range_or_list(args.test_init_states)

    include_families = []
    if args.include_families.strip():
        include_families = [x.strip() for x in args.include_families.split(",") if x.strip()]

    binary_col = args.binary_col
    if not binary_col:
        if args.target_col in {
            "bowl2_grasp_success_proxy",
            "bowl2_clean_grasp_proxy",
            "bowl2_source_lifted",
            "bowl2_source_moved",
        }:
            binary_col = args.target_col
        else:
            binary_col = "bowl2_grasp_success_proxy"

    print("=" * 120)
    print(f"[data_dir] {data_dir}")
    print(f"[csv_name] {args.csv_name}")
    print(f"[horizon] {args.horizon}")
    print(f"[target_col] {args.target_col}")
    print(f"[target_transform] {args.target_transform}")
    print(f"[binary_col] {binary_col}")
    print(f"[train_init_states] {train_init_states}")
    print(f"[test_init_states] {test_init_states}")
    print(f"[include_families] {include_families if include_families else 'ALL'}")
    print(f"[exclude_controls] {args.exclude_controls}")
    print("=" * 120)

    rows, arr = build_dataset(
        data_dir=data_dir,
        csv_name=args.csv_name,
        target_col=args.target_col,
        target_transform=args.target_transform,
        binary_col=binary_col,
        horizon=args.horizon,
        include_families=include_families,
        exclude_controls=args.exclude_controls,
    )

    X_state = arr["X_state"]
    X_action = arr["X_action"]
    X_state_action = np.concatenate([X_state, X_action], axis=1)

    y_raw = arr["y_raw"]
    y_score = arr["y_score"]
    y_bin = arr["y_bin"]
    seeds = arr["seeds"]
    families = arr["families"]

    train_mask, test_mask = make_state_holdout_split(
        seeds=seeds,
        train_init_states=train_init_states,
        test_init_states=test_init_states,
    )
    unused_mask = ~(train_mask | test_mask)

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples selected.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test samples selected.")

    print(f"[num samples] {len(rows)}")
    print(f"[train samples] {int(train_mask.sum())}")
    print(f"[test samples] {int(test_mask.sum())}")
    print(f"[unused samples] {int(unused_mask.sum())}")
    print(f"[state dim] {X_state.shape[1]}")
    print(f"[action dim] {X_action.shape[1]}")
    print(f"[target raw mean all] {float(np.mean(y_raw)):+.6f}")
    print(f"[target score mean all] {float(np.mean(y_score)):+.6f}")
    print(f"[binary positive rate all] {float(np.mean(y_bin)):.4f}")

    print(f"[train seeds] {sorted(set(int(s) for s in seeds[train_mask].tolist()))}")
    print(f"[test seeds] {sorted(set(int(s) for s in seeds[test_mask].tolist()))}")

    model_inputs = {
        "state_only": X_state,
        "action_only": X_action,
        "state_action": X_state_action,
    }

    all_metrics = []
    pred_by_model: Dict[str, np.ndarray] = {}

    for model_name, X in model_inputs.items():
        print("\n" + "=" * 120)
        print(f"[model] {model_name}")

        metrics, pred = evaluate_model(
            model_name=model_name,
            X=X,
            y_score=y_score,
            y_raw=y_raw,
            y_bin=y_bin,
            seeds=seeds,
            families=families,
            train_mask=train_mask,
            test_mask=test_mask,
            ridge_alpha=args.ridge_alpha,
            rank_margin=args.rank_margin,
            target_transform=args.target_transform,
            binary_threshold=args.binary_threshold,
        )

        all_metrics.append(metrics)
        pred_by_model[model_name] = pred
        print_metrics(metrics)

    split = np.array(["unused"] * len(rows), dtype=object)
    split[train_mask] = "train"
    split[test_mask] = "test"

    safe_target = args.target_col.replace("/", "_").replace(".", "p")
    result_json = data_dir / f"{args.out_prefix}_{safe_target}_state_holdout.json"
    pred_csv = data_dir / f"{args.out_prefix}_{safe_target}_state_holdout_predictions.csv"

    with open(result_json, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "args": vars(args),
                "binary_col_resolved": binary_col,
                "num_samples": int(len(rows)),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "unused_samples": int(unused_mask.sum()),
                "train_seeds": sorted(set(int(s) for s in seeds[train_mask].tolist())),
                "test_seeds": sorted(set(int(s) for s in seeds[test_mask].tolist())),
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    save_predictions_csv(
        path=pred_csv,
        rows=rows,
        y_score=y_score,
        y_raw=y_raw,
        y_bin=y_bin,
        split=split,
        pred_by_model=pred_by_model,
        target_transform=args.target_transform,
    )

    print("\n" + "=" * 120)
    print(f"[saved metrics] {result_json}")
    print(f"[saved predictions] {pred_csv}")
    print("=" * 120)


if __name__ == "__main__":
    main()