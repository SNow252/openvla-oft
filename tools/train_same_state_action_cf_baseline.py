#!/usr/bin/env python3
"""
Train minimal baselines for same-state action counterfactual dataset.

Goal:
  Verify that the new dataset is not reducible to initial-state prediction.

Input data structure:
  DATA_DIR/
    summary.csv
    init_000/
      initial_state.json
      candidate_00_move_to_bowl2/traj.npz
      candidate_00_move_to_bowl2/meta.json
      ...

Targets:
  source_advantage = progress_to_bowl2 - progress_to_bowl1

Models:
  state_only:
    initial object/robot state only
  action_only:
    action chunk features only
  state_action:
    state + action

Main metric:
  within-seed pairwise ranking accuracy
  For each held-out init_state_idx, compare all candidate pairs.
  A model succeeds if it assigns higher score to the candidate with higher
  source_advantage.

This is intentionally dependency-light: numpy + Python stdlib only.
No pandas / sklearn required.
"""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "akita_black_bowl_1_pos",
    "akita_black_bowl_2_pos",
    "plate_1_pos",
    "glazed_rim_porcelain_ramekin_1_pos",
]

OBJECT_POS_KEYS = {
    "bowl1": "akita_black_bowl_1_pos",
    "bowl2": "akita_black_bowl_2_pos",
    "plate": "plate_1_pos",
    "ramekin": "glazed_rim_porcelain_ramekin_1_pos",
}


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_summary_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out: Dict[str, Any] = {}
            for k, v in r.items():
                if k in {"candidate_name"}:
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

            if "source_advantage_bowl2_over_bowl1" in out:
                source_adv = safe_float(out["source_advantage_bowl2_over_bowl1"])
            else:
                source_adv = safe_float(out["progress_to_bowl2"]) - safe_float(out["progress_to_bowl1"])

            out["target_source_advantage"] = source_adv
            out["target_binary"] = float(source_adv > 0.0)
            rows.append(out)

    return rows


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_vec(summary: Dict[str, Any], key: str, dim: int | None = None) -> np.ndarray:
    if key not in summary:
        if dim is None:
            raise KeyError(f"Missing key: {key}")
        return np.zeros(dim, dtype=np.float32)

    arr = np.asarray(summary[key], dtype=np.float32).reshape(-1)
    if dim is not None:
        if arr.size >= dim:
            return arr[:dim].astype(np.float32)
        padded = np.zeros(dim, dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr.astype(np.float32)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_state_features(init_json: Dict[str, Any]) -> np.ndarray:
    summary = init_json["summary"]

    feats: List[np.ndarray] = []

    # Raw state fields.
    feats.append(get_vec(summary, "robot0_eef_pos", 3))
    feats.append(get_vec(summary, "robot0_gripper_qpos", 2))
    feats.append(get_vec(summary, "robot0_joint_pos", 7))

    for key in [
        "akita_black_bowl_1_pos",
        "akita_black_bowl_2_pos",
        "plate_1_pos",
        "glazed_rim_porcelain_ramekin_1_pos",
    ]:
        feats.append(get_vec(summary, key, 3))

    eef = get_vec(summary, "robot0_eef_pos", 3)
    bowl1 = get_vec(summary, "akita_black_bowl_1_pos", 3)
    bowl2 = get_vec(summary, "akita_black_bowl_2_pos", 3)
    plate = get_vec(summary, "plate_1_pos", 3)
    ramekin = get_vec(summary, "glazed_rim_porcelain_ramekin_1_pos", 3)

    objs = [bowl1, bowl2, plate, ramekin]

    # EEF-to-object vectors and distances.
    for obj in objs:
        feats.append(obj - eef)
    feats.append(np.array([l2(eef, obj) for obj in objs], dtype=np.float32))

    # Object-to-object relation vectors.
    relation_pairs = [
        (bowl2, bowl1),
        (bowl2, plate),
        (bowl2, ramekin),
        (bowl1, plate),
        (bowl1, ramekin),
        (plate, ramekin),
    ]
    for a, b in relation_pairs:
        feats.append(a - b)
    feats.append(np.array([l2(a, b) for a, b in relation_pairs], dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)


def find_candidate_dir(data_dir: Path, init_state_idx: int, candidate_name: str) -> Path:
    seed_dir = data_dir / f"init_{init_state_idx:03d}"
    pattern = str(seed_dir / f"candidate_*_{candidate_name}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No candidate dir for init={init_state_idx}, candidate={candidate_name}, pattern={pattern}")
    if len(matches) > 1:
        # Should not happen, but keep deterministic.
        print(f"[warning] multiple candidate dirs for {candidate_name}: {matches}; using first")
    return Path(matches[0])


def pad_or_truncate_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape (T,7), got {actions.shape}")

    out = np.zeros((horizon, 7), dtype=np.float32)
    t = min(horizon, actions.shape[0])
    out[:t] = actions[:t]
    return out


def build_action_features(traj_npz_path: Path, horizon: int) -> np.ndarray:
    data = np.load(traj_npz_path)
    actions = pad_or_truncate_actions(data["actions"], horizon)

    xyz = actions[:, :3]
    grip = actions[:, 6:7]

    flat = actions.reshape(-1)

    summary_feats = [
        actions.mean(axis=0),
        actions.std(axis=0),
        actions.min(axis=0),
        actions.max(axis=0),
        actions[0],
        actions[-1],
        actions.sum(axis=0),
        np.array(
            [
                float(np.linalg.norm(xyz, axis=1).mean()),
                float(np.linalg.norm(xyz, axis=1).std()),
                float(np.linalg.norm(xyz.sum(axis=0))),
                float(grip.mean()),
                float(grip.std()),
            ],
            dtype=np.float32,
        ),
    ]

    return np.concatenate([flat] + summary_feats).astype(np.float32)


def build_dataset(data_dir: Path, horizon: int) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows = read_summary_csv(data_dir / "summary.csv")

    X_state: List[np.ndarray] = []
    X_action: List[np.ndarray] = []
    y: List[float] = []
    y_bin: List[float] = []
    seeds: List[int] = []
    candidate_names: List[str] = []

    for r in rows:
        init_idx = int(r["init_state_idx"])
        cand = str(r["candidate_name"])

        init_json = load_json(data_dir / f"init_{init_idx:03d}" / "initial_state.json")
        cand_dir = find_candidate_dir(data_dir, init_idx, cand)
        traj_path = cand_dir / "traj.npz"

        X_state.append(build_state_features(init_json))
        X_action.append(build_action_features(traj_path, horizon))
        y.append(float(r["target_source_advantage"]))
        y_bin.append(float(r["target_binary"]))
        seeds.append(init_idx)
        candidate_names.append(cand)

    arrays = {
        "X_state": np.stack(X_state, axis=0).astype(np.float32),
        "X_action": np.stack(X_action, axis=0).astype(np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "y_bin": np.asarray(y_bin, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "candidate_names": np.asarray(candidate_names, dtype=object),
    }
    return rows, arrays


def group_train_test_split(seeds: np.ndarray, test_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(seeds)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(unique) * test_frac)))
    n_test = min(n_test, len(unique) - 1)

    test_seeds = set(shuffled[:n_test].tolist())
    train_mask = np.array([s not in test_seeds for s in seeds], dtype=bool)
    test_mask = np.array([s in test_seeds for s in seeds], dtype=bool)
    return train_mask, test_mask


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean.squeeze(0), std.squeeze(0)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression with bias term unregularized."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    d = Xb.shape[1]
    reg = alpha * np.eye(d, dtype=np.float32)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return beta.astype(np.float32)


def predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    return (Xb @ beta).astype(np.float32)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_acc_from_score(y_bin: np.ndarray, score: np.ndarray) -> float:
    pred = (score > 0.0).astype(np.float32)
    return float(np.mean(pred == y_bin))


def auc_score(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC, dependency-free."""
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


def within_seed_pairwise_ranking(
    seeds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margin: float,
) -> Dict[str, float]:
    total = 0
    correct = 0.0
    used_seeds = 0

    posneg_total = 0
    posneg_correct = 0.0

    for seed in sorted(np.unique(seeds).tolist()):
        idx = np.where(seeds == seed)[0]
        if len(idx) < 2:
            continue

        seed_pairs = 0
        for a_i in range(len(idx)):
            for b_i in range(a_i + 1, len(idx)):
                i = idx[a_i]
                j = idx[b_i]

                dy = float(y_true[i] - y_true[j])
                if abs(dy) <= margin:
                    continue

                dp = float(y_pred[i] - y_pred[j])

                if dp == 0:
                    score = 0.5
                else:
                    score = 1.0 if math.copysign(1.0, dp) == math.copysign(1.0, dy) else 0.0

                correct += score
                total += 1
                seed_pairs += 1

                # stricter pair type: one positive source_adv and one non-positive
                if (y_true[i] > 0) != (y_true[j] > 0):
                    posneg_correct += score
                    posneg_total += 1

        if seed_pairs > 0:
            used_seeds += 1

    return {
        "within_seed_pairwise_acc": float(correct / total) if total else float("nan"),
        "within_seed_pairwise_pairs": int(total),
        "within_seed_pairwise_used_seeds": int(used_seeds),
        "within_seed_posneg_acc": float(posneg_correct / posneg_total) if posneg_total else float("nan"),
        "within_seed_posneg_pairs": int(posneg_total),
    }


def top1_metrics(
    seeds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    candidate_names: np.ndarray,
) -> Dict[str, float]:
    exact = 0
    positive = 0
    total = 0

    for seed in sorted(np.unique(seeds).tolist()):
        idx = np.where(seeds == seed)[0]
        if len(idx) == 0:
            continue

        pred_top_local = idx[int(np.argmax(y_pred[idx]))]
        true_top_local = idx[int(np.argmax(y_true[idx]))]

        if candidate_names[pred_top_local] == candidate_names[true_top_local]:
            exact += 1
        if y_true[pred_top_local] > 0:
            positive += 1
        total += 1

    return {
        "top1_exact_candidate_acc": float(exact / total) if total else float("nan"),
        "top1_true_positive_rate": float(positive / total) if total else float("nan"),
        "top1_num_seeds": int(total),
    }


def candidate_mean_report(
    candidate_names: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cand in sorted(set(candidate_names.tolist())):
        idx = np.where(candidate_names == cand)[0]
        out[cand] = {
            "n": int(len(idx)),
            "true_mean": float(np.mean(y_true[idx])),
            "pred_mean": float(np.mean(y_pred[idx])),
            "true_pos_rate": float(np.mean(y_true[idx] > 0)),
            "pred_pos_rate": float(np.mean(y_pred[idx] > 0)),
        }
    return out


def evaluate_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    y_bin: np.ndarray,
    seeds: np.ndarray,
    candidate_names: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha: float,
    rank_margin: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    X_train, X_test, _, _ = standardize_train_test(X_train_raw, X_test_raw)
    beta = fit_ridge(X_train, y_train, alpha=alpha)

    pred_train = predict_ridge(X_train, beta)
    pred_test = predict_ridge(X_test, beta)

    test_seeds = seeds[test_mask]
    test_candidates = candidate_names[test_mask]

    metrics: Dict[str, Any] = {
        "model": name,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "dim": int(X.shape[1]),
        "train_r2": r2_score(y_train, pred_train),
        "test_r2": r2_score(y_test, pred_test),
        "train_mae": mae(y_train, pred_train),
        "test_mae": mae(y_test, pred_test),
        "test_binary_acc_score_gt_0": binary_acc_from_score(y_bin[test_mask], pred_test),
        "test_auc": auc_score(y_bin[test_mask], pred_test),
    }

    metrics.update(within_seed_pairwise_ranking(test_seeds, y_test, pred_test, margin=rank_margin))
    metrics.update(top1_metrics(test_seeds, y_test, pred_test, test_candidates))
    metrics["candidate_report_test"] = candidate_mean_report(test_candidates, y_test, pred_test)

    full_pred = np.zeros_like(y, dtype=np.float32)
    # Only test predictions are used for saved comparison; train pred optional.
    full_pred[train_mask] = pred_train
    full_pred[test_mask] = pred_test

    return metrics, full_pred


def save_predictions_csv(
    path: Path,
    seeds: np.ndarray,
    candidate_names: np.ndarray,
    y: np.ndarray,
    y_bin: np.ndarray,
    split: np.ndarray,
    pred_by_model: Dict[str, np.ndarray],
) -> None:
    fields = ["init_state_idx", "candidate_name", "target_source_advantage", "target_binary", "split"]
    for model_name in pred_by_model:
        fields.append(f"pred_{model_name}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(len(y)):
            row = {
                "init_state_idx": int(seeds[i]),
                "candidate_name": str(candidate_names[i]),
                "target_source_advantage": float(y[i]),
                "target_binary": int(y_bin[i]),
                "split": str(split[i]),
            }
            for model_name, pred in pred_by_model.items():
                row[f"pred_{model_name}"] = float(pred[i])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.30)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--out_prefix", type=str, default="same_state_cf_baseline")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print(f"[data_dir] {data_dir}")
    print(f"[horizon] {args.horizon}")
    print(f"[seed] {args.seed}")
    print("=" * 80)

    rows, arr = build_dataset(data_dir, horizon=args.horizon)

    X_state = arr["X_state"]
    X_action = arr["X_action"]
    X_state_action = np.concatenate([X_state, X_action], axis=1)

    y = arr["y"]
    y_bin = arr["y_bin"]
    seeds = arr["seeds"]
    candidate_names = arr["candidate_names"]

    train_mask, test_mask = group_train_test_split(seeds, args.test_frac, rng)
    split = np.array(["train"] * len(y), dtype=object)
    split[test_mask] = "test"

    print(f"[num samples] {len(y)}")
    print(f"[unique seeds] {len(np.unique(seeds))}")
    print(f"[train samples] {int(train_mask.sum())}")
    print(f"[test samples] {int(test_mask.sum())}")
    print(f"[state dim] {X_state.shape[1]}")
    print(f"[action dim] {X_action.shape[1]}")
    print(f"[target mean] {float(np.mean(y)):+.6f}")
    print(f"[target positive rate] {float(np.mean(y > 0)):.4f}")

    print("\n[target mean by candidate]")
    for cand in sorted(set(candidate_names.tolist())):
        idx = np.where(candidate_names == cand)[0]
        print(
            f"  {cand:16s} "
            f"n={len(idx):3d} "
            f"mean={float(np.mean(y[idx])):+.6f} "
            f"pos_rate={float(np.mean(y[idx] > 0)):.3f}"
        )

    model_inputs = {
        "state_only": X_state,
        "action_only": X_action,
        "state_action": X_state_action,
    }

    all_metrics: List[Dict[str, Any]] = []
    pred_by_model: Dict[str, np.ndarray] = {}

    for model_name, X in model_inputs.items():
        print("\n" + "-" * 80)
        print(f"[model] {model_name}")
        metrics, pred = evaluate_model(
            name=model_name,
            X=X,
            y=y,
            y_bin=y_bin,
            seeds=seeds,
            candidate_names=candidate_names,
            train_mask=train_mask,
            test_mask=test_mask,
            alpha=args.ridge_alpha,
            rank_margin=args.rank_margin,
        )
        all_metrics.append(metrics)
        pred_by_model[model_name] = pred

        print(f"  dim: {metrics['dim']}")
        print(f"  test_r2: {metrics['test_r2']:+.4f}")
        print(f"  test_mae: {metrics['test_mae']:.6f}")
        print(f"  test_auc: {metrics['test_auc']:.4f}")
        print(f"  binary_acc(score>0): {metrics['test_binary_acc_score_gt_0']:.4f}")
        print(f"  within_seed_pairwise_acc: {metrics['within_seed_pairwise_acc']:.4f} "
              f"({metrics['within_seed_pairwise_pairs']} pairs)")
        print(f"  within_seed_posneg_acc: {metrics['within_seed_posneg_acc']:.4f} "
              f"({metrics['within_seed_posneg_pairs']} pairs)")
        print(f"  top1_exact_candidate_acc: {metrics['top1_exact_candidate_acc']:.4f}")
        print(f"  top1_true_positive_rate: {metrics['top1_true_positive_rate']:.4f}")

        print("  candidate report on test:")
        for cand, rep in metrics["candidate_report_test"].items():
            print(
                f"    {cand:16s} "
                f"n={rep['n']:2d} "
                f"true_mean={rep['true_mean']:+.5f} "
                f"pred_mean={rep['pred_mean']:+.5f} "
                f"true_pos={rep['true_pos_rate']:.2f} "
                f"pred_pos={rep['pred_pos_rate']:.2f}"
            )

    result_json = data_dir / f"{args.out_prefix}_seed{args.seed}.json"
    pred_csv = data_dir / f"{args.out_prefix}_predictions_seed{args.seed}.csv"

    with open(result_json, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "args": vars(args),
                "num_samples": int(len(y)),
                "num_seeds": int(len(np.unique(seeds))),
                "train_seeds": sorted(set(seeds[train_mask].tolist())),
                "test_seeds": sorted(set(seeds[test_mask].tolist())),
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )

    save_predictions_csv(
        pred_csv,
        seeds=seeds,
        candidate_names=candidate_names,
        y=y,
        y_bin=y_bin,
        split=split,
        pred_by_model=pred_by_model,
    )

    print("\n" + "=" * 80)
    print(f"[saved metrics] {result_json}")
    print(f"[saved predictions] {pred_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()