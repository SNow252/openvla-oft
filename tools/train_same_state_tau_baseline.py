#!/usr/bin/env python3
"""
Train baselines for same-state action counterfactual hitting-time / reachability labels.

Inputs:
  v2-tau dataset:
    summary.csv
    init_XXX/initial_state.json
    init_XXX/candidate_YYY_*/traj.npz

Models:
  state_only
  action_only
  state_action

Supported targets:
  reached_source_eps_0p18                    # classification-style regression target
  tau_source_eps_0p18                        # use --target_transform neg
  min_dist_to_bowl2                          # use --target_transform neg
  min_dist_advantage_bowl1_minus_bowl2       # higher is better

Default split:
  dual holdout over both init_state and alpha.

Main metrics:
  R2 / MAE on transformed target score
  raw MAE on original target
  AUC using --binary_col
  within-seed pairwise ranking accuracy
  within-seed binary-pair ranking accuracy
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.train_same_state_action_cf_baseline import (  # noqa: E402
    build_state_features,
    build_action_features,
    load_json,
    standardize_train_test,
    fit_ridge,
    predict_ridge,
    r2_score,
    mae,
    auc_score,
)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_range_or_list(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return parse_int_list(s)


def almost_in(x: float, values: List[float], tol: float = 1e-5) -> bool:
    return any(abs(float(x) - float(v)) <= tol for v in values)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_summary_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                if k in {"candidate_name", "candidate_family"}:
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
        raise FileNotFoundError(f"No candidate dir for init={init_state_idx}, candidate={candidate_name}, pattern={pattern}")
    if len(matches) > 1:
        print(f"[warning] multiple candidate dirs for {candidate_name}: {matches}; using first")
    return Path(matches[0])


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


def build_dataset(
    data_dir: Path,
    horizon: int,
    target_col: str,
    target_transform: str,
    binary_col: str,
    interp_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    rows_all = read_summary_csv(data_dir / "summary.csv")

    rows = []
    X_state = []
    X_action = []
    y_raw = []
    y_bin = []
    seeds = []
    alphas = []
    candidate_names = []

    for r in rows_all:
        if interp_only and str(r.get("candidate_family", "")) != "interp_bowl1_bowl2":
            continue

        if target_col not in r:
            raise KeyError(f"Missing target_col={target_col} in summary.csv row keys: {sorted(r.keys())}")
        if binary_col and binary_col not in r:
            raise KeyError(f"Missing binary_col={binary_col} in summary.csv row keys: {sorted(r.keys())}")

        init_idx = int(r["init_state_idx"])
        cand = str(r["candidate_name"])

        init_json = load_json(data_dir / f"init_{init_idx:03d}" / "initial_state.json")
        cand_dir = find_candidate_dir(data_dir, init_idx, cand)
        traj_path = cand_dir / "traj.npz"

        X_state.append(build_state_features(init_json))
        X_action.append(build_action_features(traj_path, horizon))

        raw = safe_float(r[target_col])
        y_raw.append(raw)

        if binary_col:
            b = safe_float(r[binary_col])
            y_bin.append(float(b > 0.5))
        else:
            y_bin.append(float(raw > 0.0))

        seeds.append(init_idx)
        alphas.append(safe_float(r.get("alpha", float("nan"))))
        candidate_names.append(cand)

        rr = dict(r)
        rr["target_raw"] = raw
        rows.append(rr)

    y_raw_arr = np.asarray(y_raw, dtype=np.float32)
    y_score = transform_target(y_raw_arr, target_transform)

    arrays = {
        "X_state": np.stack(X_state, axis=0).astype(np.float32),
        "X_action": np.stack(X_action, axis=0).astype(np.float32),
        "y_raw": y_raw_arr,
        "y_score": y_score.astype(np.float32),
        "y_bin": np.asarray(y_bin, dtype=np.float32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "alphas": np.asarray(alphas, dtype=np.float32),
        "candidate_names": np.asarray(candidate_names, dtype=object),
    }

    return rows, arrays


def make_split(
    seeds: np.ndarray,
    alphas: np.ndarray,
    split_mode: str,
    train_init_states: List[int],
    test_init_states: List[int],
    test_alphas: List[float],
    train_alphas: List[float],
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if split_mode == "dual":
        train_state_mask = np.asarray([int(s) in train_init_states for s in seeds], dtype=bool)
        test_state_mask = np.asarray([int(s) in test_init_states for s in seeds], dtype=bool)

        test_alpha_mask = np.asarray([almost_in(a, test_alphas) for a in alphas], dtype=bool)
        if train_alphas:
            train_alpha_mask = np.asarray([almost_in(a, train_alphas) for a in alphas], dtype=bool)
        else:
            train_alpha_mask = ~test_alpha_mask

        train_mask = train_state_mask & train_alpha_mask
        test_mask = test_state_mask & test_alpha_mask
        return train_mask, test_mask

    if split_mode == "holdout_alpha":
        test_mask = np.asarray([almost_in(a, test_alphas) for a in alphas], dtype=bool)
        if train_alphas:
            train_mask = np.asarray([almost_in(a, train_alphas) for a in alphas], dtype=bool)
        else:
            train_mask = ~test_mask
        return train_mask, test_mask

    if split_mode == "random_state":
        unique = np.unique(seeds)
        shuffled = unique.copy()
        rng.shuffle(shuffled)

        n_test = max(1, int(round(len(unique) * test_frac)))
        n_test = min(n_test, len(unique) - 1)

        test_seed_set = set(shuffled[:n_test].tolist())
        train_mask = np.asarray([int(s) not in test_seed_set for s in seeds], dtype=bool)
        test_mask = np.asarray([int(s) in test_seed_set for s in seeds], dtype=bool)
        return train_mask, test_mask

    raise ValueError(f"Unknown split_mode: {split_mode}")


def within_seed_ranking(
    seeds: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
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
                    dp = float(y_pred[i] - y_pred[j])
                    if abs(dp) <= 1e-12:
                        score = 0.5
                    else:
                        score = 1.0 if math.copysign(1.0, dp) == math.copysign(1.0, dy) else 0.0

                    correct += score
                    total += 1
                    seed_pairs += 1

                if y_bin[i] != y_bin[j]:
                    # Positive class should get higher predicted score.
                    desired = float(y_bin[i] - y_bin[j])
                    dp = float(y_pred[i] - y_pred[j])
                    if abs(dp) <= 1e-12:
                        score = 0.5
                    else:
                        score = 1.0 if math.copysign(1.0, dp) == math.copysign(1.0, desired) else 0.0

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


def binary_accuracy_from_auc_score(y_bin: np.ndarray, pred: np.ndarray, threshold: float = 0.0) -> float:
    pred_bin = (pred > threshold).astype(np.float32)
    return float(np.mean(pred_bin == y_bin))


def evaluate_model(
    model_name: str,
    X: np.ndarray,
    y_score: np.ndarray,
    y_raw: np.ndarray,
    y_bin: np.ndarray,
    seeds: np.ndarray,
    alphas: np.ndarray,
    candidate_names: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    ridge_alpha: float,
    rank_margin: float,
    target_transform: str,
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
    }

    metrics.update(
        within_seed_ranking(
            seeds=seeds[test_mask],
            y_score=y_score[test_mask],
            y_pred=pred_test,
            y_bin=y_bin[test_mask],
            margin=rank_margin,
        )
    )

    # By-alpha report on test.
    by_alpha = {}
    for a in sorted(set(float(x) for x in alphas[test_mask].tolist())):
        idx = np.where(test_mask & (np.abs(alphas - a) <= 1e-5))[0]
        by_alpha[f"{a:.2f}"] = {
            "n": int(len(idx)),
            "true_score_mean": float(np.mean(y_score[idx])) if len(idx) else float("nan"),
            "pred_score_mean": float(np.mean(full_pred[idx])) if len(idx) else float("nan"),
            "true_raw_mean": float(np.mean(y_raw[idx])) if len(idx) else float("nan"),
            "pred_raw_mean": float(np.mean(invert_prediction(full_pred[idx], target_transform))) if len(idx) else float("nan"),
            "true_bin_rate": float(np.mean(y_bin[idx])) if len(idx) else float("nan"),
            "pred_score_pos_rate": float(np.mean(full_pred[idx] > 0)) if len(idx) else float("nan"),
        }

    metrics["by_alpha_test"] = by_alpha

    return metrics, full_pred


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
        "alpha",
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
                "alpha": r.get("alpha", float("nan")),
                "target_raw": float(y_raw[i]),
                "target_score": float(y_score[i]),
                "target_binary": int(y_bin[i]),
                "split": str(split[i]),
            }

            for model, pred in pred_by_model.items():
                row[f"pred_score_{model}"] = float(pred[i])
                row[f"pred_raw_{model}"] = float(invert_prediction(np.asarray([pred[i]]), target_transform)[0])

            writer.writerow(row)


def print_metrics(metrics: Dict[str, Any]) -> None:
    print(f"  dim: {metrics['dim']}")
    print(f"  test_r2_score: {metrics['test_r2_score']:+.4f}")
    print(f"  test_mae_score: {metrics['test_mae_score']:.6f}")
    print(f"  test_mae_raw: {metrics['test_mae_raw']:.6f}")
    print(f"  test_auc_binary_col: {metrics['test_auc_binary_col']:.4f}")
    print(
        f"  within_seed_pairwise_acc: {metrics['within_seed_pairwise_acc']:.4f} "
        f"({metrics['within_seed_pairwise_pairs']} pairs)"
    )
    print(
        f"  within_seed_binpair_acc: {metrics['within_seed_binpair_acc']:.4f} "
        f"({metrics['within_seed_binpair_pairs']} pairs)"
    )

    print("  by alpha on test:")
    print(
        f"    {'alpha':>8} {'n':>5} "
        f"{'true_raw':>12} {'pred_raw':>12} "
        f"{'true_score':>12} {'pred_score':>12} "
        f"{'bin_rate':>10}"
    )

    for a, rep in metrics["by_alpha_test"].items():
        print(
            f"    {a:>8} {rep['n']:5d} "
            f"{rep['true_raw_mean']:+12.5f} {rep['pred_raw_mean']:+12.5f} "
            f"{rep['true_score_mean']:+12.5f} {rep['pred_score_mean']:+12.5f} "
            f"{rep['true_bin_rate']:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=48)

    parser.add_argument("--target_col", type=str, default="reached_source_eps_0p18")
    parser.add_argument("--target_transform", type=str, default="identity", choices=["identity", "neg"])
    parser.add_argument(
        "--binary_col",
        type=str,
        default="",
        help="Binary label column for AUC / bin-pair ranking. If empty, use target_raw > 0.",
    )

    parser.add_argument("--split_mode", type=str, default="dual", choices=["dual", "holdout_alpha", "random_state"])
    parser.add_argument("--train_init_states", type=str, default="0-13")
    parser.add_argument("--test_init_states", type=str, default="14-19")
    parser.add_argument("--test_alphas", type=str, default="0.75,0.95,1.05,1.15")
    parser.add_argument("--train_alphas", type=str, default="")
    parser.add_argument("--test_frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--interp_only", action="store_true", default=True)
    parser.add_argument("--include_non_interp", action="store_true")

    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--out_prefix", type=str, default="same_state_tau_baseline")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)

    interp_only = args.interp_only and not args.include_non_interp

    binary_col = args.binary_col
    if not binary_col:
        if args.target_col.startswith("reached_"):
            binary_col = args.target_col
        elif args.target_col.startswith("tau_source_eps_"):
            eps_key = args.target_col.replace("tau_source_eps_", "")
            binary_col = f"reached_source_eps_{eps_key}"
        elif args.target_col in {"min_dist_to_bowl2", "min_dist_to_bowl2_rebuilt"}:
            binary_col = "reached_source_eps_0p18"
        else:
            binary_col = ""

    train_init_states = parse_int_range_or_list(args.train_init_states)
    test_init_states = parse_int_range_or_list(args.test_init_states)
    test_alphas = parse_float_list(args.test_alphas)
    train_alphas = parse_float_list(args.train_alphas) if args.train_alphas else []

    print("=" * 120)
    print(f"[data_dir] {data_dir}")
    print(f"[horizon] {args.horizon}")
    print(f"[target_col] {args.target_col}")
    print(f"[target_transform] {args.target_transform}")
    print(f"[binary_col] {binary_col if binary_col else 'target_raw>0'}")
    print(f"[split_mode] {args.split_mode}")
    print(f"[train_init_states] {train_init_states}")
    print(f"[test_init_states] {test_init_states}")
    print(f"[test_alphas] {test_alphas}")
    print(f"[train_alphas] {train_alphas if train_alphas else 'ALL_NON_TEST_ALPHAS'}")
    print("=" * 120)

    rows, arr = build_dataset(
        data_dir=data_dir,
        horizon=args.horizon,
        target_col=args.target_col,
        target_transform=args.target_transform,
        binary_col=binary_col,
        interp_only=interp_only,
    )

    X_state = arr["X_state"]
    X_action = arr["X_action"]
    X_state_action = np.concatenate([X_state, X_action], axis=1)

    y_score = arr["y_score"]
    y_raw = arr["y_raw"]
    y_bin = arr["y_bin"]
    seeds = arr["seeds"]
    alphas = arr["alphas"]
    candidate_names = arr["candidate_names"]

    train_mask, test_mask = make_split(
        seeds=seeds,
        alphas=alphas,
        split_mode=args.split_mode,
        train_init_states=train_init_states,
        test_init_states=test_init_states,
        test_alphas=test_alphas,
        train_alphas=train_alphas,
        test_frac=args.test_frac,
        rng=rng,
    )

    unused_mask = ~(train_mask | test_mask)

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples selected.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test samples selected.")

    print(f"[num samples] {len(y_score)}")
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
    print(f"[train alphas] {sorted(set(float(a) for a in alphas[train_mask].tolist()))}")
    print(f"[test alphas] {sorted(set(float(a) for a in alphas[test_mask].tolist()))}")

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
            alphas=alphas,
            candidate_names=candidate_names,
            train_mask=train_mask,
            test_mask=test_mask,
            ridge_alpha=args.ridge_alpha,
            rank_margin=args.rank_margin,
            target_transform=args.target_transform,
        )

        all_metrics.append(metrics)
        pred_by_model[model_name] = pred

        print_metrics(metrics)

    split = np.array(["unused"] * len(y_score), dtype=object)
    split[train_mask] = "train"
    split[test_mask] = "test"

    safe_target = args.target_col.replace(".", "p").replace("/", "_")
    result_json = data_dir / f"{args.out_prefix}_{safe_target}_{args.split_mode}.json"
    pred_csv = data_dir / f"{args.out_prefix}_{safe_target}_{args.split_mode}_predictions.csv"

    with open(result_json, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "args": vars(args),
                "binary_col_resolved": binary_col,
                "interp_only": interp_only,
                "num_samples": int(len(y_score)),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "unused_samples": int(unused_mask.sum()),
                "train_seeds": sorted(set(int(s) for s in seeds[train_mask].tolist())),
                "test_seeds": sorted(set(int(s) for s in seeds[test_mask].tolist())),
                "train_alphas": sorted(set(float(a) for a in alphas[train_mask].tolist())),
                "test_alphas": sorted(set(float(a) for a in alphas[test_mask].tolist())),
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