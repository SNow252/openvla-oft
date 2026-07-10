#!/usr/bin/env python3
"""
Dual holdout baseline for same-state continuous action counterfactuals.

Purpose:
  Test generalization to BOTH unseen initial states and unseen action directions.

Data:
  v1 dense same-state action counterfactual dataset.

Split:
  Use only interpolation candidates:
      candidate_family == "interp_bowl1_bowl2"

  Train:
      init_state_idx in train states
      alpha in train alphas

  Test:
      init_state_idx in test states
      alpha in test alphas

Default:
  train init states: 0-13
  test init states: 14-19

  test alphas:
      0.75, 0.95, 1.05, 1.15

  train alphas:
      all interpolation alphas not in test_alphas

Models:
  state_only
  action_only
  state_action

Main metrics:
  test_r2
  test_auc
  within_seed_pairwise_acc
  within_seed_posneg_acc
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.train_same_state_action_cf_baseline import (  # noqa: E402
    build_dataset,
    candidate_mean_report,
    evaluate_model,
    save_predictions_csv,
)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_range_or_list(s: str) -> List[int]:
    """
    Accept:
      "0,1,2,3"
      "0-13"
    """
    s = s.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return parse_int_list(s)


def almost_in(x: float, values: List[float], tol: float = 1e-5) -> bool:
    return any(abs(float(x) - float(v)) <= tol for v in values)


def summarize_by_alpha(rows: List[Dict[str, Any]], indices: np.ndarray, title: str) -> None:
    print("\n" + title)
    print("-" * 100)

    by_alpha: Dict[float, List[float]] = {}

    for idx in indices.tolist():
        r = rows[idx]
        alpha = float(r["alpha"])
        y = float(r["target_source_advantage"])
        by_alpha.setdefault(alpha, []).append(y)

    for alpha in sorted(by_alpha.keys()):
        vals = by_alpha[alpha]
        print(
            f"alpha={alpha:+.2f} "
            f"n={len(vals):4d} "
            f"mean_target={float(np.mean(vals)):+.5f} "
            f"pos_rate={float(np.mean(np.asarray(vals) > 0)):.3f}"
        )


def summarize_by_seed(rows: List[Dict[str, Any]], indices: np.ndarray, title: str) -> None:
    print("\n" + title)
    print("-" * 100)

    by_seed: Dict[int, List[float]] = {}

    for idx in indices.tolist():
        r = rows[idx]
        seed = int(r["init_state_idx"])
        y = float(r["target_source_advantage"])
        by_seed.setdefault(seed, []).append(y)

    for seed in sorted(by_seed.keys()):
        vals = by_seed[seed]
        print(
            f"init_state_idx={seed:3d} "
            f"n={len(vals):4d} "
            f"mean_target={float(np.mean(vals)):+.5f} "
            f"pos_rate={float(np.mean(np.asarray(vals) > 0)):.3f}"
        )


def summarize_predictions_by_alpha(
    rows: List[Dict[str, Any]],
    indices: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> None:
    print("\n" + title)
    print("-" * 120)

    by_alpha: Dict[float, List[int]] = {}

    for idx in indices.tolist():
        alpha = float(rows[idx]["alpha"])
        by_alpha.setdefault(alpha, []).append(idx)

    print(
        f"{'alpha':>8} {'n':>5} "
        f"{'true_mean':>12} {'pred_mean':>12} "
        f"{'true_pos':>10} {'pred_pos':>10}"
    )

    for alpha in sorted(by_alpha.keys()):
        idxs = by_alpha[alpha]
        y_true = np.asarray([float(rows[i]["target_source_advantage"]) for i in idxs])
        y_pred = pred[idxs]

        print(
            f"{alpha:+8.2f} {len(idxs):5d} "
            f"{float(np.mean(y_true)):+12.5f} "
            f"{float(np.mean(y_pred)):+12.5f} "
            f"{float(np.mean(y_true > 0)):10.3f} "
            f"{float(np.mean(y_pred > 0)):10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=24)

    parser.add_argument(
        "--train_init_states",
        type=str,
        default="0-13",
        help='Training init states, e.g. "0-13" or "0,1,2".',
    )
    parser.add_argument(
        "--test_init_states",
        type=str,
        default="14-19",
        help='Testing init states, e.g. "14-19" or "14,15,16".',
    )

    parser.add_argument(
        "--test_alphas",
        type=str,
        default="0.75,0.95,1.05,1.15",
    )
    parser.add_argument(
        "--train_alphas",
        type=str,
        default="",
        help="Optional. If empty, train on all interpolation alphas not in test_alphas.",
    )

    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--out_prefix", type=str, default="same_state_cf_dual_holdout")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_init_states = parse_int_range_or_list(args.train_init_states)
    test_init_states = parse_int_range_or_list(args.test_init_states)

    test_alphas = parse_float_list(args.test_alphas)
    train_alphas = parse_float_list(args.train_alphas) if args.train_alphas else []

    print("=" * 110)
    print(f"[data_dir] {data_dir}")
    print(f"[horizon] {args.horizon}")
    print(f"[train_init_states] {train_init_states}")
    print(f"[test_init_states] {test_init_states}")
    print(f"[test_alphas] {test_alphas}")
    print(f"[train_alphas] {train_alphas if train_alphas else 'ALL_NON_TEST_INTERP_ALPHAS'}")
    print("=" * 110)

    rows_all, arr_all = build_dataset(data_dir, horizon=args.horizon)

    # Keep interpolation candidates only.
    interp_indices = []
    for i, r in enumerate(rows_all):
        if str(r.get("candidate_family", "")) == "interp_bowl1_bowl2":
            interp_indices.append(i)

    if not interp_indices:
        raise RuntimeError("No interpolation candidates found.")

    interp_indices = np.asarray(interp_indices, dtype=np.int64)
    rows = [rows_all[i] for i in interp_indices.tolist()]

    X_state = arr_all["X_state"][interp_indices]
    X_action = arr_all["X_action"][interp_indices]
    X_state_action = np.concatenate([X_state, X_action], axis=1)

    y = arr_all["y"][interp_indices]
    y_bin = arr_all["y_bin"][interp_indices]
    seeds = arr_all["seeds"][interp_indices]
    candidate_names = arr_all["candidate_names"][interp_indices]

    alphas = np.asarray([float(r["alpha"]) for r in rows], dtype=np.float32)

    train_state_mask = np.asarray([int(s) in train_init_states for s in seeds], dtype=bool)
    test_state_mask = np.asarray([int(s) in test_init_states for s in seeds], dtype=bool)

    test_alpha_mask = np.asarray([almost_in(a, test_alphas) for a in alphas], dtype=bool)

    if train_alphas:
        train_alpha_mask = np.asarray([almost_in(a, train_alphas) for a in alphas], dtype=bool)
    else:
        train_alpha_mask = ~test_alpha_mask

    train_mask = train_state_mask & train_alpha_mask
    test_mask = test_state_mask & test_alpha_mask

    unused_mask = ~(train_mask | test_mask)

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples selected.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test samples selected.")

    train_alpha_set = sorted(set(float(a) for a in alphas[train_mask].tolist()))
    test_alpha_set = sorted(set(float(a) for a in alphas[test_mask].tolist()))
    train_seed_set = sorted(set(int(s) for s in seeds[train_mask].tolist()))
    test_seed_set = sorted(set(int(s) for s in seeds[test_mask].tolist()))

    print(f"[total interp samples] {len(y)}")
    print(f"[train samples] {int(train_mask.sum())}")
    print(f"[test samples] {int(test_mask.sum())}")
    print(f"[unused samples] {int(unused_mask.sum())}")

    print(f"[train seeds observed] {train_seed_set}")
    print(f"[test seeds observed] {test_seed_set}")
    print(f"[train alphas observed] {train_alpha_set}")
    print(f"[test alphas observed] {test_alpha_set}")

    print(f"[state dim] {X_state.shape[1]}")
    print(f"[action dim] {X_action.shape[1]}")
    print(f"[target mean all] {float(np.mean(y)):+.6f}")
    print(f"[target positive rate all] {float(np.mean(y > 0)):.4f}")

    summarize_by_alpha(rows, np.where(train_mask)[0], "[train target by alpha]")
    summarize_by_alpha(rows, np.where(test_mask)[0], "[test target by alpha]")

    summarize_by_seed(rows, np.where(train_mask)[0], "[train target by seed]")
    summarize_by_seed(rows, np.where(test_mask)[0], "[test target by seed]")

    model_inputs = {
        "state_only": X_state,
        "action_only": X_action,
        "state_action": X_state_action,
    }

    all_metrics: List[Dict[str, Any]] = []
    pred_by_model: Dict[str, np.ndarray] = {}

    for model_name, X in model_inputs.items():
        print("\n" + "=" * 110)
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
        print(
            f"  within_seed_pairwise_acc: {metrics['within_seed_pairwise_acc']:.4f} "
            f"({metrics['within_seed_pairwise_pairs']} pairs)"
        )
        print(
            f"  within_seed_posneg_acc: {metrics['within_seed_posneg_acc']:.4f} "
            f"({metrics['within_seed_posneg_pairs']} pairs)"
        )

        summarize_predictions_by_alpha(
            rows,
            np.where(test_mask)[0],
            pred,
            title=f"[test prediction by alpha] {model_name}",
        )

        print("  candidate report on test:")
        rep = candidate_mean_report(candidate_names[test_mask], y[test_mask], pred[test_mask])
        for cand, item in rep.items():
            print(
                f"    {cand:36s} "
                f"n={item['n']:3d} "
                f"true_mean={item['true_mean']:+.5f} "
                f"pred_mean={item['pred_mean']:+.5f} "
                f"true_pos={item['true_pos_rate']:.2f} "
                f"pred_pos={item['pred_pos_rate']:.2f}"
            )

    split = np.array(["unused"] * len(y), dtype=object)
    split[train_mask] = "train"
    split[test_mask] = "test"

    result_json = data_dir / f"{args.out_prefix}.json"
    pred_csv = data_dir / f"{args.out_prefix}_predictions.csv"

    with open(result_json, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "args": vars(args),
                "num_interp_samples": int(len(y)),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "unused_samples": int(unused_mask.sum()),
                "train_seed_set": train_seed_set,
                "test_seed_set": test_seed_set,
                "train_alpha_set": train_alpha_set,
                "test_alpha_set": test_alpha_set,
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

    print("\n" + "=" * 110)
    print(f"[saved metrics] {result_json}")
    print(f"[saved predictions] {pred_csv}")
    print("=" * 110)


if __name__ == "__main__":
    main()