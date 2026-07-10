#!/usr/bin/env python3
"""
Evaluate analytical / calibrated kinematic baselines for same-state action counterfactual data.

Purpose:
  Test whether current action-consequence prediction can be explained by a
  simple kinematic shortcut:
      eef_pred[t] = eef_0 + scale * cumsum(action_xyz)

Baselines:
  1. raw_scale1:
      scale = 1.0
  2. fit_scalar:
      fit one scalar scale on train split using final displacement.
  3. fit_xyz:
      fit one scale per xyz dimension on train split.

Targets:
  source_advantage_bowl2_over_bowl1
  reached_source_eps_0p18
  tau_source_eps_0p18
  min_dist_to_bowl2

Default split:
  dual holdout:
    train init_state: 0-13
    test init_state: 14-19
    test alpha: 0.75,0.95,1.05,1.15
"""

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


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
        print(f"[warning] multiple candidate dirs for {candidate_name}; using first")
    return Path(matches[0])


def load_dataset(data_dir: Path, interp_only: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_all = read_summary_csv(data_dir / "summary.csv")
    samples = []

    for idx, r in enumerate(rows_all):
        if interp_only and str(r.get("candidate_family", "")) != "interp_bowl1_bowl2":
            continue

        init_idx = int(r["init_state_idx"])
        cand_name = str(r["candidate_name"])
        cand_dir = find_candidate_dir(data_dir, init_idx, cand_name)
        traj_path = cand_dir / "traj.npz"

        data = np.load(traj_path)

        required = [
            "actions",
            "eef_pos_seq",
            "bowl1_pos_seq",
            "bowl2_pos_seq",
            "dist_eef_to_bowl1_seq",
            "dist_eef_to_bowl2_seq",
        ]
        for k in required:
            if k not in data:
                raise KeyError(f"{traj_path} missing key: {k}")

        sample = {
            "row_idx": idx,
            "row": r,
            "traj_path": str(traj_path),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "eef_pos_seq": np.asarray(data["eef_pos_seq"], dtype=np.float32),
            "bowl1_pos_seq": np.asarray(data["bowl1_pos_seq"], dtype=np.float32),
            "bowl2_pos_seq": np.asarray(data["bowl2_pos_seq"], dtype=np.float32),
            "dist_eef_to_bowl1_seq": np.asarray(data["dist_eef_to_bowl1_seq"], dtype=np.float32),
            "dist_eef_to_bowl2_seq": np.asarray(data["dist_eef_to_bowl2_seq"], dtype=np.float32),
        }
        samples.append(sample)

    return rows_all, samples


def make_split(
    samples: List[Dict[str, Any]],
    train_init_states: List[int],
    test_init_states: List[int],
    test_alphas: List[float],
    train_alphas: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = []
    test_mask = []

    for s in samples:
        r = s["row"]
        init_idx = int(r["init_state_idx"])
        alpha = float(r["alpha"])

        train_state = init_idx in train_init_states
        test_state = init_idx in test_init_states

        test_alpha = almost_in(alpha, test_alphas)
        if train_alphas:
            train_alpha = almost_in(alpha, train_alphas)
        else:
            train_alpha = not test_alpha

        train_mask.append(train_state and train_alpha)
        test_mask.append(test_state and test_alpha)

    return np.asarray(train_mask, dtype=bool), np.asarray(test_mask, dtype=bool)


def fit_scalar_scale(samples: List[Dict[str, Any]], train_mask: np.ndarray) -> float:
    num = 0.0
    den = 0.0

    for sample, is_train in zip(samples, train_mask):
        if not is_train:
            continue

        actions = sample["actions"]
        eef_seq = sample["eef_pos_seq"]

        u = actions[:, :3].sum(axis=0)
        dx = eef_seq[-1] - eef_seq[0]

        num += float(np.dot(u, dx))
        den += float(np.dot(u, u))

    if den < 1e-12:
        return 1.0
    return num / den


def fit_xyz_scale(samples: List[Dict[str, Any]], train_mask: np.ndarray) -> np.ndarray:
    num = np.zeros(3, dtype=np.float64)
    den = np.zeros(3, dtype=np.float64)

    for sample, is_train in zip(samples, train_mask):
        if not is_train:
            continue

        actions = sample["actions"]
        eef_seq = sample["eef_pos_seq"]

        u = actions[:, :3].sum(axis=0).astype(np.float64)
        dx = (eef_seq[-1] - eef_seq[0]).astype(np.float64)

        num += u * dx
        den += u * u

    scale = np.ones(3, dtype=np.float32)
    for i in range(3):
        if den[i] >= 1e-12:
            scale[i] = float(num[i] / den[i])

    return scale


def predict_eef_seq(sample: Dict[str, Any], scale: np.ndarray | float) -> np.ndarray:
    actions = sample["actions"]
    eef0 = sample["eef_pos_seq"][0].astype(np.float32)

    xyz = actions[:, :3].astype(np.float32)

    if isinstance(scale, float):
        step_disp = xyz * float(scale)
    else:
        step_disp = xyz * scale.reshape(1, 3)

    cum = np.cumsum(step_disp, axis=0)
    pred = np.zeros((actions.shape[0] + 1, 3), dtype=np.float32)
    pred[0] = eef0
    pred[1:] = eef0.reshape(1, 3) + cum
    return pred


def compute_kinematic_outputs(sample: Dict[str, Any], pred_eef_seq: np.ndarray, eps: float) -> Dict[str, float]:
    bowl1_seq = sample["bowl1_pos_seq"]
    bowl2_seq = sample["bowl2_pos_seq"]

    # Objects are mostly static during source approach, but use per-step seq for consistency.
    T = min(len(pred_eef_seq), len(bowl1_seq), len(bowl2_seq))

    pred = pred_eef_seq[:T]
    b1 = bowl1_seq[:T]
    b2 = bowl2_seq[:T]

    d1 = np.linalg.norm(pred - b1, axis=1)
    d2 = np.linalg.norm(pred - b2, axis=1)

    start_d1 = float(d1[0])
    start_d2 = float(d2[0])
    final_d1 = float(d1[-1])
    final_d2 = float(d2[-1])

    progress_bowl1 = start_d1 - final_d1
    progress_bowl2 = start_d2 - final_d2
    source_adv = progress_bowl2 - progress_bowl1

    min_d2 = float(np.min(d2))
    final_d2 = float(d2[-1])

    hits = np.where(d2 <= eps)[0]
    if len(hits) > 0:
        reached = 1.0
        tau = float(int(hits[0]))
    else:
        reached = 0.0
        tau = float(T)

    return {
        "pred_progress_to_bowl1": float(progress_bowl1),
        "pred_progress_to_bowl2": float(progress_bowl2),
        "pred_source_advantage": float(source_adv),
        "pred_min_dist_to_bowl2": float(min_d2),
        "pred_final_dist_to_bowl2": float(final_d2),
        "pred_reached_source": float(reached),
        "pred_tau_source": float(tau),
        "pred_score_reached": float(-min_d2),
        "pred_score_tau": float(-tau),
        "pred_score_min_dist": float(-min_d2),
    }


def true_values(row: Dict[str, Any], eps_key: str) -> Dict[str, float]:
    reached_col = f"reached_source_eps_{eps_key}"
    tau_col = f"tau_source_eps_{eps_key}"

    return {
        "source_advantage": safe_float(row["source_advantage_bowl2_over_bowl1"]),
        "reached": safe_float(row[reached_col]),
        "tau": safe_float(row[tau_col]),
        "min_dist": safe_float(row["min_dist_to_bowl2"]),
        "score_source_advantage": safe_float(row["source_advantage_bowl2_over_bowl1"]),
        "score_reached": safe_float(row[reached_col]),
        "score_tau": -safe_float(row[tau_col]),
        "score_min_dist": -safe_float(row["min_dist_to_bowl2"]),
    }


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

    for seed in sorted(np.unique(seeds).tolist()):
        idx = np.where(seeds == seed)[0]
        if len(idx) < 2:
            continue

        for ai in range(len(idx)):
            for bi in range(ai + 1, len(idx)):
                i = idx[ai]
                j = idx[bi]

                dy = float(y_score[i] - y_score[j])
                if abs(dy) > margin:
                    dp = float(pred_score[i] - pred_score[j])
                    if abs(dp) <= 1e-12:
                        s = 0.5
                    else:
                        s = 1.0 if math.copysign(1.0, dy) == math.copysign(1.0, dp) else 0.0
                    correct += s
                    total += 1

                if y_bin[i] != y_bin[j]:
                    desired = float(y_bin[i] - y_bin[j])
                    dp = float(pred_score[i] - pred_score[j])
                    if abs(dp) <= 1e-12:
                        s = 0.5
                    else:
                        s = 1.0 if math.copysign(1.0, desired) == math.copysign(1.0, dp) else 0.0
                    bin_correct += s
                    bin_total += 1

    return {
        "within_seed_pairwise_acc": float(correct / total) if total else float("nan"),
        "within_seed_pairwise_pairs": int(total),
        "within_seed_binpair_acc": float(bin_correct / bin_total) if bin_total else float("nan"),
        "within_seed_binpair_pairs": int(bin_total),
    }


def evaluate_target(
    samples: List[Dict[str, Any]],
    test_mask: np.ndarray,
    kin_outputs: List[Dict[str, float]],
    eps_key: str,
    target_name: str,
    rank_margin: float,
) -> Dict[str, float]:
    y_raw = []
    pred_raw = []
    y_score = []
    pred_score = []
    y_bin = []
    seeds = []

    for sample, is_test, pred in zip(samples, test_mask, kin_outputs):
        if not is_test:
            continue

        row = sample["row"]
        tv = true_values(row, eps_key)

        if target_name == "source_advantage":
            y_raw.append(tv["source_advantage"])
            pred_raw.append(pred["pred_source_advantage"])
            y_score.append(tv["score_source_advantage"])
            pred_score.append(pred["pred_source_advantage"])
            y_bin.append(float(tv["source_advantage"] > 0.0))

        elif target_name == "reached":
            y_raw.append(tv["reached"])
            pred_raw.append(pred["pred_reached_source"])
            y_score.append(tv["score_reached"])
            pred_score.append(pred["pred_score_reached"])
            y_bin.append(tv["reached"])

        elif target_name == "tau":
            y_raw.append(tv["tau"])
            pred_raw.append(pred["pred_tau_source"])
            y_score.append(tv["score_tau"])
            pred_score.append(pred["pred_score_tau"])
            y_bin.append(tv["reached"])

        elif target_name == "min_dist":
            y_raw.append(tv["min_dist"])
            pred_raw.append(pred["pred_min_dist_to_bowl2"])
            y_score.append(tv["score_min_dist"])
            pred_score.append(pred["pred_score_min_dist"])
            y_bin.append(tv["reached"])

        else:
            raise ValueError(f"Unknown target_name: {target_name}")

        seeds.append(int(row["init_state_idx"]))

    y_raw = np.asarray(y_raw, dtype=np.float32)
    pred_raw = np.asarray(pred_raw, dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    pred_score = np.asarray(pred_score, dtype=np.float32)
    y_bin = np.asarray(y_bin, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.int64)

    metrics = {
        "target": target_name,
        "n_test": int(len(y_raw)),
        "r2_raw": r2_score(y_raw, pred_raw),
        "mae_raw": mae(y_raw, pred_raw),
        "auc_binary": auc_score(y_bin, pred_score),
    }
    metrics.update(within_seed_pairwise(seeds, y_score, pred_score, y_bin, rank_margin))
    return metrics


def summarize_predictions_by_alpha(
    samples: List[Dict[str, Any]],
    test_mask: np.ndarray,
    kin_outputs: List[Dict[str, float]],
    eps_key: str,
    baseline_name: str,
) -> Dict[str, Dict[str, float]]:
    by_alpha: Dict[float, List[int]] = {}
    test_indices = [i for i, m in enumerate(test_mask) if m]

    for i in test_indices:
        alpha = float(samples[i]["row"]["alpha"])
        by_alpha.setdefault(alpha, []).append(i)

    report = {}
    for alpha, idxs in sorted(by_alpha.items()):
        true_reach = []
        pred_reach = []
        true_tau = []
        pred_tau = []
        true_min = []
        pred_min = []
        true_adv = []
        pred_adv = []

        for i in idxs:
            row = samples[i]["row"]
            tv = true_values(row, eps_key)
            pred = kin_outputs[i]

            true_reach.append(tv["reached"])
            pred_reach.append(pred["pred_reached_source"])
            true_tau.append(tv["tau"])
            pred_tau.append(pred["pred_tau_source"])
            true_min.append(tv["min_dist"])
            pred_min.append(pred["pred_min_dist_to_bowl2"])
            true_adv.append(tv["source_advantage"])
            pred_adv.append(pred["pred_source_advantage"])

        report[f"{alpha:.2f}"] = {
            "n": int(len(idxs)),
            "true_reach_mean": float(np.mean(true_reach)),
            "pred_reach_mean": float(np.mean(pred_reach)),
            "true_tau_mean": float(np.mean(true_tau)),
            "pred_tau_mean": float(np.mean(pred_tau)),
            "true_min_dist_mean": float(np.mean(true_min)),
            "pred_min_dist_mean": float(np.mean(pred_min)),
            "true_adv_mean": float(np.mean(true_adv)),
            "pred_adv_mean": float(np.mean(pred_adv)),
        }

    print(f"\n[by alpha on test] {baseline_name}")
    print("-" * 140)
    print(
        f"{'alpha':>8} {'n':>5} "
        f"{'true_reach':>12} {'pred_reach':>12} "
        f"{'true_tau':>10} {'pred_tau':>10} "
        f"{'true_min':>10} {'pred_min':>10} "
        f"{'true_adv':>10} {'pred_adv':>10}"
    )
    for alpha, r in report.items():
        print(
            f"{alpha:>8} {r['n']:5d} "
            f"{r['true_reach_mean']:12.3f} {r['pred_reach_mean']:12.3f} "
            f"{r['true_tau_mean']:10.2f} {r['pred_tau_mean']:10.2f} "
            f"{r['true_min_dist_mean']:10.4f} {r['pred_min_dist_mean']:10.4f} "
            f"{r['true_adv_mean']:+10.4f} {r['pred_adv_mean']:+10.4f}"
        )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eps_key", type=str, default="0p18")
    parser.add_argument("--eps_value", type=float, default=0.18)

    parser.add_argument("--train_init_states", type=str, default="0-13")
    parser.add_argument("--test_init_states", type=str, default="14-19")
    parser.add_argument("--test_alphas", type=str, default="0.75,0.95,1.05,1.15")
    parser.add_argument("--train_alphas", type=str, default="")

    parser.add_argument("--rank_margin", type=float, default=1e-6)
    parser.add_argument("--interp_only", action="store_true", default=True)
    parser.add_argument("--include_non_interp", action="store_true")
    parser.add_argument("--out_prefix", type=str, default="kinematic_baseline")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    interp_only = args.interp_only and not args.include_non_interp

    train_init_states = parse_int_range_or_list(args.train_init_states)
    test_init_states = parse_int_range_or_list(args.test_init_states)
    test_alphas = parse_float_list(args.test_alphas)
    train_alphas = parse_float_list(args.train_alphas) if args.train_alphas else []

    print("=" * 120)
    print(f"[data_dir] {data_dir}")
    print(f"[eps_key] {args.eps_key}")
    print(f"[eps_value] {args.eps_value}")
    print(f"[train_init_states] {train_init_states}")
    print(f"[test_init_states] {test_init_states}")
    print(f"[test_alphas] {test_alphas}")
    print(f"[train_alphas] {train_alphas if train_alphas else 'ALL_NON_TEST_ALPHAS'}")
    print("=" * 120)

    _rows_all, samples = load_dataset(data_dir, interp_only=interp_only)

    train_mask, test_mask = make_split(samples, train_init_states, test_init_states, test_alphas, train_alphas)

    if train_mask.sum() == 0:
        raise RuntimeError("No train samples selected.")
    if test_mask.sum() == 0:
        raise RuntimeError("No test samples selected.")

    print(f"[samples] {len(samples)}")
    print(f"[train samples] {int(train_mask.sum())}")
    print(f"[test samples] {int(test_mask.sum())}")
    print(f"[unused samples] {int((~(train_mask | test_mask)).sum())}")

    scalar_scale = fit_scalar_scale(samples, train_mask)
    xyz_scale = fit_xyz_scale(samples, train_mask)

    print(f"[fit_scalar scale] {scalar_scale:.8f}")
    print(f"[fit_xyz scale] {xyz_scale}")

    baselines = {
        "raw_scale1": 1.0,
        "fit_scalar": float(scalar_scale),
        "fit_xyz": xyz_scale,
    }

    all_results = []

    for baseline_name, scale in baselines.items():
        print("\n" + "=" * 120)
        print(f"[baseline] {baseline_name}")

        kin_outputs = []
        for sample in samples:
            pred_seq = predict_eef_seq(sample, scale=scale)
            pred = compute_kinematic_outputs(sample, pred_seq, eps=args.eps_value)
            kin_outputs.append(pred)

        target_metrics = []
        for target in ["source_advantage", "reached", "tau", "min_dist"]:
            m = evaluate_target(
                samples=samples,
                test_mask=test_mask,
                kin_outputs=kin_outputs,
                eps_key=args.eps_key,
                target_name=target,
                rank_margin=args.rank_margin,
            )
            target_metrics.append(m)

            print(
                f"  target={target:<16s} "
                f"R2={m['r2_raw']:+.4f} "
                f"MAE={m['mae_raw']:.6f} "
                f"AUC={m['auc_binary']:.4f} "
                f"pair={m['within_seed_pairwise_acc']:.4f} "
                f"binpair={m['within_seed_binpair_acc']:.4f}"
            )

        by_alpha = summarize_predictions_by_alpha(
            samples=samples,
            test_mask=test_mask,
            kin_outputs=kin_outputs,
            eps_key=args.eps_key,
            baseline_name=baseline_name,
        )

        all_results.append(
            {
                "baseline": baseline_name,
                "scale": scale.tolist() if isinstance(scale, np.ndarray) else float(scale),
                "target_metrics": target_metrics,
                "by_alpha": by_alpha,
            }
        )

    result_path = data_dir / f"{args.out_prefix}_eps_{args.eps_key}.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "data_dir": str(data_dir),
                "args": vars(args),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "fit_scalar_scale": float(scalar_scale),
                "fit_xyz_scale": xyz_scale.tolist(),
                "results": all_results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 120)
    print(f"[saved] {result_path}")
    print("=" * 120)


if __name__ == "__main__":
    main()