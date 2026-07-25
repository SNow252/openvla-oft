#!/usr/bin/env python3
"""
Make paper-ready held-out ranking baseline table with bootstrap 95% CI.

Input:
  baseline_joined_rows.csv
from:
  tools/compare_heldout_ranking_baselines_v4.py

Output:
  heldout_ranking_paper_table_with_ci.csv
  heldout_ranking_paper_table_with_ci.md
  heldout_ranking_bootstrap_long.csv
  heldout_ranking_task_level_table_with_ci.md

Metrics:
  Pearson
  Spearman
  Kendall tau
  Pairwise ranking accuracy
  Top-1 regret

Bootstrap:
  resample policy-condition rows with replacement.
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np


SELECTED_BASE = [
    "baseline_calibration_success",
    "efficacy_grasp",
    "efficacy_q",
    "efficacy_score",
    "native_selectivity_q",
    "profile_score_native",
    "baseline_language_profile",
]


DISPLAY_NAMES = {
    "baseline_calibration_success": "Calibration success",
    "efficacy_grasp": "Replay efficacy grasp",
    "efficacy_q": "Replay efficacy q",
    "efficacy_score": "Replay efficacy score",
    "native_selectivity_q": "Replay native selectivity",
    "profile_score_native": "Replay native profile",
    "baseline_language_profile": "Replay language profile",
}


def safe_float(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace("+", ""))
    except Exception:
        return default


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r.keys()} - set(fields))
    fields += extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x = x[m]
    y = y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return pearson(rankdata_average(x[m]), rankdata_average(y[m]))


def kendall_tau(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    if n < 2:
        return float("nan")

    c = 0
    d = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) < 1e-12 or abs(dy) < 1e-12:
                continue
            total += 1
            if dx * dy > 0:
                c += 1
            else:
                d += 1

    if total == 0:
        return float("nan")

    return float((c - d) / total)


def pairwise_acc(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    m = np.isfinite(pred) & np.isfinite(target)
    pred = pred[m]
    target = target[m]
    n = len(pred)
    if n < 2:
        return float("nan")

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            dp = pred[i] - pred[j]
            dt = target[i] - target[j]
            if abs(dp) < 1e-12 or abs(dt) < 1e-12:
                continue
            total += 1
            if dp * dt > 0:
                correct += 1

    if total == 0:
        return float("nan")

    return float(correct / total)


def top1_regret(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    m = np.isfinite(pred) & np.isfinite(target)
    pred = pred[m]
    target = target[m]
    if len(pred) == 0:
        return float("nan")

    selected_idx = int(np.argmax(pred))
    return float(np.max(target) - target[selected_idx])


def compute_metrics(pred, target):
    return {
        "pearson": pearson(pred, target),
        "spearman": spearman(pred, target),
        "kendall": kendall_tau(pred, target),
        "pairacc": pairwise_acc(pred, target),
        "top1_regret": top1_regret(pred, target),
    }


def ci(vals):
    vals = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def fmt(x, nd=3):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{nd}f}"


def fmt_ci(mid, lo, hi, nd=3):
    return f"{fmt(mid, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def extract_rows(rows, predictor, target_key="target"):
    pred = []
    target = []
    kept = []

    for r in rows:
        p = safe_float(r.get(predictor))
        t = safe_float(r.get(target_key))
        if not math.isfinite(t):
            t = safe_float(r.get("closedloop_target_score"))
        if math.isfinite(p) and math.isfinite(t):
            pred.append(p)
            target.append(t)
            kept.append(r)

    return np.asarray(pred, dtype=float), np.asarray(target, dtype=float), kept


def best_kinematic_predictor(rows):
    candidates = sorted({k for r in rows for k in r.keys() if k.startswith("baseline_kin_")})
    best = None

    for k in candidates:
        pred, target, _ = extract_rows(rows, k)
        if len(pred) < 2:
            continue
        s = spearman(pred, target)
        if not math.isfinite(s):
            continue
        if best is None or s > best[1]:
            best = (k, s)

    return best[0] if best else None


def summarize_predictor(rows, predictor, n_boot=5000, seed=0, group="all", value="all"):
    pred, target, kept = extract_rows(rows, predictor)
    point = compute_metrics(pred, target)

    rng = np.random.default_rng(seed)
    boot = defaultdict(list)
    boot_rows = []

    n = len(kept)
    if n >= 2:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            pred_b = pred[idx]
            target_b = target[idx]
            m = compute_metrics(pred_b, target_b)
            for metric, val in m.items():
                if math.isfinite(val):
                    boot[metric].append(val)
                    boot_rows.append({
                        "group": group,
                        "value": value,
                        "predictor": predictor,
                        "bootstrap_id": b,
                        "metric": metric,
                        "value_metric": val,
                    })

    out = {
        "group": group,
        "value": value,
        "predictor": predictor,
        "display_name": DISPLAY_NAMES.get(predictor, predictor),
        "n": n,
    }

    for metric in ["pearson", "spearman", "kendall", "pairacc", "top1_regret"]:
        lo, hi = ci(boot.get(metric, []))
        out[metric] = point[metric]
        out[f"{metric}_ci_low"] = lo
        out[f"{metric}_ci_high"] = hi
        out[f"{metric}_with_ci"] = fmt_ci(point[metric], lo, hi)

    return out, boot_rows


def write_paper_md(rows, path, title):
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Predictor | n | Pearson | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['display_name']} | {r['n']} | "
            f"{r['pearson_with_ci']} | "
            f"{r['spearman_with_ci']} | "
            f"{r['kendall_with_ci']} | "
            f"{r['pairacc_with_ci']} | "
            f"{r['top1_regret_with_ci']} |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- Bootstrap confidence intervals resample policy-condition rows with replacement.")
    lines.append("- Calibration success is a closed-loop rollout baseline on calibration manifests.")
    lines.append("- Kinematic++ is the best pre-action kinematic baseline selected by all-config Spearman.")
    lines.append("- Language profile measures language-conditioned object selectivity and is not expected to predict native BDDL success.")

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_joined_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.baseline_joined_csv)

    best_kin = best_kinematic_predictor(rows)
    if best_kin is None:
        print("[WARN] no Kinematic++ predictor found")
        predictors = list(SELECTED_BASE)
    else:
        DISPLAY_NAMES[best_kin] = "Best Kinematic++"
        predictors = [
            "baseline_calibration_success",
            "efficacy_grasp",
            "efficacy_q",
            "efficacy_score",
            "native_selectivity_q",
            "profile_score_native",
            best_kin,
            "baseline_language_profile",
        ]

    summary_rows = []
    boot_long = []

    for i, pred in enumerate(predictors):
        r, b = summarize_predictor(
            rows,
            pred,
            n_boot=args.n_boot,
            seed=args.seed + i * 17,
            group="all",
            value="all",
        )
        summary_rows.append(r)
        boot_long.extend(b)

    # Sort by Spearman descending, except keep language profile near end if weak.
    summary_rows = sorted(summary_rows, key=lambda x: safe_float(x["spearman"], -999), reverse=True)

    write_csv(summary_rows, out / "heldout_ranking_paper_table_with_ci.csv")
    write_csv(boot_long, out / "heldout_ranking_bootstrap_long.csv")
    write_paper_md(
        summary_rows,
        out / "heldout_ranking_paper_table_with_ci.md",
        "Held-out ranking baseline comparison with bootstrap 95% CI",
    )

    # Task-level compact table for main replay predictors.
    by_task = defaultdict(list)
    for r in rows:
        by_task[r.get("task", "unknown")].append(r)

    task_summary = []
    for task, rs in sorted(by_task.items()):
        for pred in ["efficacy_score", "profile_score_native", "efficacy_grasp"]:
            rr, bb = summarize_predictor(
                rs,
                pred,
                n_boot=args.n_boot,
                seed=args.seed + len(task_summary) * 29,
                group="task",
                value=task,
            )
            task_summary.append(rr)
            boot_long.extend(bb)

    write_csv(task_summary, out / "heldout_ranking_task_level_with_ci.csv")

    lines = []
    lines.append("# Task-level held-out ranking metrics with bootstrap 95% CI")
    lines.append("")
    lines.append("| Task | Predictor | n | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in task_summary:
        lines.append(
            f"| {r['value']} | {r['display_name']} | {r['n']} | "
            f"{r['spearman_with_ci']} | {r['kendall_with_ci']} | "
            f"{r['pairacc_with_ci']} | {r['top1_regret_with_ci']} |"
        )
    Path(out / "heldout_ranking_task_level_table_with_ci.md").write_text("\n".join(lines) + "\n")

    print("[INFO] best_kinematic_predictor:", best_kin)
    print("[saved]", out / "heldout_ranking_paper_table_with_ci.csv")
    print("[saved]", out / "heldout_ranking_paper_table_with_ci.md")
    print("[saved]", out / "heldout_ranking_bootstrap_long.csv")
    print("[saved]", out / "heldout_ranking_task_level_with_ci.csv")
    print("[saved]", out / "heldout_ranking_task_level_table_with_ci.md")
    print()
    print((out / "heldout_ranking_paper_table_with_ci.md").read_text())
    print()
    print((out / "heldout_ranking_task_level_table_with_ci.md").read_text())


if __name__ == "__main__":
    main()