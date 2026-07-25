#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from collections import defaultdict
import numpy as np


def safe_float(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace("+", ""))
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(str(x)))
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


def first(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in ["", None]:
            return row[k]
    return default


def infer_policy_from_path(path: Path):
    s = str(path).lower()
    if "openvla" in s:
        return "openvla_oft"
    if "smolvla" in s:
        return "smolvla"
    return "unknown"


def infer_task_from_condition(cond, task_id=None):
    cond = str(cond)
    if cond.startswith("task3"):
        return "task3"
    if cond.startswith("task9"):
        return "task9"
    if cond.startswith("task8") or cond.startswith("task1_native"):
        return "task8_scene"
    if task_id is not None:
        task_id = int(task_id)
        if task_id == 3:
            return "task3"
        if task_id == 8 or task_id == 1:
            return "task8_scene"
        if task_id == 9:
            return "task9"
    return "unknown"


def native_side(condition):
    if str(condition) == "task1_native_next_to_ramekin":
        return "alternative"
    return "primary"


def language_side(condition):
    cond = str(condition)
    if cond == "task1_native_next_to_ramekin":
        return "alternative"
    if "wrong_source" in cond:
        return "alternative"
    if cond.endswith("_original") or cond == "task8_original":
        return "primary"
    if "wrong_target_language_only" in cond and "wrong_source" not in cond:
        return "primary"
    return "none"


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
        ranks[order[i:j+1]] = avg
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
    c = d = total = 0
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
    correct = total = 0
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
    return float(np.max(target) - target[int(np.argmax(pred))])


def load_heldout_joined(path):
    rows = []
    for r in read_csv(path):
        out = dict(r)
        for k in list(out.keys()):
            if k not in ["policy", "task", "condition", "target_source"]:
                v = safe_float(out[k])
                if math.isfinite(v):
                    out[k] = v
        out["target"] = safe_float(out.get("closedloop_target_score"))
        rows.append(out)
    return rows


def load_calibration_success(fresh_roots):
    by = defaultdict(list)

    for root_s in fresh_roots:
        root = Path(root_s)
        if not root.exists():
            print("[WARN] missing calibration root:", root)
            continue

        policy_default = infer_policy_from_path(root)

        for m in sorted(root.glob("runs/*/manifest.csv")):
            cond = m.parent.name
            for i, r in enumerate(read_csv(m)):
                policy = first(r, ["policy", "policy_name", "fresh_policy_name"], policy_default)
                if policy in ["openvla", "openvla-oft"]:
                    policy = "openvla_oft"
                task_id = safe_int(first(r, ["task_id", "fresh_task_id"]), None)
                task = infer_task_from_condition(cond, task_id)

                val = safe_float(first(r, ["success", "episode_success", "is_success", "reward", "done"]))
                if math.isfinite(val):
                    by[(policy, task, cond)].append(val)

    out = {}
    for k, vals in by.items():
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_kinematic_condition_scores(csv_paths):
    """
    Reads preaction_kinematicpp_object_scores.csv files and builds condition-level
    native/effectiveness scores for each method/scale.
    """
    groups = defaultdict(lambda: {"primary": [], "alternative": []})

    for path_s in csv_paths:
        path = Path(path_s)
        if not path.exists():
            print("[WARN] missing kinematic score csv:", path)
            continue
        for r in read_csv(path):
            policy = r.get("policy", infer_policy_from_path(path))
            if policy in ["openvla", "openvla-oft"]:
                policy = "openvla_oft"
            cond = r.get("condition", "unknown")
            task = infer_task_from_condition(cond)
            obj = r.get("object", "")
            side = "primary" if obj in ["bowl1", "akita_black_bowl_1"] else "alternative"
            scale = str(r.get("scale", ""))
            for method in ["eef_min_dist", "close_min_dist", "relative_margin", "grasp_heuristic"]:
                v = safe_float(r.get(method))
                if math.isfinite(v):
                    groups[(policy, task, cond, method, scale)][side].append(v)

    out = {}
    for (policy, task, cond, method, scale), d in groups.items():
        p = np.mean(d["primary"]) if d["primary"] else float("nan")
        a = np.mean(d["alternative"]) if d["alternative"] else float("nan")
        side = native_side(cond)
        if side == "primary":
            native = p + 0.5 * (p - a)
        else:
            native = a + 0.5 * (a - p)
        efficacy = max(p, a)
        out[(policy, task, cond, method, scale, "native")] = float(native)
        out[(policy, task, cond, method, scale, "efficacy")] = float(efficacy)
    return out


def add_baselines(joined, calibration_success, kinematic_scores):
    out = []

    methods_scales = sorted({(k[3], k[4]) for k in kinematic_scores.keys()})

    for r in joined:
        policy = r["policy"]
        task = r["task"]
        cond = r["condition"]
        key = (policy, task, cond)

        row = dict(r)
        row["baseline_calibration_success"] = calibration_success.get(key, float("nan"))

        # Language fidelity is already in joined, but expose as baseline name.
        row["baseline_language_profile"] = safe_float(r.get("profile_score_language"))

        for method, scale in methods_scales:
            name = f"baseline_kin_{method}_s{scale}"
            row[name + "_native"] = kinematic_scores.get((policy, task, cond, method, scale, "native"), float("nan"))
            row[name + "_efficacy"] = kinematic_scores.get((policy, task, cond, method, scale, "efficacy"), float("nan"))

        out.append(row)

    return out


def compute_metrics(rows):
    predictors = [
        "profile_score_native",
        "efficacy_score",
        "efficacy_q",
        "efficacy_grasp",
        "native_selectivity_q",
        "baseline_calibration_success",
        "baseline_language_profile",
    ]

    # Add kinematic predictors.
    for k in rows[0].keys():
        if k.startswith("baseline_kin_"):
            predictors.append(k)

    metrics = []

    def add_group(group_name, group_value, rs):
        target = [safe_float(r["target"]) for r in rs]
        for pred in predictors:
            x = [safe_float(r.get(pred)) for r in rs]
            valid = [math.isfinite(a) and math.isfinite(b) for a, b in zip(x, target)]
            if sum(valid) < 2:
                continue
            xv = [a for a, m in zip(x, valid) if m]
            yv = [b for b, m in zip(target, valid) if m]
            metrics.append({
                "group": group_name,
                "value": group_value,
                "predictor": pred,
                "n": len(xv),
                "pearson": pearson(xv, yv),
                "spearman": spearman(xv, yv),
                "kendall": kendall_tau(xv, yv),
                "pairacc": pairwise_acc(xv, yv),
                "top1_regret": top1_regret(xv, yv),
            })

    add_group("all", "all", rows)

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for task, rs in sorted(by_task.items()):
        add_group("task", task, rs)

    return metrics


def fmt(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "NA"
    return f"{x:.3f}"


def write_metrics_md(metrics, path, top_k=80):
    # Sort useful first: all group, then by Spearman.
    metrics = sorted(metrics, key=lambda r: (
        0 if r["group"] == "all" else 1,
        -safe_float(r["spearman"], -999)
    ))

    lines = []
    lines.append("| Group | Value | Predictor | n | Pearson | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in metrics[:top_k]:
        lines.append(
            f"| {r['group']} | {r['value']} | {r['predictor']} | {r['n']} | "
            f"{fmt(r['pearson'])} | {fmt(r['spearman'])} | {fmt(r['kendall'])} | "
            f"{fmt(r['pairacc'])} | {fmt(r['top1_regret'])} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_paper_md(metrics, path):
    keep = [
        "profile_score_native",
        "efficacy_score",
        "efficacy_q",
        "efficacy_grasp",
        "native_selectivity_q",
        "baseline_calibration_success",
        "baseline_language_profile",
    ]

    # Best kinematic by all-group Spearman.
    all_kin = [m for m in metrics if m["group"] == "all" and m["predictor"].startswith("baseline_kin_")]
    if all_kin:
        best_kin = max(all_kin, key=lambda x: safe_float(x["spearman"], -999))
        keep.append(best_kin["predictor"])

    all_rows = [m for m in metrics if m["group"] == "all" and m["predictor"] in keep]
    all_rows = sorted(all_rows, key=lambda x: safe_float(x["spearman"], -999), reverse=True)

    lines = []
    lines.append("# Held-out ranking baseline comparison")
    lines.append("")
    lines.append("| Predictor | n | Pearson | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        lines.append(
            f"| {r['predictor']} | {r['n']} | {fmt(r['pearson'])} | {fmt(r['spearman'])} | "
            f"{fmt(r['kendall'])} | {fmt(r['pairacc'])} | {fmt(r['top1_regret'])} |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- `baseline_calibration_success` is closed-loop success on calibration manifests, used as a rollout-based baseline.")
    lines.append("- `baseline_kin_*` predictors are pre-action Kinematic++ baselines computed from initial state and action chunk.")
    lines.append("- `baseline_language_profile` measures instruction/object selectivity and is not expected to predict native BDDL success.")

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout_joined_csv", required=True)
    ap.add_argument("--calibration_fresh_roots", nargs="+", required=True)
    ap.add_argument("--kinematic_object_csvs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joined = load_heldout_joined(args.heldout_joined_csv)
    calibration_success = load_calibration_success(args.calibration_fresh_roots)
    kin_scores = load_kinematic_condition_scores(args.kinematic_object_csvs)

    rows = add_baselines(joined, calibration_success, kin_scores)
    metrics = compute_metrics(rows)

    write_csv(rows, out / "baseline_joined_rows.csv")
    write_csv(metrics, out / "baseline_ranking_metrics.csv")
    write_metrics_md(metrics, out / "baseline_ranking_metrics.md")
    write_paper_md(metrics, out / "baseline_ranking_paper_table.md")

    print("[saved]", out / "baseline_joined_rows.csv")
    print("[saved]", out / "baseline_ranking_metrics.csv")
    print("[saved]", out / "baseline_ranking_metrics.md")
    print("[saved]", out / "baseline_ranking_paper_table.md")
    print()
    print((out / "baseline_ranking_paper_table.md").read_text())


if __name__ == "__main__":
    main()