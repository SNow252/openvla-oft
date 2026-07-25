#!/usr/bin/env python3
"""
Held-out closed-loop ranking analysis for same-state replay profiles.

This script does NOT run simulation.
It joins:

  1. Calibration replay profile scores:
       policy_profile_scores.csv

  2. Closed-loop evaluation manifests:
       fresh_policy_dumps_v4/*/runs/<condition>/manifest.csv

Then it evaluates whether replay profile scores predict closed-loop performance.

Important:
- If the manifest roots come from the same init states used for replay profiles,
  this is only a sanity check.
- For a true held-out ranking experiment, pass fresh roots collected on disjoint
  init_state_idx ranges, e.g. init 20-49.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from collections import defaultdict
import numpy as np


SUCCESS_KEYS = [
    "success", "episode_success", "is_success", "done_success",
    "task_success", "final_success"
]

REWARD_KEYS = [
    "reward", "total_reward", "episode_reward", "return"
]

DONE_KEYS = [
    "done", "final_done"
]

STEP_KEYS = [
    "num_steps", "steps", "episode_len", "episode_length", "timestep", "timesteps"
]

INIT_KEYS = [
    "init_state_idx", "init_idx", "episode_id", "episode_idx", "trial_id"
]

POLICY_KEYS = [
    "policy", "policy_name", "fresh_policy_name"
]

TASK_ID_KEYS = [
    "task_id", "fresh_task_id"
]


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


def first(row, keys, default=""):
    for k in keys:
        if k in row and row[k] not in ["", None]:
            return row[k]
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
        if int(task_id) == 3:
            return "task3"
        if int(task_id) == 8:
            return "task8_scene"
        if int(task_id) == 9:
            return "task9"
        if int(task_id) == 1:
            return "task8_scene"

    return "unknown"


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
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return pearson(rankdata_average(x[mask]), rankdata_average(y[mask]))


def kendall_tau(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 2:
        return float("nan")

    conc = 0
    disc = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if abs(dx) < 1e-12 or abs(dy) < 1e-12:
                continue
            total += 1
            if dx * dy > 0:
                conc += 1
            else:
                disc += 1
    if total == 0:
        return float("nan")
    return float((conc - disc) / total)


def pairwise_acc(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    n = len(pred)
    if n < 2:
        return float("nan")

    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = target[i] - target[j]
            dp = pred[i] - pred[j]
            if abs(dt) < 1e-12 or abs(dp) < 1e-12:
                continue
            total += 1
            if dt * dp > 0:
                correct += 1
    if total == 0:
        return float("nan")
    return float(correct / total)


def top1_regret(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    if len(pred) == 0:
        return float("nan")
    pred_best = int(np.argmax(pred))
    true_best = float(np.max(target))
    selected = float(target[pred_best])
    return float(true_best - selected)


def bootstrap_ci(values, fn, n_boot=1000, seed=0):
    values = np.asarray(values)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    outs = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            outs.append(fn(values[idx]))
        except Exception:
            pass
    outs = np.asarray([x for x in outs if np.isfinite(x)], dtype=float)
    if len(outs) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(outs, 0.025)), float(np.quantile(outs, 0.975)))


def load_profile_rows(profile_csv):
    rows = []
    for r in read_csv(profile_csv):
        out = dict(r)
        out["policy"] = out.get("policy", "")
        out["task"] = out.get("task", "")
        out["condition"] = out.get("condition", "")
        for k in [
            "profile_score_native",
            "profile_score_language",
            "efficacy_score",
            "native_selectivity_q",
            "language_selectivity_q",
            "efficacy_q",
            "efficacy_grasp",
            "collapse_flag",
        ]:
            out[k] = safe_float(out.get(k))
        rows.append(out)
    return rows


def load_manifest_rows(fresh_roots, min_init=None, max_init=None, exclude_regex=None):
    rows = []

    for root_s in fresh_roots:
        root = Path(root_s)
        if not root.exists():
            print(f"[WARN] missing fresh root: {root}")
            continue

        for manifest in sorted(root.glob("runs/*/manifest.csv")):
            condition = manifest.parent.name
            if exclude_regex and re.search(exclude_regex, condition):
                continue

            raw = read_csv(manifest)
            policy_from_path = infer_policy_from_path(root)

            for i, r in enumerate(raw):
                init_idx = safe_int(first(r, INIT_KEYS), default=i)
                if min_init is not None and init_idx is not None and init_idx < min_init:
                    continue
                if max_init is not None and init_idx is not None and init_idx > max_init:
                    continue

                policy = first(r, POLICY_KEYS, default=policy_from_path)
                if policy in ["openvla", "openvla-oft", "openvla_oft"]:
                    policy = "openvla_oft"
                if policy.lower() == "smolvla":
                    policy = "smolvla"

                task_id = safe_int(first(r, TASK_ID_KEYS), default=None)
                task = infer_task_from_condition(condition, task_id=task_id)

                success = safe_float(first(r, SUCCESS_KEYS))
                reward = safe_float(first(r, REWARD_KEYS))
                done = safe_float(first(r, DONE_KEYS))
                steps = safe_float(first(r, STEP_KEYS))

                # If success is missing, use reward if it looks binary, then done.
                target_score = success
                target_source = "success"
                if not math.isfinite(target_score):
                    if math.isfinite(reward):
                        target_score = reward
                        target_source = "reward"
                    elif math.isfinite(done):
                        target_score = done
                        target_source = "done"

                rows.append({
                    "policy": policy,
                    "task": task,
                    "condition": condition,
                    "init_state_idx": init_idx,
                    "success": success,
                    "reward": reward,
                    "done": done,
                    "steps": steps,
                    "target_score": target_score,
                    "target_source": target_source,
                    "manifest_path": str(manifest),
                    "fresh_root": str(root),
                })

    return rows


def aggregate_targets(manifest_rows):
    by = defaultdict(list)
    for r in manifest_rows:
        key = (r["policy"], r["task"], r["condition"])
        by[key].append(r)

    out = []
    for (policy, task, condition), rs in sorted(by.items()):
        def mean(k):
            vals = [safe_float(r.get(k)) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        out.append({
            "policy": policy,
            "task": task,
            "condition": condition,
            "n_closedloop": len(rs),
            "closedloop_success": mean("success"),
            "closedloop_reward": mean("reward"),
            "closedloop_done": mean("done"),
            "closedloop_steps": mean("steps"),
            "closedloop_target_score": mean("target_score"),
            "target_source": rs[0].get("target_source", ""),
        })
    return out


def join_profile_targets(profile_rows, target_rows):
    target_by_key = {
        (r["policy"], r["task"], r["condition"]): r
        for r in target_rows
    }

    joined = []
    missing = []

    for p in profile_rows:
        key = (p["policy"], p["task"], p["condition"])
        t = target_by_key.get(key)
        if t is None:
            missing.append({
                "policy": p["policy"],
                "task": p["task"],
                "condition": p["condition"],
            })
            continue
        row = dict(p)
        row.update(t)
        joined.append(row)

    return joined, missing


def metric_rows_for_group(rows, group_name, group_value):
    predictors = [
        "profile_score_native",
        "profile_score_language",
        "efficacy_score",
        "native_selectivity_q",
        "language_selectivity_q",
        "efficacy_q",
        "efficacy_grasp",
    ]

    target = [safe_float(r["closedloop_target_score"]) for r in rows]
    out = []

    for pred_name in predictors:
        pred = [safe_float(r.get(pred_name)) for r in rows]
        mask = [math.isfinite(p) and math.isfinite(t) for p, t in zip(pred, target)]
        pred_m = [p for p, m in zip(pred, mask) if m]
        target_m = [t for t, m in zip(target, mask) if m]

        if len(pred_m) < 2:
            continue

        out.append({
            "group_name": group_name,
            "group_value": group_value,
            "predictor": pred_name,
            "n": len(pred_m),
            "pearson": pearson(pred_m, target_m),
            "spearman": spearman(pred_m, target_m),
            "kendall_tau": kendall_tau(pred_m, target_m),
            "pairwise_acc": pairwise_acc(pred_m, target_m),
            "top1_regret": top1_regret(pred_m, target_m),
        })

    return out


def compute_metrics(joined_rows):
    metrics = []

    metrics += metric_rows_for_group(joined_rows, "all", "all")

    by_task = defaultdict(list)
    for r in joined_rows:
        by_task[r["task"]].append(r)

    for task, rs in sorted(by_task.items()):
        metrics += metric_rows_for_group(rs, "task", task)

    return metrics


def fmt(x, nd=3, signed=False):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "NA"
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def write_joined_md(rows, path):
    lines = []
    lines.append("| Policy | Task | Condition | n eval | Closed-loop target | Native profile | Lang profile | Efficacy | Selectivity | Collapse |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (x["task"], x["condition"], x["policy"])):
        lines.append(
            f"| {r['policy']} | {r['task']} | {r['condition']} | {r['n_closedloop']} | "
            f"{fmt(r['closedloop_target_score'],3)} | "
            f"{fmt(r.get('profile_score_native'),3,True)} | "
            f"{fmt(r.get('profile_score_language'),3,True)} | "
            f"{fmt(r.get('efficacy_score'),3,True)} | "
            f"{fmt(r.get('native_selectivity_q'),3,True)} | "
            f"{int(safe_float(r.get('collapse_flag'),0))} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_metrics_md(rows, path):
    lines = []
    lines.append("| Group | Value | Predictor | n | Pearson | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['group_name']} | {r['group_value']} | {r['predictor']} | {r['n']} | "
            f"{fmt(r['pearson'])} | {fmt(r['spearman'])} | {fmt(r['kendall_tau'])} | "
            f"{fmt(r['pairwise_acc'])} | {fmt(r['top1_regret'])} |"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_rankings_md(rows, path):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    lines = []
    lines.append("# Closed-loop target vs replay-profile ranking")
    lines.append("")
    for task, rs in sorted(by_task.items()):
        lines.append(f"## {task}")
        lines.append("")
        lines.append("### Ranking by closed-loop target")
        lines.append("")
        lines.append("| Rank | Policy | Condition | Target | Native profile | Lang profile | Efficacy |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for i, r in enumerate(sorted(rs, key=lambda x: safe_float(x["closedloop_target_score"]), reverse=True), 1):
            lines.append(
                f"| {i} | {r['policy']} | {r['condition']} | "
                f"{fmt(r['closedloop_target_score'])} | {fmt(r.get('profile_score_native'),3,True)} | "
                f"{fmt(r.get('profile_score_language'),3,True)} | {fmt(r.get('efficacy_score'),3,True)} |"
            )
        lines.append("")
        lines.append("### Ranking by native profile")
        lines.append("")
        lines.append("| Rank | Policy | Condition | Native profile | Target | Efficacy |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for i, r in enumerate(sorted(rs, key=lambda x: safe_float(x.get("profile_score_native")), reverse=True), 1):
            lines.append(
                f"| {i} | {r['policy']} | {r['condition']} | "
                f"{fmt(r.get('profile_score_native'),3,True)} | {fmt(r['closedloop_target_score'])} | "
                f"{fmt(r.get('efficacy_score'),3,True)} |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def write_interpretation(metrics, joined, missing, path):
    best = sorted(
        [m for m in metrics if math.isfinite(safe_float(m.get("spearman")))],
        key=lambda x: x["spearman"],
        reverse=True,
    )[:5]

    lines = []
    lines.append("# Held-out closed-loop ranking interpretation")
    lines.append("")
    lines.append("This analysis joins calibration replay profile scores with closed-loop evaluation manifests.")
    lines.append("")
    lines.append("Important caveat: if the manifest roots use the same init states as the replay profiles, this is only a ranking sanity check. For a true held-out policy-evaluation result, rerun closed-loop evaluation on disjoint init states and pass those manifest roots to this script.")
    lines.append("")
    lines.append(f"Joined rows: {len(joined)}")
    lines.append(f"Missing profile-target joins: {len(missing)}")
    lines.append("")
    lines.append("## Best predictor/group combinations by Spearman")
    lines.append("")
    lines.append("| Rank | Group | Value | Predictor | n | Spearman | Kendall | PairAcc | Top1 regret |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
    for i, m in enumerate(best, 1):
        lines.append(
            f"| {i} | {m['group_name']} | {m['group_value']} | {m['predictor']} | {m['n']} | "
            f"{fmt(m['spearman'])} | {fmt(m['kendall_tau'])} | {fmt(m['pairwise_acc'])} | {fmt(m['top1_regret'])} |"
        )
    lines.append("")
    lines.append("## Next decision")
    lines.append("")
    lines.append("If replay profile scores correlate with closed-loop targets in this sanity check, run a true held-out version with calibration init states and evaluation init states disjoint. The paper-level evidence should report Spearman, Kendall tau, pairwise ranking accuracy, top-1 regret, and bootstrap confidence intervals.")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_csv", required=True)
    ap.add_argument("--fresh_roots", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_init", type=int, default=None)
    ap.add_argument("--max_init", type=int, default=None)
    ap.add_argument("--exclude_condition_regex", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_rows = load_profile_rows(args.profile_csv)
    manifest_rows = load_manifest_rows(
        args.fresh_roots,
        min_init=args.min_init,
        max_init=args.max_init,
        exclude_regex=args.exclude_condition_regex or None,
    )
    target_rows = aggregate_targets(manifest_rows)
    joined, missing = join_profile_targets(profile_rows, target_rows)
    metrics = compute_metrics(joined)

    write_csv(manifest_rows, out_dir / "closedloop_manifest_rows.csv")
    write_csv(target_rows, out_dir / "closedloop_targets.csv")
    write_csv(joined, out_dir / "profile_vs_closedloop_joined.csv")
    write_csv(missing, out_dir / "missing_profile_target_joins.csv")
    write_csv(metrics, out_dir / "ranking_metrics.csv")

    write_joined_md(joined, out_dir / "profile_vs_closedloop_joined.md")
    write_metrics_md(metrics, out_dir / "ranking_metrics.md")
    write_rankings_md(joined, out_dir / "closedloop_vs_profile_rankings.md")
    write_interpretation(metrics, joined, missing, out_dir / "heldout_closedloop_ranking_interpretation.md")

    print("[saved]", out_dir / "closedloop_manifest_rows.csv")
    print("[saved]", out_dir / "closedloop_targets.csv")
    print("[saved]", out_dir / "profile_vs_closedloop_joined.csv")
    print("[saved]", out_dir / "ranking_metrics.csv")
    print("[saved]", out_dir / "profile_vs_closedloop_joined.md")
    print("[saved]", out_dir / "ranking_metrics.md")
    print("[saved]", out_dir / "closedloop_vs_profile_rankings.md")
    print("[saved]", out_dir / "heldout_closedloop_ranking_interpretation.md")
    print()
    print((out_dir / "ranking_metrics.md").read_text())
    print()
    print((out_dir / "heldout_closedloop_ranking_interpretation.md").read_text())


if __name__ == "__main__":
    main()