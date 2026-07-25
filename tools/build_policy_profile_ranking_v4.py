#!/usr/bin/env python3
"""
Build oracle/profile ranking tables from existing same-state replay consequence tables.

This script does NOT run simulation.
It reads:
  1. task8 n=20 paper_policy_consequence_table.csv
  2. task3/task9 multitask_minimal_policy_consequence_table.csv

It normalizes both into:
  policy, task, condition, primary/alternative consequence

Then computes:
  - efficacy
  - native-goal selectivity
  - language-goal selectivity when applicable
  - collapse flag
  - heuristic profile score

The output is a sanity check for whether replay consequence profiles can support
policy/configuration ranking before doing held-out closed-loop evaluation.
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict


def safe_float(x, default=float("nan")):
    try:
        if x is None or x == "":
            return default
        return float(str(x).replace("+", ""))
    except Exception:
        return default


def first(row, keys, default=""):
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r.keys()} - set(fields))
    fields += extra
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def infer_task_from_condition(cond):
    cond = str(cond)
    if cond.startswith("task3"):
        return "task3"
    if cond.startswith("task8") or cond.startswith("task1_native"):
        return "task8_scene"
    if cond.startswith("task9"):
        return "task9"
    if cond.startswith("task1"):
        return "task1"
    return "unknown"


def normalize_task8_rows(rows):
    out = []
    for r in rows:
        policy = first(r, ["policy", "Policy"])
        cond = first(r, ["condition", "Condition"])
        n = safe_float(first(r, ["n", "N"]), 0)

        primary_grasp = safe_float(first(r, [
            "primary_grasp", "bowl1_grasp", "B1 grasp", "b1_grasp",
            "bowl1_grasp_success_proxy"
        ]))
        primary_q = safe_float(first(r, [
            "primary_q", "bowl1_quality", "B1 q", "b1_q",
            "bowl1_grasp_quality_score"
        ]))
        alt_grasp = safe_float(first(r, [
            "alternative_grasp", "bowl2_grasp", "B2 grasp", "b2_grasp",
            "bowl2_grasp_success_proxy"
        ]))
        alt_q = safe_float(first(r, [
            "alternative_q", "bowl2_quality", "B2 q", "b2_q",
            "bowl2_grasp_quality_score"
        ]))
        alt_minus_primary_q = safe_float(first(r, [
            "alternative_minus_primary_q",
            "bowl2_minus_bowl1_quality",
            "B2-B1 q",
            "b2_minus_b1_q"
        ]))

        if not math.isfinite(alt_minus_primary_q):
            alt_minus_primary_q = alt_q - primary_q

        out.append({
            "policy": policy,
            "task": infer_task_from_condition(cond),
            "condition": cond,
            "n": int(n),
            "primary_grasp": primary_grasp,
            "primary_q": primary_q,
            "alternative_grasp": alt_grasp,
            "alternative_q": alt_q,
            "alternative_minus_primary_q": alt_minus_primary_q,
            "source_table": "task8_n20",
        })
    return out


def normalize_multitask_rows(rows):
    out = []
    for r in rows:
        policy = first(r, ["policy", "Policy"])
        task = first(r, ["task", "Task"])
        cond = first(r, ["condition", "Condition"])
        n = safe_float(first(r, ["n", "N"]), 0)

        primary_grasp = safe_float(first(r, ["primary_grasp", "Primary grasp"]))
        primary_q = safe_float(first(r, ["primary_q", "Primary q"]))
        alt_grasp = safe_float(first(r, ["alternative_grasp", "Alt grasp", "alternative_grasp"]))
        alt_q = safe_float(first(r, ["alternative_q", "Alt q", "alternative_q"]))
        alt_minus_primary_q = safe_float(first(r, [
            "alternative_minus_primary_q", "Alt-Primary q"
        ]))

        if not math.isfinite(alt_minus_primary_q):
            alt_minus_primary_q = alt_q - primary_q

        out.append({
            "policy": policy,
            "task": task,
            "condition": cond,
            "n": int(n),
            "primary_grasp": primary_grasp,
            "primary_q": primary_q,
            "alternative_grasp": alt_grasp,
            "alternative_q": alt_q,
            "alternative_minus_primary_q": alt_minus_primary_q,
            "source_table": "multitask_minimal",
        })
    return out


def native_intended_side(task, condition):
    """
    Native-goal side means the object intended by the policy's native condition.

    For task8_scene, task1_native_next_to_ramekin is a positive-control condition
    whose intended object is alternative/bowl2 in the task8 scene.
    For task3/task8/task9 original/corruption conditions, the native BDDL object
    is primary/bowl1.
    """
    cond = str(condition)
    if cond == "task1_native_next_to_ramekin":
        return "alternative"
    return "primary"


def language_intended_side(condition):
    """
    Language-directed side when the condition has a clear object instruction.

    empty/nonsense/unrelated have no well-defined language target, so return none.
    wrong_source conditions are treated as alternative-directed.
    wrong_target only changes target, not source, so source side remains primary.
    """
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


def get_side_values(row, side):
    if side == "primary":
        return (
            row["primary_grasp"],
            row["primary_q"],
            row["alternative_grasp"],
            row["alternative_q"],
        )
    if side == "alternative":
        return (
            row["alternative_grasp"],
            row["alternative_q"],
            row["primary_grasp"],
            row["primary_q"],
        )
    return (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )


def add_profile_scores(row):
    p_g = row["primary_grasp"]
    p_q = row["primary_q"]
    a_g = row["alternative_grasp"]
    a_q = row["alternative_q"]

    efficacy_q = max(p_q, a_q)
    efficacy_grasp = max(p_g, a_g)
    abs_object_bias_q = abs(p_q - a_q)
    abs_object_bias_grasp = abs(p_g - a_g)

    native_side = native_intended_side(row["task"], row["condition"])
    native_g, native_q, native_unintended_g, native_unintended_q = get_side_values(row, native_side)
    native_selectivity_q = native_q - native_unintended_q
    native_selectivity_grasp = native_g - native_unintended_g

    lang_side = language_intended_side(row["condition"])
    if lang_side == "none":
        lang_g = lang_q = lang_unintended_g = lang_unintended_q = float("nan")
        lang_selectivity_q = float("nan")
        lang_selectivity_grasp = float("nan")
    else:
        lang_g, lang_q, lang_unintended_g, lang_unintended_q = get_side_values(row, lang_side)
        lang_selectivity_q = lang_q - lang_unintended_q
        lang_selectivity_grasp = lang_g - lang_unintended_g

    # A simple, transparent oracle profile score.
    # High when the intended object has high quality and the unintended object is not manipulated.
    profile_score_native = (
        native_q
        + 0.5 * native_selectivity_q
        + 0.20 * native_g
        - 0.20 * native_unintended_g
    )

    # Efficacy-only score ignores whether the policy manipulated the intended object.
    efficacy_score = efficacy_q + 0.20 * efficacy_grasp

    collapse_flag = int((efficacy_grasp < 0.10) and (efficacy_q < 0.030))

    out = dict(row)
    out.update({
        "native_intended_side": native_side,
        "language_intended_side": lang_side,
        "native_intended_grasp": native_g,
        "native_intended_q": native_q,
        "native_unintended_grasp": native_unintended_g,
        "native_unintended_q": native_unintended_q,
        "native_selectivity_q": native_selectivity_q,
        "native_selectivity_grasp": native_selectivity_grasp,
        "language_intended_q": lang_q,
        "language_selectivity_q": lang_selectivity_q,
        "efficacy_q": efficacy_q,
        "efficacy_grasp": efficacy_grasp,
        "efficacy_score": efficacy_score,
        "abs_object_bias_q": abs_object_bias_q,
        "abs_object_bias_grasp": abs_object_bias_grasp,
        "collapse_flag": collapse_flag,
        "profile_score_native": profile_score_native,
    })
    return out


def fmt(x, nd=3, signed=False):
    if isinstance(x, str):
        return x
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "NA"
    if signed:
        return f"{x:+.{nd}f}"
    return f"{x:.{nd}f}"


def write_profile_md(rows, path):
    lines = []
    lines.append("| Policy | Task | Condition | n | Intended | P grasp | P q | A grasp | A q | Efficacy | Selectivity | Profile | Collapse |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in sorted(rows, key=lambda x: (x["task"], x["condition"], x["policy"])):
        lines.append(
            f"| {r['policy']} | {r['task']} | {r['condition']} | {r['n']} | {r['native_intended_side']} | "
            f"{fmt(r['primary_grasp'],2)} | {fmt(r['primary_q'],3,True)} | "
            f"{fmt(r['alternative_grasp'],2)} | {fmt(r['alternative_q'],3,True)} | "
            f"{fmt(r['efficacy_score'],3,True)} | {fmt(r['native_selectivity_q'],3,True)} | "
            f"{fmt(r['profile_score_native'],3,True)} | {r['collapse_flag']} |"
        )

    Path(path).write_text("\n".join(lines) + "\n")


def write_rankings_by_task(rows, path):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    lines = []
    lines.append("# Rankings by task")
    lines.append("")

    for task in sorted(by_task):
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| Rank | Policy | Condition | Profile | Efficacy | Selectivity | Collapse |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")

        ranked = sorted(by_task[task], key=lambda x: x["profile_score_native"], reverse=True)
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"| {i} | {r['policy']} | {r['condition']} | "
                f"{fmt(r['profile_score_native'],3,True)} | "
                f"{fmt(r['efficacy_score'],3,True)} | "
                f"{fmt(r['native_selectivity_q'],3,True)} | "
                f"{r['collapse_flag']} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


def write_same_condition_policy_comparison(rows, path):
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["condition"])].append(r)

    lines = []
    lines.append("# Same task-condition policy comparison")
    lines.append("")
    lines.append("| Task | Condition | OpenVLA profile | SmolVLA profile | Δ profile | OpenVLA efficacy | SmolVLA efficacy | Δ efficacy |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    csv_rows = []

    for key in sorted(by):
        rs = by[key]
        d = {r["policy"]: r for r in rs}
        if "openvla_oft" not in d or "smolvla" not in d:
            continue

        o = d["openvla_oft"]
        s = d["smolvla"]

        delta_profile = o["profile_score_native"] - s["profile_score_native"]
        delta_efficacy = o["efficacy_score"] - s["efficacy_score"]

        lines.append(
            f"| {key[0]} | {key[1]} | "
            f"{fmt(o['profile_score_native'],3,True)} | {fmt(s['profile_score_native'],3,True)} | {fmt(delta_profile,3,True)} | "
            f"{fmt(o['efficacy_score'],3,True)} | {fmt(s['efficacy_score'],3,True)} | {fmt(delta_efficacy,3,True)} |"
        )

        csv_rows.append({
            "task": key[0],
            "condition": key[1],
            "openvla_profile_score": o["profile_score_native"],
            "smolvla_profile_score": s["profile_score_native"],
            "delta_profile_openvla_minus_smolvla": delta_profile,
            "openvla_efficacy_score": o["efficacy_score"],
            "smolvla_efficacy_score": s["efficacy_score"],
            "delta_efficacy_openvla_minus_smolvla": delta_efficacy,
        })

    Path(path).write_text("\n".join(lines) + "\n")
    return csv_rows


def write_interpretation(rows, comparison_rows, path):
    n = len(rows)
    n_collapse = sum(int(r["collapse_flag"]) for r in rows)

    # Find strongest profiles.
    top = sorted(rows, key=lambda x: x["profile_score_native"], reverse=True)[:5]
    bottom = sorted(rows, key=lambda x: x["profile_score_native"])[:5]

    lines = []
    lines.append("# Policy profile ranking interpretation")
    lines.append("")
    lines.append(f"Rows: {n}")
    lines.append(f"Collapsed profiles: {n_collapse}")
    lines.append("")
    lines.append("## What this sanity check means")
    lines.append("")
    lines.append(
        "This is an oracle/profile ranking sanity check based on replayed object-level consequences. "
        "It is not yet a held-out closed-loop policy-ranking result. "
        "The goal is to verify whether the existing same-state replay profiles induce sensible rankings before running expensive independent closed-loop evaluation."
    )
    lines.append("")
    lines.append("## Top profiles by native profile score")
    lines.append("")
    lines.append("| Rank | Policy | Task | Condition | Profile | Efficacy | Selectivity |")
    lines.append("|---:|---|---|---|---:|---:|---:|")
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | {r['policy']} | {r['task']} | {r['condition']} | "
            f"{fmt(r['profile_score_native'],3,True)} | {fmt(r['efficacy_score'],3,True)} | {fmt(r['native_selectivity_q'],3,True)} |"
        )

    lines.append("")
    lines.append("## Lowest profiles by native profile score")
    lines.append("")
    lines.append("| Rank | Policy | Task | Condition | Profile | Efficacy | Selectivity |")
    lines.append("|---:|---|---|---|---:|---:|---:|")
    for i, r in enumerate(bottom, 1):
        lines.append(
            f"| {i} | {r['policy']} | {r['task']} | {r['condition']} | "
            f"{fmt(r['profile_score_native'],3,True)} | {fmt(r['efficacy_score'],3,True)} | {fmt(r['native_selectivity_q'],3,True)} |"
        )

    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append(
        "If these rankings look reasonable, the next step is to test whether profile scores computed from a small calibration set predict independent held-out closed-loop performance across multiple policy/configuration candidates. "
        "That requires Spearman correlation, Kendall tau, pairwise ranking accuracy, and top-1 regret against a closed-loop evaluation set."
    )
    lines.append("")

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task8_csv", required=True)
    ap.add_argument("--multitask_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += normalize_task8_rows(read_csv(args.task8_csv))
    rows += normalize_multitask_rows(read_csv(args.multitask_csv))
    rows = [add_profile_scores(r) for r in rows]

    profile_csv = out_dir / "policy_profile_scores.csv"
    profile_md = out_dir / "policy_profile_scores.md"
    rankings_md = out_dir / "rankings_by_task.md"
    comparison_md = out_dir / "same_condition_policy_comparison.md"
    comparison_csv = out_dir / "same_condition_policy_comparison.csv"
    interp_md = out_dir / "policy_profile_ranking_interpretation.md"

    write_csv(rows, profile_csv)
    write_profile_md(rows, profile_md)
    write_rankings_by_task(rows, rankings_md)
    comparison_rows = write_same_condition_policy_comparison(rows, comparison_md)
    write_csv(comparison_rows, comparison_csv)
    write_interpretation(rows, comparison_rows, interp_md)

    print("[saved]", profile_csv)
    print("[saved]", profile_md)
    print("[saved]", rankings_md)
    print("[saved]", comparison_md)
    print("[saved]", comparison_csv)
    print("[saved]", interp_md)
    print()
    print(profile_md.read_text())
    print()
    print(rankings_md.read_text())
    print()
    print(comparison_md.read_text())
    print()
    print(interp_md.read_text())


if __name__ == "__main__":
    main()