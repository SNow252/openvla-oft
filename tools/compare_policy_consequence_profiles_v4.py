#!/usr/bin/env python3
"""
Compare OpenVLA vs SmolVLA object-level consequence profiles.

Input:
  each replay dir should contain:
    summary_dual_source.csv

Output:
  combined_policy_consequence_table.csv
  combined_policy_consequence_table.md
  policy_delta_smolvla_minus_openvla.csv

Metrics:
  official/replay:
    mean sum_reward
    mean any_done
  object consequence:
    bowl1/bowl2 grasp
    bowl1/bowl2 lifted
    bowl1/bowl2 quality
    bowl2 - bowl1 quality
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITION_ORDER = [
    "task8_original",
    "task8_empty_language",
    "task8_nonsense_language",
    "task8_unrelated_language",
    "task8_wrong_source_language_only",
    "task8_wrong_target_language_only",
    "task8_wrong_source_wrong_target_language_only",
    "task1_native_next_to_ramekin",
]


METRICS = [
    "sum_reward",
    "any_done",
    "bowl1_grasp_success_proxy",
    "bowl1_source_lifted",
    "bowl1_grasp_quality_score",
    "bowl2_grasp_success_proxy",
    "bowl2_source_lifted",
    "bowl2_grasp_quality_score",
    "bowl2_minus_bowl1_grasp_success_proxy",
    "bowl2_minus_bowl1_quality",
]


OUT_FIELDS = [
    "policy",
    "condition",
    "n",
    "reward",
    "done",
    "b1_grasp",
    "b1_lift",
    "b1_q",
    "b2_grasp",
    "b2_lift",
    "b2_q",
    "b2_minus_b1_grasp",
    "b2_minus_b1_q",
]


def try_float(x):
    try:
        return float(x)
    except Exception:
        return x


def read_rows(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {k: try_float(v) for k, v in r.items()}
            rows.append(out)
    return rows


def vals(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, float) and not math.isnan(v):
            out.append(v)
    return out


def mean(rows, key):
    xs = vals(rows, key)
    return float(np.mean(xs)) if xs else float("nan")


def summarize_policy(policy_name: str, replay_dir: Path):
    path = replay_dir / "summary_dual_source.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary_dual_source.csv: {path}")

    rows = read_rows(path)

    by_cond = defaultdict(list)
    for r in rows:
        cond = str(r.get("condition", "unknown"))
        by_cond[cond].append(r)

    summary = []
    for cond, rs in by_cond.items():
        summary.append(
            {
                "policy": policy_name,
                "condition": cond,
                "n": len(rs),
                "reward": mean(rs, "sum_reward"),
                "done": mean(rs, "any_done"),
                "b1_grasp": mean(rs, "bowl1_grasp_success_proxy"),
                "b1_lift": mean(rs, "bowl1_source_lifted"),
                "b1_q": mean(rs, "bowl1_grasp_quality_score"),
                "b2_grasp": mean(rs, "bowl2_grasp_success_proxy"),
                "b2_lift": mean(rs, "bowl2_source_lifted"),
                "b2_q": mean(rs, "bowl2_grasp_quality_score"),
                "b2_minus_b1_grasp": mean(rs, "bowl2_minus_bowl1_grasp_success_proxy"),
                "b2_minus_b1_q": mean(rs, "bowl2_minus_bowl1_quality"),
            }
        )

    return summary


def cond_key(row):
    cond = row["condition"]
    try:
        ci = CONDITION_ORDER.index(cond)
    except ValueError:
        ci = 999
    return (ci, row["policy"], cond)


def write_csv(rows, path: Path, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def fmt(x, nd=3):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(rows, path: Path):
    lines = []
    lines.append("| Policy | Condition | n | Reward | Done | B1 grasp | B1 lift | B1 q | B2 grasp | B2 lift | B2 q | B2-B1 q |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["policy"]),
                    str(r["condition"]),
                    str(r["n"]),
                    fmt(r["reward"], 2),
                    fmt(r["done"], 2),
                    fmt(r["b1_grasp"], 2),
                    fmt(r["b1_lift"], 2),
                    fmt(r["b1_q"], 3),
                    fmt(r["b2_grasp"], 2),
                    fmt(r["b2_lift"], 2),
                    fmt(r["b2_q"], 3),
                    fmt(r["b2_minus_b1_q"], 3),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n")


def build_delta_table(rows):
    by_policy_cond = {(r["policy"], r["condition"]): r for r in rows}
    deltas = []

    shared_conds = sorted(
        {
            cond
            for _, cond in by_policy_cond.keys()
            if ("openvla_oft", cond) in by_policy_cond and ("smolvla", cond) in by_policy_cond
        },
        key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else 999,
    )

    delta_metrics = [
        "reward",
        "done",
        "b1_grasp",
        "b1_lift",
        "b1_q",
        "b2_grasp",
        "b2_lift",
        "b2_q",
        "b2_minus_b1_grasp",
        "b2_minus_b1_q",
    ]

    for cond in shared_conds:
        o = by_policy_cond[("openvla_oft", cond)]
        s = by_policy_cond[("smolvla", cond)]
        row = {"condition": cond}
        for m in delta_metrics:
            row[f"smolvla_minus_openvla_{m}"] = s[m] - o[m]
        deltas.append(row)

    return deltas


def print_key_findings(rows):
    by = {(r["policy"], r["condition"]): r for r in rows}

    def get(policy, cond, key):
        r = by.get((policy, cond))
        return None if r is None else r.get(key)

    print("\n[Key findings]")
    for cond in CONDITION_ORDER:
        if ("openvla_oft", cond) in by and ("smolvla", cond) in by:
            o_b1 = get("openvla_oft", cond, "b1_grasp")
            o_b2 = get("openvla_oft", cond, "b2_grasp")
            s_b1 = get("smolvla", cond, "b1_grasp")
            s_b2 = get("smolvla", cond, "b2_grasp")
            print(
                f"{cond:48s} | "
                f"OpenVLA B1/B2={o_b1:.2f}/{o_b2:.2f} | "
                f"SmolVLA B1/B2={s_b1:.2f}/{s_b2:.2f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla_dir", required=True)
    parser.add_argument("--smolvla_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(summarize_policy("openvla_oft", Path(args.openvla_dir)))
    rows.extend(summarize_policy("smolvla", Path(args.smolvla_dir)))
    rows = sorted(rows, key=cond_key)

    combined_csv = out_dir / "combined_policy_consequence_table.csv"
    combined_md = out_dir / "combined_policy_consequence_table.md"
    delta_csv = out_dir / "policy_delta_smolvla_minus_openvla.csv"

    write_csv(rows, combined_csv, OUT_FIELDS)
    write_markdown(rows, combined_md)

    deltas = build_delta_table(rows)
    if deltas:
        delta_fields = list(deltas[0].keys())
        write_csv(deltas, delta_csv, delta_fields)

    print("[saved]", combined_csv)
    print("[saved]", combined_md)
    print("[saved]", delta_csv)

    print("\n[Combined Markdown Table]")
    print(combined_md.read_text())

    print_key_findings(rows)


if __name__ == "__main__":
    main()