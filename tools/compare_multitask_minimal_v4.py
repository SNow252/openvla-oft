#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def read_table(path, policy, task_key):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {
                "policy": policy,
                "task": task_key,
            }
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
            rows.append(out)
    return rows


def fmt(x, nd=3):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:+.{nd}f}" if nd == 3 else f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvla_task3_csv", required=True)
    ap.add_argument("--openvla_task9_csv", required=True)
    ap.add_argument("--smolvla_task3_csv", required=True)
    ap.add_argument("--smolvla_task9_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += read_table(args.openvla_task3_csv, "openvla_oft", "task3")
    rows += read_table(args.openvla_task9_csv, "openvla_oft", "task9")
    rows += read_table(args.smolvla_task3_csv, "smolvla", "task3")
    rows += read_table(args.smolvla_task9_csv, "smolvla", "task9")

    csv_path = out_dir / "multitask_minimal_policy_consequence_table.csv"
    md_path = out_dir / "multitask_minimal_policy_consequence_table.md"
    interp_path = out_dir / "multitask_minimal_interpretation.md"

    fields = [
        "policy",
        "task",
        "condition",
        "n",
        "primary_grasp",
        "primary_q",
        "alternative_grasp",
        "alternative_q",
        "alternative_minus_primary_q",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = []
    lines.append("| Policy | Task | Condition | n | Primary grasp | Primary q | Alt grasp | Alt q | Alt-Primary q |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['task']} | {r['condition']} | {int(r['n'])} | "
            f"{r['primary_grasp']:.2f} | {r['primary_q']:+.3f} | "
            f"{r['alternative_grasp']:.2f} | {r['alternative_q']:+.3f} | "
            f"{r['alternative_minus_primary_q']:+.3f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    interp = """# Multi-task minimal interpretation

The multi-task minimal replay results show that same-state interventional replay is not specific to task8.

For task3, OpenVLA-OFT preserves a strong primary-object manipulation template even under empty and nonsense language, whereas SmolVLA collapses under the same language corruptions.

For task9, both policies execute the primary-object manipulation under the original instruction. Under language corruption, SmolVLA mostly collapses to ineffective actions, while OpenVLA-OFT loses the primary-object template and shows partial alternative-object consequences.

Together with the task8 n=20 results, this suggests that interventional replay exposes policy-, task-, and language-condition-specific action consequence profiles. OpenVLA-OFT often preserves strong task-conditioned templates, but this robustness is task-dependent. SmolVLA is more language-sensitive and tends to collapse under corrupted language rather than maintaining a stable default template.
"""
    interp_path.write_text(interp)

    print("[saved]", csv_path)
    print("[saved]", md_path)
    print("[saved]", interp_path)
    print()
    print(md_path.read_text())
    print()
    print(interp)


if __name__ == "__main__":
    main()
