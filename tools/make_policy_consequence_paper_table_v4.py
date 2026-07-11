#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

ORDER = [
    "task8_original",
    "task8_empty_language",
    "task8_nonsense_language",
    "task8_unrelated_language",
    "task8_wrong_source_language_only",
    "task8_wrong_target_language_only",
    "task8_wrong_source_wrong_target_language_only",
    "task1_native_next_to_ramekin",
]

KEEP = [
    "policy",
    "condition",
    "n",
    "b1_grasp",
    "b1_q",
    "b2_grasp",
    "b2_q",
    "b2_minus_b1_q",
]

def f3(x):
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:+.3f}"
    except Exception:
        return str(x)

def f2(x):
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:.2f}"
    except Exception:
        return str(x)

def cond_key(row):
    cond = row["condition"]
    try:
        ci = ORDER.index(cond)
    except ValueError:
        ci = 999
    pi = 0 if row["policy"] == "openvla_oft" else 1
    return (ci, pi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.combined_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    rows = sorted(rows, key=cond_key)

    csv_out = out_dir / "paper_policy_consequence_table.csv"
    md_out = out_dir / "paper_policy_consequence_table.md"

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in KEEP})

    lines = []
    lines.append("| Policy | Condition | n | B1 grasp | B1 q | B2 grasp | B2 q | B2-B1 q |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['condition']} | {r['n']} | "
            f"{f2(r['b1_grasp'])} | {f3(r['b1_q'])} | "
            f"{f2(r['b2_grasp'])} | {f3(r['b2_q'])} | {f3(r['b2_minus_b1_q'])} |"
        )

    md_out.write_text("\n".join(lines) + "\n")

    print("[saved]", csv_out)
    print("[saved]", md_out)
    print()
    print(md_out.read_text())

if __name__ == "__main__":
    main()#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

ORDER = [
    "task8_original",
    "task8_empty_language",
    "task8_nonsense_language",
    "task8_unrelated_language",
    "task8_wrong_source_language_only",
    "task8_wrong_target_language_only",
    "task8_wrong_source_wrong_target_language_only",
    "task1_native_next_to_ramekin",
]

KEEP = [
    "policy",
    "condition",
    "n",
    "b1_grasp",
    "b1_q",
    "b2_grasp",
    "b2_q",
    "b2_minus_b1_q",
]

def f3(x):
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:+.3f}"
    except Exception:
        return str(x)

def f2(x):
    try:
        x = float(x)
        if math.isnan(x):
            return "nan"
        return f"{x:.2f}"
    except Exception:
        return str(x)

def cond_key(row):
    cond = row["condition"]
    try:
        ci = ORDER.index(cond)
    except ValueError:
        ci = 999
    pi = 0 if row["policy"] == "openvla_oft" else 1
    return (ci, pi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.combined_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    rows = sorted(rows, key=cond_key)

    csv_out = out_dir / "paper_policy_consequence_table.csv"
    md_out = out_dir / "paper_policy_consequence_table.md"

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in KEEP})

    lines = []
    lines.append("| Policy | Condition | n | B1 grasp | B1 q | B2 grasp | B2 q | B2-B1 q |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['condition']} | {r['n']} | "
            f"{f2(r['b1_grasp'])} | {f3(r['b1_q'])} | "
            f"{f2(r['b2_grasp'])} | {f3(r['b2_q'])} | {f3(r['b2_minus_b1_q'])} |"
        )

    md_out.write_text("\n".join(lines) + "\n")

    print("[saved]", csv_out)
    print("[saved]", md_out)
    print()
    print(md_out.read_text())

if __name__ == "__main__":
    main()