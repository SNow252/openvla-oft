#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math


def try_float(x):
    try:
        return float(x)
    except Exception:
        return x


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def pos_rate(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(x > 0 for x in xs) / len(xs) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--csv_name", default="summary.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv_name

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append({k: try_float(v) for k, v in r.items()})

    print("=" * 100)
    print("DATA_DIR:", data_dir)
    print("CSV:", csv_path)
    print("Samples:", len(rows))
    print("=" * 100)

    reach_cols = sorted([c for c in fieldnames if c.startswith("reached_source_eps_")])
    tau_cols = sorted([c for c in fieldnames if c.startswith("tau_source_eps_")])

    print("\nDetected reach columns:")
    for c in reach_cols:
        print(" ", c)

    print("\nDetected tau columns:")
    for c in tau_cols:
        print(" ", c)

    interp = [r for r in rows if r.get("candidate_family") == "interp_bowl1_bowl2"]
    print("\nInterp samples:", len(interp))

    for reach_col in reach_cols:
        eps_key = reach_col.replace("reached_source_eps_", "")
        tau_col = f"tau_source_eps_{eps_key}"

        if tau_col not in fieldnames:
            print(f"\n[skip] {reach_col}, missing {tau_col}")
            continue

        print("\n" + "=" * 100)
        print(f"EPS = {eps_key}")
        print("-" * 100)

        by_alpha = defaultdict(list)
        for r in interp:
            by_alpha[r["alpha"]].append(r)

        for alpha in sorted(by_alpha):
            rs = by_alpha[alpha]
            reached = [r[reach_col] for r in rs]
            tau = [r[tau_col] for r in rs]
            min_d = [r.get("min_dist_to_bowl2", r.get("min_dist_to_bowl2_rebuilt", float("nan"))) for r in rs]
            adv = [r.get("source_advantage_bowl2_over_bowl1", float("nan")) for r in rs]

            print(
                f"alpha={alpha:+.2f} "
                f"n={len(rs):4d} "
                f"reach_rate={mean(reached):.3f} "
                f"mean_tau={mean(tau):.2f} "
                f"mean_min_d={mean(min_d):.4f} "
                f"mean_adv={mean(adv):+.4f} "
                f"adv_pos_rate={pos_rate(adv):.3f}"
            )

    print("\n" + "=" * 100)
    print("Family summary")
    print("-" * 100)

    by_family = defaultdict(list)
    for r in rows:
        by_family[r.get("candidate_family", "UNKNOWN")].append(r)

    primary_reach = "reached_source_eps_0p18" if "reached_source_eps_0p18" in fieldnames else (reach_cols[0] if reach_cols else None)
    primary_tau = primary_reach.replace("reached_source", "tau_source") if primary_reach else None

    for fam in sorted(by_family):
        rs = by_family[fam]
        adv = [r.get("source_advantage_bowl2_over_bowl1", float("nan")) for r in rs]
        min_d = [r.get("min_dist_to_bowl2", r.get("min_dist_to_bowl2_rebuilt", float("nan"))) for r in rs]

        msg = (
            f"{fam:24s} "
            f"n={len(rs):4d} "
            f"mean_adv={mean(adv):+.4f} "
            f"adv_pos_rate={pos_rate(adv):.3f} "
            f"mean_min_d={mean(min_d):.4f}"
        )

        if primary_reach and primary_tau:
            msg += (
                f" "
                f"{primary_reach}={mean([r[primary_reach] for r in rs]):.3f} "
                f"{primary_tau}={mean([r[primary_tau] for r in rs]):.2f}"
            )

        print(msg)


if __name__ == "__main__":
    main()