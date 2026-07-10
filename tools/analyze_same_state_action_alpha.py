import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_same_state_action_alpha.py DATA_DIR")
        return

    data_dir = Path(sys.argv[1])
    csv_path = data_dir / "summary.csv"

    rows = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for r in reader:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except Exception:
                    pass

            rows.append(r)

    print("=" * 80)
    print("Data:", data_dir)
    print("Samples:", len(rows))
    print("=" * 80)

    # interpolation candidates
    interp = [
        r for r in rows
        if r["candidate_family"] == "interp_bowl1_bowl2"
    ]

    alpha_dict = defaultdict(list)

    for r in interp:
        alpha_dict[r["alpha"]].append(
            r["source_advantage_bowl2_over_bowl1"]
        )

    print("\nMean source_advantage by alpha:")
    print("-" * 80)

    for alpha in sorted(alpha_dict.keys()):
        vals = alpha_dict[alpha]

        mean_adv = sum(vals) / len(vals)
        pos_rate = sum(v > 0 for v in vals) / len(vals)

        print(
            f"alpha={alpha:+.2f} "
            f"n={len(vals):4d} "
            f"mean_adv={mean_adv:+.5f} "
            f"pos_rate={pos_rate:.3f}"
        )


    family_dict = defaultdict(list)

    for r in rows:
        family_dict[r["candidate_family"]].append(
            r["source_advantage_bowl2_over_bowl1"]
        )


    print("\nMean source_advantage by family:")
    print("-" * 80)

    for fam in sorted(family_dict.keys()):
        vals = family_dict[fam]

        print(
            f"{fam:25s} "
            f"n={len(vals):4d} "
            f"mean_adv={sum(vals)/len(vals):+.5f} "
            f"pos_rate={sum(v>0 for v in vals)/len(vals):.3f}"
        )


if __name__ == "__main__":
    main()