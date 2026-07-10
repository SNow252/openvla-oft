#!/usr/bin/env python3
"""
Summarize v3 learned and kinematic baselines into one paper-style table.

Inputs expected in --data_dir:
  Learned baseline JSONs:
    grasp_macro_only_success_bowl2_grasp_success_proxy_state_holdout.json
    grasp_macro_only_lifted_bowl2_source_lifted_state_holdout.json
    grasp_macro_only_quality_bowl2_grasp_quality_score_state_holdout.json

  Kinematic baseline JSON:
    v3_kinematic_macro_only_state_holdout.json

Outputs:
  v3_baseline_summary_table.csv
  v3_baseline_summary_table.md
  v3_baseline_summary_notes.txt
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def find_one(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {data_dir / pattern}")
    if len(matches) > 1:
        print(f"[warning] multiple matches for {pattern}; using {matches[0]}")
    return matches[0]


def get_model_metrics(learned_json: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    for m in learned_json["metrics"]:
        if m["model"] == model_name:
            return m
    raise KeyError(f"Model {model_name} not found in learned metrics.")


def get_kin_metric(
    kin_json: Dict[str, Any],
    target: str,
    mode: str,
    group: str,
) -> Dict[str, Any]:
    return kin_json["targets"][target]["modes"][mode][group]


def safe_get(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(d[key])
    except Exception:
        return default


def fmt(x: float, ndigits: int = 3) -> str:
    try:
        if x != x:
            return "nan"
        return f"{x:.{ndigits}f}"
    except Exception:
        return "nan"


def add_learned_row(
    rows: List[Dict[str, Any]],
    row_name: str,
    model_name: str,
    learned_success: Dict[str, Any],
    learned_lifted: Dict[str, Any],
    learned_quality: Dict[str, Any],
) -> None:
    ms = get_model_metrics(learned_success, model_name)
    ml = get_model_metrics(learned_lifted, model_name)
    mq = get_model_metrics(learned_quality, model_name)

    rows.append(
        {
            "group": "learned",
            "method": row_name,
            "grasp_auc": safe_get(ms, "test_auc_binary_col"),
            "grasp_pairwise": safe_get(ms, "within_seed_pairwise_acc"),
            "grasp_binpair": safe_get(ms, "within_seed_binpair_acc"),
            "lifted_auc": safe_get(ml, "test_auc_binary_col"),
            "lifted_pairwise": safe_get(ml, "within_seed_pairwise_acc"),
            "lifted_binpair": safe_get(ml, "within_seed_binpair_acc"),
            "quality_r2": safe_get(mq, "test_r2_score"),
            "quality_mae": safe_get(mq, "test_mae_raw"),
            "quality_pairwise": safe_get(mq, "within_seed_pairwise_acc"),
            "quality_binpair": safe_get(mq, "within_seed_binpair_acc"),
        }
    )


def add_kin_row(
    rows: List[Dict[str, Any]],
    row_name: str,
    mode: str,
    group: str,
    kin_json: Dict[str, Any],
) -> None:
    ms = get_kin_metric(
        kin_json,
        target="bowl2_grasp_success_proxy",
        mode=mode,
        group=group,
    )
    ml = get_kin_metric(
        kin_json,
        target="bowl2_source_lifted",
        mode=mode,
        group=group,
    )
    mq = get_kin_metric(
        kin_json,
        target="bowl2_grasp_quality_score",
        mode=mode,
        group=group,
    )

    rows.append(
        {
            "group": "kinematic",
            "method": row_name,
            "grasp_auc": safe_get(ms, "auc"),
            "grasp_pairwise": safe_get(ms, "within_seed_pairwise_acc"),
            "grasp_binpair": safe_get(ms, "within_seed_binpair_acc"),
            "lifted_auc": safe_get(ml, "auc"),
            "lifted_pairwise": safe_get(ml, "within_seed_pairwise_acc"),
            "lifted_binpair": safe_get(ml, "within_seed_binpair_acc"),
            "quality_r2": safe_get(mq, "r2"),
            "quality_mae": safe_get(mq, "mae"),
            "quality_pairwise": safe_get(mq, "within_seed_pairwise_acc"),
            "quality_binpair": safe_get(mq, "within_seed_binpair_acc"),
        }
    )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "group",
        "method",
        "grasp_auc",
        "grasp_pairwise",
        "grasp_binpair",
        "lifted_auc",
        "lifted_pairwise",
        "lifted_binpair",
        "quality_r2",
        "quality_mae",
        "quality_pairwise",
        "quality_binpair",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    headers = [
        "Method",
        "Grasp AUC",
        "Grasp Pair",
        "Lift AUC",
        "Lift Pair",
        "Quality R2",
        "Quality Pair",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        method = r["method"]
        if r["group"] == "learned":
            method = f"**{method}**"

        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    fmt(r["grasp_auc"]),
                    fmt(r["grasp_pairwise"]),
                    fmt(r["lifted_auc"]),
                    fmt(r["lifted_pairwise"]),
                    fmt(r["quality_r2"]),
                    fmt(r["quality_pairwise"]),
                ]
            )
            + " |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def best_value(rows: List[Dict[str, Any]], metric: str, group_filter: str | None = None) -> tuple[str, float]:
    best_method = ""
    best_val = float("-inf")

    for r in rows:
        if group_filter is not None and r["group"] != group_filter:
            continue
        val = r.get(metric, float("nan"))
        try:
            val = float(val)
        except Exception:
            continue
        if val != val:
            continue
        if val > best_val:
            best_val = val
            best_method = r["method"]

    return best_method, best_val


def write_notes(rows: List[Dict[str, Any]], path: Path) -> None:
    metrics = [
        ("grasp_auc", "grasp success AUC"),
        ("grasp_pairwise", "grasp success within-seed pairwise"),
        ("lifted_auc", "lifted AUC"),
        ("lifted_pairwise", "lifted within-seed pairwise"),
        ("quality_r2", "grasp quality R2"),
        ("quality_pairwise", "grasp quality within-seed pairwise"),
    ]

    lines = []
    lines.append("V3 baseline summary notes")
    lines.append("=" * 80)

    for metric, label in metrics:
        best_all = best_value(rows, metric, group_filter=None)
        best_learned = best_value(rows, metric, group_filter="learned")
        best_kin = best_value(rows, metric, group_filter="kinematic")

        lines.append("")
        lines.append(label)
        lines.append(f"  best overall:   {best_all[0]} = {fmt(best_all[1])}")
        lines.append(f"  best learned:   {best_learned[0]} = {fmt(best_learned[1])}")
        lines.append(f"  best kinematic: {best_kin[0]} = {fmt(best_kin[1])}")

        if best_learned[1] == best_learned[1] and best_kin[1] == best_kin[1]:
            lines.append(f"  learned - kinematic gap: {fmt(best_learned[1] - best_kin[1])}")

    lines.append("")
    lines.append("Suggested interpretation:")
    lines.append(
        "  EEF kinematics is a strong but insufficient baseline. It explains part of the "
        "object-level outcome, but action-conditioned learned baselines generally provide "
        "stronger within-state ranking for grasp/lift consequences."
    )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_prefix", default="v3_baseline_summary_table")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    success_path = find_one(
        data_dir,
        "grasp_macro_only_success_bowl2_grasp_success_proxy_state_holdout.json",
    )
    lifted_path = find_one(
        data_dir,
        "grasp_macro_only_lifted_bowl2_source_lifted_state_holdout.json",
    )
    quality_path = find_one(
        data_dir,
        "grasp_macro_only_quality_bowl2_grasp_quality_score_state_holdout.json",
    )
    kin_path = find_one(
        data_dir,
        "v3_kinematic_macro_only_state_holdout.json",
    )

    print("[success]", success_path)
    print("[lifted]", lifted_path)
    print("[quality]", quality_path)
    print("[kinematic]", kin_path)

    learned_success = load_json(success_path)
    learned_lifted = load_json(lifted_path)
    learned_quality = load_json(quality_path)
    kin_json = load_json(kin_path)

    rows: List[Dict[str, Any]] = []

    # Learned baselines.
    add_learned_row(
        rows,
        row_name="State-only",
        model_name="state_only",
        learned_success=learned_success,
        learned_lifted=learned_lifted,
        learned_quality=learned_quality,
    )
    add_learned_row(
        rows,
        row_name="Action-only",
        model_name="action_only",
        learned_success=learned_success,
        learned_lifted=learned_lifted,
        learned_quality=learned_quality,
    )
    add_learned_row(
        rows,
        row_name="State+Action",
        model_name="state_action",
        learned_success=learned_success,
        learned_lifted=learned_lifted,
        learned_quality=learned_quality,
    )

    # Kinematic baselines.
    add_kin_row(
        rows,
        row_name="EEF min-dist oracle",
        mode="actual_eef_oracle",
        group="min_dist",
        kin_json=kin_json,
    )
    add_kin_row(
        rows,
        row_name="Kinematic fit-scalar EEF-full",
        mode="fit_scalar",
        group="eef_full",
        kin_json=kin_json,
    )
    add_kin_row(
        rows,
        row_name="Kinematic fit-xyz EEF-full",
        mode="fit_xyz",
        group="eef_full",
        kin_json=kin_json,
    )
    add_kin_row(
        rows,
        row_name="Actual EEF oracle EEF-full",
        mode="actual_eef_oracle",
        group="eef_full",
        kin_json=kin_json,
    )

    csv_path = data_dir / f"{args.out_prefix}.csv"
    md_path = data_dir / f"{args.out_prefix}.md"
    notes_path = data_dir / f"{args.out_prefix}_notes.txt"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_notes(rows, notes_path)

    print("\n" + "=" * 100)
    print("[saved csv]", csv_path)
    print("[saved md]", md_path)
    print("[saved notes]", notes_path)
    print("=" * 100)

    print("\nMarkdown table:")
    with open(md_path) as f:
        print(f.read())

    print("\nNotes:")
    with open(notes_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()