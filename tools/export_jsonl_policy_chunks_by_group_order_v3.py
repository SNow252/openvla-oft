#!/usr/bin/env python3
"""
Export old JSONL OpenVLA dumps into standard v3 policy chunk format by filename group order.

Observed dump layout:
  80 jsonl files = 8 groups × 10 trajectories.

Filename:
  traj_<timestamp>_<group_token>_<hash>.jsonl

Example:
  group_token=483424, files 0-9
  group_token=484491, files 10-19
  ...

Mapping:
  each group_token = one dump condition
  within each group, sorted file order maps to init_state_idx 0..9

This is safer than obs matching when the JSONL lacks explicit init_state_idx.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def safe_name(x: str) -> str:
    x = str(x)
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_") or "unnamed"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["_line_idx"] = i
                    records.append(obj)
            except Exception:
                pass
    return records


def get_group_token(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return "unknown_group"


def get_hash_token(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 4:
        return safe_name(parts[3])
    return safe_name(path.stem)[-8:]


def get_line0_meta(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    rec0 = records[0]
    if rec0.get("type") == "metadata":
        return rec0
    return rec0


def extract_actions(records: List[Dict[str, Any]], action_field: str, fallback_field: str) -> np.ndarray:
    actions = []

    for rec in records:
        value = rec.get(action_field, None)
        if value is None and fallback_field:
            value = rec.get(fallback_field, None)

        if value is None:
            continue

        try:
            arr = np.asarray(value, dtype=np.float32)
        except Exception:
            continue

        if arr.shape == (7,):
            actions.append(arr)

    if not actions:
        return np.zeros((0, 7), dtype=np.float32)

    return np.stack(actions, axis=0).astype(np.float32)


def clip_actions(actions: np.ndarray, start_step: int, max_steps: int) -> np.ndarray:
    start = max(0, int(start_step))
    end = actions.shape[0]
    if max_steps > 0:
        end = min(end, start + int(max_steps))
    return actions[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--policy_name", default="openvla_oft")
    parser.add_argument("--chunk_name", default="mismatch_v2_processed")

    parser.add_argument("--action_field", default="processed_action")
    parser.add_argument("--fallback_action_field", default="raw_action_before_process")

    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_inits_per_group", type=int, default=10)

    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=140)

    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dump_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No jsonl files found in {dump_dir}")

    grouped = defaultdict(list)
    global_order = {p: i for i, p in enumerate(files)}

    for p in files:
        grouped[get_group_token(p)].append(p)

    # Preserve group order by first occurrence in sorted filename order.
    group_tokens = sorted(grouped.keys(), key=lambda g: min(global_order[p] for p in grouped[g]))

    policy_name = safe_name(args.policy_name)
    base_chunk_name = safe_name(args.chunk_name)

    print("=" * 100)
    print("[dump_dir]", dump_dir)
    print("[out_dir]", out_dir)
    print("[num files]", len(files))
    print("[num groups]", len(group_tokens))
    print("[groups]")
    for gi, g in enumerate(group_tokens):
        print(f"  group_idx={gi:02d} token={g} n={len(grouped[g])}")
    print("=" * 100)

    rows = []
    missing_or_empty = []

    for group_idx, group_token in enumerate(group_tokens):
        group_files = sorted(grouped[group_token])

        if len(group_files) != args.num_inits_per_group:
            print(
                f"[warning] group {group_token} has {len(group_files)} files, "
                f"expected {args.num_inits_per_group}"
            )

        for local_idx, path in enumerate(group_files):
            init_idx = args.start_init_state_idx + local_idx

            records = read_jsonl(path)
            meta = get_line0_meta(records)

            actions = extract_actions(
                records=records,
                action_field=args.action_field,
                fallback_field=args.fallback_action_field,
            )
            actions = clip_actions(actions, args.start_step, args.max_steps)

            if actions.shape[0] == 0:
                missing_or_empty.append(str(path))
                continue

            dump_condition = str(meta.get("dump_condition", ""))
            custom_language = str(meta.get("custom_language", ""))
            model_family = str(meta.get("model_family", ""))
            num_open_loop_steps = meta.get("num_open_loop_steps", "")

            condition_name = safe_name(dump_condition) if dump_condition else f"group{group_idx:02d}_{group_token}"
            hash_token = get_hash_token(path)

            final_chunk_name = safe_name(
                f"{policy_name}_{base_chunk_name}_{condition_name}_{hash_token}"
            )

            seed_dir = out_dir / f"init_{init_idx:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            out_path = seed_dir / f"{final_chunk_name}.npz"

            np.savez_compressed(
                out_path,
                actions=actions.astype(np.float32),
                policy_name=policy_name,
                chunk_name=final_chunk_name,
                language=custom_language,
                condition=dump_condition,
                source_path=str(path),
                init_state_idx=np.asarray(init_idx, dtype=np.int32),
                inferred_by="group_order",
                group_idx=np.asarray(group_idx, dtype=np.int32),
                group_token=group_token,
                local_idx=np.asarray(local_idx, dtype=np.int32),
                start_step=np.asarray(args.start_step, dtype=np.int32),
                used_num_steps=np.asarray(actions.shape[0], dtype=np.int32),
                action_field=args.action_field,
                fallback_action_field=args.fallback_action_field,
                model_family=model_family,
                num_open_loop_steps=str(num_open_loop_steps),
            )

            row = {
                "group_idx": group_idx,
                "group_token": group_token,
                "local_idx": local_idx,
                "init_state_idx": init_idx,
                "steps": int(actions.shape[0]),
                "condition": dump_condition,
                "language": custom_language,
                "model_family": model_family,
                "num_open_loop_steps": num_open_loop_steps,
                "chunk_name": final_chunk_name,
                "source_path": str(path),
                "out_path": str(out_path),
            }
            rows.append(row)

            print(
                f"[export] group={group_idx:02d} token={group_token} "
                f"local={local_idx:02d} init={init_idx:03d} "
                f"steps={actions.shape[0]:03d} condition={condition_name} -> {out_path.name}"
            )

    manifest_json = out_dir / f"manifest_{policy_name}_{base_chunk_name}_group_order.json"
    with open(manifest_json, "w") as f:
        json.dump(
            {
                "dump_dir": str(dump_dir),
                "out_dir": str(out_dir),
                "args": vars(args),
                "groups": {
                    g: [str(p) for p in sorted(grouped[g])]
                    for g in group_tokens
                },
                "exported": rows,
                "missing_or_empty": missing_or_empty,
            },
            f,
            indent=2,
        )

    manifest_csv = out_dir / f"manifest_{policy_name}_{base_chunk_name}_group_order.csv"
    fields = [
        "group_idx",
        "group_token",
        "local_idx",
        "init_state_idx",
        "steps",
        "condition",
        "language",
        "model_family",
        "num_open_loop_steps",
        "chunk_name",
        "source_path",
        "out_path",
    ]

    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    print("\n" + "=" * 100)
    print("Exported:", len(rows))
    print("Missing/empty:", len(missing_or_empty))
    print("Manifest JSON:", manifest_json)
    print("Manifest CSV:", manifest_csv)
    print("=" * 100)

    if missing_or_empty:
        print("\nMissing/empty examples:")
        for x in missing_or_empty[:20]:
            print(" ", x)


if __name__ == "__main__":
    main()