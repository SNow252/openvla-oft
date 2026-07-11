#!/usr/bin/env python3
"""
Convert OpenVLA / SmolVLA JSONL action dumps into standard v3 policy chunk format.

Input:
  DUMP_DIR/*.jsonl

Expected useful fields:
  processed_action: [7]                # preferred, executed env action
  raw_action_before_process: [7]
  raw_actions_chunk: [T, 7]

Output layout:
  OUT_DIR/init_000/openvla_oft_<chunk_name>_<fileid>.npz
  OUT_DIR/init_001/openvla_oft_<chunk_name>_<fileid>.npz
  ...

Each output npz:
  actions: [T, 7]
  policy_name
  chunk_name
  language
  source_path
  init_state_idx

The script tries to auto-detect init_state_idx from JSONL metadata.
If auto-detection fails, it reports missing files instead of silently assigning wrong init states.
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


INIT_KEYS = [
    "init_state_idx",
    "initial_state_idx",
    "init_idx",
    "state_idx",
    "libero_init_state_idx",
    "episode_init_state_idx",
]

LANGUAGE_KEYS = [
    "language",
    "instruction",
    "task_language",
    "prompt",
    "query_language",
]

CONDITION_KEYS = [
    "condition",
    "condition_name",
    "variant",
    "eval_condition",
    "prompt_name",
    "tag",
]


def safe_name(x: str) -> str:
    x = str(x)
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_") or "unnamed"


def try_float_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def is_action_vec(x: Any) -> bool:
    arr = try_float_array(x)
    return arr is not None and arr.shape == (7,)


def is_action_chunk(x: Any) -> bool:
    arr = try_float_array(x)
    return arr is not None and arr.ndim == 2 and arr.shape[1] == 7 and arr.shape[0] > 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["_line_idx"] = line_idx
                    records.append(obj)
            except Exception:
                continue
    return records


def iter_nested(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from iter_nested(v, p)
    elif isinstance(obj, list):
        # Avoid huge recursion through numeric arrays.
        if len(obj) > 0 and all(isinstance(x, (int, float, bool)) for x in obj[: min(20, len(obj))]):
            return
        for i, v in enumerate(obj[:50]):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from iter_nested(v, p)


def find_first_key_value(records: List[Dict[str, Any]], keys: List[str]) -> str:
    for rec in records:
        for path, value in iter_nested(rec):
            last = path.split(".")[-1]
            # strip list suffix if any
            last = re.sub(r"\[.*\]$", "", last)
            if last in keys:
                if isinstance(value, (str, int, float)):
                    return str(value)
    return ""


def find_init_state_idx(records: List[Dict[str, Any]]) -> Optional[int]:
    val = find_first_key_value(records, INIT_KEYS)
    if val == "":
        return None
    try:
        return int(float(val))
    except Exception:
        return None


def find_language(records: List[Dict[str, Any]]) -> str:
    return find_first_key_value(records, LANGUAGE_KEYS)


def find_condition(records: List[Dict[str, Any]]) -> str:
    return find_first_key_value(records, CONDITION_KEYS)


def get_by_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]
    return cur


def extract_step_actions(
    records: List[Dict[str, Any]],
    action_field: str,
    fallback_action_field: str,
) -> np.ndarray:
    actions = []

    for rec in records:
        value = get_by_path(rec, action_field)
        if value is None and fallback_action_field:
            value = get_by_path(rec, fallback_action_field)

        if value is None:
            continue

        arr = try_float_array(value)
        if arr is None:
            continue

        if arr.shape == (7,):
            actions.append(arr.astype(np.float32))
        elif arr.ndim == 2 and arr.shape[1] == 7:
            # Some records might contain a chunk; only use this in chunk mode,
            # so ignore here to avoid accidentally duplicating chunk predictions.
            continue

    if not actions:
        return np.zeros((0, 7), dtype=np.float32)

    return np.stack(actions, axis=0).astype(np.float32)


def extract_first_chunk(records: List[Dict[str, Any]], chunk_field: str) -> np.ndarray:
    for rec in records:
        value = get_by_path(rec, chunk_field)
        if value is None:
            continue

        arr = try_float_array(value)
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 7:
            return arr.astype(np.float32)

    return np.zeros((0, 7), dtype=np.float32)


def infer_file_short_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("traj_"):
        stem = stem[len("traj_") :]
    # Keep the last token if it looks like hash, otherwise short full stem.
    parts = stem.split("_")
    if parts:
        return safe_name(parts[-1])[:16]
    return safe_name(stem)[:16]


def maybe_clip_steps(actions: np.ndarray, start_step: int, max_steps: int) -> np.ndarray:
    start = max(0, int(start_step))
    end = actions.shape[0]
    if max_steps > 0:
        end = min(end, start + int(max_steps))
    return actions[start:end].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--policy_name", default="openvla_oft")
    parser.add_argument("--chunk_name", default="jsonl_dump")

    parser.add_argument(
        "--mode",
        choices=["step_actions", "first_chunk"],
        default="step_actions",
        help="step_actions uses processed_action per line; first_chunk uses raw_actions_chunk.",
    )
    parser.add_argument("--action_field", default="processed_action")
    parser.add_argument("--fallback_action_field", default="raw_action_before_process")
    parser.add_argument("--chunk_field", default="raw_actions_chunk")

    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=140)

    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument(
        "--allow_missing_init",
        action="store_true",
        help="If set, export files without detected init_state_idx by assigning them in sorted order modulo num_seeds. Use only as last resort.",
    )

    parser.add_argument(
        "--filter_contains",
        default="",
        help="Only export jsonl files whose path contains this substring.",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(dump_dir.glob("*.jsonl"))
    if args.filter_contains:
        files = [p for p in files if args.filter_contains in str(p)]

    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise RuntimeError(f"No jsonl files found in {dump_dir}")

    policy_name = safe_name(args.policy_name)
    base_chunk_name = safe_name(args.chunk_name)

    exported = []
    missing_init = []
    empty_actions = []
    skipped_out_of_range = []

    print("=" * 100)
    print("[dump_dir]", dump_dir)
    print("[out_dir]", out_dir)
    print("[files]", len(files))
    print("[mode]", args.mode)
    print("[policy_name]", policy_name)
    print("[chunk_name]", base_chunk_name)
    print("=" * 100)

    for file_i, path in enumerate(files):
        records = read_jsonl(path)

        if not records:
            empty_actions.append(str(path))
            continue

        init_idx = find_init_state_idx(records)

        if init_idx is None:
            if args.allow_missing_init:
                init_idx = args.start_init_state_idx + (file_i % args.num_seeds)
            else:
                missing_init.append(str(path))
                continue

        if not (args.start_init_state_idx <= init_idx < args.start_init_state_idx + args.num_seeds):
            skipped_out_of_range.append((str(path), init_idx))
            continue

        if args.mode == "step_actions":
            actions = extract_step_actions(
                records=records,
                action_field=args.action_field,
                fallback_action_field=args.fallback_action_field,
            )
        else:
            actions = extract_first_chunk(records=records, chunk_field=args.chunk_field)

        actions = maybe_clip_steps(actions, start_step=args.start_step, max_steps=args.max_steps)

        if actions.shape[0] == 0:
            empty_actions.append(str(path))
            continue

        language = find_language(records)
        condition = find_condition(records)

        short_id = infer_file_short_id(path)
        condition_name = safe_name(condition) if condition else "unknown_condition"

        final_chunk_name = safe_name(f"{policy_name}_{base_chunk_name}_{condition_name}_{short_id}")

        seed_dir = out_dir / f"init_{init_idx:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        out_path = seed_dir / f"{final_chunk_name}.npz"

        np.savez_compressed(
            out_path,
            actions=actions.astype(np.float32),
            policy_name=policy_name,
            chunk_name=final_chunk_name,
            language=language,
            condition=condition,
            source_path=str(path),
            init_state_idx=np.asarray(init_idx, dtype=np.int32),
            start_step=np.asarray(args.start_step, dtype=np.int32),
            used_num_steps=np.asarray(actions.shape[0], dtype=np.int32),
            mode=args.mode,
            action_field=args.action_field,
            fallback_action_field=args.fallback_action_field,
            chunk_field=args.chunk_field,
        )

        exported.append(
            {
                "init_state_idx": init_idx,
                "source_path": str(path),
                "out_path": str(out_path),
                "steps": int(actions.shape[0]),
                "language": language,
                "condition": condition,
                "chunk_name": final_chunk_name,
            }
        )

        print(
            f"[export] file={file_i:04d} init={init_idx:03d} "
            f"steps={actions.shape[0]:03d} condition={condition_name} -> {out_path}"
        )

    manifest = {
        "dump_dir": str(dump_dir),
        "out_dir": str(out_dir),
        "args": vars(args),
        "exported": exported,
        "missing_init": missing_init,
        "empty_actions": empty_actions,
        "skipped_out_of_range": skipped_out_of_range,
    }

    manifest_path = out_dir / f"manifest_{policy_name}_{base_chunk_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    csv_path = out_dir / f"manifest_{policy_name}_{base_chunk_name}.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["init_state_idx", "steps", "condition", "language", "chunk_name", "source_path", "out_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in exported:
            writer.writerow({k: r.get(k, "") for k in fields})

    print("\n" + "=" * 100)
    print("Exported:", len(exported))
    print("Missing init:", len(missing_init))
    print("Empty actions:", len(empty_actions))
    print("Skipped out of range:", len(skipped_out_of_range))
    print("Manifest:", manifest_path)
    print("CSV:", csv_path)
    print("=" * 100)

    if missing_init:
        print("\nMissing init examples:")
        for x in missing_init[:20]:
            print(" ", x)

    if empty_actions:
        print("\nEmpty action examples:")
        for x in empty_actions[:20]:
            print(" ", x)


if __name__ == "__main__":
    main()