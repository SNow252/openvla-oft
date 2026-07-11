#!/usr/bin/env python3
"""
Scan a dump directory and find files/keys that contain action-like arrays.

It detects:
  - .npz keys with shape [..., 7]
  - .npy arrays with shape [..., 7]
  - .json / .jsonl fields containing numeric arrays with shape [..., 7]

Use this before writing a converter from OpenVLA/SmolVLA dumps to standard
policy_chunks/init_XXX/*.npz format.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


VALID_SUFFIXES = {".npz", ".npy", ".json", ".jsonl"}


def looks_like_action_array(arr: np.ndarray) -> bool:
    if arr.ndim == 1:
        return arr.shape[0] == 7
    if arr.ndim >= 2:
        return arr.shape[-1] == 7 and arr.shape[0] > 0
    return False


def safe_shape(x: Any) -> str:
    try:
        return str(tuple(np.asarray(x).shape))
    except Exception:
        return "UNKNOWN"


def try_numeric_array(x: Any):
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 0:
            return None
        return arr
    except Exception:
        return None


def scan_json_obj(
    obj: Any,
    path: str = "",
    max_depth: int = 8,
) -> List[Tuple[str, Tuple[int, ...]]]:
    hits = []

    if max_depth < 0:
        return hits

    arr = try_numeric_array(obj)
    if arr is not None and looks_like_action_array(arr):
        hits.append((path or "$", tuple(arr.shape)))
        return hits

    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{path}.{k}" if path else str(k)
            hits.extend(scan_json_obj(v, child, max_depth=max_depth - 1))

    elif isinstance(obj, list):
        # Avoid recursively expanding huge arrays too deeply.
        if len(obj) > 0:
            for i, v in enumerate(obj[:20]):
                child = f"{path}[{i}]"
                hits.extend(scan_json_obj(v, child, max_depth=max_depth - 1))

    return hits


def scan_npz(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return [{"file": str(path), "kind": "npz_error", "key": "", "shape": "", "error": repr(e)}]

    for key in data.files:
        try:
            arr = np.asarray(data[key])
            shape = tuple(arr.shape)
            is_action = looks_like_action_array(arr)
            rows.append(
                {
                    "file": str(path),
                    "kind": "npz",
                    "key": key,
                    "shape": str(shape),
                    "is_action_like": int(is_action),
                    "error": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "file": str(path),
                    "kind": "npz_key_error",
                    "key": key,
                    "shape": "",
                    "is_action_like": 0,
                    "error": repr(e),
                }
            )

    return rows


def scan_npy(path: Path) -> List[Dict[str, Any]]:
    try:
        arr = np.load(path, allow_pickle=True)
        shape = tuple(np.asarray(arr).shape)
        return [
            {
                "file": str(path),
                "kind": "npy",
                "key": "$",
                "shape": str(shape),
                "is_action_like": int(looks_like_action_array(np.asarray(arr))),
                "error": "",
            }
        ]
    except Exception as e:
        return [{"file": str(path), "kind": "npy_error", "key": "$", "shape": "", "is_action_like": 0, "error": repr(e)}]


def scan_json(path: Path, max_lines: int) -> List[Dict[str, Any]]:
    rows = []

    try:
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line_idx, line in enumerate(f):
                    if line_idx >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    hits = scan_json_obj(obj, path=f"line[{line_idx}]")
                    for key, shape in hits:
                        rows.append(
                            {
                                "file": str(path),
                                "kind": "jsonl",
                                "key": key,
                                "shape": str(shape),
                                "is_action_like": 1,
                                "error": "",
                            }
                        )

        else:
            with open(path) as f:
                obj = json.load(f)

            hits = scan_json_obj(obj, path="$")
            for key, shape in hits:
                rows.append(
                    {
                        "file": str(path),
                        "kind": "json",
                        "key": key,
                        "shape": str(shape),
                        "is_action_like": 1,
                        "error": "",
                    }
                )

    except Exception as e:
        rows.append(
            {
                "file": str(path),
                "kind": "json_error",
                "key": "",
                "shape": "",
                "is_action_like": 0,
                "error": repr(e),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_csv", default="action_array_scan.csv")
    parser.add_argument("--max_files", type=int, default=20000)
    parser.add_argument("--max_json_lines", type=int, default=200)
    parser.add_argument("--only_hits", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix in VALID_SUFFIXES]
    files = files[: args.max_files]

    print("Root:", root)
    print("Candidate files:", len(files))

    rows = []

    for i, p in enumerate(files):
        if i % 500 == 0:
            print(f"[scan] {i}/{len(files)}")

        if p.suffix == ".npz":
            rs = scan_npz(p)
        elif p.suffix == ".npy":
            rs = scan_npy(p)
        elif p.suffix in {".json", ".jsonl"}:
            rs = scan_json(p, max_lines=args.max_json_lines)
        else:
            rs = []

        rows.extend(rs)

    if args.only_hits:
        rows_out = [r for r in rows if int(r.get("is_action_like", 0)) == 1]
    else:
        rows_out = rows

    out_path = root / args.out_csv
    fields = ["file", "kind", "key", "shape", "is_action_like", "error"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    hits = [r for r in rows if int(r.get("is_action_like", 0)) == 1]

    print("\n" + "=" * 100)
    print("Total scanned rows:", len(rows))
    print("Action-like hits:", len(hits))
    print("Saved:", out_path)
    print("=" * 100)

    print("\nTop hits:")
    for r in hits[:100]:
        print(f"{r['kind']:6s} shape={r['shape']:18s} key={r['key']} file={r['file']}")


if __name__ == "__main__":
    main()