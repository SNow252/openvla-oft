#!/usr/bin/env python3
"""
Export saved rollout actions into the standard v3 policy chunk format.

Standard output layout:
  OUT_DIR/init_000/openvla_original.npz
  OUT_DIR/init_001/openvla_original.npz
  ...

Each output npz contains:
  actions: [T, 7]
  policy_name: str
  chunk_name: str
  language: str
  source_path: str

Supported input:
  --source_glob can include {init:03d}, for example:
    /path/to/openvla_rollouts/init_{init:03d}/traj.npz
    /path/to/openvla_rollouts/init_{init:03d}/actions.npy
    /path/to/openvla_rollouts/init_{init:03d}/**/traj.npz

The script reads:
  - .npz with key "actions"
  - .npy as raw action array
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def safe_name(x: str) -> str:
    x = str(x)
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    return x


def load_actions(path: Path, action_key: str = "actions") -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        actions = np.asarray(arr, dtype=np.float32)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if action_key not in data:
            keys = list(data.keys())
            raise KeyError(f"{path} missing key '{action_key}'. Available keys: {keys}")
        actions = np.asarray(data[action_key], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if actions.ndim != 2:
        raise ValueError(f"{path} actions should be [T, A], got shape={actions.shape}")

    if actions.shape[1] < 7:
        raise ValueError(f"{path} actions should have at least 7 dims, got shape={actions.shape}")

    actions = actions[:, :7].astype(np.float32)
    return actions


def find_source_for_init(source_glob: str, init_idx: int) -> List[Path]:
    pattern = source_glob.format(init=init_idx)
    return sorted(Path(p) for p in glob.glob(pattern, recursive=True))


def maybe_load_language(language: str, language_file: str, init_idx: int) -> str:
    if language:
        return language

    if not language_file:
        return ""

    path = Path(language_file.format(init=init_idx))
    if not path.exists():
        return ""

    try:
        if path.suffix == ".json":
            with open(path) as f:
                obj = json.load(f)
            for key in ["language", "instruction", "task_language", "prompt"]:
                if key in obj:
                    return str(obj[key])
            return json.dumps(obj, ensure_ascii=False)
        else:
            return path.read_text().strip()
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_glob", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--start_init_state_idx", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=20)

    parser.add_argument("--policy_name", required=True)
    parser.add_argument("--chunk_name", required=True)

    parser.add_argument("--action_key", default="actions")
    parser.add_argument("--max_steps", type=int, default=140)
    parser.add_argument("--start_step", type=int, default=0)

    parser.add_argument("--language", default="")
    parser.add_argument(
        "--language_file",
        default="",
        help="Optional template path, e.g. /path/init_{init:03d}/language.txt or .json",
    )

    parser.add_argument(
        "--take_first_match",
        action="store_true",
        help="If multiple source files match one init, take the first one.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    missing = []
    skipped_multi = []

    policy_name = safe_name(args.policy_name)
    chunk_name = safe_name(args.chunk_name)

    for init_idx in range(args.start_init_state_idx, args.start_init_state_idx + args.num_seeds):
        matches = find_source_for_init(args.source_glob, init_idx)

        if not matches:
            print(f"[missing] init={init_idx:03d}")
            missing.append(init_idx)
            continue

        if len(matches) > 1 and not args.take_first_match:
            print(f"[multiple] init={init_idx:03d}, n={len(matches)}")
            for m in matches[:10]:
                print("  ", m)
            skipped_multi.append(init_idx)
            continue

        src = matches[0]
        actions = load_actions(src, action_key=args.action_key)

        start = max(0, args.start_step)
        end = actions.shape[0]
        if args.max_steps > 0:
            end = min(end, start + args.max_steps)

        actions = actions[start:end]

        if actions.shape[0] == 0:
            print(f"[empty] init={init_idx:03d}, src={src}")
            continue

        language = maybe_load_language(args.language, args.language_file, init_idx)

        seed_dir = out_dir / f"init_{init_idx:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        out_path = seed_dir / f"{policy_name}_{chunk_name}.npz"

        np.savez_compressed(
            out_path,
            actions=actions.astype(np.float32),
            policy_name=policy_name,
            chunk_name=f"{policy_name}_{chunk_name}",
            language=language,
            source_path=str(src),
            init_state_idx=np.asarray(init_idx, dtype=np.int32),
            start_step=np.asarray(start, dtype=np.int32),
            used_num_steps=np.asarray(actions.shape[0], dtype=np.int32),
        )

        exported.append((init_idx, src, out_path, actions.shape[0]))

        print(
            f"[exported] init={init_idx:03d} "
            f"steps={actions.shape[0]:03d} "
            f"{src} -> {out_path}"
        )

    manifest = {
        "source_glob": args.source_glob,
        "out_dir": str(out_dir),
        "policy_name": policy_name,
        "chunk_name": chunk_name,
        "start_init_state_idx": args.start_init_state_idx,
        "num_seeds": args.num_seeds,
        "max_steps": args.max_steps,
        "start_step": args.start_step,
        "exported": [
            {
                "init_state_idx": i,
                "source_path": str(src),
                "out_path": str(out),
                "steps": int(steps),
            }
            for i, src, out, steps in exported
        ],
        "missing": missing,
        "skipped_multi": skipped_multi,
    }

    with open(out_dir / f"manifest_{policy_name}_{chunk_name}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 100)
    print("Exported:", len(exported))
    print("Missing:", len(missing))
    print("Skipped multiple:", len(skipped_multi))
    print("Saved manifest:", out_dir / f"manifest_{policy_name}_{chunk_name}.json")
    print("=" * 100)


if __name__ == "__main__":
    main()