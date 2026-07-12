#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


def np_list(x):
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def short_goal_from_bddl(text: str, max_chars: int = 1200):
    m = re.search(r"\(:goal(.*?)(?:\n\s*\)|$)", text, flags=re.S)
    if not m:
        return ""
    s = "(:goal" + m.group(1)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def make_env(task_suite_name: str, task_id: int, resolution: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark.get_benchmark(benchmark_dict[task_suite_name])()
    task = task_suite.get_task(task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    init_states = task_suite.get_task_init_states(task_id)

    return env, task_suite, task, bddl_file, init_states


def reset_to_init(env, init_states, init_state_idx: int, settle_steps: int):
    env.reset()
    obs = env.set_init_state(init_states[init_state_idx])

    zero = np.zeros(7, dtype=np.float32)
    for _ in range(settle_steps):
        step_out = env.step(zero)
        if len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs = step_out[0]

    try:
        obs = env.get_observation()
    except Exception:
        pass

    return obs


def collect_pos_keys(obs):
    out = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        if arr.shape == (3,) and k.endswith("_pos"):
            out[k] = arr.astype(float)
    return out


def classify_pos_key(k):
    if k.startswith("robot") or "eef" in k or "gripper" in k:
        return "robot"
    return "object"


def inspect_one(task_suite_name, task_id, init_indices, resolution, settle_steps):
    env, task_suite, task, bddl_file, init_states = make_env(
        task_suite_name, task_id, resolution
    )

    bddl_text = Path(bddl_file).read_text(errors="ignore")

    result = {
        "task_suite_name": task_suite_name,
        "task_id": task_id,
        "language": getattr(task, "language", ""),
        "problem_folder": getattr(task, "problem_folder", ""),
        "bddl_file": getattr(task, "bddl_file", ""),
        "bddl_file_path": bddl_file,
        "bddl_goal_excerpt": short_goal_from_bddl(bddl_text),
        "num_init_states": len(init_states),
        "inits": [],
    }

    for init_idx in init_indices:
        obs = reset_to_init(env, init_states, init_idx, settle_steps)
        pos = collect_pos_keys(obs)

        eef = None
        for k, v in pos.items():
            if "eef" in k:
                eef = v
                break

        objects = []
        robots = []

        for k, v in sorted(pos.items()):
            item = {
                "key": k,
                "xyz": np_list(v),
            }
            if eef is not None and classify_pos_key(k) == "object":
                item["dist_to_eef"] = float(np.linalg.norm(v - eef))

            if classify_pos_key(k) == "robot":
                robots.append(item)
            else:
                objects.append(item)

        result["inits"].append(
            {
                "init_state_idx": init_idx,
                "robot_pos_keys": robots,
                "object_pos_keys": objects,
            }
        )

    try:
        env.close()
    except Exception:
        pass

    return result


def write_markdown(results, out_md):
    lines = []
    lines.append("# LIBERO task object inspection\n")

    for r in results:
        lines.append(f"## Task {r['task_id']}")
        lines.append("")
        lines.append(f"- suite: `{r['task_suite_name']}`")
        lines.append(f"- language: **{r['language']}**")
        lines.append(f"- bddl: `{r['bddl_file_path']}`")
        lines.append(f"- num_init_states: {r['num_init_states']}")
        lines.append("")
        lines.append("### BDDL goal excerpt")
        lines.append("")
        lines.append("```")
        lines.append(r.get("bddl_goal_excerpt", ""))
        lines.append("```")
        lines.append("")

        for init in r["inits"]:
            lines.append(f"### init_state_idx = {init['init_state_idx']}")
            lines.append("")
            lines.append("| key | x | y | z | dist_to_eef |")
            lines.append("|---|---:|---:|---:|---:|")
            for obj in init["object_pos_keys"]:
                xyz = obj["xyz"]
                dist = obj.get("dist_to_eef", float("nan"))
                lines.append(
                    f"| `{obj['key']}` | {xyz[0]:+.4f} | {xyz[1]:+.4f} | {xyz[2]:+.4f} | {dist:.4f} |"
                )
            lines.append("")

    Path(out_md).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_suite_name", default="libero_spatial")
    ap.add_argument("--task_ids", nargs="+", type=int, default=[1, 3, 8, 9])
    ap.add_argument("--init_indices", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--settle_steps", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task_id in args.task_ids:
        print(f"[INFO] Inspect task_id={task_id}")
        r = inspect_one(
            task_suite_name=args.task_suite_name,
            task_id=task_id,
            init_indices=args.init_indices,
            resolution=args.resolution,
            settle_steps=args.settle_steps,
        )
        results.append(r)

    out_json = out_dir / "task_object_inspection.json"
    out_md = out_dir / "task_object_inspection.md"

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_markdown(results, out_md)

    print("[saved]", out_json)
    print("[saved]", out_md)
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()