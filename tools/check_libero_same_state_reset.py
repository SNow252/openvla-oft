import argparse
import json
import os
from types import SimpleNamespace

import numpy as np


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return None


def summarize_obs(obs):
    """Extract useful robot/object positions from LIBERO obs dict."""
    summary = {}
    if not isinstance(obs, dict):
        return summary

    wanted_exact = {
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "akita_black_bowl_1_pos",
        "akita_black_bowl_2_pos",
        "plate_1_pos",
        "glazed_rim_porcelain_ramekin_1_pos",
    }

    for k, v in obs.items():
        arr = to_numpy(v)
        if arr is None:
            continue

        # Keep exact keys plus most object position keys.
        if k in wanted_exact or k.endswith("_pos"):
            arr = arr.astype(float).reshape(-1)
            if arr.size <= 10:
                summary[k] = arr.tolist()

    return summary


def l2_diff(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def make_env_with_openvla_utils(args, task):
    """Prefer OpenVLA-OFT's own LIBERO env builder, if available."""
    from experiments.robot.libero.libero_utils import get_libero_env

    dummy_args = SimpleNamespace(
        task_suite_name=args.task_suite_name,
        center_crop=False,
    )

    # get_libero_env signature in OpenVLA-OFT commonly expects:
    # get_libero_env(args, task, model_family, resolution=...)
    env, task_description = get_libero_env(
        dummy_args,
        task,
        model_family="openvla",
        resolution=args.resolution,
    )
    return env, task_description


def make_env_with_libero(args, task):
    """Fallback: construct LIBERO env directly.

    In --no-render mode, disable all camera/offscreen rendering so this can run
    on compute servers without /dev/dri render permission.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": args.resolution,
        "camera_widths": args.resolution,
    }

    if args.no_render:
        env_args.update(
            {
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "use_camera_obs": False,
                "use_object_obs": True,
            }
        )
    else:
        env_args.update(
            {
                "render_gpu_device_id": args.render_gpu_device_id,
            }
        )

    env = OffScreenRenderEnv(**env_args)
    return env, getattr(task, "language", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=8)
    parser.add_argument("--init_state_idx", type=int, default=0)
    parser.add_argument("--num_resets", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render_gpu_device_id", type=int, default=0)
    parser.add_argument("--no_render", "--no-render", action="store_true")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    print("=" * 80)
    print("[check] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[check] MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("=" * 80)

    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    init_states = task_suite.get_task_init_states(args.task_id)

    if args.init_state_idx >= len(init_states):
        raise ValueError(
            f"init_state_idx={args.init_state_idx} out of range; "
            f"num init states={len(init_states)}"
        )

    print(f"[task_suite] {args.task_suite_name}")
    print(f"[task_id] {args.task_id}")
    print(f"[init_state_idx] {args.init_state_idx}")
    print(f"[num_init_states] {len(init_states)}")
    print(f"[task language] {getattr(task, 'language', '')}")
    print("=" * 80)

    if args.no_render:
        print("[env] no-render mode: skip OpenVLA-OFT get_libero_env")
        env, task_description = make_env_with_libero(args, task)
        print("[env] created with LIBERO env, camera/offscreen rendering disabled")
    else:
        try:
            env, task_description = make_env_with_openvla_utils(args, task)
            print("[env] created with OpenVLA-OFT get_libero_env")
        except Exception as e:
            print("[env] OpenVLA-OFT get_libero_env failed, fallback to LIBERO direct env")
            print("[env] error:", repr(e))
            env, task_description = make_env_with_libero(args, task)
            print("[env] created with LIBERO OffScreenRenderEnv")

    all_summaries = []

    for i in range(args.num_resets):
        env.reset()

        # This is the important part: force exactly the same LIBERO init state.
        obs = env.set_init_state(init_states[args.init_state_idx])

        summary = summarize_obs(obs)
        all_summaries.append(summary)

        print(f"\n[reset {i}] extracted keys:")
        for k in sorted(summary.keys()):
            print(f"  {k}: {np.array(summary[k])}")

        if i > 0:
            print(f"[reset {i}] L2 diff vs reset 0:")
            base = all_summaries[0]
            for k in sorted(set(base.keys()) & set(summary.keys())):
                print(f"  {k}: {l2_diff(base[k], summary[k]):.8f}")

    if hasattr(env, "close"):
        env.close()

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "task_suite_name": args.task_suite_name,
                    "task_id": args.task_id,
                    "init_state_idx": args.init_state_idx,
                    "num_resets": args.num_resets,
                    "summaries": all_summaries,
                },
                f,
                indent=2,
            )
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()