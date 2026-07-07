import os
import subprocess
from pathlib import Path
from datetime import datetime


OPENVLA_ROOT = Path.home() / "projects/vla_lang_generalization/openvla-oft"
CKPT = Path.home() / "projects/vla_lang_generalization/hf_models/openvla-7b-oft-finetuned-libero-spatial"

EPISODES = int(os.environ.get("OPENVLA_NUM_EPISODES", "10"))

DUMP_ROOT = (
    Path.home()
    / "projects/vla_lang_generalization/goal_controllability_verifier/data"
    / f"openvla_mismatch_v2_dumps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "PYTHONPATH": str(OPENVLA_ROOT / "LIBERO"),
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OPENVLA_DUMP_TRAJ": "1",
        "OPENVLA_DUMP_DIR": str(DUMP_ROOT),
    }
)


CONDITIONS = [
    {
        "name": "task8_original",
        "task_id": "8",
        "custom_language": None,
        "episodes": EPISODES,
    },
    {
        "name": "task8_wrong_target_language_only",
        "task_id": "8",
        "custom_language": "pick up the black bowl next to the plate and place it on the ramekin",
        "episodes": EPISODES,
    },
    {
        "name": "task8_wrong_source_language_only",
        "task_id": "8",
        "custom_language": "pick up the black bowl next to the ramekin and place it on the plate",
        "episodes": EPISODES,
    },
    {
        "name": "task8_wrong_source_wrong_target_language_only",
        "task_id": "8",
        "custom_language": "pick up the black bowl next to the ramekin and place it on the ramekin",
        "episodes": EPISODES,
    },
    {
        "name": "task8_empty_language",
        "task_id": "8",
        "custom_language": "",
        "episodes": EPISODES,
    },
    {
        "name": "task8_nonsense_language",
        "task_id": "8",
        "custom_language": "dax blicket wug zorp flarn",
        "episodes": EPISODES,
    },
    {
        "name": "task8_unrelated_language",
        "task_id": "8",
        "custom_language": "open the drawer and put the book on the shelf",
        "episodes": EPISODES,
    },
    {
        "name": "task1_native_next_to_ramekin",
        "task_id": "1",
        "custom_language": None,
        "episodes": EPISODES,
    },
]


def run_condition(cond):
    env = BASE_ENV.copy()
    env["OPENVLA_TASK_IDS"] = cond["task_id"]
    env["OPENVLA_DUMP_CONDITION"] = cond["name"]

    if cond["custom_language"] is None:
        env.pop("OPENVLA_CUSTOM_LANGUAGE", None)
    else:
        env["OPENVLA_CUSTOM_LANGUAGE"] = cond["custom_language"]

    DUMP_ROOT.mkdir(parents=True, exist_ok=True)

    log_path = DUMP_ROOT / f"{cond['name']}.log"
    run_id_note = f"dump_{cond['name']}"

    cmd = [
        "python",
        "-u",
        "experiments/robot/libero/run_libero_eval_dump.py",
        "--pretrained_checkpoint",
        str(CKPT),
        "--task_suite_name",
        "libero_spatial",
        "--num_trials_per_task",
        str(cond["episodes"]),
        "--center_crop",
        "True",
        "--run_id_note",
        run_id_note,
    ]

    print("\n" + "=" * 100)
    print(f"Running condition: {cond['name']}")
    print(f"Task ID: {cond['task_id']}")
    print(f"Episodes: {cond['episodes']}")
    print(f"Custom language: {repr(cond['custom_language'])}")
    print(f"Dump root: {DUMP_ROOT}")
    print(f"Log path: {log_path}")
    print("=" * 100)

    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(OPENVLA_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Condition failed: {cond['name']}, see {log_path}")

    print(f"Finished: {cond['name']}")


def main():
    print(f"Writing dumps to: {DUMP_ROOT}")
    print(f"Episodes per condition: {EPISODES}")

    for cond in CONDITIONS:
        run_condition(cond)

    print("\nAll done.")
    print(f"Dump root: {DUMP_ROOT}")


if __name__ == "__main__":
    main()