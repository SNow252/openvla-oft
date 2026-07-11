# ===== VLA FRESH POLICY DUMP PATCH START =====
import csv as _vla_csv
import json as _vla_json
import os as _vla_os
import re as _vla_re
import time as _vla_time
import uuid as _vla_uuid
from pathlib import Path as _vla_Path

import numpy as _vla_np


def _vla_safe_name(x):
    """Make a string safe for filenames."""
    x = str(x)
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = _vla_re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    x = _vla_re.sub(r"_+", "_", x)
    return x.strip("_") or "unnamed"


def _vla_dump_sanitize(x):
    """Convert numpy / torch / Python objects into JSON-safe values."""
    try:
        if x is None or isinstance(x, (str, int, float, bool)):
            return x

        if isinstance(x, dict):
            return {str(k): _vla_dump_sanitize(v) for k, v in x.items()}

        if isinstance(x, (list, tuple)):
            return [_vla_dump_sanitize(v) for v in x]

        # torch.Tensor-like
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            try:
                x = x.detach().cpu().numpy()
            except Exception:
                x = x.detach().cpu().tolist()
                return _vla_dump_sanitize(x)

        # numpy ndarray
        if isinstance(x, _vla_np.ndarray):
            return _vla_dump_sanitize(x.tolist())

        # numpy scalar
        if isinstance(x, _vla_np.generic):
            return x.item()

        # generic array-like
        if hasattr(x, "tolist"):
            return _vla_dump_sanitize(x.tolist())

        # scalar-like
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass

        return str(x)

    except Exception as e:
        return {"dump_error": str(e), "type": str(type(x))}


def _vla_obs_summary(obs):
    """Keep small non-image observations only; avoid dumping full images."""
    try:
        if not isinstance(obs, dict):
            return str(type(obs))
        out = {}
        for k, v in obs.items():
            ks = str(k).lower()
            if "image" in ks or "pixel" in ks or "rgb" in ks:
                try:
                    out[str(k)] = {"shape": list(_vla_np.asarray(v).shape), "skipped": "image"}
                except Exception:
                    out[str(k)] = {"skipped": "image"}
                continue
            arr = _vla_np.asarray(v)
            if arr.size <= 512:
                out[str(k)] = arr.tolist()
            else:
                out[str(k)] = {"shape": list(arr.shape), "skipped": "large_array"}
        return out
    except Exception as e:
        return {"obs_summary_error": str(e)}


def _vla_write_jsonl(path, obj):
    """Append one JSON-safe object to a JSONL file."""
    if path is None:
        return
    with open(path, "a") as f:
        f.write(_vla_json.dumps(_vla_dump_sanitize(obj), ensure_ascii=False) + "\n")


def _vla_stack_or_empty(items, shape_tail=(7,), dtype=_vla_np.float32):
    """Stack a list of arrays or return an empty array."""
    if not items:
        return _vla_np.zeros((0,) + tuple(shape_tail), dtype=dtype)
    try:
        return _vla_np.stack([_vla_np.asarray(x, dtype=dtype) for x in items], axis=0)
    except Exception:
        return _vla_np.asarray([_vla_np.asarray(x, dtype=dtype) for x in items], dtype=object)


def _vla_stack_chunks_or_empty(chunks):
    """Stack raw action chunks, normally [num_queries, chunk_len, 7]."""
    if not chunks:
        return _vla_np.zeros((0, 0, 7), dtype=_vla_np.float32)
    try:
        return _vla_np.stack([_vla_np.asarray(x, dtype=_vla_np.float32) for x in chunks], axis=0)
    except Exception:
        return _vla_np.asarray([_vla_np.asarray(x, dtype=_vla_np.float32) for x in chunks], dtype=object)


def _vla_manifest_fields():
    return [
        "episode_id",
        "policy_name",
        "model_family",
        "task_suite_name",
        "task_id",
        "condition",
        "condition_safe",
        "init_state_idx",
        "language",
        "success",
        "num_steps",
        "sum_reward",
        "jsonl_path",
        "npz_path",
    ]


def _vla_append_manifest(root, row):
    """Append one row to root/manifest.csv."""
    root = _vla_Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"
    exists = manifest_path.exists()
    fields = _vla_manifest_fields()
    with open(manifest_path, "a", newline="") as f:
        writer = _vla_csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _vla_get_dump_cfg(cfg, task_id, init_state_idx, task_description):
    """
    Build dump paths and metadata.

    Compatible controls:
      New CLI/draccus fields:
        --fresh_dump_dir
        --fresh_policy_name
        --fresh_condition

      Old env controls:
        OPENVLA_DUMP_TRAJ=1
        OPENVLA_DUMP_DIR=...
        OPENVLA_DUMP_CONDITION=...
        OPENVLA_POLICY_NAME=...
    """
    fresh_dump_dir = str(getattr(cfg, "fresh_dump_dir", "") or "").strip()
    env_dump_enabled = _vla_os.environ.get("OPENVLA_DUMP_TRAJ", "0") == "1"
    enabled = bool(fresh_dump_dir) or env_dump_enabled

    if not enabled:
        return {"enabled": False}

    root = (
        fresh_dump_dir
        or _vla_os.environ.get("OPENVLA_FRESH_DUMP_DIR", "")
        or _vla_os.environ.get("OPENVLA_DUMP_DIR", "/tmp/openvla_oft_fresh_policy_dumps")
    )
    root = _vla_Path(root)

    policy_name = (
        str(getattr(cfg, "fresh_policy_name", "") or "").strip()
        or _vla_os.environ.get("OPENVLA_POLICY_NAME", "")
        or str(getattr(cfg, "model_family", "policy"))
    )
    policy_name = _vla_safe_name(policy_name)

    condition = (
        str(getattr(cfg, "fresh_condition", "") or "").strip()
        or _vla_os.environ.get("OPENVLA_DUMP_CONDITION", "")
        or f"task{task_id}"
    )
    condition_safe = _vla_safe_name(condition)

    episode_id = f"{int(_vla_time.time() * 1000)}_{_vla_os.getpid()}_{_vla_uuid.uuid4().hex[:8]}"

    jsonl_dir = root / "jsonl" / policy_name / condition_safe
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = jsonl_dir / f"init_{int(init_state_idx):03d}_{episode_id}.jsonl"

    chunk_dir = root / "policy_chunks" / f"init_{int(init_state_idx):03d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_name = _vla_safe_name(
        f"{policy_name}_{condition_safe}_task{int(task_id)}_init{int(init_state_idx):03d}_{episode_id}"
    )
    npz_path = chunk_dir / f"{chunk_name}.npz"

    return {
        "enabled": True,
        "root": str(root),
        "policy_name": policy_name,
        "condition": condition,
        "condition_safe": condition_safe,
        "episode_id": episode_id,
        "jsonl_path": str(jsonl_path),
        "npz_path": str(npz_path),
        "chunk_name": chunk_name,
        "task_id": int(task_id),
        "init_state_idx": int(init_state_idx),
        "language": str(task_description),
        "model_family": str(getattr(cfg, "model_family", "")),
        "task_suite_name": str(getattr(cfg, "task_suite_name", "")),
        "num_open_loop_steps": int(getattr(cfg, "num_open_loop_steps", -1)),
        "pretrained_checkpoint": str(getattr(cfg, "pretrained_checkpoint", "")),
    }


def _vla_start_episode_dump(ctx, obs):
    """Write metadata + initial observation summary."""
    if not ctx.get("enabled", False):
        return

    _vla_write_jsonl(
        ctx["jsonl_path"],
        {
            "type": "metadata",
            "episode_id": ctx["episode_id"],
            "dump_path": ctx["jsonl_path"],
            "npz_path": ctx["npz_path"],
            "policy_name": ctx["policy_name"],
            "model_family": ctx["model_family"],
            "task_suite_name": ctx["task_suite_name"],
            "task_id": ctx["task_id"],
            "condition": ctx["condition"],
            "condition_safe": ctx["condition_safe"],
            "init_state_idx": ctx["init_state_idx"],
            "language": ctx["language"],
            "custom_language": _vla_os.environ.get("OPENVLA_CUSTOM_LANGUAGE", ""),
            "task_ids": _vla_os.environ.get("OPENVLA_TASK_IDS", ""),
            "num_open_loop_steps": ctx["num_open_loop_steps"],
            "pretrained_checkpoint": ctx["pretrained_checkpoint"],
            "timestamp": _vla_time.time(),
        },
    )

    _vla_write_jsonl(
        ctx["jsonl_path"],
        {
            "type": "initial_obs",
            "episode_id": ctx["episode_id"],
            "init_state_idx": ctx["init_state_idx"],
            "obs_summary": _vla_obs_summary(obs),
        },
    )


def _vla_finalize_episode_dump(
    ctx,
    *,
    processed_actions,
    raw_actions_before_process,
    raw_action_chunks,
    query_steps,
    rewards,
    dones,
    success,
    final_info=None,
    episode_error=None,
):
    """Save standard replay-compatible .npz and append manifest.csv."""
    if not ctx.get("enabled", False):
        return None

    actions_arr = _vla_stack_or_empty(processed_actions, shape_tail=(7,), dtype=_vla_np.float32)
    raw_actions_arr = _vla_stack_or_empty(raw_actions_before_process, shape_tail=(7,), dtype=_vla_np.float32)
    raw_chunks_arr = _vla_stack_chunks_or_empty(raw_action_chunks)
    rewards_arr = _vla_np.asarray(rewards, dtype=_vla_np.float32)
    dones_arr = _vla_np.asarray(dones, dtype=_vla_np.bool_)
    query_steps_arr = _vla_np.asarray(query_steps, dtype=_vla_np.int32)

    success_int = -1 if success is None else int(bool(success))

    meta = {
        "episode_id": ctx["episode_id"],
        "policy_name": ctx["policy_name"],
        "model_family": ctx["model_family"],
        "task_suite_name": ctx["task_suite_name"],
        "task_id": ctx["task_id"],
        "condition": ctx["condition"],
        "condition_safe": ctx["condition_safe"],
        "init_state_idx": ctx["init_state_idx"],
        "language": ctx["language"],
        "success": success_int,
        "num_steps": int(actions_arr.shape[0]),
        "sum_reward": float(_vla_np.sum(rewards_arr)) if rewards_arr.size else 0.0,
        "jsonl_path": ctx["jsonl_path"],
        "npz_path": ctx["npz_path"],
        "episode_error": episode_error,
        "final_info": _vla_dump_sanitize(final_info),
    }

    _vla_np.savez_compressed(
        ctx["npz_path"],
        actions=actions_arr,
        processed_actions=actions_arr,
        raw_actions_before_process=raw_actions_arr,
        raw_actions_chunks=raw_chunks_arr,
        query_steps=query_steps_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        policy_name=ctx["policy_name"],
        model_family=ctx["model_family"],
        task_suite_name=ctx["task_suite_name"],
        task_id=_vla_np.asarray(ctx["task_id"], dtype=_vla_np.int32),
        condition=ctx["condition"],
        condition_safe=ctx["condition_safe"],
        init_state_idx=_vla_np.asarray(ctx["init_state_idx"], dtype=_vla_np.int32),
        language=ctx["language"],
        chunk_name=ctx["chunk_name"],
        source_path=ctx["jsonl_path"],
        success=_vla_np.asarray(success_int, dtype=_vla_np.int32),
        used_num_steps=_vla_np.asarray(actions_arr.shape[0], dtype=_vla_np.int32),
        meta_json=_vla_json.dumps(_vla_dump_sanitize(meta), ensure_ascii=False),
    )

    _vla_write_jsonl(
        ctx["jsonl_path"],
        {
            "type": "episode_end",
            **meta,
        },
    )

    _vla_append_manifest(
        ctx["root"],
        {
            "episode_id": ctx["episode_id"],
            "policy_name": ctx["policy_name"],
            "model_family": ctx["model_family"],
            "task_suite_name": ctx["task_suite_name"],
            "task_id": ctx["task_id"],
            "condition": ctx["condition"],
            "condition_safe": ctx["condition_safe"],
            "init_state_idx": ctx["init_state_idx"],
            "language": ctx["language"],
            "success": success_int,
            "num_steps": int(actions_arr.shape[0]),
            "sum_reward": float(_vla_np.sum(rewards_arr)) if rewards_arr.size else 0.0,
            "jsonl_path": ctx["jsonl_path"],
            "npz_path": ctx["npz_path"],
        },
    )

    return ctx["npz_path"]
# ===== VLA FRESH POLICY DUMP PATCH END =====


"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Fresh policy action dump
    #################################################################################################################
    fresh_dump_dir: str = ""                          # If non-empty, save JSONL + replay-compatible .npz policy chunks
    fresh_policy_name: str = ""                       # Optional policy name, e.g. openvla_oft / smolvla
    fresh_condition: str = ""                         # Optional condition name, e.g. task8_original

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    task_id: int = -1,
    init_state_idx: int = -1,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Fresh VLA / policy action dump config.
    # This keeps compatibility with the old env-based dump switch:
    #   OPENVLA_DUMP_TRAJ=1
    #   OPENVLA_DUMP_DIR=...
    # and adds replay-compatible .npz chunks under:
    #   <dump_root>/policy_chunks/init_XXX/*.npz
    _vla_dump_ctx = _vla_get_dump_cfg(
        cfg=cfg,
        task_id=task_id,
        init_state_idx=init_state_idx,
        task_description=task_description,
    )
    _vla_dump_enabled = bool(_vla_dump_ctx.get("enabled", False))
    _vla_dump_step = 0

    _vla_processed_actions = []
    _vla_raw_actions_before_process = []
    _vla_raw_action_chunks = []
    _vla_query_steps = []
    _vla_rewards = []
    _vla_dones = []
    _vla_last_info = None
    _vla_episode_error = None

    if _vla_dump_enabled:
        _vla_start_episode_dump(_vla_dump_ctx, obs)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                _vla_query_obs_summary = _vla_obs_summary(obs) if _vla_dump_enabled else None
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                if _vla_dump_enabled:
                    _vla_actions_chunk_arr = _vla_np.asarray(actions, dtype=_vla_np.float32)
                    if _vla_actions_chunk_arr.ndim == 1 and _vla_actions_chunk_arr.shape[0] == 7:
                        _vla_actions_chunk_arr = _vla_actions_chunk_arr.reshape(1, 7)

                    _vla_raw_action_chunks.append(_vla_actions_chunk_arr.copy())
                    _vla_query_steps.append(int(_vla_dump_step))

                    _vla_write_jsonl(
                        _vla_dump_ctx["jsonl_path"],
                        {
                            "type": "chunk",
                            "episode_id": _vla_dump_ctx["episode_id"],
                            "step": _vla_dump_step,
                            "raw_actions_chunk": _vla_actions_chunk_arr,
                            "query_obs_summary": _vla_query_obs_summary,
                        },
                    )

                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            _vla_raw_action_before_process = _vla_np.asarray(action).copy()
            _vla_obs_before_summary = _vla_obs_summary(obs) if _vla_dump_enabled else None

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            if _vla_dump_enabled:
                _vla_processed_action_arr = _vla_np.asarray(action, dtype=_vla_np.float32).copy()
                _vla_raw_action_arr = _vla_np.asarray(_vla_raw_action_before_process, dtype=_vla_np.float32).copy()

                _vla_processed_actions.append(_vla_processed_action_arr)
                _vla_raw_actions_before_process.append(_vla_raw_action_arr)
                _vla_rewards.append(float(reward))
                _vla_dones.append(bool(done))
                _vla_last_info = info

                _vla_write_jsonl(
                    _vla_dump_ctx["jsonl_path"],
                    {
                        "type": "step",
                        "episode_id": _vla_dump_ctx["episode_id"],
                        "step": _vla_dump_step,
                        "raw_action_before_process": _vla_raw_action_arr,
                        "processed_action": _vla_processed_action_arr,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "obs_before_summary": _vla_obs_before_summary,
                        "obs_after_summary": _vla_obs_summary(obs),
                        "obs_summary": _vla_obs_summary(obs),
                    },
                )
                _vla_dump_step += 1
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        _vla_episode_error = str(e)
        log_message(f"Episode error: {e}", log_file)

    finally:
        if _vla_dump_enabled:
            try:
                _vla_finalize_episode_dump(
                    _vla_dump_ctx,
                    processed_actions=_vla_processed_actions,
                    raw_actions_before_process=_vla_raw_actions_before_process,
                    raw_action_chunks=_vla_raw_action_chunks,
                    query_steps=_vla_query_steps,
                    rewards=_vla_rewards,
                    dones=_vla_dones,
                    success=success,
                    final_info=_vla_last_info,
                    episode_error=_vla_episode_error,
                )
            except Exception as dump_e:
                log_message(f"Fresh policy dump finalize error: {dump_e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    if "OPENVLA_CUSTOM_LANGUAGE" in os.environ:
        custom_language = os.environ.get("OPENVLA_CUSTOM_LANGUAGE", "")
        log_message(f"[LANG DEBUG] old task: {task_description}", log_file)
        task_description = custom_language
        log_message(f"[LANG DEBUG] new task: {repr(task_description)}", log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            initial_state=initial_state,
            task_id=task_id,
            init_state_idx=episode_idx,
            log_file=log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    task_ids_env = os.environ.get("OPENVLA_TASK_IDS", "").strip()
    if task_ids_env:
        task_ids = [int(x) for x in task_ids_env.split(",")]
    else:
        task_ids = list(range(num_tasks))

    for task_id in tqdm.tqdm(task_ids):


        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
