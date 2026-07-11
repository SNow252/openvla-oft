#!/usr/bin/env python3
"""
Fresh policy action chunk logger for OpenVLA / SmolVLA / other learned policies.

Purpose:
  Save policy action chunks with explicit metadata:
    - policy_name
    - model_family
    - task_suite_name
    - task_id
    - condition
    - init_state_idx
    - language
    - processed action sequence
    - raw action chunks
    - rewards / dones / success
    - obs summaries

Outputs:
  OUT_DIR/jsonl/<policy>/<condition>/init_000_<episode_id>.jsonl
  OUT_DIR/policy_chunks/init_000/<policy>_<condition>_<episode_id>.npz
  OUT_DIR/manifest.csv

The .npz format is compatible with replay_policy_action_chunks_v3.py.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


SUMMARY_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "akita_black_bowl_1_pos",
    "akita_black_bowl_2_pos",
    "plate_1_pos",
    "glazed_rim_porcelain_ramekin_1_pos",
]


def safe_name(x: Any) -> str:
    x = str(x)
    x = x.replace("-", "m").replace("+", "p").replace(".", "p")
    x = re.sub(r"[^a-zA-Z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_") or "unnamed"


def to_numpy(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def to_float_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    return str(x)


def summarize_obs(obs: Any) -> Dict[str, Any]:
    """
    Robust obs summarizer.

    It first tries known LIBERO/robosuite state keys.
    Then it also keeps small numeric fields that look state-like.
    """
    out: Dict[str, Any] = {}

    if not isinstance(obs, dict):
        return out

    for k in SUMMARY_KEYS:
        if k in obs:
            arr = to_float_array(obs[k])
            if arr is not None and arr.size <= 16:
                out[k] = arr.reshape(-1).tolist()

    # Keep additional compact state-like keys.
    for k, v in obs.items():
        if k in out:
            continue

        if not isinstance(k, str):
            continue

        key_lower = k.lower()
        looks_state_like = (
            key_lower.endswith("_pos")
            or key_lower.endswith("_quat")
            or key_lower.endswith("_qpos")
            or key_lower.endswith("_vel")
            or "eef" in key_lower
            or "gripper" in key_lower
            or "joint" in key_lower
        )

        if not looks_state_like:
            continue

        arr = to_float_array(v)
        if arr is None:
            continue

        if 0 < arr.size <= 16:
            out[k] = arr.reshape(-1).tolist()

    return out


class PolicyEpisodeLogger:
    def __init__(
        self,
        parent: "FreshPolicyChunkLogger",
        *,
        policy_name: str,
        model_family: str,
        task_suite_name: str,
        task_id: int,
        condition: str,
        init_state_idx: int,
        language: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.parent = parent

        self.policy_name = safe_name(policy_name)
        self.model_family = str(model_family)
        self.task_suite_name = str(task_suite_name)
        self.task_id = int(task_id)
        self.condition = safe_name(condition)
        self.condition_raw = str(condition)
        self.init_state_idx = int(init_state_idx)
        self.language = str(language)
        self.extra_metadata = extra_metadata or {}

        self.episode_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        self.jsonl_dir = (
            self.parent.out_dir
            / "jsonl"
            / self.policy_name
            / self.condition
        )
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_dir = (
            self.parent.out_dir
            / "policy_chunks"
            / f"init_{self.init_state_idx:03d}"
        )
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.jsonl_dir / f"init_{self.init_state_idx:03d}_{self.episode_id}.jsonl"

        self.processed_actions: List[np.ndarray] = []
        self.raw_actions_before_process: List[np.ndarray] = []
        self.raw_action_chunks: List[np.ndarray] = []
        self.query_steps: List[int] = []

        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.infos: List[Any] = []

        self.start_time = time.time()
        self.closed = False

        self._write_jsonl(
            {
                "type": "metadata",
                "timestamp": self.start_time,
                "episode_id": self.episode_id,
                "policy_name": self.policy_name,
                "model_family": self.model_family,
                "task_suite_name": self.task_suite_name,
                "task_id": self.task_id,
                "condition": self.condition_raw,
                "condition_safe": self.condition,
                "init_state_idx": self.init_state_idx,
                "language": self.language,
                "extra_metadata": self.extra_metadata,
            }
        )

    def _write_jsonl(self, obj: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(json_safe(obj), ensure_ascii=False) + "\n")

    def log_initial_obs(self, obs: Any) -> None:
        self._write_jsonl(
            {
                "type": "initial_obs",
                "init_state_idx": self.init_state_idx,
                "obs_summary": summarize_obs(obs),
            }
        )

    def log_action_query(
        self,
        *,
        step: int,
        raw_actions_chunk: Any,
        query_obs: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        arr = to_float_array(raw_actions_chunk)

        if arr is not None:
            if arr.ndim == 1 and arr.shape[0] == 7:
                arr = arr.reshape(1, 7)
            if arr.ndim == 2 and arr.shape[1] == 7:
                self.raw_action_chunks.append(arr.astype(np.float32))
                self.query_steps.append(int(step))

        self._write_jsonl(
            {
                "type": "action_query",
                "step": int(step),
                "raw_actions_chunk": arr.tolist() if arr is not None else None,
                "query_obs_summary": summarize_obs(query_obs),
                "extra": extra or {},
            }
        )

    def log_step(
        self,
        *,
        step: int,
        raw_action_before_process: Any,
        processed_action: Any,
        obs_before: Any,
        obs_after: Any,
        reward: float,
        done: bool,
        info: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_arr = to_float_array(raw_action_before_process)
        proc_arr = to_float_array(processed_action)

        if raw_arr is not None and raw_arr.shape == (7,):
            self.raw_actions_before_process.append(raw_arr.astype(np.float32))

        if proc_arr is not None and proc_arr.shape == (7,):
            self.processed_actions.append(proc_arr.astype(np.float32))

        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.infos.append(info)

        self._write_jsonl(
            {
                "type": "step",
                "step": int(step),
                "raw_action_before_process": raw_arr.tolist() if raw_arr is not None else None,
                "processed_action": proc_arr.tolist() if proc_arr is not None else None,
                "reward": float(reward),
                "done": bool(done),
                "info": json_safe(info),
                "obs_before_summary": summarize_obs(obs_before),
                "obs_after_summary": summarize_obs(obs_after),
                "obs_summary": summarize_obs(obs_after),
                "extra": extra or {},
            }
        )

    def close(
        self,
        *,
        success: Optional[bool] = None,
        final_info: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        if self.closed:
            raise RuntimeError("Episode already closed")

        self.closed = True
        elapsed = time.time() - self.start_time

        actions = (
            np.stack(self.processed_actions, axis=0).astype(np.float32)
            if self.processed_actions
            else np.zeros((0, 7), dtype=np.float32)
        )

        raw_actions = (
            np.stack(self.raw_actions_before_process, axis=0).astype(np.float32)
            if self.raw_actions_before_process
            else np.zeros((0, 7), dtype=np.float32)
        )

        if self.raw_action_chunks:
            try:
                raw_chunks = np.stack(self.raw_action_chunks, axis=0).astype(np.float32)
            except Exception:
                raw_chunks = np.asarray(self.raw_action_chunks, dtype=object)
        else:
            raw_chunks = np.zeros((0, 0, 7), dtype=np.float32)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.bool_)

        chunk_name = safe_name(
            f"{self.policy_name}_{self.condition}_task{self.task_id}_init{self.init_state_idx:03d}_{self.episode_id}"
        )

        npz_path = self.chunk_dir / f"{chunk_name}.npz"

        meta = {
            "episode_id": self.episode_id,
            "policy_name": self.policy_name,
            "model_family": self.model_family,
            "task_suite_name": self.task_suite_name,
            "task_id": self.task_id,
            "condition": self.condition_raw,
            "condition_safe": self.condition,
            "init_state_idx": self.init_state_idx,
            "language": self.language,
            "success": success,
            "elapsed_sec": elapsed,
            "jsonl_path": str(self.jsonl_path),
            "extra_metadata": self.extra_metadata,
            "final_info": json_safe(final_info),
            "extra": extra or {},
        }

        np.savez_compressed(
            npz_path,
            actions=actions,
            processed_actions=actions,
            raw_actions_before_process=raw_actions,
            raw_actions_chunks=raw_chunks,
            query_steps=np.asarray(self.query_steps, dtype=np.int32),
            rewards=rewards,
            dones=dones,
            policy_name=self.policy_name,
            model_family=self.model_family,
            task_suite_name=self.task_suite_name,
            task_id=np.asarray(self.task_id, dtype=np.int32),
            condition=self.condition_raw,
            condition_safe=self.condition,
            init_state_idx=np.asarray(self.init_state_idx, dtype=np.int32),
            language=self.language,
            chunk_name=chunk_name,
            source_path=str(self.jsonl_path),
            success=np.asarray(-1 if success is None else int(bool(success)), dtype=np.int32),
            used_num_steps=np.asarray(actions.shape[0], dtype=np.int32),
            meta_json=json.dumps(json_safe(meta), ensure_ascii=False),
        )

        self._write_jsonl(
            {
                "type": "episode_end",
                "episode_id": self.episode_id,
                "success": success,
                "num_steps": int(actions.shape[0]),
                "sum_reward": float(np.sum(rewards)) if rewards.size else 0.0,
                "elapsed_sec": elapsed,
                "npz_path": str(npz_path),
                "final_info": json_safe(final_info),
                "extra": extra or {},
            }
        )

        self.parent.append_manifest(
            {
                "episode_id": self.episode_id,
                "policy_name": self.policy_name,
                "model_family": self.model_family,
                "task_suite_name": self.task_suite_name,
                "task_id": self.task_id,
                "condition": self.condition_raw,
                "condition_safe": self.condition,
                "init_state_idx": self.init_state_idx,
                "language": self.language,
                "success": "" if success is None else int(bool(success)),
                "num_steps": int(actions.shape[0]),
                "sum_reward": float(np.sum(rewards)) if rewards.size else 0.0,
                "jsonl_path": str(self.jsonl_path),
                "npz_path": str(npz_path),
            }
        )

        return npz_path


class FreshPolicyChunkLogger:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.csv"

        if not self.manifest_path.exists():
            with open(self.manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()

    def append_manifest(self, row: Dict[str, Any]) -> None:
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
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
                ],
            )
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    def start_episode(
        self,
        *,
        policy_name: str,
        model_family: str,
        task_suite_name: str,
        task_id: int,
        condition: str,
        init_state_idx: int,
        language: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> PolicyEpisodeLogger:
        return PolicyEpisodeLogger(
            parent=self,
            policy_name=policy_name,
            model_family=model_family,
            task_suite_name=task_suite_name,
            task_id=task_id,
            condition=condition,
            init_state_idx=init_state_idx,
            language=language,
            extra_metadata=extra_metadata,
        )