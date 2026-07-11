#!/usr/bin/env bash
set -euo pipefail

cd /home/asd/projects/vla_lang_generalization/openvla-oft

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYTHONPATH=$PWD/LIBERO:$PWD:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

export OPENVLA_TASK_IDS=8
unset OPENVLA_CUSTOM_LANGUAGE

export OPENVLA_CKPT=/home/asd/projects/vla_lang_generalization/hf_models/openvla-7b-oft-finetuned-libero-spatial

export FRESH_DUMP_DIR=/home/asd/projects/vla_lang_generalization/goal_controllability_verifier/data/fresh_policy_dumps_v4/openvla_task8_original_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FRESH_DUMP_DIR"

echo "[INFO] FRESH_DUMP_DIR=$FRESH_DUMP_DIR"

python experiments/robot/libero/run_libero_eval_fresh_dump_v4.py \
  --model_family openvla \
  --pretrained_checkpoint "$OPENVLA_CKPT" \
  --task_suite_name libero_spatial \
  --num_trials_per_task 2 \
  --num_open_loop_steps 8 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --center_crop True \
  --lora_rank 32 \
  --load_in_8bit False \
  --load_in_4bit False \
  --env_img_res 256 \
  --fresh_dump_dir "$FRESH_DUMP_DIR" \
  --fresh_policy_name openvla_oft \
  --fresh_condition task8_original \
  2>&1 | tee "$FRESH_DUMP_DIR/run.log"

echo "[INFO] Done."
echo "[INFO] FRESH_DUMP_DIR=$FRESH_DUMP_DIR"
echo "[INFO] Files:"
find "$FRESH_DUMP_DIR" -maxdepth 5 -type f | sort | head -100