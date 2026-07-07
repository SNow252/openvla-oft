#!/usr/bin/env bash
set -e

cd ~/projects/vla_lang_generalization/openvla-oft

source /home/asd/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

export PYTHONPATH=$PWD/LIBERO
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CKPT=/home/asd/projects/vla_lang_generalization/hf_models/openvla-7b-oft-finetuned-libero-spatial

# language templates from LIBERO-spatial task IDs
LANG_3="pick up the black bowl on the cookie box and place it on the plate"
LANG_8="pick up the black bowl next to the plate and place it on the plate"
LANG_9="pick up the black bowl on the wooden cabinet and place it on the plate"

ENVS=(3 8 9)
LANGS=(3 8 9)

for ENV_ID in "${ENVS[@]}"; do
  export OPENVLA_TASK_IDS=$ENV_ID

  for LANG_ID in "${LANGS[@]}"; do
    if [ "$LANG_ID" = "3" ]; then
      export OPENVLA_CUSTOM_LANGUAGE="$LANG_3"
    elif [ "$LANG_ID" = "8" ]; then
      export OPENVLA_CUSTOM_LANGUAGE="$LANG_8"
    elif [ "$LANG_ID" = "9" ]; then
      export OPENVLA_CUSTOM_LANGUAGE="$LANG_9"
    fi

    echo "============================================================"
    echo "OpenVLA-OFT 3x3 matrix"
    echo "ENV_ID=$ENV_ID"
    echo "LANG_ID=$LANG_ID"
    echo "PROMPT=$OPENVLA_CUSTOM_LANGUAGE"
    echo "============================================================"

    python -u experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "$CKPT" \
      --task_suite_name libero_spatial \
      --num_trials_per_task 20 \
      --center_crop True \
      --run_id_note "matrix_env${ENV_ID}_lang${LANG_ID}" \
      2>&1 | tee "/tmp/openvla_oft_matrix_env${ENV_ID}_lang${LANG_ID}.log"
  done
done
