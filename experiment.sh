#!/bin/bash

# 设定超参数范围
ADV_TYPES=("none" "fgm" "pgd" "freelb" "smartp")
FEATURE_TYPES=(1 2)
FUSION_TYPES=("weighted" "gate" "residual" "cat")

BERT_DIRS=("hf_hub/models--hfl--chinese-roberta-wwm-ext" \
           "hf_hub/models--hfl--chinese-macbert-base" \
           "hf_hub/models--hfl--chinese-bert-wwm-ext" \
           "hf_hub/models--Langboat--mengzi-bert-base" \
           "hf_hub/models--nghuyong--ernie-3.0-base-zh")

# ADV_TYPES=("none")
# FEATURE_TYPES=(1)
# FUSION_TYPES=("cat")
# BERT_DIRS=("hf_hub/models--nghuyong--ernie-3.0-base-zh")

# 其他默认参数
BASE_URL="data/qunaer_20250226_balance19"

# 生成所有实验命令
declare -a COMMANDS
for BERT_DIR in "${BERT_DIRS[@]}"; do
  for ADV in "${ADV_TYPES[@]}"; do
    for FEATURE in "${FEATURE_TYPES[@]}"; do
      if [ "$FEATURE" -eq 1 ]; then
        # feature_type=1 时，不使用 fusion_type
        CMD="/data/wuchao/miniconda3/envs/llamaf/bin/python main.py \
            --base_url $BASE_URL \
            --feature_type $FEATURE \
            --adv_type $ADV \
            --bert_dir $BERT_DIR" 
        COMMANDS+=("$CMD")
      else
        # feature_type=2 时，遍历 fusion_type
        for FUSION in "${FUSION_TYPES[@]}"; do
          CMD="/data/wuchao/miniconda3/envs/llamaf/bin/python main.py \
              --base_url $BASE_URL \
              --feature_type $FEATURE \
              --adv_type $ADV \
              --fusion_type $FUSION \
              --bert_dir $BERT_DIR" 
          COMMANDS+=("$CMD")
        done
      fi
    done
  done
done

# 设置GPU列表和命令队列
GPUS=(4 5 6 7)
declare -a GPU_QUEUE_4 GPU_QUEUE_5 GPU_QUEUE_6 GPU_QUEUE_7

# 将命令分配到不同GPU的队列
for i in "${!COMMANDS[@]}"; do
  GPU_INDEX=$((i % 4))
  case "${GPUS[$GPU_INDEX]}" in
    4) GPU_QUEUE_4+=("${COMMANDS[$i]}") ;;
    5) GPU_QUEUE_5+=("${COMMANDS[$i]}") ;;
    6) GPU_QUEUE_6+=("${COMMANDS[$i]}") ;;
    7) GPU_QUEUE_7+=("${COMMANDS[$i]}") ;;
  esac
done

# 定义运行函数
run_gpu_queue() {
  local GPU_ID=$1
  shift
  local QUEUE=("$@")
  for CMD in "${QUEUE[@]}"; do
    echo "Starting experiment on GPU $GPU_ID: $CMD"
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD
  done
}

# 并行执行不同GPU队列
run_gpu_queue 4 "${GPU_QUEUE_4[@]}" &
run_gpu_queue 5 "${GPU_QUEUE_5[@]}" &
run_gpu_queue 6 "${GPU_QUEUE_6[@]}" &
run_gpu_queue 7 "${GPU_QUEUE_7[@]}" &

# 等待所有后台任务完成
wait

echo "All experiments completed!"