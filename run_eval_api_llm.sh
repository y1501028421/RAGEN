#!/bin/bash

# Run API evaluation for Sokoban
# Usage: bash run_eval_api_llm.sh

cd "$(dirname "$0")"

export SWANLAB_MODE='cloud'

python ragen/eval_api.py --config-path "../config_eval_api_llm" --config-name evaluate_api_llm trainer.experiment_name=Qwen3-0-6B-origin-test



#!/bin/bash

# 运行 Sokoban API 评估脚本
# Usage: bash run_eval_api_llm.sh

#!/bin/bash

# Run API evaluation for Sokoban
# Usage: bash run_eval_api_llm.sh

# cd "$(dirname "$0")"

# export SWANLAB_MODE='cloud'

# run_experiment() {
#     local EXP_SUFFIX=$1
#     local MAX_TURN=$2
#     local ACTIONS_PER_TURN=$3
#     local ENABLE_THINK=$4
    
#     local EXP_NAME="Qwen3-0-6B-${EXP_SUFFIX}"

#     echo "========================================================"
#     echo "Start Experiment: ${EXP_NAME}"
#     echo "Settings: Turns=${MAX_TURN}, Acts/Turn=${ACTIONS_PER_TURN}, Thinking=${ENABLE_THINK}"
#     echo "========================================================"

#     # 修正说明：
#     # 1. stream 移到了 generation_kwargs 下
#     # 2. stream_options 保持原位（因为它在 YAML 中是与 generation_kwargs 同级的）

#     python ragen/eval_api.py \
#         --config-path "../config_eval_api_llm" \
#         --config-name evaluate_api_llm \
#         trainer.experiment_name="${EXP_NAME}" \
#         agent_proxy.max_turn=${MAX_TURN} \
#         agent_proxy.max_actions_per_turn=${ACTIONS_PER_TURN} \
#         model_info.Qwen3-0-6B.generation_kwargs.extra_body.enable_thinking=${ENABLE_THINK} \
#         model_info.Qwen3-0-6B.generation_kwargs.stream=${ENABLE_THINK} \
#         model_info.Qwen3-0-6B.stream_options.include_usage=${ENABLE_THINK}
# }

# # --- 实验组 1 & 2 ---
# # run_experiment "mt20-act1-noThink" 20 1 False

# run_experiment "no-limit" 20 10 True

# run_experiment "mt10-act3-Think" 10 3 True

# # --- 实验组 3 & 4 ---
# # run_experiment "mt10-act2-noThink" 10 2 False
# run_experiment "mt10-act2-Think" 10 2 True

# # --- 实验组 5 & 6 ---
# # run_experiment "mt10-act3-noThink" 10 3 False
# run_experiment "mt20-act1-Think" 20 1 True

# echo "All experiments completed."