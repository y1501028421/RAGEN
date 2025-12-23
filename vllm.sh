CUDA_VISIBLE_DEVICES=6,7 vllm serve /mnt/public/yixiangmin/model/Qwen3-0.6B \
  --port 8003 \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 2 \
  --reasoning-parser deepseek_r1 \
  --served-model-name Qwen3-0.6B