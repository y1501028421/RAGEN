set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo"
# USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

# Section 3.1&3.2 - General Observations
# python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-ppo $USE_PPO $USE_BASE &
# python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE &
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE &
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-grpo $USE_GRPO $USE_BASE &
# python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=frozen_lake-ppo $USE_PPO $USE_BASE &
# python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=frozen_lake-grpo $USE_GRPO $USE_BASE &

# Section 4.1 - Filtering and critic
# 0.25
# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="5" trainer.n_gpus_per_node=1 trainer.experiment_name=Instruct-sokoban-ppo-rolloutfilter0.25 actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO
python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="6" trainer.n_gpus_per_node=1 trainer.experiment_name=Instruct-sokoban-grpo actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_GRPO

# python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES=\"3,4,5\" trainer.n_gpus_per_node=2 trainer.experiment_name=qwen3-4b-sokoban-ppo-rolloutfilter0.25 actor_rollout_ref.rollout.rollout_filter_ratio=0.25 $USE_PPO