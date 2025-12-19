from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
import time
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from ragen.llm_agent.base_llm import ConcurrentLLM
from ragen.llm_agent.agent_proxy import ApiCallingWrapperWg, VllmWrapperWg, LLMAgentProxy
from hydra.utils import to_absolute_path
from verl.utils.tracking import ValidationGenerationsLogger


def init_swanlab(config):
    """Initialize SwanLab if configured"""
    loggers = config.trainer.logger if hasattr(config.trainer, 'logger') else []
    if 'swanlab' not in loggers:
        return None
    
    try:
        import swanlab
        
        SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
        SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
        # SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
        SWANLAB_MODE = 'offline'
        # if SWANLAB_API_KEY:
        #     swanlab.login(SWANLAB_API_KEY)
        
        swanlab.init(
            project=config.trainer.project_name,
            experiment_name=f"{config.trainer.experiment_name}",
            config=dict(
                model_name=config.model_config.model_name,
                env_groups=config.es_manager.val.env_groups,
                env_tags=list(config.es_manager.val.env_configs.tags),
            ),
            logdir=SWANLAB_LOG_DIR,
            mode=SWANLAB_MODE,
        )
        print("SwanLab initialized successfully")
        return swanlab
    except Exception as e:
        print(f"Failed to initialize SwanLab: {e}")
        import traceback
        traceback.print_exc()
        return None


def log_to_swanlab(swanlab_instance, metrics, rollouts, config, tokenizer):
    """Log results to SwanLab using the same format as verl's ValidationGenerationsLogger"""
    if swanlab_instance is None:
        return
    
    try:
        # Log scalar metrics
        swanlab_instance.log(metrics, step=0)
        
        # Log sample generations using ValidationGenerationsLogger format
        n_samples_to_log = config.trainer.generations_to_log_to_wandb.val
        
        if "input_ids" in rollouts.batch:
            input_ids = rollouts.batch["input_ids"]
            rm_scores = rollouts.batch["rm_scores"].sum(-1).cpu().tolist()
            
            # 使用 skip_special_tokens=True 解码，与 verl 保持一致
            inputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            outputs = [""] * len(inputs)  # API 模式下 output 已经包含在 input 中
            scores = rm_scores
            
            # 构建 samples: [[input, output, score], ...]
            samples = list(zip(inputs, outputs, scores))
            
            # 随机选取 n_samples_to_log 个样本
            import numpy as np
            rng = np.random.RandomState(42)
            rng.shuffle(samples)
            samples = samples[:n_samples_to_log]
            
            # 使用 verl 的 ValidationGenerationsLogger 格式记录
            if samples:
                swanlab_table = swanlab_instance.echarts.Table()
                headers = ["step", "input", "output", "score"]
                swanlab_row_list = [[0, sample[0], sample[1], sample[2]] for sample in samples]
                swanlab_table.add(headers=headers, rows=swanlab_row_list)
                swanlab_instance.log({"val/generations": swanlab_table}, step=0)
                print(f"Logged {len(samples)} samples to SwanLab")
        
        swanlab_instance.finish()
        print("Results logged to SwanLab")
    except Exception as e:
        print(f"Failed to log to SwanLab: {e}")
        import traceback
        traceback.print_exc()


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_api_llm")
def main(config):
    """
    Main function to evaluate LLM performance via API calls.
    
    Config parameters:
    - test_freq: 在训练时每隔多少步进行一次验证（eval_api.py 中未使用）
    - generations_to_log_to_wandb.val: 记录到 wandb/swanlab 的样本数量
    """
    print("="*60)
    print("Starting API Evaluation")
    print("="*60)
    print(f"Model: {config.model_config.model_name}")
    print(f"Provider: {config.model_info[config.model_config.model_name].provider_name}")
    print(f"Env groups: {config.es_manager.val.env_groups}")
    print(f"Env tags: {config.es_manager.val.env_configs.tags}")
    print(f"Logger: {config.trainer.logger}")
    print(f"Samples to log: {config.trainer.generations_to_log_to_wandb.val}")
    print("="*60)
    
    # Initialize SwanLab
    swanlab_instance = init_swanlab(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    
    # Initialize API wrapper
    actor_wg = ApiCallingWrapperWg(config, tokenizer)
    
    # Initialize agent proxy
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    
    # Run rollout
    start_time = time.time()
    rollouts = proxy.rollout(
        DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                'eos_token_id': 151645,
                'pad_token_id': 151643,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True
            }
        ),
        val=True
    )
    end_time = time.time()
    
    # Debug: print available keys
    print(f"\n[DEBUG] rollouts.batch keys: {list(rollouts.batch.keys())}")
    print(f"[DEBUG] rollouts.non_tensor_batch keys: {list(rollouts.non_tensor_batch.keys())}")
    
    # Calculate results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Rollout time: {end_time - start_time:.2f} seconds")
    
    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    avg_reward = rm_scores.sum(-1).mean().item()
    
    print(f"\nAverage Reward: {avg_reward:.4f}")
    print(f"\nDetailed Metrics:")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Prepare metrics for logging
    log_metrics = {
        "val/avg_reward": avg_reward,
        "val/rollout_time": end_time - start_time,
    }
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            log_metrics[f"val/{k}"] = v
    
    # Log to SwanLab
    log_to_swanlab(swanlab_instance, log_metrics, rollouts, config, tokenizer)
    
    print("="*60)


if __name__ == "__main__":
    main()
