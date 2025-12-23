"""
Conversation logger for recording multi-turn dialogue histories.
Logs complete tokenized conversations with special tokens preserved.
Auto-detects SwanLab run directory for storage.

Author: AI Assistant
Date: 2025-12-21
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class ConversationLogger:
    """
    Records complete tokenized multi-turn conversation histories.
    Saves the exact input_ids with all special tokens (<|im_start|>, <|im_end|>, etc.) preserved.
    Automatically finds the current SwanLab run directory to save logs.
    """
    
    def __init__(self, tokenizer, log_dir: Optional[str] = None, experiment_name: str = "experiment"):
        """
        Initialize conversation logger.
        
        Args:
            tokenizer: Tokenizer instance for decoding
            log_dir: (Optional) Force specific directory. If None, auto-detects SwanLab run dir.
            experiment_name: Name of the experiment for organizing logs
        """
        self.tokenizer = tokenizer
        self.experiment_name = experiment_name
        self.log_dir = None
        self.logger = None
        self.readable_logger = None
        self._initialized = False
        
        # 如果显式传入了路径，则立即初始化；否则等到第一次 log 时再动态查找
        if log_dir is not None:
            self._setup_loggers(Path(log_dir))
    
    def _setup_loggers(self, run_dir: Path):
        """Setup file loggers inside the run directory."""
        # 保存到 run-xxx/logs 目录下
        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup JSON logger (machine readable)
        self.logger = logging.getLogger(f"conversation_logger_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        log_file = self.log_dir / "conversations.jsonl"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Setup readable logger (human readable)
        readable_log_file = self.log_dir / "conversations_readable.txt"
        self.readable_logger = logging.getLogger(f"conversation_readable_{self.experiment_name}")
        self.readable_logger.setLevel(logging.INFO)
        self.readable_logger.handlers.clear()
        
        readable_handler = logging.FileHandler(readable_log_file, mode='a', encoding='utf-8')
        self.readable_logger.addHandler(readable_handler)
        
        self._initialized = True
        print(f"[ConversationLogger] ✅ Initialized. Conversation logs will be saved to: {self.log_dir}")
    
    def _ensure_initialized(self):
        """
        Ensure logger is initialized. 
        If not, try to find the specific SwanLab run directory dynamically.
        """
        if self._initialized:
            return True
        
        # 1. 尝试直接从 SwanLab API 获取当前运行对象 (最准确)
        try:
            import swanlab
            run = swanlab.get_run()
            if run and hasattr(run, 'settings'):
                # console_dir 通常是 .../swanlog/run-xxx/console
                # 我们想存到 .../swanlog/run-xxx/logs
                target_dir = getattr(run.settings, 'console_dir', None) or getattr(run.settings, 'run_dir', None)
                
                if target_dir:
                    # 如果拿到的是 console 目录，取父级；否则直接用
                    path_obj = Path(target_dir)
                    run_root = path_obj.parent if path_obj.name == "console" else path_obj
                    
                    self._setup_loggers(run_root)
                    return True
        except (ImportError, AttributeError):
            pass # 如果没装 swanlab 或者没 init，继续往下走

        # 2. 降级方案：扫描文件系统 (SWANLAB_LOG_DIR)
        swanlab_log_dir = os.environ.get('SWANLAB_LOG_DIR', None)
        if swanlab_log_dir and os.path.exists(swanlab_log_dir):
            root_path = Path(swanlab_log_dir)
            
            # 如果环境变量指向的是根目录 'swanlog'
            if root_path.name == "swanlog":
                try:
                    # 找到所有 run- 开头的子目录
                    subdirs = [
                        x for x in root_path.iterdir() 
                        if x.is_dir() and x.name.startswith("run-")
                    ]
                    if subdirs:
                        # 找到修改时间最新的那个文件夹
                        latest_run_dir = max(subdirs, key=lambda p: p.stat().st_mtime)
                        self._setup_loggers(latest_run_dir)
                        return True
                except Exception as e:
                    print(f"[ConversationLogger] Error scanning log dir: {e}")
            else:
                # 如果环境变量本身就已经指向了具体的 run 目录
                self._setup_loggers(root_path)
                return True
        
        # 如果什么都没找到，可以选择不记录，或者记录到临时目录（这里选择直接返回 False）
        return False
    
    def log_conversation(self, 
                        env_id: int,
                        env_output: Dict[str, Any],
                        input_ids: List[int],
                        step: int,
                        phase: str = "train"):
        """
        Log a complete conversation with original tokenized input.
        """
        # 关键点：每次写日志前检查初始化。
        # 由于这是在训练循环中调用的，此时 swanlab.init 肯定已经完成，_ensure_initialized 基本能一次成功。
        if not self._ensure_initialized():
            return  
        
        # Decode the full input with special tokens preserved
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            non_pad_indices = [i for i, token_id in enumerate(input_ids) if token_id != pad_token_id]
            if non_pad_indices:
                first_non_pad = non_pad_indices[0]
                input_ids = input_ids[first_non_pad:]
        
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "phase": phase,
            "env_id": env_id,
            "group_id": env_output.get("group_id", -1),
            "tag": env_output.get("tag", "unknown"),
            "full_tokenized_conversation": full_text,
            "input_ids_length": len(input_ids),
            "turns": []
        }
        
        # Parse turns from history
        history = env_output.get("history", [])
        for turn_idx, turn_data in enumerate(history):
            turn_info = {
                "turn_number": turn_idx + 1,
            }
            
            if "state" in turn_data:
                turn_info["state"] = turn_data["state"]
                turn_info["actions_left"] = turn_data.get("actions_left", 0)
            
            if "llm_response" in turn_data:
                turn_info["llm_response"] = turn_data["llm_response"]
                
                # Parse thinking and answer
                llm_response = turn_data["llm_response"]
                if '<think>' in llm_response and '</think>' in llm_response:
                    think_start = llm_response.find('<think>') + 7
                    think_end = llm_response.find('</think>')
                    turn_info["thinking"] = llm_response[think_start:think_end].strip()
                
                if '<answer>' in llm_response and '</answer>' in llm_response:
                    answer_start = llm_response.find('<answer>') + 8
                    answer_end = llm_response.find('</answer>')
                    turn_info["answer"] = llm_response[answer_start:answer_end].strip()
            
            if "reward" in turn_data:
                turn_info["reward"] = turn_data["reward"]
            
            if "actions" in turn_data:
                turn_info["actions"] = turn_data["actions"]
            
            conversation_data["turns"].append(turn_info)
        
        # Add metrics
        if "metrics" in env_output:
            conversation_data["metrics"] = env_output["metrics"]
        
        # Log as JSON line
        if self.logger:
            self.logger.info(json.dumps(conversation_data, ensure_ascii=False))
        
        # Log human-readable version
        self._log_readable_conversation(conversation_data)
    
    def _log_readable_conversation(self, conversation_data: Dict[str, Any]):
        """Create a human-readable version of the conversation."""
        if not self.readable_logger:
            return

        lines = []
        lines.append("=" * 80)
        lines.append(f"Step: {conversation_data['step']} | Phase: {conversation_data['phase']} | "
                    f"Env ID: {conversation_data['env_id']} | Group: {conversation_data['group_id']}")
        lines.append(f"Tag: {conversation_data.get('tag', 'N/A')} | Time: {conversation_data['timestamp']}")
        lines.append(f"Input IDs Length: {conversation_data['input_ids_length']}")
        lines.append("-" * 80)
        
        full_conv = conversation_data.get("full_tokenized_conversation", "")
        if len(full_conv) > 2000:
            lines.append(f"Full Conversation (truncated): {full_conv[:2000]}...")
        else:
            lines.append(f"Full Conversation: {full_conv}")
        lines.append("-" * 80)
        
        for turn in conversation_data.get("turns", []):
            lines.append(f"\n--- Turn {turn['turn_number']} ---")
            
            if "state" in turn:
                state = turn['state']
                if len(state) > 300:
                    state = state[:300] + "..."
                lines.append(f"State: {state}")
                
            if "thinking" in turn:
                lines.append(f"Thinking: {turn['thinking']}")
            
            if "answer" in turn:
                lines.append(f"Answer: {turn['answer']}")
            
            if "llm_response" in turn and "thinking" not in turn and "answer" not in turn:
                lines.append(f"LLM Response: {turn['llm_response']}")
            
            if "actions" in turn:
                lines.append(f"Actions: {turn['actions']}")
            
            if "reward" in turn:
                lines.append(f"Reward: {turn['reward']}")
        
        if "metrics" in conversation_data:
            lines.append("\n" + "-" * 80)
            lines.append("Metrics:")
            for key, value in conversation_data["metrics"].items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 80)
        lines.append("")
        
        self.readable_logger.info("\n".join(lines))
    
    def log_batch_conversations(self,
                                env_outputs: List[Dict[str, Any]],
                                input_ids_batch: Any,
                                step: int,
                                phase: str = "train",
                                max_samples: Optional[int] = None):
        """
        Log conversations from a batch of environments.
        """
        if not self._ensure_initialized():
            return
        
        if hasattr(input_ids_batch, 'tolist'):
            input_ids_batch = input_ids_batch.tolist()
        
        num_samples = len(env_outputs)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        
        for i in range(num_samples):
            env_output = env_outputs[i]
            input_ids = input_ids_batch[i] if i < len(input_ids_batch) else []
            env_id = env_output.get("env_id", i)
            
            self.log_conversation(
                env_id=env_id,
                env_output=env_output,
                input_ids=input_ids,
                step=step,
                phase=phase
            )
    
    def close(self):
        """Close all file handlers."""
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        if self.readable_logger:
            for handler in self.readable_logger.handlers[:]:
                handler.close()
                self.readable_logger.removeHandler(handler)