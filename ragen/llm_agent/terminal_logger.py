"""
Terminal logger with Ray Compatibility, Non-blocking Flush, and Loop Prevention.
Fixes the infinite logging loop by disabling propagation.

Author: AI Assistant
Date: 2025-12-21
"""
import os
import sys
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

class TerminalLogger:
    def __init__(self, experiment_name: str = "experiment", wait_timeout: int = 600):
        self.experiment_name = experiment_name
        self.wait_timeout = wait_timeout
        self.log_dir = None
        self._initialized = False 
        
        # 内存缓存区
        self._log_buffer: List[str] = []
        self._buffer_lock = threading.Lock()
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Loggers
        self.full_logger = None
        self.error_logger = None
        
        self.start_capture()
        
        self._monitor_thread = threading.Thread(target=self._wait_for_swanlab, daemon=True)
        self._monitor_thread.start()
    
    def _wait_for_swanlab(self):
        """Background thread: Wait for SwanLab directory to be created."""
        start_time = time.time()
        
        while time.time() - start_time < self.wait_timeout:
            try:
                import swanlab
                if swanlab.get_run() and hasattr(swanlab.get_run(), 'settings'):
                    run = swanlab.get_run()
                    target_dir = getattr(run.settings, 'console_dir', None) or getattr(run.settings, 'run_dir', None)
                    if target_dir:
                        run_dir = Path(target_dir).parent if str(target_dir).endswith("console") else Path(target_dir)
                        self._setup_loggers(run_dir)
                        return
            except Exception:
                pass
            
            swanlab_log_dir = os.environ.get('SWANLAB_LOG_DIR', None)
            if swanlab_log_dir and os.path.exists(swanlab_log_dir):
                if os.path.basename(swanlab_log_dir) == "swanlog":
                    try:
                        subdirs = [
                            os.path.join(swanlab_log_dir, d) 
                            for d in os.listdir(swanlab_log_dir) 
                            if os.path.isdir(os.path.join(swanlab_log_dir, d)) and d.startswith("run-")
                        ]
                        if subdirs:
                            latest_run_dir = max(subdirs, key=os.path.getmtime)
                            if os.path.getmtime(latest_run_dir) > start_time - 30: 
                                self._setup_loggers(Path(latest_run_dir))
                                return
                    except OSError:
                        pass
                else:
                    self._setup_loggers(Path(swanlab_log_dir))
                    return

            time.sleep(1)
        
        self.original_stdout.write(f"[TerminalLogger] Warning: SwanLab not initialized after {self.wait_timeout}s.\n")

    def _setup_loggers(self, run_dir: Path):
        """Initialize file loggers and flush buffer safely."""
        try:
            self.log_dir = run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_log_path = self.log_dir / f"terminal_output.log"
            error_log_path = self.log_dir / f"terminal_errors.log"
            
            self.full_logger = self._create_file_logger("full", full_log_path, logging.INFO)
            self.error_logger = self._create_file_logger("error", error_log_path, logging.WARNING)
            
            self.original_stdout.write(f"\n[TerminalLogger] ✅ Detected SwanLab! Saving logs to: {self.log_dir}\n")
            
            messages_to_write = []
            with self._buffer_lock:
                self._initialized = True
                messages_to_write = list(self._log_buffer)
                self._log_buffer = []
            
            if messages_to_write:
                self.original_stdout.write(f"[TerminalLogger] Flushing {len(messages_to_write)} buffered lines to file...\n")
                for line in messages_to_write:
                    self._write_to_file(line)
                    
        except Exception as e:
            self.original_stdout.write(f"[TerminalLogger] Error setting up loggers: {e}\n")

    def _create_file_logger(self, name_suffix, path, level):
        """Helper to create a standard file logger."""
        logger = logging.getLogger(f"term_{self.experiment_name}_{name_suffix}")
        
        # -------------------------------------------------------------
        # 核心修复：禁止日志向上传播，防止打印回终端造成死循环
        # -------------------------------------------------------------
        logger.propagate = False 
        
        logger.setLevel(level)
        logger.handlers.clear()
        
        handler = logging.FileHandler(path, mode='a', encoding='utf-8')
        handler.setLevel(level)
        # 保持格式简洁，因为原始 output 已经包含了必要信息
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        return logger

    def _write_to_file(self, message: str):
        if not self.full_logger: return
        try:
            clean_msg = message.rstrip()
            self.full_logger.info(clean_msg)
            
            msg_lower = clean_msg.lower()
            if self.error_logger and any(k in msg_lower for k in ['error', 'exception', 'traceback']):
                self.error_logger.error(clean_msg)
        except Exception:
            pass

    class StreamCapture:
        def __init__(self, original_stream, parent_logger):
            self.original_stream = original_stream
            self.parent = parent_logger
        
        def write(self, message):
            # 1. 写回原始终端
            try:
                self.original_stream.write(message)
                self.original_stream.flush()
            except Exception:
                pass 
            
            # 2. 写日志
            if message:
                try:
                    if self.parent._initialized:
                         self.parent._write_to_file(message)
                    else:
                        with self.parent._buffer_lock:
                            if self.parent._initialized:
                                self.parent._write_to_file(message)
                            else:
                                self.parent._log_buffer.append(message)
                except Exception:
                    pass
        
        def flush(self):
            try:
                self.original_stream.flush()
            except Exception:
                pass

        def fileno(self):
            return self.original_stream.fileno()

        def isatty(self):
            return getattr(self.original_stream, 'isatty', lambda: False)()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)
    
    def start_capture(self):
        sys.stdout = self.StreamCapture(self.original_stdout, self)
        sys.stderr = self.StreamCapture(self.original_stderr, self)
    
    def stop_capture(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# Global instance
_global_terminal_logger = None

def setup_terminal_logging(experiment_name: str = "experiment") -> TerminalLogger:
    global _global_terminal_logger
    if _global_terminal_logger is None:
        _global_terminal_logger = TerminalLogger(experiment_name=experiment_name)
    return _global_terminal_logger