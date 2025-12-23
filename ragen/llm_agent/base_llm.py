from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import os
import asyncio
import time
import os
import re
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from together import AsyncTogether

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation"""
    
    def __init__(self, model_name: str = "deepseek-reasoner", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
        from omegaconf import OmegaConf
        if OmegaConf.is_config(kwargs):
            # 情况 A: 纯 OmegaConf 对象 -> 彻底转为 dict
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
        elif isinstance(kwargs, dict):
            # 情况 B: 外层是 dict，但里层(extra_body)可能还是 OmegaConf 对象
            # 重新构建字典，确保所有值都被递归转换
            kwargs = {
                k: (OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v)
                for k, v in kwargs.items()
            }
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            print(f'[DEBUG] DeepSeek response finish_reason: {response.choices[0].finish_reason}')
            raise ValueError("Content filtered or length exceeded")
        # print(f'[DEBUG] final content: {response.choices[0].message.content}')
        # print(f'[DEBUG] final reasoning_content: {response.choices[0].message.reasoning_content}')
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation
    Refer to https://github.com/anthropics/anthropic-sdk-python
    """
    
    def __init__(self, model_name: str = "claude-3.5-sonnet-20240620", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message if present
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                # Map to Anthropic's format
                chat_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
        response = await self.client.messages.create(
            model=self.model_name,
            system=system_content,
            messages=chat_messages,
            **kwargs
        )
        if response.stop_reason == "max_tokens":
            raise ValueError("Max tokens exceeded")
        return LLMResponse(
            content=response.content[0].text,
            model_name=response.model
        )

class TogetherProvider(LLMProvider):
    """Together AI API provider implementation"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided and not found in environment variables")
        
        self.client = AsyncTogether(api_key=self.api_key)
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class XiaomiProvider(LLMProvider):
    """XiaomiProvider API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError("XiaomiProvider API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.xiaomimimo.com/v1")
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # 1. 检测是否开启了流式传输，默认为 False
        is_stream = kwargs.get("stream", False)
        
        if is_stream:
            response_stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            full_content = []
            full_reasoning = []
            final_model = self.model_name
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # 拼接常规内容
                if delta.content:
                    full_content.append(delta.content)
                
                # 拼接思考内容 (DeepSeek/R1 风格)
                # 注意：不同库版本可能放在 delta.reasoning_content 或 delta.model_extra['reasoning_content']
                reasoning_chunk = getattr(delta, 'reasoning_content', None)
                if reasoning_chunk:
                    full_reasoning.append(reasoning_chunk)
                    
                if chunk.model:
                    final_model = chunk.model

            content="".join(full_content)
            reasoning_content="".join(full_reasoning) if full_reasoning else None
            # print(f'[DEBUG] final streamed content: {content}')
            # print(f'[DEBUG] final streamed reasoning_content: {reasoning_content}')
            return LLMResponse(
                content="".join(full_content),
                model_name=final_model,
                # 将思考内容通过 reasoning_content 字段返回
                # reasoning_content="".join(full_reasoning) if full_reasoning else None
            )

        else:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            choice = response.choices[0]
            
            # 检查过滤原因
            if choice.finish_reason in ['length', 'content_filter']:
                raise ValueError(f"Content filtered or length exceeded: {choice.finish_reason}")
            
            message = choice.message
            
            # 安全获取 reasoning_content
            # OpenAI SDK 可能会把非标准字段藏在 model_extra 或直接属性中，使用 getattr 最稳妥
            reasoning = getattr(message, 'reasoning_content', None)
            
            return LLMResponse(
                content=message.content,
                model_name=response.model,
                # 将思考内容通过 reasoning_content 字段返回
                # reasoning_content=reasoning
            )
        

class InfiniaiProvider(LLMProvider):
    """InfiniaiProvider API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("INFINIAI_API_KEY")
        if not self.api_key:
            raise ValueError("InfiniaiProvider API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://cloud.infini-ai.com/maas/v1")
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # 1. 检测是否开启了流式传输，默认为 False
        is_stream = kwargs.get("stream", False)
        
        if is_stream:
            response_stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            full_content = []
            full_reasoning = []
            final_model = self.model_name
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # 拼接常规内容
                if delta.content:
                    full_content.append(delta.content)
                
                # 拼接思考内容 (DeepSeek/R1 风格)
                # 注意：不同库版本可能放在 delta.reasoning_content 或 delta.model_extra['reasoning_content']
                reasoning_chunk = getattr(delta, 'reasoning_content', None)
                if reasoning_chunk:
                    full_reasoning.append(reasoning_chunk)
                    
                if chunk.model:
                    final_model = chunk.model

            content="".join(full_content)
            reasoning_content="".join(full_reasoning) if full_reasoning else None
            print(f'[DEBUG] final messages content: {messages}')
            print(f'[DEBUG] final content: {content}')
            print(f'[DEBUG] final reasoning_content: {reasoning_content}')
            return LLMResponse(
                content="".join(full_content),
                model_name=final_model,
                # 将思考内容通过 reasoning_content 字段返回
                # reasoning_content="".join(full_reasoning) if full_reasoning else None
            )

        else:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            choice = response.choices[0]
            
            # 检查过滤原因
            if choice.finish_reason in ['length', 'content_filter']:
                raise ValueError(f"Content filtered or length exceeded: {choice.finish_reason}")
            
            message = choice.message
            
            # 安全获取 reasoning_content
            # OpenAI SDK 可能会把非标准字段藏在 model_extra 或直接属性中，使用 getattr 最稳妥
            reasoning = getattr(message, 'reasoning_content', None)
            
            return LLMResponse(
                content=message.content,
                model_name=response.model,
                # 将思考内容通过 reasoning_content 字段返回
                # reasoning_content=reasoning
            )


# 假设 LLMResponse 是你自定义的类，保持不变
# from your_module import LLMResponse 

class VLLMProvider: # (这里略去父类继承定义以便展示)
    """VLLMProvider API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("INFINIAI_API_KEY")
        if not self.api_key:
            raise ValueError("VLLMProvider API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="http://0.0.0.0:8003/v1")

    def _extract_reasoning(self, content: str, current_reasoning: Optional[str] = None):
        """
        辅助函数：从文本内容中分离 <think>...</think>
        如果 API 已经返回了 current_reasoning，则优先保留 API 的（虽然 vllm 目前可能为 None）
        """
        if not content:
            return content, current_reasoning

        # 正则匹配 <think> 内容，re.DOTALL 让 . 可以匹配换行符
        pattern = r"<think>(.*?)</think>"
        match = re.search(pattern, content, flags=re.DOTALL)
        
        if match:
            # 提取标签内的思考内容
            extracted_reasoning = match.group(1).strip()
            
            # 从原始内容中删除 <think>...</think> 块，并去除首尾空白
            clean_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
            
            # 如果 API 没有返回 reasoning，就用提取出来的；否则拼接（视具体需求而定）
            final_reasoning = current_reasoning or extracted_reasoning
            
            return clean_content, final_reasoning
        
        # 如果没有匹配到标签，直接返回原始内容
        return content, current_reasoning

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> "LLMResponse": # type: ignore
        is_stream = kwargs.get("stream", False)
        
        if is_stream:
            response_stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            full_content = []
            full_reasoning = []
            final_model = self.model_name
            
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                if delta.content:
                    full_content.append(delta.content)
                
                # 尝试获取 API 返回的 reasoning (DeepSeek 格式)
                reasoning_chunk = getattr(delta, 'reasoning_content', None)
                if reasoning_chunk:
                    full_reasoning.append(reasoning_chunk)
                    
                if chunk.model:
                    final_model = chunk.model

            # 1. 拼接原始文本
            raw_content = "".join(full_content)
            api_reasoning = "".join(full_reasoning) if full_reasoning else None

            # 2. 调用清洗函数：分离 <think> 内容
            final_content, final_reasoning = self._extract_reasoning(raw_content, api_reasoning)

            # print(f'[DEBUG] final streamed clean content: {final_content}')
            # print(f'[DEBUG] final streamed extracted reasoning: {final_reasoning}')
            
            return LLMResponse(
                content=final_content, # 返回清洗后的内容（不含 think）
                model_name=final_model,
                # reasoning_content=final_reasoning # 将思考过程单独存储
            )

        else:
            # 非流式处理
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            choice = response.choices[0]
            if choice.finish_reason in ['length', 'content_filter']:
                raise ValueError(f"Content filtered or length exceeded: {choice.finish_reason}")
            
            message = choice.message
            
            raw_content = message.content
            api_reasoning = getattr(message, 'reasoning_content', None)

            # 同样调用清洗函数
            final_content, final_reasoning = self._extract_reasoning(raw_content, api_reasoning)

            return LLMResponse(
                content=final_content,
                model_name=response.model,
                # reasoning_content=final_reasoning
            )
        
class ConcurrentLLM:
    """Unified concurrent interface for multiple LLM providers"""
    
    def __init__(self, provider: Union[str, LLMProvider], model_name: Optional[str] = None, 
                api_key: Optional[str] = None, max_concurrency: int = 4):
        """
        Initialize the concurrent LLM client.
        
        Args:
            provider: Either a provider instance or a string ('openai', 'anthropic', 'together')
            model_name: Model name (if provider is a string)
            api_key: API key (if provider is a string)
            max_concurrency: Maximum number of concurrent requests
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            if provider.lower() == "openai":
                self.provider = OpenAIProvider(model_name or "gpt-4o", api_key)
            elif provider.lower() == "deepseek":
                self.provider = DeepSeekProvider(model_name or "deepseek-reasoner", api_key)
            elif provider.lower() == "anthropic":
                self.provider = AnthropicProvider(model_name or "claude-3-7-sonnet-20250219", api_key)
            elif provider.lower() == "together":
                self.provider = TogetherProvider(model_name or "meta-llama/Llama-3-70b-chat-hf", api_key)
            elif provider.lower() == "infiniai":
                self.provider = InfiniaiProvider(model_name or "qwen3-8b", api_key)
            elif provider.lower() == "xiaomi":
                self.provider = XiaomiProvider(model_name or "mimo-v2-flash", api_key)
            elif provider.lower() == "local":
                self.provider = VLLMProvider(model_name or "qwen3-4b", api_key)
                
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Store max_concurrency but don't create the semaphore yet
        self.max_concurrency = max_concurrency
        self._semaphore = None
    
    @property
    def semaphore(self):
        """
        Lazy initialization of the semaphore.
        This ensures the semaphore is created in the event loop where it's used.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response with concurrency control"""
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)
    
    def run_batch(self, 
                messages_list: List[List[Dict[str, str]]], 
                **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Process batches with retries in separate event loops, using id() to track messages"""

        results = [None] * len(messages_list)
        position_map = {id(messages): i for i, messages in enumerate(messages_list)}
        
        # Queue to store unfinished or failed tasks
        current_batch = messages_list.copy()
        max_retries = kwargs.get("max_retries", 100)
        retry_count = 0
        
        while current_batch and retry_count < max_retries:
            async def process_batch():
                self._semaphore = None  # Reset semaphore for this event loop
                batch_results = []
                failures = []
                
                tasks_with_messages = [(msg, asyncio.create_task(self.generate(msg, **kwargs))) 
                                    for msg in current_batch]
                for messages, task in tasks_with_messages:
                    try:
                        response = await task
                        position = position_map[id(messages)]
                        batch_results.append((position, {
                            "messages": messages,
                            "response": response.content,
                            "model": response.model_name,
                            "success": True
                        }))
                    except Exception as e:
                        print(f'[DEBUG] error: {e}')
                        failures.append(messages)
                
                return batch_results, failures
            
            # Run in fresh event loop
            batch_results, next_batch = asyncio.run(process_batch())
            
            # Update results with successful responses
            for position, result in batch_results:
                results[position] = result
            
            # Update for next iteration
            if next_batch:
                retry_count += 1
                # Update position map for failed messages
                position_map = {id(messages): position_map[id(messages)] 
                            for messages in next_batch}
                
                current_batch = next_batch
                time.sleep(5)
                print(f'[DEBUG] {len(next_batch)} failed messages, retry_count: {retry_count}')
            else:
                break

        return results, next_batch



if __name__ == "__main__":
    # llm = ConcurrentLLM(provider="openai", model_name="gpt-4o")
    # llm = ConcurrentLLM(provider="anthropic", model_name="claude-3-5-sonnet-20240620")
    llm = ConcurrentLLM(provider="together", model_name="Qwen/Qwen2.5-7B-Instruct-Turbo")
    messages = [
        [{"role": "user", "content": "what is 2+2?"}],
        [{"role": "user", "content": "what is 2+3?"}],
        [{"role": "user", "content": "what is 2+4?"}],
        [{"role": "user", "content": "what is 2+5?"}],
        [{"role": "user", "content": "what is 2+6?"}],
        [{"role": "user", "content": "what is 2+7?"}],
        [{"role": "user", "content": "what is 2+8?"}],
        [{"role": "user", "content": "what is 2+9?"}],
        [{"role": "user", "content": "what is 2+10?"}],
        [{"role": "user", "content": "what is 2+11?"}],
        [{"role": "user", "content": "what is 2+12?"}],
        [{"role": "user", "content": "what is 2+13?"}],
        [{"role": "user", "content": "what is 2+14?"}],
        [{"role": "user", "content": "what is 2+15?"}],
        [{"role": "user", "content": "what is 2+16?"}],
        [{"role": "user", "content": "what is 2+17?"}],
    ]
    response = llm.run_batch(messages, max_tokens=100)
    print(f"final response: {response}")
