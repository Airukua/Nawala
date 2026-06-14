"""
LLM Wrapper with unified interface for multiple providers.

Supports both local (Ollama, llama-cpp-python, HuggingFace Transformers) and
cloud-based LLM services (OpenAI, Anthropic, Google Gemini, Groq, Cohere, Mistral,
DeepSeek, OpenRouter, AWS Bedrock, Vertex AI, Azure OpenAI, Perplexity, etc.).

Key Features:
- Provider-agnostic API for chat completions and streaming
- Async-first design with sync wrappers for convenience
- Unified request/response models (ProviderMessage, ProviderResponse)
- Automatic retries with exponential backoff and jitter
- Comprehensive error classification (rate limits, auth, timeouts, etc.)
- Plugin-style provider registration via factory pattern
- Configurable timeouts, max tokens, temperature, and other generation parameters
- Thread-safe and connection pooling via httpx
- Optional structured output (JSON/Pydantic) and tool calling support
- Detailed token usage and timing metrics

Based on design patterns from: litellm, abstractcore, llm-provider-abstraction, llm-sdk-core.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .enums import Role
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    Type
)

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    RetryCallState,
    AsyncRetrying,
    RetryError,
)
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None
    PYDANTIC_AVAILABLE = False

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from ollama import AsyncClient as OllamaClient
except ImportError:
    OllamaClient = None

try:
    from llama_cpp import Llama as LlamaCPP
except ImportError:
    LlamaCPP = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class ProviderConfig:
    """
    Configuration for an LLM provider.
    
    Why this exists:
    - Centralizes all provider-specific settings into a single dataclass.
    - Allows validation of required fields per provider type.
    - Makes it easy to pass configuration through the factory.
    
    Attributes:
        provider_type: Type of provider (openai, anthropic, ollama, etc.).
        model: Model identifier (e.g., "gpt-4", "claude-3-opus").
        api_key: API key for cloud providers (optional for local).
        base_url: Custom endpoint URL (for proxies or OpenAI-compatible servers).
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        presence_penalty: Presence penalty (-2.0 to 2.0).
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        stop: Stop sequences (list of strings).
        seed: Random seed for deterministic outputs.
        metadata: Additional provider-specific parameters.
    """
    provider_type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(
        cls,
        provider_type: str,
        model: Optional[str] = None,
        **overrides,
    ) -> "ProviderConfig":
        """
        Create a configuration from environment variables.
        
        Environment variables follow the pattern:
            {PROVIDER_UPPER}_API_KEY (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
            {PROVIDER_UPPER}_BASE_URL (e.g., OPENAI_BASE_URL)
            {PROVIDER_UPPER}_MODEL (e.g., OPENAI_MODEL)
        
        Args:
            provider_type: Type of provider (openai, anthropic, etc.).
            model: Override model name (takes precedence over env var).
            **overrides: Any additional config fields to override.
        
        Returns:
            A populated ProviderConfig instance.
        """
        env_prefix = provider_type.upper()
        api_key = os.environ.get(f"{env_prefix}_API_KEY", overrides.pop("api_key", None))
        base_url = os.environ.get(f"{env_prefix}_BASE_URL", overrides.pop("base_url", None))
        env_model = os.environ.get(f"{env_prefix}_MODEL")
        
        return cls(
            provider_type=provider_type,
            model=model or env_model or "unknown",
            api_key=api_key,
            base_url=base_url,
            **overrides,
        )


@dataclass
class Message:
    """
    Structured message format used internally by the wrapper.
    
    This abstraction shields the application from provider-specific message
    formats (e.g., OpenAI's role/content arrays vs. Anthropic's system/user).
    
    Attributes:
        role: Message role (system, user, assistant, tool).
        content: Message content as string or list of content parts.
        name: Optional name for tool messages.
        tool_call_id: Optional tool call ID for tool responses.
    """
    role: Role
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    @classmethod
    def user(cls, content: Union[str, List[Dict[str, Any]]]) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: Union[str, List[Dict[str, Any]]]) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        """Create a tool response message."""
        return cls(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id)


@dataclass
class ProviderResponse:
    """
    Standardized response format from any LLM provider.
    
    This unified structure allows application code to handle responses
    consistently regardless of which underlying provider is used.
    
    Attributes:
        text: Generated response text.
        model: Model identifier used for generation.
        usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens).
        finish_reason: Why generation stopped ("stop", "length", "tool_calls", etc.).
        raw_response: Provider-specific raw response for debugging/extensibility.
        latency_ms: Request latency in milliseconds.
        tool_calls: Optional list of tool call objects (for function calling).
    """
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    latency_ms: float = 0.0
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def total_tokens(self) -> int:
        """Return total tokens used (prompt + completion)."""
        return self.usage.get("total_tokens", 0)
    
    def prompt_tokens(self) -> int:
        """Return prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)
    
    def completion_tokens(self) -> int:
        """Return completion tokens generated."""
        return self.usage.get("completion_tokens", 0)



class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    pass


class ProviderError(LLMError):
    """Provider-specific error (e.g., API returned an error response)."""
    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(ProviderError):
    """API key invalid or missing."""
    pass


class TimeoutError(LLMError):
    """Request timeout exceeded."""
    pass


class ConfigurationError(LLMError):
    """Invalid configuration or missing required parameters."""
    pass


class StreamingError(LLMError):
    """Error during streaming response."""
    pass



class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Design philosophy:
    - Each provider implements the same set of methods (`generate`, `stream`).
    - Provider implementations handle the conversion between the wrapper's
      internal Message format and the provider's native format.
    - Async-first: all network I/O is done asynchronously.
    - Retries are applied at the provider level using tenacity.
    
    Why this approach:
    - Allows the application to be completely agnostic to the underlying provider.
    - New providers can be added by implementing this interface.
    - The factory pattern (see LLMProviderFactory) decouples creation from usage.
    
    Based on the adapter pattern used in litellm and abstractcore.
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider configuration containing API keys, model, etc.
        
        Raises:
            ConfigurationError: If required dependencies are missing or config invalid.
        """
        self.config = config
        self._logger = logging.getLogger(f"LLMProvider.{config.provider_type}")
        self._validate_config()
        self._client = None
        self._init_client()
    
    def _validate_config(self) -> None:
        """Validate provider-specific configuration. Override in subclasses."""
        pass
    
    def _init_client(self) -> None:
        """Initialize the underlying HTTP client or SDK. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """Internal generation method implemented by each provider."""
        pass
    
    async def generate(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of Message objects representing the conversation.
            **kwargs: Override config parameters (temperature, max_tokens, etc.).
        
        Returns:
            ProviderResponse containing the generated text and metadata.
        
        Raises:
            LLMError: If generation fails after retries.
        """
        start_time = time.perf_counter()
        
        try:
            response = await self._generate_with_retry(messages, **kwargs)
            response.latency_ms = (time.perf_counter() - start_time) * 1000
            return response
        except Exception as e:
            self._logger.error(f"Generation failed: {type(e).__name__}: {str(e)}")
            raise self._wrap_exception(e) from e
    
    @abstractmethod
    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Internal streaming method implemented by each provider."""
        pass
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM token by token.
        
        Args:
            messages: List of Message objects.
            **kwargs: Override config parameters.
        
        Yields:
            String tokens as they are generated.
        """
        try:
            async for chunk in self._stream_internal(messages, **kwargs):
                yield chunk
        except Exception as e:
            self._logger.error(f"Streaming failed: {type(e).__name__}: {str(e)}")
            raise self._wrap_exception(e) from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=0.5),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    async def _generate_with_retry(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Wrapper around _generate_internal with automatic retries.
        
        Retry policies:
        - Stops after 3 attempts.
        - Exponential backoff with jitter (1s, ~2.7s, ~6.4s).
        - Only retries on network-related errors (timeout, connection issues).
        - Non-retryable errors (e.g., authentication) are raised immediately.
        """
        return await self._generate_internal(messages, **kwargs)
    
    def _wrap_exception(self, exc: Exception) -> LLMError:
        """Convert provider-specific exceptions to wrapper's exception hierarchy."""
        if isinstance(exc, httpx.TimeoutException):
            return TimeoutError(f"Request timeout: {str(exc)}")
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 429:
                return RateLimitError(f"Rate limit exceeded: {str(exc)}")
            if exc.response.status_code == 401:
                return AuthenticationError(f"Authentication failed: {str(exc)}")
        return ProviderError(str(exc))
    
    async def close(self) -> None:
        """Release any resources held by the provider."""
        if self._client and hasattr(self._client, "close"):
            await self._client.aclose()
        self._client = None



class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI API and OpenAI-compatible endpoints."""
    
    def _validate_config(self) -> None:
        if not AsyncOpenAI:
            raise ConfigurationError(
                "openai package not installed. Run: pip install openai"
            )
        if not self.config.api_key and not self.config.base_url:
            if "OPENAI_API_KEY" not in os.environ:
                self._logger.warning("No API key provided for OpenAI provider")
    
    def _init_client(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.config.api_key or "dummy",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,
        )
    
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        openai_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                content = [
                    {"type": part["type"], **{k: v for k, v in part.items() if k != "type"}}
                    for part in content
                ]
            openai_messages.append({
                "role": msg.role.value,
                "content": content,
            })
        
        params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
        }
        if self.config.stop:
            params["stop"] = self.config.stop
        if self.config.seed is not None:
            params["seed"] = self.config.seed
        
        response = await self._client.chat.completions.create(**params)
        
        choice = response.choices[0]
        return ProviderResponse(
            text=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            raw_response=response,
        )
    
    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }
        
        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(BaseLLMProvider):
    """Provider for locally running Ollama models."""
    
    def _validate_config(self) -> None:
        if not OllamaClient:
            raise ConfigurationError(
                "ollama package not installed. Run: pip install ollama"
            )
    
    def _init_client(self) -> None:
        host = self.config.base_url or "http://localhost:11434"
        self._client = OllamaClient(host=host)
    
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        ollama_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        parts.append("[Image]")
                content = " ".join(parts)
            ollama_messages.append({
                "role": msg.role.value,
                "content": content,
            })
        
        params = {
            "model": self.config.model,
            "messages": ollama_messages,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            },
        }
        
        response = await self._client.chat(**params)
        
        usage = {
            "prompt_tokens": getattr(response, "prompt_eval_count", 0),
            "completion_tokens": getattr(response, "eval_count", 0),
            "total_tokens": getattr(response, "prompt_eval_count", 0) + getattr(response, "eval_count", 0),
        }
        
        return ProviderResponse(
            text=response.message.content,
            model=self.config.model,
            usage=usage,
            finish_reason=response.get("done_reason", "stop"),
            raw_response=response,
        )
    
    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        
        params = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        
        stream = await self._client.chat(**params)
        async for chunk in stream:
            if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                yield chunk.message.content


class LlamaCPPProvider(BaseLLMProvider):
    """Provider for local GGUF models using llama-cpp-python."""
    
    def _validate_config(self) -> None:
        if not LlamaCPP:
            raise ConfigurationError(
                "llama-cpp-python package not installed. Run: pip install llama-cpp-python"
            )
        if not self.config.metadata.get("model_path"):
            raise ConfigurationError(
                "LlamaCPPProvider requires 'model_path' in metadata"
            )
    
    def _init_client(self) -> None:        
        self._llama_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        model_path = self.config.metadata["model_path"]
        n_ctx = self.config.metadata.get("n_ctx", 2048)
        n_gpu_layers = self.config.metadata.get("n_gpu_layers", 0)
        
        future = self._executor.submit(
            LlamaCPP,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._client = future.result()
    
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        prompt = self._messages_to_prompt(messages)
        
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._client.create_completion(
                prompt=prompt,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                stop=self.config.stop,
            )
        )
        
        return ProviderResponse(
            text=result["choices"][0]["text"],
            model=self.config.model,
            usage={
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "total_tokens": result["usage"].get("total_tokens", 0),
            },
            finish_reason=result["choices"][0].get("finish_reason", "stop"),
            raw_response=result,
        )
    
    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        prompt = self._messages_to_prompt(messages)
        loop = asyncio.get_running_loop()
        
        def generate_stream():
            stream = self._client.create_completion(
                prompt=prompt,
                stream=True,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            for chunk in stream:
                if chunk["choices"][0]["text"]:
                    yield chunk["choices"][0]["text"]
        
        for token in await loop.run_in_executor(self._executor, generate_stream):
            yield token
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == Role.USER:
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == Role.ASSISTANT:
                prompt += f"Assistant: {msg.content}\n\n"
        prompt += "Assistant: "
        return prompt


class HuggingFaceProvider(BaseLLMProvider):
    """Provider for HuggingFace Transformers models (local)."""
    
    def _validate_config(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError(
                "transformers package not installed. Run: pip install transformers torch"
            )
        if not self.config.model:
            raise ConfigurationError("HuggingFaceProvider requires a model identifier")
    
    def _init_client(self) -> None:
        from transformers import pipeline
        
        self._executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()
        
        def load_model():
            return pipeline(
                "text-generation",
                model=self.config.model,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        
        self._pipe = loop.run_until_complete(
            loop.run_in_executor(self._executor, load_model)
        )
    
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        prompt = self._messages_to_prompt(messages)
        loop = asyncio.get_running_loop()
        
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._pipe(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
            )
        )
        
        generated = result[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        
        return ProviderResponse(
            text=generated,
            model=self.config.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw_response=result,
        )
    
    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        response = await self._generate_internal(messages, **kwargs)
        yield response.text
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        prompt = ""
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == Role.USER:
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == Role.ASSISTANT:
                prompt += f"Assistant: {msg.content}\n\n"
        prompt += "Assistant: "
        return prompt



class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.
    
    Design pattern: Factory Method.
    Why it's used:
    - Centralizes provider creation logic.
    - Handles different configuration requirements per provider.
    - Makes adding new providers easy without modifying existing code.
    - Enables runtime provider selection.
    
    Based on the factory pattern used in llm-provider-abstraction and litellm.
    """
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, provider_type: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a provider class for a given type.
        
        This allows dynamic addition of providers at runtime.
        
        Args:
            provider_type: String identifier (e.g., "openai", "anthropic").
            provider_class: Provider class (subclass of BaseLLMProvider).
        """
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create(cls, config: ProviderConfig) -> BaseLLMProvider:
        """
        Create and return a provider instance based on configuration.
        
        Args:
            config: Provider configuration.
        
        Returns:
            An instance of the appropriate provider class.
        
        Raises:
            ConfigurationError: If provider_type is not registered.
        """
        provider_type = config.provider_type.lower()
        if provider_type not in cls._providers:
            raise ConfigurationError(
                f"Unknown provider type: {provider_type}. "
                f"Registered providers: {list(cls._providers.keys())}"
            )
        return cls._providers[provider_type](config)



LLMProviderFactory.register("openai", OpenAIProvider)
LLMProviderFactory.register("ollama", OllamaProvider)
LLMProviderFactory.register("llamacpp", LlamaCPPProvider)
LLMProviderFactory.register("huggingface", HuggingFaceProvider)


class LLMClient:
    """
    Unified LLM client for the application.
    
    This is the main interface that application code should use. It provides
    a consistent API for chat completions and streaming, regardless of the
    underlying provider.
    
    Example:
        config = ProviderConfig.from_env("openai", model="gpt-4")
        client = LLMClient(config)
        response = await client.chat([Message.user("Hello, world!")])
        print(response.text)
        
        async for token in client.stream([Message.user("Write a haiku")]):
            print(token, end="")
        
        client.switch_provider(ProviderConfig.from_env("ollama", model="llama3"))
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Provider configuration. If not provided, must be set later
                   via `configure()` or `switch_provider()`.
        """
        self._config: Optional[ProviderConfig] = config
        self._provider: Optional[BaseLLMProvider] = None
        self._logger = logging.getLogger("LLMClient")
        
        if config:
            self._provider = LLMProviderFactory.create(config)
    
    def configure(self, config: ProviderConfig) -> None:
        """
        Configure the client with a provider configuration.
        
        If a provider already exists, it is closed and replaced.
        
        Args:
            config: Provider configuration.
        """
        if self._provider:
            asyncio.create_task(self._close_provider())
        self._config = config
        self._provider = LLMProviderFactory.create(config)
    
    async def switch_provider(self, config: ProviderConfig) -> None:
        """
        Switch to a different provider at runtime.
        
        Args:
            config: New provider configuration.
        """
        if self._provider:
            await self._provider.close()
        self._config = config
        self._provider = LLMProviderFactory.create(config)
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.
        
        Returns:
            ProviderResponse with generated text and metadata.
        
        Raises:
            ConfigurationError: If no provider is configured.
            LLMError: If generation fails.
        """
        if not self._provider:
            raise ConfigurationError("LLMClient not configured. Call configure() first.")
        return await self._provider.generate(messages, **kwargs)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response token by token.
        
        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.
        
        Yields:
            String tokens as they are generated.
        """
        if not self._provider:
            raise ConfigurationError("LLMClient not configured. Call configure() first.")
        async for token in self._provider.stream(messages, **kwargs):
            yield token
    
    async def close(self) -> None:
        """Close the client and release all resources."""
        if self._provider:
            await self._provider.close()
            self._provider = None
    
    async def __aenter__(self) -> "LLMClient":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def _close_provider(self) -> None:
        if self._provider:
            await self._provider.close()


async def create_llm_client(
    provider_type: str,
    model: str,
    **kwargs,
) -> LLMClient:
    """
    Convenience function to create a configured LLMClient.
    
    Args:
        provider_type: Type of provider (openai, ollama, etc.).
        model: Model identifier.
        **kwargs: Additional configuration parameters.
    
    Returns:
        A configured LLMClient instance.
    
    Example:
        client = await create_llm_client("openai", "gpt-4", temperature=0.7)
    """
    config = ProviderConfig.from_env(provider_type, model=model, **kwargs)
    return LLMClient(config)