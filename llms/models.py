import os
from utils.enums import Role
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

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
