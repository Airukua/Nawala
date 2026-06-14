import asyncio
from .models import ProviderConfig, ProviderResponse, Message
from typing import Optional, List, AsyncGenerator
from utils.logger import get_logger
from .base import BaseLLMProvider
from .factory import LLMProviderFactory
from .exceptions import ConfigurationError

class LLMClient:
    """
    Unified LLM client for the application.

    This is the main interface that application code should use. It provides
    a consistent API for chat completions and streaming, regardless of the
    underlying provider.

    Why this exists:
        - Decouples application logic from provider-specific implementations.
        - Manages provider lifecycle (creation, switching, cleanup).
        - Provides a simple, ergonomic API (`chat`, `stream`) that hides
          the complexity of provider instantiation and configuration.

    Design philosophy:
        - The client holds a single active provider instance at any time.
        - Switching providers closes the old provider and creates a new one,
          ensuring resources are freed.
        - Configuration can be applied lazily (client created without a provider
          and later configured via `configure()`).
        - The client is async-first; all network I/O is done via `await`.
        - The client implements async context manager protocol for safe resource
          cleanup.

    Lifecycle:
        1. Create client with optional initial config.
        2. Optionally call `configure()` or `switch_provider()` to set/change provider.
        3. Call `chat()` or `stream()` zero or more times.
        4. Call `close()` (or use async context manager) to release resources.

    State:
        - `_config`: Currently active provider configuration (may be None).
        - `_provider`: Active provider instance (None if not configured).
        - `_logger`: Logger instance for client-level events.

    Invariants:
        - If `_config` is not None, `_provider` is not None and corresponds to it.
        - `_provider` is always created via `LLMProviderFactory`.
        - Calls to `chat`/`stream` raise `ConfigurationError` if `_provider` is None.

    Thread safety:
        - The client is designed for asyncio single-threaded usage.
        - Switching providers while a request is in flight would cause undefined
          behaviour (the in-flight request uses the old provider, which is then
          closed). It is the caller's responsibility to avoid concurrent switching.
        - For multi-threaded asyncio environments (e.g., multiple event loops),
          each loop should have its own client instance.

    Example:
        >>> # Create client with OpenAI
        >>> config = ProviderConfig.from_env("openai", model="gpt-4")
        >>> client = LLMClient(config)
        >>>
        >>> # Generate a response
        >>> response = await client.chat([Message.user("Hello, world!")])
        >>> print(response.text)
        >>>
        >>> # Stream a response
        >>> async for token in client.stream([Message.user("Write a haiku")]):
        ...     print(token, end="")
        >>>
        >>> # Switch to Ollama at runtime
        >>> await client.switch_provider(ProviderConfig.from_env("ollama", model="llama3"))
        >>>
        >>> # Use as context manager
        >>> async with LLMClient(config) as client:
        ...     response = await client.chat(messages)
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize the LLM client.

        Why this method:
            - Allows creation without an immediate provider (lazy configuration).
            - If a config is provided, the provider is created immediately.
            - Does not start any background tasks or network connections until
              first request.

        Args:
            config: Provider configuration. If not provided, must be set later
                    via `configure()` or `switch_provider()`.

        Side Effects:
            - If `config` is given, `self._provider` is set by calling
              `LLMProviderFactory.create(config)`.
            - Otherwise, `self._provider` remains `None`.

        Edge Cases:
            - If `config` is provided but contains invalid settings, the factory
              will raise `ConfigurationError` during creation. The client will
              not be usable until reconfigured.
        """
        self._config: Optional[ProviderConfig] = config
        self._provider: Optional[BaseLLMProvider] = None
        self._logger = get_logger("LLMClient")

        if config:
            self._provider = LLMProviderFactory.create(config)

    def configure(self, config: ProviderConfig) -> None:
        """
        Configure the client with a provider configuration.

        Why this method:
            - Allows setting or updating the provider after client creation.
            - Closes the existing provider (if any) in a fire‑and‑forget manner.
            - Useful for scenarios where the client is created first and
              configuration is loaded later (e.g., from a config file).

        Important:
            - This method is synchronous and does not await the closure of the
              old provider. A background task is created to close it. This avoids
              blocking the caller, but means that the old provider may still be
              alive briefly after this method returns. If you need to ensure
              the old provider is fully closed, use `await switch_provider(config)`
              instead.
            - If the client was previously configured, the new provider replaces
              the old one. Any in‑flight requests using the old provider will
              continue (the provider is not forcibly cancelled). However, new
              requests will use the new provider.

        Args:
            config: Provider configuration.

        Side Effects:
            - Sets `self._config` and `self._provider` to the new values.
            - Schedules a task to close the old provider (if any).

        Edge Cases:
            - If the old provider's `close()` method raises an exception, it is
              logged but not propagated (fire‑and‑forget). This is intentional
              to avoid disrupting the new configuration.
        """
        if self._provider:
            # Fire-and-forget close to avoid blocking the caller.
            asyncio.create_task(self._close_provider())
        self._config = config
        self._provider = LLMProviderFactory.create(config)

    async def switch_provider(self, config: ProviderConfig) -> None:
        """
        Switch to a different provider at runtime.

        Why this method:
            - Provides a clean, awaitable way to change providers.
            - Ensures the old provider is fully closed before the new one is
              created, preventing resource leaks.
            - Useful when you need to change credentials, models, or providers
              based on user input or runtime conditions.

        Differences from `configure()`:
            - `configure()` does not await the old provider's closure.
            - `switch_provider()` does, guaranteeing that the old provider's
              resources (e.g., HTTP connections) are released before proceeding.

        Args:
            config: New provider configuration.

        Side Effects:
            - Closes the existing provider (if any).
            - Creates a new provider with the new config.
            - Updates `self._config` and `self._provider`.

        Important:
            - If the old provider's `close()` method raises an exception, it is
              logged and the exception is **not** propagated (the method continues
              to create the new provider). This ensures that even if cleanup
              fails, the client can still switch to the new provider.
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

        Why this method:
            - Primary interface for non‑streaming generation.
            - Delegates directly to the active provider's `generate` method.
            - Applies method‑level parameter overrides (e.g., temperature).

        Args:
            messages: List of Message objects representing the conversation.
            **kwargs: Override generation parameters (temperature, max_tokens, etc.).
                      These are passed through to the provider.

        Returns:
            ProviderResponse with generated text, token usage, latency, etc.

        Raises:
            ConfigurationError: If no provider is configured (i.e., `_provider` is None).
            LLMError: If generation fails (after retries). Subclasses include
                      `TimeoutError`, `RateLimitError`, `AuthenticationError`.

        Important:
            - This method is idempotent (same input yields same output if
              parameters are deterministic). However, LLMs are non‑deterministic
              by default unless you set a seed.
            - The method does not modify the client state.

        Example:
            >>> response = await client.chat([
            ...     Message.system("You are a helpful assistant."),
            ...     Message.user("What is Python?")
            ... ], temperature=0.5)
            >>> print(response.text)
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

        Why this method:
            - Provides real‑time output for long generations.
            - Reduces perceived latency for interactive applications.
            - Delegates to the active provider's `stream` method.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.

        Yields:
            String tokens as they are generated (typically word pieces or whole words).

        Raises:
            ConfigurationError: If no provider is configured.
            LLMError: If streaming fails (including network errors, authentication
                      errors, or malformed chunks).

        Important:
            - The generator must be fully consumed (or closed) to avoid resource
              leaks. If you break out of the loop early, you may want to cancel
              the generator (the provider should handle cancellation, but it's
              provider‑dependent).
            - Streams are not automatically retried; if an error occurs mid‑stream,
              the generator will raise an exception and further tokens are lost.
              The caller should handle the error and possibly restart the request.

        Example:
            >>> async for token in client.stream([Message.user("Write a haiku about coding")]):
            ...     print(token, end="", flush=True)
        """
        if not self._provider:
            raise ConfigurationError("LLMClient not configured. Call configure() first.")
        async for token in self._provider.stream(messages, **kwargs):
            yield token

    async def close(self) -> None:
        """
        Close the client and release all resources.

        Why this method:
            - Ensures proper cleanup of the underlying provider (closes HTTP
              connections, thread pools, etc.).
            - Should be called when the client is no longer needed (e.g., at
              application shutdown).

        Important:
            - After closing, the client cannot be used again. To use it again,
              create a new instance or call `configure()` with a new config.
            - This method is idempotent: calling it twice has no effect.

        Example:
            >>> await client.close()
        """
        if self._provider:
            await self._provider.close()
            self._provider = None

    async def __aenter__(self) -> "LLMClient":
        """
        Enter the async context manager.

        Why:
            - Enables `async with LLMClient(...) as client:` pattern.
            - Returns the client itself; no special setup needed.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the async context manager, ensuring the client is closed.

        Why:
            - Automatically calls `close()` even if an exception occurred in the
              context block.
            - Prevents resource leaks in error scenarios.

        Args:
            exc_type: Exception type (if any).
            exc_val: Exception value (if any).
            exc_tb: Exception traceback (if any).
        """
        await self.close()

    async def _close_provider(self) -> None:
        """
        Internal helper to close the current provider if it exists.

        Why:
            - Used by `configure()` to close the old provider in a fire‑and‑forget
              task.
            - Prevents blocking the synchronous `configure()` method.

        Important:
            - Any exception during provider closure is logged but not propagated.
        """
        if self._provider:
            await self._provider.close()


async def create_llm_client(
    provider_type: str,
    model: str,
    **kwargs,
) -> LLMClient:
    """
    Convenience function to create a configured LLMClient.

    Why this exists:
        - Reduces boilerplate when creating a client.
        - Handles environment variable loading via `ProviderConfig.from_env`.
        - Returns a ready‑to‑use client instance.

    Args:
        provider_type: Type of provider (openai, ollama, anthropic, etc.).
        model: Model identifier (e.g., "gpt-4", "llama3.2:1b", "claude-3-opus").
        **kwargs: Additional configuration parameters passed to `ProviderConfig.from_env`.
                  Common parameters:
                  - api_key (str): Override API key (instead of env var).
                  - base_url (str): Custom endpoint URL.
                  - timeout (float): Request timeout in seconds.
                  - temperature (float): Sampling temperature.
                  - max_tokens (int): Maximum tokens to generate.
                  - top_p (float): Nucleus sampling parameter.
                  - presence_penalty (float): Presence penalty.
                  - frequency_penalty (float): Frequency penalty.
                  - stop (List[str]): Stop sequences.

    Returns:
        A configured LLMClient instance (already has an active provider).

    Raises:
        ConfigurationError: If provider_type is unknown, required dependencies
                            missing, or required configuration fields not provided
                            (e.g., missing API key when needed).

    Example:
        >>> client = await create_llm_client("openai", "gpt-4", temperature=0.7)
        >>> response = await client.chat([Message.user("Tell me a joke")])
        >>> print(response.text)

        >>> client = await create_llm_client("ollama", "llama3.2:1b", base_url="http://localhost:11434")
        >>> async for token in client.stream([Message.user("Hello")]):
        ...     print(token)

    Important:
        - This function is async only for consistency; it does not perform any
          I/O (the client creation is synchronous). The `async` keyword is not
          strictly necessary, but kept for future extensibility (e.g., if provider
          initialisation becomes async).
    """
    config = ProviderConfig.from_env(provider_type, model=model, **kwargs)
    return LLMClient(config)