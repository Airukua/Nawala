import httpx
import asyncio
from abc import ABC, abstractmethod
from .models import ProviderConfig, ProviderResponse
from typing import List, AsyncGenerator
from utils.logger import get_logger
from .exceptions import ProviderError, RateLimitError, AuthenticationError, LLMError
import time
from .models import Message
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


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

    Lifecycle:
        1. Instantiate with a ProviderConfig (contains API keys, model, timeouts, etc.)
        2. Provider validates config and initialises its internal client (sync or async).
        3. Call `generate()` or `stream()` zero or more times.
        4. Call `close()` to release resources (e.g., HTTP connection pools, thread pools).

    Invariants:
        - `_client` is either `None` (before init or after close) or a provider-specific
          client object (e.g., `AsyncOpenAI`, `httpx.AsyncClient`).
        - `config` is read‑only after construction. Subclasses may add extra config
          fields via `metadata` in ProviderConfig.
        - Each call to `generate()` is timed and the latency is stored in the response.
        - Network‑related errors are automatically retried up to 3 times with
          exponential backoff and jitter.

    Edge Cases & Risks:
        - The `_generate_internal` method is called by `_generate_with_retry`. Subclasses
          must ensure it is idempotent for retryable errors – i.e., the same prompt
          can be sent multiple times without side effects. Most LLM APIs are naturally
          idempotent, but streaming may have side effects; streaming is not retried.
        - The `stream()` method does NOT use retries because a stream that fails mid‑way
          cannot be transparently resumed. The caller must handle streaming errors.
        - `_wrap_exception` maps HTTP status codes to specific exception types. Subclasses
          may override to add provider‑specific error codes (e.g., Anthropic's `overloaded_error`).
        - The `_client` may be a synchronous blocking client (e.g., for local models).
          Subclasses must handle the async‑to‑sync bridge (e.g., using `run_in_executor`).
        - `close()` is idempotent; calling it twice does nothing.

    Performance:
        - `generate()` uses `time.perf_counter()` for high‑resolution latency measurement.
        - Retry logic adds overhead only on failures (backoff sleeps). Use `max_retries=0`
          to disable retries if not needed.
        - Subclasses should avoid expensive operations in `__init__`; lazy initialisation
          (e.g., loading large models on first `generate`) is preferred.
    """

    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.

        Why:
            - Centralises all provider‑specific settings into a single `config` object.
            - Allows subclasses to validate required fields before any request is made.
            - Creates the underlying client once, reusing it for all requests.

        Args:
            config: Provider configuration containing API keys, model, timeout, etc.

        Raises:
            ConfigurationError: If required dependencies are missing or config invalid
                                (subclasses should raise this in `_validate_config`).

        Side Effects:
            - Stores `config` and initialises `_client` via `_init_client()`.
            - Creates a logger instance named `LLMProvider.{provider_type}`.

        Invariants:
            - After `__init__`, `self._client` may be `None` if the provider is
              lazy‑initialised; but typical providers create the client immediately.
        """
        self.config = config
        self._logger = get_logger(f"LLMProvider.{config.provider_type}")
        self._validate_config()
        self._client = None
        self._init_client()

    def _validate_config(self) -> None:
        """
        Validate provider‑specific configuration.

        Why override:
            - Each provider has different requirements (e.g., Ollama needs a host,
              OpenAI needs an API key unless using a local proxy).
            - Early validation prevents cryptic runtime errors.

        Default implementation:
            Does nothing. Subclasses should override and raise `ConfigurationError`
            if required fields are missing or invalid.

        Invariants:
            - Should not modify `self.config`.
            - Should not perform network I/O (only local checks).

        Example:
            >>> if not self.config.api_key and not self.config.base_url:
            >>>     raise ConfigurationError("OpenAI provider requires api_key or base_url")
        """
        pass

    def _init_client(self) -> None:
        """
        Initialize the underlying HTTP client or SDK.

        Why override:
            - Different providers use different libraries (OpenAI SDK, Anthropic SDK,
              raw httpx, llama‑cpp‑python, etc.).
            - Allows setting custom timeouts, connection pools, or proxy settings.

        Default implementation:
            Does nothing. Subclasses must implement and set `self._client`.

        Side Effects:
            - Assigns `self._client` (e.g., `AsyncOpenAI`, `httpx.AsyncClient`).
            - May raise `ConfigurationError` if initialisation fails.

        Important:
            - The client should be reusable for multiple requests.
            - The client must be `async`‑friendly (or support async wrappers).
            - The client will be closed in `self.close()`.
        """
        pass

    @abstractmethod
    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal generation method implemented by each provider.

        Why this method exists:
            - Separates the core generation logic from retries and timing.
            - Subclasses implement the provider‑specific API call here.
            - It is called by `_generate_with_retry` which adds retry behaviour.

        Important:
            - This method **must not** implement retries or timing; those are handled
              by the wrapper.
            - It may raise any exception; the retry wrapper will decide whether to retry.
            - It must be async and return a `ProviderResponse` directly.

        Args:
            messages: List of internal `Message` objects. Subclasses must convert
                      these to the provider's native message format.
            **kwargs: Overrides for generation parameters (temperature, max_tokens, etc.).
                      Subclasses should merge them with `self.config` (with kwargs
                      taking precedence).

        Returns:
            A `ProviderResponse` containing the generated text, token usage, etc.

        Raises:
            Any exception that the underlying provider may raise (e.g., `httpx.HTTPError`).

        Note:
            Subclasses may also set `response.raw_response` to the original provider
            output for debugging.
        """
        pass

    async def generate(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a response from the LLM with automatic retries and timing.

        Why this method:
            - Provides a clean, unified interface for all providers.
            - Automatically measures request latency and stores it in the response.
            - Wraps the internal generation with retry logic and exception mapping.

        Args:
            messages: List of Message objects representing the conversation.
            **kwargs: Override config parameters (temperature, max_tokens, etc.).

        Returns:
            ProviderResponse containing the generated text, token usage, latency, etc.

        Raises:
            LLMError: If generation fails after retries. Sub‑exceptions include
                      `TimeoutError`, `RateLimitError`, `AuthenticationError`, etc.

        Performance:
            - Latency is measured using `time.perf_counter()` (high resolution,
              monotonic, unaffected by system clock changes).
            - The retry wrapper only retries on network‑related exceptions; other
              errors (e.g., validation) are raised immediately.

        Thread safety:
            - This method is async and safe to call concurrently from multiple tasks,
              provided the underlying client is thread‑safe (most async clients are).
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
        """
        Internal streaming method implemented by each provider.

        Why this method:
            - Abstracts the provider‑specific streaming implementation.
            - Must return an async generator that yields string tokens.
            - Does not include retries (streams are not retried automatically).

        Important:
            - The returned async generator should yield tokens as they become available.
            - The generator may raise exceptions; the caller (`stream()`) catches and
              wraps them.
            - Subclasses must handle conversion from the provider's chunk format to
              plain text strings.

        Args:
            messages: List of internal `Message` objects.
            **kwargs: Override generation parameters.

        Yields:
            String chunks (typically tokens or word pieces) as they are generated.

        Note:
            - For providers that do not support streaming natively, subclasses may
              implement this by calling `_generate_internal` and yielding the full
              response as a single chunk.
        """
        pass

    async def stream(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM token by token.

        Why this method:
            - Provides a uniform streaming interface across all providers.
            - Handles exception wrapping and logging.
            - No retries – callers must handle stream failures.

        Args:
            messages: List of Message objects.
            **kwargs: Override config parameters.

        Yields:
            String tokens as they are generated.

        Raises:
            LLMError: If streaming fails (including network errors, authentication errors).

        Important:
            - Once an exception is raised, the generator is exhausted.
            - The caller should either close the provider or recreate it after an error.
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
        Wrapper around `_generate_internal` with automatic retries.

        Why this method:
            - Retry logic is applied only to network‑related failures (timeout, connection).
            - Uses exponential backoff with jitter to avoid thundering herd on recovery.
            - Retries are transparent to the caller; they see only the final success or
              failure.

        Retry policies:
            - Max attempts: 3 (configurable via `self.config.max_retries` in the future).
            - Backoff: initial 1 second, multiplied by ~2.7 each retry, plus jitter.
            - Retryable exceptions: `httpx.TimeoutException`, `httpx.NetworkError`.
            - Non‑retryable errors (e.g., `httpx.HTTPStatusError` for 400‑level) are
              not retried; they are raised immediately.

        Edge Cases:
            - If the provider raises a retryable error but the same request would
              always fail (e.g., a malformed request that times out due to server bug),
              retries waste time. The caller can reduce `max_retries` to 0.
            - The decorator uses `reraise=True` so that the original exception is
              re‑raised after all retries exhausted.

        Note:
            - Streaming methods do not use this retry wrapper because partial streams
              cannot be retried.
        """
        return await self._generate_internal(messages, **kwargs)

    def _wrap_exception(self, exc: Exception) -> LLMError:
        """
        Convert provider‑specific exceptions to wrapper's exception hierarchy.

        Why this method:
            - The rest of the application should not depend on provider‑specific
              exception types (e.g., `openai.APIError`).
            - This method maps common error patterns (HTTP status codes, timeouts)
              to standard exceptions: `TimeoutError`, `RateLimitError`, etc.

        Args:
            exc: The original exception raised by the provider.

        Returns:
            An instance of `LLMError` (or a subclass) that wraps the original.

        Default implementation:
            - `httpx.TimeoutException` -> `TimeoutError`
            - `httpx.HTTPStatusError` with status 429 -> `RateLimitError`
            - `httpx.HTTPStatusError` with status 401 -> `AuthenticationError`
            - Any other `httpx.HTTPStatusError` -> `ProviderError`
            - All others -> `ProviderError(exc)`

        Subclasses may override to add provider‑specific error codes (e.g., Anthropic's
        `overloaded_error` mapping to `RateLimitError`).

        Important:
            - The returned exception should preserve the original message and
              ideally set `__cause__` to the original exception (done by `from e`).
        """
        if isinstance(exc, httpx.TimeoutException):
            return TimeoutError(f"Request timeout: {str(exc)}")
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 429:
                return RateLimitError(f"Rate limit exceeded: {str(exc)}")
            if exc.response.status_code == 401:
                return AuthenticationError(f"Authentication failed: {str(exc)}")
        return ProviderError(str(exc))

    async def close(self) -> None:
        """
        Release any resources held by the provider.

        Why this method:
            - Prevents resource leaks (open HTTP connections, file handles, GPU memory).
            - Should be called when the provider is no longer needed (e.g., at
              application shutdown).

        What it does:
            - If `self._client` exists and has a `close` method (async or sync),
              it will be awaited/called appropriately.
            - Sets `self._client = None` to avoid accidental reuse.

        Idempotence:
            - Calling `close()` multiple times is safe; subsequent calls do nothing
              (since `self._client` becomes `None` after first call).

        Note:
            - Subclasses should override this if they have additional resources to
              clean up (e.g., a thread pool for local models). They must call
              `super().close()`.
            - After `close()`, the provider cannot be used again; a new instance
              must be created.
        """
        if self._client and hasattr(self._client, "close"):
            # Check if the close method is async (e.g., httpx.AsyncClient) or sync.
            close_method = getattr(self._client, "close")
            if asyncio.iscoroutinefunction(close_method):
                await close_method()
            else:
                close_method()  # sync close (e.g., some local model clients)
        self._client = None