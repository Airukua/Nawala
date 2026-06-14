class LLMError(Exception):
    """
    Base exception for all LLM-related errors.

    Why this exists:
    - Provides a single catch‑all category for any error originating from the LLM
      wrapper or underlying provider.
    - Enables application code to handle all LLM failures with `except LLMError`,
      while still allowing fine‑grained handling of specific subclasses.
    - Follows the standard Python exception hierarchy practice: a domain‑specific
      base class.

    When to raise:
        - Never directly. Subclasses should be raised instead.
        - This class exists primarily for type‑checking and broad exception catching.

    Relationship to other exceptions:
        - All other exceptions in this module inherit from `LLMError`.
        - It does not inherit from `Exception` directly? Actually it does – the
          base is `Exception`. That's correct; we don't want to suppress system‑level
          exceptions like `KeyboardInterrupt`.

    Usage example:
        >>> try:
        ...     response = await llm.generate(messages)
        ... except LLMError as e:
        ...     logger.error(f"LLM operation failed: {e}")
        ...     # Fallback to a default response.
    """
    pass


class ProviderError(LLMError):
    """
    Provider-specific error (e.g., API returned an error response).

    Why this exists:
        - Distinguishes errors that come from the external LLM provider (e.g., HTTP 500,
          malformed response) from errors in the wrapper itself (configuration, timeouts).
        - Allows callers to decide whether to retry, switch provider, or propagate.

    When to raise:
        - When the underlying LLM API returns an error response that is not a rate limit,
          authentication failure, or other more specific case.
        - When the provider's SDK raises an unexpected exception that cannot be classified
          as a timeout, rate limit, or auth error.

    Example scenarios:
        - OpenAI returns a 503 Service Unavailable.
        - Anthropic returns a 400 with a message about invalid stop sequences.
        - A local model outputs malformed JSON that cannot be parsed.

    Note:
        - This exception is not raised for network‑level errors (timeouts, connection
          resets) – those raise `TimeoutError` (or a more specific subclass).
        - It is also not raised for configuration errors – those raise `ConfigurationError`.
    """
    pass


class RateLimitError(ProviderError):
    """
    Rate limit exceeded.

    Why this exists:
        - Rate limits are a common failure mode with cloud LLM APIs.
        - Distinguishing rate limits allows the application to implement backoff,
          queue requests, or fall back to a different provider.

    When to raise:
        - When the provider returns an HTTP 429 (Too Many Requests) or any
          provider‑specific error code indicating that the request was throttled.
        - Should be raised after the retry mechanism (in the base provider) has
          exhausted its attempts, or if retries are disabled.

    Typical handling strategies:
        - Wait and retry after a delay (exponential backoff).
        - Switch to a different API key or provider.
        - Cache responses to reduce request frequency.

    Important:
        - This exception inherits from `ProviderError`, so catching `ProviderError`
          will also catch rate limit errors unless the handler is more specific.
    """
    pass


class AuthenticationError(ProviderError):
    """
    API key invalid or missing.

    Why this exists:
        - Credential errors are common during initial setup or key rotation.
        - Separating auth errors from other provider errors allows applications to
          alert administrators or fall back to a different set of credentials.

    When to raise:
        - When the provider returns HTTP 401 (Unauthorized) or 403 (Forbidden).
        - When a local model requires a license key that is missing or expired.

    Note:
        - This error is typically fatal – retrying with the same credentials will
          continue to fail. The application should either:
            * Use a different API key (from a pool).
            * Notify an operator to refresh credentials.
            * Abort the operation.
    """
    pass


class TimeoutError(LLMError):
    """
    Request timeout exceeded.

    Why this exists:
        - LLM calls can be slow, and hanging requests waste resources.
        - The wrapper enforces timeouts to prevent indefinite waits.
        - This exception is separate from `ProviderError` because timeouts are often
          network‑related and may be retryable (unlike a malformed request).

    When to raise:
        - When the configured `timeout_seconds` is exceeded while waiting for a response
          from the provider.
        - When a streaming response takes too long between chunks.

    Important:
        - This exception is NOT a subclass of `ProviderError` because timeouts are
          often not the provider's fault (e.g., network partition, DNS issues).
        - It inherits directly from `LLMError` to allow broad catching of all
          LLM failures.

    Note on naming:
        - The name `TimeoutError` is intentionally chosen to mirror the built‑in
          `asyncio.TimeoutError` and `httpx.TimeoutException`. However, this is a
          distinct exception class defined in this module. It does not conflict with
          the built‑in because the built‑in is not imported into this namespace.
    """
    pass


class ConfigurationError(LLMError):
    """
    Invalid configuration or missing required parameters.

    Why this exists:
        - Many LLM wrapper errors occur due to misconfiguration (e.g., missing API key,
          invalid model name, unsupported provider).
        - These errors should be raised as early as possible (e.g., during
          `ProviderConfig` validation or `BaseLLMProvider.__init__`), not during
          a request.
        - Separating configuration errors from runtime errors helps developers
          diagnose setup issues quickly.

    When to raise:
        - When a required configuration parameter is `None` or invalid.
        - When a required Python package is not installed (e.g., `openai` missing
          for `OpenAIProvider`).
        - When an unsupported provider type is requested.
        - When a model identifier is not found in a local registry.

    Important:
        - This exception should be considered fatal and not retryable.
        - It typically indicates a programming error or environment misconfiguration,
          not a transient condition.
        - It inherits directly from `LLMError` because it is not a provider error
          (it happens before any provider call is made).
    """
    pass


class StreamingError(LLMError):
    """
    Error during streaming response.

    Why this exists:
        - Streaming adds complexity: chunks may be malformed, connections may break
          midway, or the provider may send an error after sending some data.
        - Distinguishing streaming errors from regular generation errors allows
          callers to decide whether to salvage partially received content or to
          retry the entire request.

    When to raise:
        - When a streamed chunk cannot be parsed (e.g., invalid JSON from the provider).
        - When a provider returns an error after the first chunk has been sent.
        - When a network failure occurs after streaming has started.

    Relationship to `TimeoutError`:
        - A timeout during streaming (e.g., no new chunk for 30 seconds) should raise
          `TimeoutError`, not `StreamingError`. `StreamingError` is for protocol or
          data‑level failures.

    Important:
        - This exception is not raised for initial connection failures – those are
          `TimeoutError` or `ProviderError` depending on the cause.
        - If streaming fails, the generator is exhausted; the caller may need to
          recreate the provider or start a new request.
    """
    pass