from abc import ABC, abstractmethod
from .models import ProviderConfig, ProviderResponse

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
