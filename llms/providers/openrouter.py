import os
from typing import List, AsyncGenerator, Dict, Any
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenRouterProvider(BaseLLMProvider):
    """
    Provider for OpenRouter API – unified access to 400+ LLM models.

    Why this provider:
        - OpenRouter acts as a gateway to dozens of LLM providers (OpenAI, Anthropic,
          Google, Cohere, Mistral, local models via OSS, etc.) through a single API.
        - It supports model fallback, provider routing, cost tracking, and session
          persistence (e.g., for multi‑turn conversations).
        - This provider uses the OpenAI‑compatible endpoint of OpenRouter, allowing
          reuse of the `AsyncOpenAI` client with custom headers and base URL.

    Design decisions:
        - The `AsyncOpenAI` client is used because OpenRouter's API mirrors OpenAI's
          chat completion interface (messages, stream, etc.).
        - Default headers (`HTTP-Referer`, `X-Title`) are set from environment
          variables or sensible defaults. They help OpenRouter identify the source
          application for analytics and routing.
        - The provider supports OpenRouter‑specific parameters (`provider`, `models`,
          `session_id`) as optional keyword arguments in `generate()`/`stream()`.
        - Retries are disabled at the client level (`max_retries=0`) because the
          base provider already implements retry logic (tenacity). This avoids
          double‑retries.
        - Error handling: All exceptions from the OpenRouter API are caught and
          wrapped as `ConfigurationError` (the base class will further map them
          according to its `_wrap_exception` logic).

    Performance:
        - The `AsyncOpenAI` client is fully async and manages its own connection pool.
        - No threading overhead; all I/O is non‑blocking.
        - The client is reused across requests, benefiting from keep‑alive connections.

    Edge Cases & Risks:
        - OpenRouter requires a valid API key (from environment variable
          `OPENROUTER_API_KEY` or config). The provider will raise `ConfigurationError`
          if missing.
        - The `HTTP-Referer` header must be a valid URL; using `http://localhost` is
          acceptable for development but production should set `OPENROUTER_SITE_URL`.
        - The `X-Title` header is optional but recommended for identifying your app
          in OpenRouter logs.
        - If `config.model` is not provided, OpenRouter will use its default model
          (which may be unexpected). It is better to always specify a model either
          in config or as `model` in `**kwargs`.
        - The provider passes through OpenRouter‑specific parameters (`provider`,
          `models`, `session_id`) directly to the API; they are not validated.
          Refer to OpenRouter documentation for their semantics.
        - Streaming errors are emitted as error strings (prefixed with "[Error:") instead
          of raising exceptions. This matches the pattern in other providers and prevents
          the generator from crashing, but the caller must check for error strings.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def _validate_config(self) -> None:
        """
        Validate configuration for OpenRouter provider.

        Requirements:
            - The `openai` package must be installed (the provider uses `AsyncOpenAI`).
            - An API key must be provided either via `config.api_key` or the
              `OPENROUTER_API_KEY` environment variable.

        Raises:
            ConfigurationError: If the package is missing or no API key is found.
        """
        if not AsyncOpenAI:
            raise ConfigurationError(
                "openai package not installed. Run: pip install openai"
            )
        if not self.config.api_key:
            if "OPENROUTER_API_KEY" in os.environ:
                self.config.api_key = os.environ["OPENROUTER_API_KEY"]
            else:
                raise ConfigurationError(
                    "OpenRouter API key is required. Set OPENROUTER_API_KEY environment "
                    "variable or pass api_key in config."
                )

    def _init_client(self) -> None:
        """
        Initialise the AsyncOpenAI client for OpenRouter.

        Why:
            - OpenRouter uses an OpenAI‑compatible API; the `AsyncOpenAI` client
              works with minimal changes.
            - Custom headers (`HTTP-Referer`, `X-Title`) are set from environment
              variables or defaults. These inform OpenRouter about the origin app.
            - Timeout is taken from `self.config.timeout` (default 60s).
            - Retries are disabled (set to 0) because the base class already provides
              retry logic with exponential backoff; client‑side retries would be redundant.

        Side Effects:
            - Creates `self._client` as an `AsyncOpenAI` instance.
            - The client will be reused for all requests (connection pooling).

        Important:
            - `base_url` can be overridden via `self.config.base_url`; if not provided,
              the provider uses the standard OpenRouter endpoint.
            - The client does not validate the API key immediately; the first request
              will fail if the key is invalid.
        """
        default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "My App"),
        }
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or self.OPENROUTER_BASE_URL,
            default_headers=default_headers,
            timeout=self.config.timeout,
            max_retries=0,
        )

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message list to OpenAI/OpenRouter chat format.

        Why:
            - OpenRouter expects the same message format as OpenAI: a list of dicts
              with `role` and `content` keys.
            - Multimodal content (lists of parts) is supported by converting each part
              to a dict containing a `type` field and the part's fields (e.g., `text`,
              `image_url`). This matches OpenAI's structure.

        Args:
            messages: List of internal Message objects.

        Returns:
            A list of dicts ready to be passed to `chat.completions.create`.

        Edge Cases:
            - If content is already a string, it is used as‑is.
            - If content is a list (multimodal), each element is assumed to be a dict
              with at least a `type` key. All other key‑value pairs are copied,
              except `type` (which is kept as the discriminator).
            - No validation is performed; malformed parts will cause an API error.
        """
        openai_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                content = [
                    {"type": part["type"], **{k: v for k, v in part.items() if k != "type"}}
                    for part in content
                ]
            openai_messages.append({"role": msg.role.value, "content": content})
        return openai_messages

    async def _generate_internal(self, messages: List[Message], **kwargs) -> ProviderResponse:
        """
        Internal non‑streaming generation using OpenRouter's OpenAI‑compatible API.

        How it works:
            1. Convert messages to OpenAI format.
            2. Build parameters by merging config with method‑level overrides.
            3. Add OpenRouter‑specific parameters (`provider`, `models`, `session_id`)
               if provided in `**kwargs`.
            4. Call the OpenAI client's `chat.completions.create`.
            5. Extract the response text, token usage, and finish reason.
            6. Return a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters and OpenRouter‑specific options:
                - model (str): Override the model name.
                - provider (dict): Provider routing preferences (e.g., `{"order": ["DeepSeek", "OpenAI"]}`).
                - models (list): List of models for fallback (OpenRouter will try them in order).
                - session_id (str): Persistent session ID for conversation context.

        Returns:
            ProviderResponse containing generated text and usage metadata.

        Raises:
            ConfigurationError: If the API call fails (wrapped from any exception).
            (The base provider will retry network‑related errors via tenacity.)

        Important:
            - The `model` parameter can be overridden in `**kwargs` to use a different
              model without reconfiguring the provider. This is useful for multi‑model
              routing.
            - Token usage is provided by OpenRouter (it forwards usage from the underlying
              provider). The structure mirrors OpenAI's usage object.
        """
        openai_messages = self._convert_messages(messages)

        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Optional OpenRouter-specific parameters
        if kwargs.get("provider"):
            params["provider"] = kwargs["provider"]
        if kwargs.get("models"):
            params["models"] = kwargs["models"]
        if kwargs.get("session_id"):
            params["session_id"] = kwargs["session_id"]

        try:
            response = await self._client.chat.completions.create(**params)
            choice = response.choices[0]
            return ProviderResponse(
                text=choice.message.content or "",
                model=response.model,  # OpenRouter returns the actual model used.
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=choice.finish_reason,
                raw_response=response,
            )
        except Exception as e:
            self._logger.error(f"OpenRouter generation error: {e}")
            raise ConfigurationError(f"OpenRouter API error: {e}") from e

    async def _stream_internal(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """
        Internal streaming generation using OpenRouter's streaming API.

        How it works:
            1. Convert messages to OpenAI format.
            2. Build parameters similarly to `_generate_internal`, but with `stream=True`.
            3. Add OpenRouter‑specific parameters (`provider`, `session_id`).
            4. Call the API and iterate over the async stream.
            5. Yield `delta.content` from each chunk.
            6. If an error occurs, yield an error string (to avoid breaking the generator).

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters and OpenRouter‑specific options
                      (same as `_generate_internal`, plus any streaming‑specific flags).

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as text.
              This matches the design of other streaming providers in this wrapper
              and ensures the caller's event loop is not disrupted.
            - OpenRouter supports the same streaming chunk format as OpenAI: each
              chunk has `choices[0].delta.content`. The code assumes this structure.
            - The `models` parameter (fallback list) is not recommended for streaming
              because switching models mid‑stream is not supported; it is omitted.
            - The stream will automatically close when the response is complete or
              an error occurs.
        """
        openai_messages = self._convert_messages(messages)

        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }

        if kwargs.get("provider"):
            params["provider"] = kwargs["provider"]
        if kwargs.get("session_id"):
            params["session_id"] = kwargs["session_id"]

        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self._logger.error(f"OpenRouter streaming error: {e}")
            yield f"[Error: {e}]"