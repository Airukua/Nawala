import os
from typing import List, AsyncGenerator, Dict, Any
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError

try:
    from openai import AsyncOpenAI, APIError
except ImportError:
    AsyncOpenAI = None
    APIError = Exception


class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI API and OpenAI-compatible endpoints.

    Why this provider:
        - Interfaces with OpenAI's official API (GPT-4, GPT-3.5, etc.).
        - Also supports any OpenAI-compatible local or proxy server (e.g., vLLM,
          LocalAI, Ollama with OpenAI compatibility, Groq, DeepSeek, etc.).
        - Handles both non‑streaming and streaming completions.
        - Converts internal `Message` objects to OpenAI's chat format, including
          multimodal content (images, etc.).

    Design decisions:
        - Uses the official `openai.AsyncOpenAI` client for full compatibility.
        - API key can be provided via config or read from `OPENAI_API_KEY` env var.
        - For local endpoints (base_url provided without an API key), a dummy key
          is used to satisfy the client's validation.
        - Timeout and retries are configured; retries are disabled on the client
          side because the base provider already implements retry logic.
        - Streaming yields content chunks as they arrive; errors during streaming
          are caught and emitted as error strings (not re‑raised) to avoid breaking
          the generator. This is a design trade‑off – the caller can detect errors
          by checking for "[Error:" prefix.

    Performance:
        - The `AsyncOpenAI` client manages connection pooling and request timeouts.
        - No additional threading; all I/O is async.
        - Streaming is efficient: tokens are yielded as they arrive.

    Edge Cases & Risks:
        - If the API returns an error (e.g., rate limit, invalid model), the
          `_generate_internal` method raises `ConfigurationError` (which is then
          wrapped by the base class). For streaming, errors are emitted as text
          chunks to avoid breaking the generator – this may confuse callers that
          expect only valid tokens. A better approach would be to re-raise, but
          that would terminate the generator; the current design prioritises
          not crashing the application.
        - Multimodal content (lists of parts) is converted to OpenAI's expected
          format (each part is a dict with a `type` field). No validation is
          performed; malformed parts may cause API errors.
        - The `reasoning_effort` and other provider‑specific parameters are
          passed through via `extra_params`, allowing access to new features
          without modifying the wrapper.
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for OpenAI provider.

        Requirements:
            - The `openai` package must be installed.
            - An API key must be provided either through `config.api_key` or
              the `OPENAI_API_KEY` environment variable.
            - Exception: if `config.base_url` is set (local endpoint), a dummy
              key is allowed because local servers often ignore the key.

        Raises:
            ConfigurationError: If the openai package is missing or no API key
                                is available (and base_url not provided).
        """
        if not AsyncOpenAI:
            raise ConfigurationError(
                "openai package not installed. Run: pip install openai"
            )
        # Validate API key: required unless using a custom base_url (local endpoint)
        if not self.config.api_key:
            if self.config.base_url:
                # For local endpoints (Ollama, LocalAI, vLLM), use a dummy key.
                self.config.api_key = "dummy"
            elif "OPENAI_API_KEY" not in os.environ:
                raise ConfigurationError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                    "or pass api_key in config."
                )
            else:
                self.config.api_key = os.environ["OPENAI_API_KEY"]

    def _init_client(self) -> None:
        """
        Initialise the AsyncOpenAI client.

        Why:
            - The client is reused for all requests (connection pooling).
            - Timeout is set from `config.timeout`.
            - `max_retries=0` because the base provider already handles retries
              with exponential backoff. Disabling client‑side retries avoids
              double‑retries.
        """
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,  # Retries are handled by BaseLLMProvider
        )

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message list to OpenAI API format.

        Why:
            - OpenAI expects a specific dict structure: `{"role": "...", "content": ...}`
            - Content can be either a string (plain text) or a list of parts
              (multimodal: text, image_url, etc.).
            - This method recursively copies all fields from each part except `type`,
              which is kept as the discriminator.

        Args:
            messages: List of internal Message objects.

        Returns:
            A list of dicts ready to be passed to `chat.completions.create`.

        Edge Cases:
            - If `content` is a string, it is passed as‑is.
            - If `content` is a list, each element is assumed to be a dict with
              a `type` key (e.g., `{"type": "text", "text": "..."}`). All other
              key‑value pairs are copied verbatim.
            - No validation is performed; malformed content will cause an API error.
        """
        openai_messages = []
        for msg in messages:
            content = msg.content
            # Handle multimodal content (list of parts)
            if isinstance(content, list):
                parts = []
                for part in content:
                    # Each part is dict with 'type' and other fields (e.g. 'image_url', 'text')
                    part_dict = {"type": part["type"]}
                    # Copy all fields except 'type' (e.g. 'image_url', 'text')
                    for key, value in part.items():
                        if key != "type":
                            part_dict[key] = value
                    parts.append(part_dict)
                content = parts
            openai_messages.append({
                "role": msg.role.value,
                "content": content,
            })
        return openai_messages

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal non‑streaming generation using OpenAI API.

        How it works:
            1. Convert messages to OpenAI format.
            2. Build parameters by merging config values and method‑level overrides
               (kwargs take precedence).
            3. Pass any extra kwargs (e.g., `reasoning_effort`) that are not
               explicitly handled.
            4. Call the API and convert the response to a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters (temperature, max_tokens, etc.)
                      as well as provider‑specific parameters (e.g., `reasoning_effort`).

        Returns:
            ProviderResponse with generated text, token usage, finish reason, etc.

        Raises:
            ConfigurationError: If the API call fails (wraps OpenAI's APIError).
            (The base provider will further wrap it in retry logic.)
        """
        openai_messages = self._convert_messages(messages)

        # Base parameters from config + kwargs (kwargs override config)
        params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
        }

        # Optional parameters
        if self.config.stop:
            params["stop"] = self.config.stop
        if self.config.seed is not None:
            params["seed"] = self.config.seed

        # Additional kwargs passed by user (e.g., reasoning_effort, modalities)
        # These override any conflicting keys above (if they were inadvertently included).
        extra_params = {k: v for k, v in kwargs.items()
                        if k not in params and not hasattr(self.config, k)}
        params.update(extra_params)

        try:
            response = await self._client.chat.completions.create(**params)
        except APIError as e:
            self._logger.error(f"OpenAI API error: {e}")
            raise ConfigurationError(f"OpenAI API error: {e}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error in OpenAI generate: {e}")
            raise

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
        """
        Internal streaming generation using OpenAI API.

        How it works:
            1. Convert messages to OpenAI format.
            2. Build parameters similarly to `_generate_internal`, but with
               `stream=True`.
            3. Iterate over the async stream, yielding `delta.content` for each chunk.
            4. If an error occurs, yield an error string instead of crashing the
               generator. This is a trade‑off: the caller sees the error but the
               generator remains alive.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters and extra options
                      (e.g., `stream_options`).

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as
              text tokens. This allows the caller to continue processing other
              tasks, but may be unexpected. In production, consider re‑raising
              or having a separate error stream.
            - The stream is closed automatically when the API response is exhausted
              or when an error occurs.
        """
        openai_messages = self._convert_messages(messages)

        params = {
            "model": self.config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
            "stream": True,
        }

        # Optional parameters
        if self.config.stop:
            params["stop"] = self.config.stop
        if self.config.seed is not None:
            params["seed"] = self.config.seed

        # Optional streaming options (e.g., include usage)
        if "stream_options" in kwargs:
            params["stream_options"] = kwargs["stream_options"]

        # Extra kwargs (like reasoning_effort, modalities)
        extra_params = {k: v for k, v in kwargs.items()
                        if k not in params and not hasattr(self.config, k)}
        params.update(extra_params)

        try:
            stream = await self._client.chat.completions.create(**params)
            async for chunk in stream:
                # Safety check: ensure choices and delta exist
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APIError as e:
            self._logger.error(f"OpenAI streaming API error: {e}")
            # Yield error as a string to avoid breaking the generator.
            yield f"[Error: {e}]"
        except Exception as e:
            self._logger.error(f"Unexpected error in OpenAI stream: {e}")
            yield f"[Error: {e}]"