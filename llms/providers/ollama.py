import asyncio
from typing import List, AsyncGenerator, Dict, Any
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError

try:
    from ollama import AsyncClient as OllamaClient
except ImportError:
    OllamaClient = None


class OllamaProvider(BaseLLMProvider):
    """
    Provider for locally running Ollama models.

    Why this provider:
        - Enables local LLM inference using Ollama (supports models like Llama 3,
          Mistral, Gemma, Phi, etc.) without external API calls.
        - Ollama provides a REST API (default http://localhost:11434) and an
          async Python client.
        - Supports both standard generation and token streaming.
        - Handles multimodal inputs (images) when the model supports it.

    Design decisions:
        - Uses the official `ollama.AsyncClient` for fully async operations.
        - The Ollama client is lightweight and does not maintain persistent
          connections; it creates a new HTTP session per request (still fine).
        - A custom timeout is applied via `asyncio.wait_for` because the Ollama
          client does not expose a native timeout parameter.
        - Both `_generate_internal` and `_stream_internal` handle the conversion
          from internal `Message` objects to Ollama's expected format, including
          multimodal image content (base64 or URL).
        - Error handling: timeouts raise `ConfigurationError` (wrapped by base
          class retry logic); other exceptions are wrapped similarly.
        - Streaming yields partial tokens as they become available; errors during
          streaming are caught and emitted as error strings (to avoid breaking
          the generator). This is a pragmatic trade‑off – the caller must check
          for "[Error:" prefix.

    Performance:
        - The Ollama server must be running separately; network latency is low
          (localhost) but the model inference time dominates.
        - No threading overhead; all calls are async and non‑blocking.
        - Streaming reduces perceived latency.

    Edge Cases & Risks:
        - The client may hang indefinitely if the Ollama server does not respond;
          the timeout (from `self.config.timeout`) prevents this.
        - `_convert_messages` tries to handle image data: if the image URL is a
          data URI (base64), it extracts the base64 payload; otherwise passes the
          URL as‑is (Ollama expects either a path, URL, or base64 string). This
          logic is simplistic and may fail for some image formats.
        - Token usage statistics (`prompt_eval_count`, `eval_count`) are not
          guaranteed to be present in all Ollama versions; the code falls back
          to 0 if missing.
        - The `__aexit__` method is defined to close the client when used as an
          async context manager, but note that `OllamaProvider` is typically used
          inside `LLMClient`, which also calls `close()`. Double‑closing is safe.
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for Ollama provider.

        Requirements:
            - The `ollama` package must be installed.

        Raises:
            ConfigurationError: If the ollama package is missing.
        """
        if not OllamaClient:
            raise ConfigurationError(
                "ollama package not installed. Run: pip install ollama"
            )

    def _init_client(self) -> None:
        """
        Initialise the Ollama async client.

        Why:
            - The client is configured with the server host (from `config.base_url`
              or default localhost:11434).
            - No authentication or custom timeouts are set at the client level;
              timeouts are enforced via `asyncio.wait_for` in the generation methods.
        """
        host = self.config.base_url or "http://localhost:11434"
        self._client = OllamaClient(host=host)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message list to Ollama API format.

        Why:
            - Ollama expects messages as a list of dicts with `role` and `content`.
            - Content can be a plain string or a list of parts (multimodal).
            - This method handles both text and image parts.

        How it works:
            1. For each message, if `content` is a string, keep it as‑is.
            2. If `content` is a list (multimodal):
                - For each part with type "text": convert to `{"type": "text", "text": ...}`
                - For each part with type "image_url":
                    * If the URL is a data URI (starts with "data:image"), extract
                      the base64 payload.
                    * Otherwise, keep the URL as a string.
                    * Ollama expects `{"type": "image", "image": base64_or_url}`.
            3. If after conversion the list contains exactly one text part, simplify
               to a plain string (Ollama accepts both).

        Args:
            messages: List of internal Message objects.

        Returns:
            A list of dicts compatible with Ollama's chat API.

        Edge Cases:
            - If an image URL is not a data URI, it is passed as‑is; Ollama will
              attempt to fetch the image. This may fail if the URL is not accessible
              from the Ollama server.
            - Malformed parts may cause an Ollama API error; this method does not
              validate deeply.
        """
        ollama_messages = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        # Extract base64 payload if it's a data URI.
                        if image_url.startswith("data:image"):
                            base64_str = image_url.split(",", 1)[-1]
                            parts.append({"type": "image", "image": base64_str})
                        else:
                            parts.append({"type": "image", "image": image_url})
                # Simplify if only one text part (Ollama can accept plain string).
                if len(parts) == 1 and parts[0]["type"] == "text":
                    content = parts[0]["text"]
                else:
                    content = parts
            ollama_messages.append({
                "role": msg.role.value,
                "content": content,
            })
        return ollama_messages

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal non‑streaming generation using Ollama API.

        How it works:
            1. Convert messages to Ollama format.
            2. Build parameters: model, messages, options (temperature, num_predict, top_p).
            3. Apply a timeout via `asyncio.wait_for` using `self.config.timeout`.
            4. Extract the generated text, token usage, and finish reason from the response.
            5. Return a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters (temperature, max_tokens, top_p).

        Returns:
            ProviderResponse with generated text and metadata.

        Raises:
            ConfigurationError: If the request times out or the Ollama API returns an error.
            (Base provider will wrap and possibly retry.)

        Important:
            - The Ollama client does not support timeouts natively; we use
              `asyncio.wait_for`. The timeout applies to the entire request,
              including model inference.
            - Token usage is extracted from `prompt_eval_count` and `eval_count`,
              which may be missing in older Ollama versions; missing values default to 0.
            - The finish reason defaults to "stop" if not provided by the API.
        """
        ollama_messages = self._convert_messages(messages)

        params = {
            "model": self.config.model,
            "messages": ollama_messages,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            },
        }

        timeout = self.config.timeout or 60.0
        try:
            response = await asyncio.wait_for(
                self._client.chat(**params),
                timeout=timeout
            )
        except asyncio.TimeoutError as e:
            raise ConfigurationError(f"Ollama request timed out after {timeout}s") from e
        except Exception as e:
            self._logger.error(f"Ollama generation error: {e}")
            raise ConfigurationError(f"Ollama API error: {e}") from e

        # Extract generated text safely (response may be object or dict).
        if hasattr(response, "message") and hasattr(response.message, "content"):
            generated_text = response.message.content
        elif isinstance(response, dict):
            generated_text = response.get("message", {}).get("content", "")
        else:
            generated_text = ""

        # Extract token usage (Ollama may return these as attributes or dict keys).
        prompt_tokens = getattr(response, "prompt_eval_count", 0)
        if prompt_tokens == 0 and isinstance(response, dict):
            prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = getattr(response, "eval_count", 0)
        if completion_tokens == 0 and isinstance(response, dict):
            completion_tokens = response.get("eval_count", 0)

        # Extract finish reason (default "stop").
        finish_reason = "stop"
        if hasattr(response, "done_reason"):
            finish_reason = response.done_reason
        elif isinstance(response, dict):
            finish_reason = response.get("done_reason", "stop")

        return ProviderResponse(
            text=generated_text,
            model=self.config.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason=finish_reason,
            raw_response=response,
        )

    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Internal streaming generation using Ollama API.

        How it works:
            1. Convert messages to Ollama format.
            2. Build parameters with `stream=True`.
            3. Iterate over the async stream, extracting content from each chunk.
            4. Yield content as it arrives; stop when `done` flag is True.
            5. If an error occurs, yield an error string (to avoid breaking the generator).

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as text.
              This design choice prevents the calling code from crashing but
              requires the caller to check for error strings.
            - The `done` flag is checked after each chunk; when True, the stream
              is finished.
            - The Ollama client's async generator may throw exceptions; they are
              caught and converted to error strings.
        """
        ollama_messages = self._convert_messages(messages)

        params = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
            },
        }

        try:
            stream = await self._client.chat(**params)
            async for chunk in stream:
                content = None
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                elif isinstance(chunk, dict):
                    content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

                # Check if this chunk indicates the end of the stream.
                done = False
                if hasattr(chunk, "done"):
                    done = chunk.done
                elif isinstance(chunk, dict):
                    done = chunk.get("done", False)
                if done:
                    break
        except Exception as e:
            self._logger.error(f"Ollama streaming error: {e}")
            yield f"[Error: {e}]"

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Clean up resources when exiting an async context manager.

        Why:
            - Allows using `async with OllamaProvider(...) as provider:`.
            - The Ollama client should be closed to release the underlying
              HTTP session and connection pool.

        Note:
            - This method is called automatically when exiting the `async with` block.
            - It is also safe to call `close()` directly; double‑closing is harmless.
        """
        if hasattr(self, "_client") and self._client:
            await self._client.close()
            self._logger.debug("Ollama client closed")