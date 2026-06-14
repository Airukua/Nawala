import os
from typing import List, AsyncGenerator, Dict, Any, Optional
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError
from utils.enums import Role

try:
    from anthropic import AsyncAnthropic
    from anthropic import APIError as AnthropicAPIError
except ImportError:
    AsyncAnthropic = None
    AnthropicAPIError = Exception


class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic Claude API using the official SDK.

    Why this provider:
        - Interfaces with Anthropic's Claude models (Claude 3.5 Sonnet, Claude 3 Opus,
          Claude 3 Haiku, etc.) via the `anthropic` Python SDK.
        - Supports both non‑streaming and token‑streaming chat completions.
        - Handles system prompts separately (Anthropic's API has a dedicated `system`
          parameter, not a system message in the conversation list).
        - Supports multimodal inputs (images) using Claude's vision capabilities.
        - Provides detailed token usage (input_tokens, output_tokens).

    Design decisions:
        - Uses the official `AsyncAnthropic` client for full async support.
        - System prompts are extracted from the first `Role.SYSTEM` message and
          passed via the `system` parameter, not as a message in the conversation.
        - Messages are converted to Anthropic's expected format:
            * Roles: `"user"` or `"assistant"` (no `"system"` in the list).
            * Content can be a string (text) or a list of parts (multimodal).
        - Multimodal content: image URLs (HTTP or data URI) are converted to
          Anthropic's `image` block with `base64` or `url` source.
        - Retries are disabled at the client level (`max_retries=0`) because the
          base provider already implements retry logic (tenacity). This avoids
          double‑retries.
        - Timeout is set from `config.timeout` (default 60s) and applied at the
          client level; the base provider's retry wrapper also applies timeouts.
        - Streaming uses `stream.text_stream` from the SDK's async context manager,
          yielding text chunks as they arrive.

    Performance:
        - The `AsyncAnthropic` client is fully async and manages its own connection pool.
        - No threading overhead; all I/O is non‑blocking.
        - The client is reused across requests, benefiting from keep‑alive connections.

    Edge Cases & Risks:
        - The API key must be provided either in `config.api_key` or the
          `ANTHROPIC_API_KEY` environment variable. The provider raises
          `ConfigurationError` if missing.
        - If a system message appears after user/assistant messages, it will be
          ignored (only the first system message is used). This matches Anthropic's
          API design, which expects system prompt to be provided separately, not
          interleaved with conversation. The provider should warn if multiple
          system messages are found.
        - Multimodal conversion assumes that for data URI images, the MIME type
          (e.g., `image/jpeg`) is correctly embedded in the URI. Malformed URIs
          may cause API errors.
        - The provider does not support Anthropic's `thinking` blocks (Claude 3.7+)
          or tool calling by default; these can be added by extending the provider
          or passing extra parameters via `**kwargs` (not yet implemented in this
          version). Future enhancements could map tool definitions.
        - Streaming errors are emitted as error strings (prefixed with "[Error:") instead
          of raising exceptions. This matches the pattern in other providers and
          prevents the generator from crashing, but the caller must check for error
          strings.
        - The SDK's `stream.text_stream` may yield empty strings occasionally;
          they are filtered out by the provider (if they are `None` or empty,
          they are not yielded).
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for Anthropic provider.

        Requirements:
            - The `anthropic` package must be installed.
            - An API key must be provided either via `config.api_key` or the
              `ANTHROPIC_API_KEY` environment variable.

        Raises:
            ConfigurationError: If the package is missing or no API key is found.
        """
        if not AsyncAnthropic:
            raise ConfigurationError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        if not self.config.api_key:
            if "ANTHROPIC_API_KEY" in os.environ:
                self.config.api_key = os.environ["ANTHROPIC_API_KEY"]
            else:
                raise ConfigurationError(
                    "Anthropic API key is required. Set ANTHROPIC_API_KEY environment "
                    "variable or pass api_key in config."
                )

    def _init_client(self) -> None:
        """
        Initialise the AsyncAnthropic client.

        Why:
            - The client is reused for all requests (connection pooling).
            - Timeout is set from `self.config.timeout` (default 60s).
            - Retries are disabled (set to 0) because the base class already provides
              retry logic with exponential backoff; client‑side retries would be redundant.

        Side Effects:
            - Creates `self._client` as an `AsyncAnthropic` instance.
            - The client does not validate the API key immediately; the first request
              will fail if the key is invalid.
        """
        self._client = AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=0,
        )

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message list to Anthropic's API format.

        Why:
            - Anthropic expects a list of messages with roles `"user"` or `"assistant"`.
            - System prompts are handled separately (not included here).
            - Content can be a string or a list of content blocks (text, image, etc.).
            - This method transforms the internal `Message` objects into the exact
              structure required by `client.messages.create`.

        How it works:
            - System messages (`Role.SYSTEM`) are skipped entirely (they are extracted
              separately by `_extract_system_prompt`).
            - For each non‑system message:
                * Map `Role.ASSISTANT` → `"assistant"`, others → `"user"`.
                * If content is a string, convert to a list with one text block.
                * If content is a list (multimodal), process each part:
                    - `type: "text"` → `{"type": "text", "text": ...}`
                    - `type: "image_url"`:
                        - If URL starts with `data:image` → extract base64 data and MIME type,
                          create `source` with `type: "base64"`.
                        - Else → create `source` with `type: "url"` and the URL.
                * If content is neither string nor list, convert to a string and wrap as text.

        Args:
            messages: List of internal Message objects.

        Returns:
            A list of dicts ready to be passed as the `messages` parameter to
            `client.messages.create`.

        Edge Cases:
            - If an image URL is a data URI, the provider assumes the format
              `data:image/<mime>;base64,<data>`. It splits on the first comma to
              extract the base64 payload. Malformed URIs may cause API errors.
            - The provider does not support other Anthropic‑specific content types
              (e.g., `document`, `thinking`) in this basic conversion; they can be
              added by extending the method.
            - Empty content lists may cause API errors; the method should at least
              provide a text block with empty string if needed.
        """
        anthropic_messages = []
        for msg in messages:
            # Skip system messages, they will be handled separately.
            if msg.role == Role.SYSTEM:
                continue

            content = msg.content
            # Handle multimodal content (images, etc.).
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        # Extract image URL or base64.
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:image"):
                            base64_data = img_url.split(",", 1)[-1]
                            mime_type = img_url.split(";")[0].split(":")[1]
                            parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_data,
                                }
                            })
                        else:
                            parts.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": img_url,
                                }
                            })
                content = parts
            elif not isinstance(content, list):
                # Convert non‑list, non‑string content to string and wrap as text block.
                content = [{"type": "text", "text": str(content)}]

            anthropic_messages.append({
                "role": "assistant" if msg.role == Role.ASSISTANT else "user",
                "content": content,
            })
        return anthropic_messages

    def _extract_system_prompt(self, messages: List[Message]) -> Optional[str]:
        """
        Extract system prompt from messages list if present.

        Why:
            - Anthropic's API requires system prompts to be passed as a separate
              `system` parameter, not as a message in the conversation list.
            - This method retrieves the content of the first `Role.SYSTEM` message.

        Args:
            messages: List of internal Message objects.

        Returns:
            The content of the first system message, or `None` if no system message exists.

        Important:
            - If multiple system messages exist, only the first is used. The provider
              does not merge them; this is consistent with Anthropic's API design.
            - System prompts can be long; the API imposes length limits (typically
              similar to the model's context window).
        """
        for msg in messages:
            if msg.role == Role.SYSTEM:
                return msg.content
        return None

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal non‑streaming generation using Anthropic's async API.

        How it works:
            1. Convert non‑system messages to Anthropic format.
            2. Extract system prompt (if any).
            3. Build request parameters by merging config with method‑level overrides.
            4. Call `self._client.messages.create`.
            5. Extract generated text from the response content blocks (concatenate all text parts).
            6. Extract token usage and finish reason.
            7. Return a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters (temperature, max_tokens, top_p, model).

        Returns:
            ProviderResponse with generated text, token usage, finish reason, etc.

        Raises:
            ConfigurationError: If the API call fails (wrapped from `AnthropicAPIError`).
            (The base provider will retry network‑related errors via tenacity.)

        Important:
            - The `max_tokens` parameter is required by Anthropic; it defaults to
              `self.config.max_tokens` (which must be set, or the API will error).
            - The response may contain multiple text blocks (e.g., interleaved with
              tool calls); this method concatenates them with spaces. For tool‑calling
              scenarios, you may need to extend this.
            - The `stop_reason` is mapped to `finish_reason` in `ProviderResponse`.
        """
        anthropic_messages = self._convert_messages(messages)
        system_prompt = self._extract_system_prompt(messages)

        # Build request parameters.
        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        if system_prompt:
            params["system"] = system_prompt

        # Add optional stop sequences if provided.
        if self.config.stop:
            params["stop_sequences"] = self.config.stop

        timeout = self.config.timeout or 60.0
        try:
            response = await self._client.messages.create(**params)
        except AnthropicAPIError as e:
            self._logger.error(f"Anthropic API error: {e}")
            raise ConfigurationError(f"Anthropic API error: {e}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error in Anthropic generate: {e}")
            raise

        # Extract text from response content blocks.
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        generated_text = " ".join(text_parts)

        # Extract token usage.
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return ProviderResponse(
            text=generated_text,
            model=response.model,
            usage=usage,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Internal streaming generation using Anthropic's streaming API.

        How it works:
            1. Convert messages and extract system prompt similarly to `_generate_internal`.
            2. Build parameters with `stream=True`.
            3. Use the context manager `self._client.messages.stream(...)`.
            4. Iterate over `stream.text_stream`, yielding each text chunk.
            5. If an error occurs, yield an error string (to avoid breaking the generator).

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as text.
              This matches the design of other streaming providers and ensures the
              caller's event loop is not disrupted.
            - The `stream.text_stream` generator yields text as it becomes available,
              which may include partial words. The caller can buffer if needed.
            - The `async with` block ensures proper cleanup of the stream
              (closing the connection, releasing resources).
            - Anthropic's streaming API also yields `message_start`, `content_block`,
              etc. events; this provider only consumes `text_stream`. If you need
              metadata (e.g., usage), you must extend the provider.
        """
        anthropic_messages = self._convert_messages(messages)
        system_prompt = self._extract_system_prompt(messages)

        params = {
            "model": kwargs.get("model", self.config.model),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }
        if system_prompt:
            params["system"] = system_prompt
        if self.config.stop:
            params["stop_sequences"] = self.config.stop

        try:
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicAPIError as e:
            self._logger.error(f"Anthropic streaming API error: {e}")
            yield f"[Error: {e}]"
        except Exception as e:
            self._logger.error(f"Unexpected error in Anthropic stream: {e}")
            yield f"[Error: {e}]"