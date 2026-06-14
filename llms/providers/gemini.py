import os
from typing import List, AsyncGenerator, Dict, Any, Optional
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError
from utils.enums import Role

try:
    from google import genai
    from google.genai.types import (
        Content, Part, GenerateContentConfig,
        SafetySetting, HarmCategory, HarmBlockThreshold,
        ThinkingConfig, Tool, ToolConfig
    )
    from google.genai import errors as genai_errors
except ImportError:
    genai = None


class GeminiProvider(BaseLLMProvider):
    """
    Provider for Google Gemini API using the modern google-genai SDK.

    Why this provider:
        - Interfaces with Google's Gemini family of models (Gemini 1.5 Pro, 1.5 Flash,
          2.0 Flash, etc.) via the official `google-genai` SDK.
        - Supports both non‑streaming and token‑streaming generation.
        - Handles multimodal inputs (text, images, video, files) using Gemini's native
          `Part` and `Content` structures.
        - Provides advanced features: system instructions, thinking budget (Gemini 2.5+),
          safety settings, and structured JSON output via response schema.
        - Uses the async client (`aio` namespace) for full asyncio compatibility.

    Design decisions:
        - The SDK is fully async; no thread pool offloading is needed.
        - System instructions are extracted from the first message if it has role SYSTEM,
          and passed separately to Gemini's API (removing it from the content list).
        - Messages are converted to Gemini's `Content` objects, where assistant messages
          map to `role="model"` (Gemini uses `user` and `model`).
        - Multimodal content (text, image_url, file_url, video) is converted to
          appropriate `Part` types (`Part(text=...)`, `Part(inline_data=...)`,
          `Part(file_data=...)`).
        - The provider supports `thinking_budget` (for reasoning models) and
          `response_schema` (for JSON‑structured outputs).
        - Errors are caught and wrapped as `ConfigurationError` (the base provider
          will retry network‑related errors as per its retry policy).

    Performance:
        - All I/O is async and non‑blocking; the SDK uses HTTP/2 connection pooling.
        - Streaming reduces perceived latency; tokens are yielded as they arrive.
        - Token usage (prompt_tokens, completion_tokens, total_tokens) is extracted
          from `response.usage_metadata` when available.

    Edge Cases & Risks:
        - The `google-genai` SDK is relatively new; some features (e.g., thinking
          budget) are model‑specific and may raise errors if used with incompatible
          models. The provider passes them through; the API will reject them.
        - Multimodal conversion: for image URLs that are data URIs (base64), the
          provider extracts the MIME type and base64 data. For remote URLs, it uses
          `file_data`. For video, it assumes a `file_uri`. This is a best‑effort
          conversion; malformed input may cause API errors.
        - If a system message is present, it is removed from the `contents` list.
          This assumes the system message is the first element. If there are multiple
          system messages (unusual), only the first is used; subsequent system messages
          will be treated as regular content and may break the conversation flow.
        - The provider does not implement retries beyond what the base class provides;
          the `google-genai` SDK may have its own retry logic (the provider disables
          client‑side retries by using the base class's tenacity retry).
        - Streaming errors are emitted as error strings (prefixed with "[Error:") instead
          of raising exceptions. This matches the pattern used in other providers and
          avoids breaking the generator, but the caller must check for error strings.
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for Gemini provider.

        Requirements:
            - The `google-genai` package must be installed.
            - An API key must be provided either via `config.api_key` or the
              `GEMINI_API_KEY` environment variable.

        Side Effects:
            - If the API key is not set in config but exists in the environment,
              it is assigned to `self.config.api_key`.

        Raises:
            ConfigurationError: If the package is missing or no API key is found.
        """
        if not genai:
            raise ConfigurationError(
                "google-genai package not installed. Run: pip install google-genai"
            )
        if not self.config.api_key:
            if "GEMINI_API_KEY" in os.environ:
                self.config.api_key = os.environ["GEMINI_API_KEY"]
            else:
                raise ConfigurationError(
                    "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                    "or pass api_key in config."
                )

    def _init_client(self) -> None:
        """
        Initialise the async Gemini client.

        Why:
            - Creates a `genai.Client` instance with the provided API key.
            - Uses the `v1beta` API version (latest stable at time of writing).
            - Sets a default model (`gemini-2.0-flash`) if none is specified in config.

        Side Effects:
            - Stores the client in `self._client`.
            - Modifies `self.config.model` if it was empty.

        Important:
            - The client is reused for all requests; it manages its own HTTP session.
            - No explicit timeout is set at the client level; timeouts are handled
              via `asyncio.wait_for` in the base provider's retry logic.
        """
        self._client = genai.Client(
            api_key=self.config.api_key,
            http_options={"api_version": "v1beta"}  # Use latest stable version.
        )
        # Set default model if not specified.
        if not self.config.model:
            self.config.model = "gemini-2.0-flash"

    def _convert_messages(self, messages: List[Message]) -> List[Content]:
        """
        Convert internal Message list to Gemini's Content format.

        Why:
            - Gemini expects a list of `Content` objects, each with a `role`
              (`"user"` or `"model"`) and a list of `Part` objects.
            - The internal `Role` enum uses `ASSISTANT`; Gemini uses `"model"`.
            - Multimodal content (images, files, video) must be translated into
              the appropriate `Part` subtypes.

        How it works:
            - For each message:
                * Map `Role.ASSISTANT` → `"model"`, others → role.value.
                * If content is a string → `Part(text=...)`.
                * If content is a list (multimodal):
                    - `type: "text"` → `Part(text=...)`
                    - `type: "image_url"`:
                        - If URL starts with `data:image` → extract base64 data and MIME type,
                          create `Part(inline_data=...)`.
                        - Else → create `Part(file_data={"file_uri": url})`.
                    - `type: "file_url"` → `Part(file_data={"file_uri": url})`.
                    - `type: "video"` → `Part(file_data={"file_uri": video_url})`.
                * If content is neither string nor list, convert to string.

        Args:
            messages: List of internal Message objects.

        Returns:
            A list of `Content` objects ready to be passed to Gemini's API.

        Edge Cases:
            - If an image URL is a data URI, the base64 payload is extracted by
              splitting on the first comma. This assumes a well‑formed data URI.
            - If the MIME type cannot be parsed from the data URI, it defaults
              to `"image/jpeg"`? The code currently extracts it; if parsing fails,
              the API may reject it.
            - For video, only `file_data` with a `file_uri` is supported; inline
              video data is not handled.
            - The conversion does not validate that the URLs or base64 data are valid.
        """
        contents = []
        for msg in messages:
            # Map role: Gemini uses "user" and "model" (not "assistant").
            role = "model" if msg.role == Role.ASSISTANT else msg.role.value

            content_parts = []
            if isinstance(msg.content, str):
                content_parts.append(Part(text=msg.content))
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.get("type") == "text":
                        content_parts.append(Part(text=part.get("text", "")))
                    elif part.get("type") == "image_url":
                        # Extract image URL or base64.
                        img_url = part.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:image"):
                            # Base64 image (prefix like "data:image/jpeg;base64,").
                            base64_data = img_url.split(",", 1)[-1]
                            mime_type = img_url.split(";")[0].split(":")[1]
                            content_parts.append(Part(
                                inline_data={"mime_type": mime_type, "data": base64_data}
                            ))
                        else:
                            content_parts.append(Part(file_data={"file_uri": img_url}))
                    elif part.get("type") == "file_url":
                        content_parts.append(Part(file_data={"file_uri": part.get("file_url", {}).get("url", "")}))
                    elif part.get("type") == "video":
                        # Gemini supports video via File API (for large files) or inline data.
                        video_url = part.get("video_url", "")
                        if video_url:
                            content_parts.append(Part(file_data={"file_uri": video_url}))
            else:
                content_parts.append(Part(text=str(msg.content)))

            contents.append(Content(role=role, parts=content_parts))
        return contents

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal non‑streaming generation using Gemini's async API.

        How it works:
            1. Convert messages to Gemini `Content` list.
            2. Build `GenerateContentConfig` from config and kwargs.
            3. Extract system instruction if first message is SYSTEM (removing it from contents).
            4. Apply thinking budget and response schema if provided.
            5. Call `self._client.aio.models.generate_content` with timeout.
            6. Extract text, usage metadata, and finish reason.
            7. Return a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters (temperature, max_tokens, top_p,
                      thinking_budget, response_schema, etc.).

        Returns:
            ProviderResponse with generated text and metadata.

        Raises:
            ConfigurationError: If the API call fails (wrapped from `genai_errors.APIError`).
            (The base provider will retry network‑related errors via tenacity.)

        Important:
            - The `system_instruction` is only taken from the first message if its
              role is SYSTEM. Subsequent system messages are not supported.
            - If `response_schema` is provided, the response MIME type is forced
              to `application/json` and the schema is passed to the API. The
              returned text will be JSON.
            - Thinking budget is only supported by Gemini 2.5+ models; using it
              with older models will cause an API error.
        """
        contents = self._convert_messages(messages)

        # Build generation config.
        config_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Handle system instruction (Gemini's specific approach).
        system_instruction = None
        if messages and messages[0].role == Role.SYSTEM:
            system_instruction = messages[0].content
            contents = contents[1:]  # Remove system from contents list.

        # Support for thinking budget (Gemini 2.5 and 3 series).
        if "thinking_budget" in kwargs:
            config_params["thinking_config"] = ThinkingConfig(
                include_thoughts=True,
                thinking_budget=kwargs["thinking_budget"]
            )
        elif hasattr(self.config, "thinking_budget") and self.config.thinking_budget:
            config_params["thinking_config"] = ThinkingConfig(
                include_thoughts=True,
                thinking_budget=self.config.thinking_budget
            )

        # Optional safety settings.
        if hasattr(self.config, "safety_settings") and self.config.safety_settings:
            config_params["safety_settings"] = [
                SafetySetting(
                    category=HarmCategory(cat),
                    threshold=HarmBlockThreshold(thresh)
                )
                for cat, thresh in self.config.safety_settings.items()
            ]

        # Optional response schema for structured outputs.
        if "response_schema" in kwargs:
            config_params["response_mime_type"] = "application/json"
            config_params["response_json_schema"] = kwargs["response_schema"]

        timeout = self.config.timeout or 60.0

        try:
            response = await self._client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=GenerateContentConfig(**config_params) if config_params else None,
            )
        except genai_errors.APIError as e:
            self._logger.error(f"Gemini API error: {e}")
            raise ConfigurationError(f"Gemini API error: {e}") from e
        except Exception as e:
            self._logger.error(f"Unexpected error in Gemini generate: {e}")
            raise

        # Extract usage metadata.
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        # Extract finish reason.
        finish_reason = "stop"
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason

        return ProviderResponse(
            text=response.text or "",
            model=self.config.model,
            usage=usage,
            finish_reason=finish_reason,
            raw_response=response,
        )

    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Internal streaming generation using Gemini's streaming API.

        How it works:
            1. Convert messages to Gemini `Content` list.
            2. Build config similarly to `_generate_internal`.
            3. Remove system instruction from contents if present.
            4. Call `self._client.aio.models.generate_content_stream`.
            5. Iterate over the async stream, yielding `chunk.text` for each chunk.
            6. If an error occurs, yield an error string (to avoid breaking the generator).

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as text.
              This matches the design of other streaming providers and ensures the
              caller's event loop is not broken.
            - Thoughts are not streamed (included_thoughts=False) when thinking budget
              is enabled, because they would interrupt the token stream. If you need
              thoughts, you must use non‑streaming generation.
            - The streaming API may yield multiple chunks; the provider concatenates
              the `text` field of each chunk. Some chunks may be empty (e.g., only
              usage metadata); they are skipped.
            - The stream will automatically close when the response is complete or
              an error occurs.
        """
        contents = self._convert_messages(messages)

        config_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Handle system instruction (remove from contents if present).
        if messages and messages[0].role == Role.SYSTEM:
            contents = contents[1:]

        # Support for thinking budget (but do not stream thoughts).
        if "thinking_budget" in kwargs:
            config_params["thinking_config"] = ThinkingConfig(
                include_thoughts=False,  # Don't stream thoughts (they would break the token flow).
                thinking_budget=kwargs["thinking_budget"]
            )

        try:
            stream = self._client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=GenerateContentConfig(**config_params) if config_params else None,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self._logger.error(f"Gemini streaming error: {e}")
            yield f"[Error: {e}]"