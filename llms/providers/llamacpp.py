import asyncio
import threading
from typing import List, AsyncGenerator, Optional, Dict, Any
from llms.models import Message, ProviderResponse
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError
from utils.enums import Role
from concurrent.futures import ThreadPoolExecutor

try:
    from llama_cpp import Llama as LlamaCPP
except ImportError:
    LlamaCPP = None


class LlamaCPPProvider(BaseLLMProvider):
    """
    Provider for local GGUF models using llama-cpp-python.

    Why this provider:
        - Enables running quantised GGUF models (e.g., Llama 2, Mistral, Mixtral)
          locally without external API calls.
        - Uses the `llama-cpp-python` library, which provides efficient CPU/GPU
          inference for GGUF format models.
        - Supports both standard chat completions and token streaming.
        - The underlying library is synchronous; this provider offloads all calls
          to a dedicated thread pool to avoid blocking the asyncio event loop.

    Design decisions:
        - A single-thread executor (`ThreadPoolExecutor(max_workers=1)`) ensures
          that only one inference runs at a time. This prevents overwhelming the
          CPU/GPU and avoids potential memory issues. If concurrent requests are
          needed, increase `max_workers` or use a model server.
        - The model is loaded synchronously in `_init_client` using the thread pool.
          Loading can take several seconds; it blocks the executor but not the
          event loop (the executor thread does the loading while the event loop
          remains free).
        - Generation methods (`_generate_internal`, `_stream_internal`) use
          `asyncio.wait_for` with a timeout to prevent hanging if the model
          stalls (rare but possible).
        - Streaming uses a synchronous generator (`stream_generator`) that is
          run in the thread pool; tokens are yielded one by one, and after each
          token `await asyncio.sleep(0)` yields control back to the event loop.
        - The provider reuses the same model instance for all requests; it is
          not re‑loaded unless the client is recreated.

    Performance:
        - Model inference is CPU/GPU bound; the thread pool prevents blocking
          the asyncio loop but does not speed up inference.
        - Using `max_workers=1` serialises requests; for higher throughput,
          consider using multiple instances of the provider (each with its own
          model copy) or a dedicated inference server.
        - Streaming reduces perceived latency but total generation time is unchanged.

    Edge Cases & Risks:
        - The llama-cpp-python library is not fully async; running it in a
          thread pool is safe but adds overhead.
        - Timeout applies to the entire generation; if the model is slow,
          `asyncio.TimeoutError` will be raised after `timeout` seconds.
        - The `metadata` dict in `ProviderConfig` must contain `model_path`
          (path to the GGUF file). Optional keys: `n_ctx` (context length,
          default 2048), `n_gpu_layers` (number of layers offloaded to GPU,
          default 0), `chat_format` (conversation template, e.g., "llama-2").
        - The model may produce malformed responses (e.g., non‑JSON); the
          provider returns raw text as received.
        - Streaming error handling: if an error occurs mid‑stream, an error string
          prefixed with "[Error:" is yielded; the generator does not raise an
          exception. The caller must check for error strings.
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for LlamaCPP provider.

        Requirements:
            - `llama-cpp-python` package must be installed.
            - `metadata` must contain `model_path` pointing to a valid GGUF file.

        Raises:
            ConfigurationError: If dependencies are missing or model_path not provided.
        """
        if not LlamaCPP:
            raise ConfigurationError(
                "llama-cpp-python package not installed. Run: pip install llama-cpp-python"
            )
        if not self.config.metadata.get("model_path"):
            raise ConfigurationError(
                "LlamaCPPProvider requires 'model_path' in metadata"
            )

    def _init_client(self) -> None:
        """
        Initialise the LlamaCPP client with proper threading setup.

        Why:
            - Loading a GGUF model can take several seconds; done in a thread
              pool to avoid blocking the asyncio event loop.
            - The model is loaded once and reused for all requests.

        Configuration from metadata:
            - `model_path`: Path to GGUF file (required).
            - `n_ctx`: Context window size (default 2048).
            - `n_gpu_layers`: Number of layers to offload to GPU (default 0).
            - `chat_format`: Chat template (e.g., "llama-2", "mistral", "chatml").

        Side Effects:
            - Creates a `ThreadPoolExecutor` with one worker.
            - Loads the model into memory (may consume several GB of RAM/VRAM).
            - Stores the model instance in `self._client`.
        """
        # No lock needed for single-threaded executor.
        self._executor = ThreadPoolExecutor(max_workers=1)
        model_path = self.config.metadata["model_path"]
        n_ctx = self.config.metadata.get("n_ctx", 2048)
        n_gpu_layers = self.config.metadata.get("n_gpu_layers", 0)
        chat_format = self.config.metadata.get("chat_format")

        # Load model in thread pool to avoid blocking the event loop.
        future = self._executor.submit(
            LlamaCPP,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
            verbose=False,
        )
        self._client = future.result()

        # Store tokenizer if available (for potential extensions).
        self._tokenizer = getattr(self._client, "tokenizer", None)

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Convert internal Message list to llama-cpp-python chat format.

        Why:
            - llama-cpp-python expects a list of dicts with keys `role` and `content`.
            - Content must be a string; if the original content is a list (multimodal),
              it is converted to a string representation. The provider does not
              support multimodal inputs (images) – only text.

        Args:
            messages: List of internal Message objects.

        Returns:
            List of dicts suitable for `create_chat_completion(messages=...)`.

        Edge Cases:
            - If content is already a string, it is used as‑is.
            - If content is a list (e.g., from multimodal input), it is converted
              to a string via `str(content)`. This may produce undesirable output
              (e.g., Python representation). A better approach would be to extract
              text parts, but that is left as future improvement.
        """
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.role.value,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            })
        return converted

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Generate a non‑streaming response using the chat completion API.

        How it works:
            1. Convert messages to llama-cpp-python format.
            2. Prepare parameters (temperature, max_tokens, top_p, stop) by merging
               config and method‑level overrides.
            3. Run `self._client.create_chat_completion` in the thread pool,
               wrapped with `asyncio.wait_for` to enforce timeout.
            4. Extract the generated text, token usage, and finish reason from the
               response.
            5. Return a `ProviderResponse`.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters (temperature, max_tokens, top_p).

        Returns:
            ProviderResponse with generated text and metadata.

        Raises:
            ConfigurationError: If the request times out or the model call fails.
            (Base provider will wrap and retry if applicable.)

        Important:
            - The llama-cpp-python call is synchronous and CPU‑bound; the thread
              pool prevents blocking the event loop.
            - Timeout applies to the entire generation; if the model is slow,
              `asyncio.TimeoutError` is caught and re‑raised as `ConfigurationError`.
            - Token usage is extracted from the `usage` field if present; otherwise
              defaults to 0.
        """
        # Convert messages to proper format.
        llama_messages = self._convert_messages(messages)

        # Prepare generation parameters (kwargs override config).
        params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stop": self.config.stop,
        }

        timeout = self.config.timeout or 60.0

        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self._client.create_chat_completion(
                        messages=llama_messages,
                        **params
                    )
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError as e:
            raise ConfigurationError(f"LlamaCPP request timed out after {timeout}s") from e
        except Exception as e:
            self._logger.error(f"LlamaCPP generation error: {e}")
            raise ConfigurationError(f"LlamaCPP API error: {e}") from e

        # Extract response data.
        choice = response["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Extract token usage (may be missing in older versions).
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ProviderResponse(
            text=content,
            model=self.config.model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=response,
        )

    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens one by one using the streaming chat completion API.

        How it works:
            1. Convert messages to llama-cpp-python format.
            2. Prepare parameters with `stream=True`.
            3. Define a synchronous generator `stream_generator` that iterates
               over `self._client.create_chat_completion` chunks and yields tokens.
            4. Run the generator in the thread pool and yield tokens from it,
               yielding control to the event loop after each token via
               `await asyncio.sleep(0)`.
            5. If an error occurs (timeout or exception), emit an error string
               prefixed with "[Error:" instead of raising.

        Args:
            messages: List of Message objects.
            **kwargs: Override generation parameters.

        Yields:
            String tokens (or error messages prefixed with "[Error:").

        Important:
            - The generator does NOT raise exceptions; errors are emitted as text.
              The caller must check for error strings.
            - The synchronous generator runs in the thread pool; each `next()`
              call may block the thread, but the event loop remains free because
              the generator is iterated in the main task (the iteration yields
              control after each token).
            - The timeout applies to the initial creation of the generator, not
              to the streaming loop. If the model stalls after generating some
              tokens, the loop may hang indefinitely. A better approach would
              implement a per‑token timeout, but that is complex and omitted.
        """
        llama_messages = self._convert_messages(messages)

        params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": True,
        }
        if self.config.stop:
            params["stop"] = self.config.stop

        timeout = self.config.timeout or 60.0

        loop = asyncio.get_running_loop()

        def stream_generator():
            """
            Synchronous generator that yields tokens from the model.

            Why separate function:
                - Allows running in the thread pool while yielding control
                  back to the event loop after each token.
                - Handles exceptions internally, yielding error strings instead
                  of raising (because raising from a thread would be messy).
            """
            try:
                stream = self._client.create_chat_completion(
                    messages=llama_messages,
                    **params
                )
                for chunk in stream:
                    # Extract content from chunk.
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    # Check for stop condition.
                    if chunk.get("stop", False):
                        break
            except Exception as e:
                self._logger.error(f"LlamaCPP streaming error: {e}")
                yield f"[Error: {e}]"

        try:
            # Run the synchronous generator in the thread pool.
            # Note: run_in_executor returns a generator object; we iterate over it.
            stream = await asyncio.wait_for(
                loop.run_in_executor(self._executor, stream_generator),
                timeout=timeout
            )
            for token in stream:
                yield token
                # Yield control back to the event loop to allow other tasks.
                await asyncio.sleep(0)
        except asyncio.TimeoutError as e:
            self._logger.error(f"LlamaCPP streaming timeout after {timeout}s")
            yield f"[Error: Streaming timeout]"
        except Exception as e:
            self._logger.error(f"LlamaCPP streaming error: {e}")
            yield f"[Error: {e}]"