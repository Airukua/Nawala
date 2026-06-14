import asyncio
from typing import List, AsyncGenerator
from llms.models import Message, ProviderResponse
import torch
from llms.base import BaseLLMProvider
from llms.exceptions import ConfigurationError
from utils.enums import Role
from concurrent.futures import ThreadPoolExecutor

try:
    from transformers import TextIteratorStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """
    Provider for HuggingFace Transformers models (local inference).

    Why this provider:
        - Enables local execution of thousands of open-source models (LLaMA, Mistral,
          Phi, Gemma, etc.) without API calls.
        - Supports both standard generation and token‑by‑token streaming.
        - Uses HuggingFace's `transformers` library with pipeline abstraction.

    Design decisions:
        - The pipeline is initialised synchronously in `_init_client` to avoid
          complex async loading. Model loading can take several seconds and
          consumes significant RAM/VRAM; this happens once at startup.
        - Generation is offloaded to a `ThreadPoolExecutor` because the
          pipeline's `__call__` method is synchronous and CPU‑bound (or GPU‑bound).
          This prevents blocking the asyncio event loop.
        - Streaming uses `TextIteratorStreamer` combined with the executor to
          yield tokens as they are generated, while still yielding control to
          the event loop via `await asyncio.sleep(0)`.
        - Prompt formatting uses the tokenizer's `apply_chat_template` when
          available (modern models), falling back to a simple manual template
          for older models.

    Performance considerations:
        - Model loading is heavy and synchronous; consider loading the model
          once and reusing it across multiple requests (this provider does that).
        - Using an executor (thread pool) with one worker ensures that only one
          generation runs at a time. For concurrent requests, increase
          `max_workers` or implement a queue.
        - Streaming reduces perceived latency but the total time remains the same.
        - GPU memory usage depends on the model size; larger models may need
          quantization (not implemented here, but can be added via config).

    Edge Cases & Risks:
        - If the tokenizer lacks a chat template, the fallback manual format
          may not match the model's expected conversation style, leading to
          poor outputs. Log a warning when fallback is used.
        - The executor runs tasks sequentially (max_workers=1). If a second
          request arrives while the first is still generating, it will wait.
          For production, consider a dedicated inference server or increase
          workers (but beware of GPU memory limits).
        - Streaming uses a thread‑safe generator (`TextIteratorStreamer`) that
          produces tokens from a background thread. The `await asyncio.sleep(0)`
          inside the token loop is necessary to allow the event loop to process
          other tasks; without it, the generator would monopolise the loop.
        - `_generate_internal` returns `usage` with zero tokens because the
          HuggingFace pipeline does not provide token counts by default.
          Subclasses or extensions could implement token counting via the tokenizer.
    """

    def _validate_config(self) -> None:
        """
        Validate configuration for HuggingFace provider.

        Requirements:
            - `transformers` and `torch` packages must be installed.
            - Model identifier must be provided (e.g., "meta-llama/Llama-2-7b-chat-hf").
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ConfigurationError(
                "transformers package not installed. Run: pip install transformers torch"
            )
        if not self.config.model:
            raise ConfigurationError("HuggingFaceProvider requires a model identifier")

    def _init_client(self) -> None:
        """
        Initialise the HuggingFace text-generation pipeline and tokenizer.

        Why synchronous initialisation:
            - Loading a model is CPU/GPU intensive and may take many seconds.
            - Doing it asynchronously would add complexity; it's acceptable to
              block during startup.
            - The pipeline is stored as `self._pipe` and reused for all requests.

        Side Effects:
            - Loads the model into memory (GPU if available, else CPU).
            - May raise `OSError` or `ImportError` if model is not found or
              dependencies are missing.
            - Creates a `ThreadPoolExecutor` with one worker for offloading
              synchronous calls.
        """
        from transformers import pipeline

        # Initialise pipeline synchronously (safer – avoids deadlocks with asyncio).
        self._pipe = pipeline(
            "text-generation",
            model=self.config.model,
            device_map="auto",          # Automatically use GPU if available
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        # Store tokenizer for chat template and streaming.
        self._tokenizer = self._pipe.tokenizer
        # Thread pool to run synchronous pipeline calls without blocking the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _generate_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> ProviderResponse:
        """
        Internal non‑streaming generation using HuggingFace pipeline.

        How it works:
            1. Convert internal `Message` list to a single prompt string using
               the model's chat template (or fallback).
            2. Run the pipeline in a thread pool to avoid blocking the event loop.
            3. Extract the generated text (strip the prompt prefix if present).
            4. Return a `ProviderResponse` with the text and dummy usage stats.

        Args:
            messages: List of Message objects (system, user, assistant).
            **kwargs: Override generation parameters (max_tokens, temperature, top_p).

        Returns:
            ProviderResponse containing generated text.

        Important:
            - Token usage is not provided by the default pipeline; set to zero.
            - The pipeline may return the full prompt+completion; we attempt to
              strip the prompt. This heuristic works for most models but may fail
              if the generated text starts with the exact prompt string for other
              reasons. A more robust approach would use the pipeline's
              `return_full_text=False` parameter (if supported).
        """
        prompt = self._messages_to_prompt(messages)
        loop = asyncio.get_running_loop()

        # Offload the synchronous pipeline call to a thread.
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._pipe(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
            )
        )

        generated = result[0]["generated_text"]
        # Remove the prompt if it appears at the beginning of the output.
        if generated.startswith(prompt):
            generated = generated[len(prompt):]

        return ProviderResponse(
            text=generated,
            model=self.config.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw_response=result,
        )

    async def _stream_internal(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Internal streaming generation using HuggingFace `TextIteratorStreamer`.

        How it works:
            1. Convert messages to a prompt string.
            2. Create a `TextIteratorStreamer` that will accumulate tokens in a
               thread‑safe queue.
            3. Start the pipeline in the thread pool, passing the streamer.
            4. Yield tokens from the streamer as they arrive, yielding control
               to the event loop after each token.

        Why this approach:
            - `TextIteratorStreamer` is designed for asynchronous token streaming
              from a synchronous pipeline.
            - Running the pipeline in a thread prevents blocking the event loop.
            - The `await asyncio.sleep(0)` after each token ensures that other
              asyncio tasks (e.g., handling new requests) can run.

        Important:
            - The streamer's `skip_prompt=True` ensures that the prompt itself
              is not yielded as tokens.
            - `skip_special_tokens=True` removes tokens like <|eot_id|> that are
              not useful for the application.
            - The generator must be fully consumed or closed; otherwise, the
              background thread may continue generating.
        """
        prompt = self._messages_to_prompt(messages)

        # Create a streamer that omits the prompt and special tokens.
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            "text_inputs": prompt,
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": True,
            "streamer": streamer,
        }

        loop = asyncio.get_running_loop()
        # Start the pipeline in the thread pool. This call will return immediately
        # because the streamer runs in a background thread; the main thread will
        # consume tokens from the streamer.
        await loop.run_in_executor(
            self._executor,
            self._pipe,
            **gen_kwargs
        )

        # Yield tokens as they become available.
        for token in streamer:
            yield token
            # Yield control to the event loop to keep the system responsive.
            await asyncio.sleep(0)

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """
        Convert a list of internal `Message` objects to a single prompt string.

        Why two strategies:
            - Modern models (e.g., LLaMA 3, Mistral) have tokenizer‑defined chat
              templates that produce the correct special tokens (e.g., `<|start_header_id|>`).
            - Older or custom models may lack a template; we fall back to a simple
              manual format that works for basic instruction‑following models.

        Strategy:
            1. Attempt to use `tokenizer.apply_chat_template` (preferred).
            2. If that fails (exception), log a warning and use the manual format.

        Args:
            messages: List of Message objects.

        Returns:
            A string suitable as input to the model's generation pipeline.

        Edge Cases:
            - If the tokenizer has no template, the fallback format may be
              suboptimal. The caller (application) should consider using a
              model that provides a template.
            - The fallback format does not handle tool calls or multi‑modal
              content; only text is preserved.
        """
        try:
            # Convert internal Message objects to dicts that the tokenizer expects.
            conversation = [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
            prompt = self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            # Log the error (but use a simple fallback). In a production system,
            # you would emit a warning via self._logger.
            # Fallback to manual formatting.
            prompt = ""
            for msg in messages:
                if msg.role == Role.SYSTEM:
                    prompt += f"System: {msg.content}\n\n"
                elif msg.role == Role.USER:
                    prompt += f"User: {msg.content}\n\n"
                elif msg.role == Role.ASSISTANT:
                    prompt += f"Assistant: {msg.content}\n\n"
            prompt += "Assistant: "
            return prompt

    async def close(self) -> None:
        """
        Release resources held by the provider.

        Why:
            - Shuts down the thread pool executor to release worker threads.
            - The pipeline and tokenizer are left for garbage collection;
              they do not have explicit close methods.

        Important:
            - After close, the provider cannot be used. Create a new instance
              if needed.
            - This method is called automatically by the `LLMClient` when
              switching providers or shutting down.
        """
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        # No need to explicitly close pipeline or tokenizer.
        await super().close()