import asyncio
import re
from typing import Dict, Any, Callable, Optional
from functools import wraps
from .logger import get_logger
logger = get_logger("Validator")


def validate_query(query: Any, max_length: int = 10000) -> str:
    """
    Sanitise and validate a user query string.

    Why this exists:
    - User‑supplied strings can be malicious (control characters, excessively long).
    - Downstream agents assume a clean, non‑empty string.
    - Centralising validation avoids repeating checks across agents.

    Validation steps (in order):
        1. Reject `None` values.
        2. Ensure type is `str` (no implicit conversion).
        3. Strip leading/trailing whitespace – whitespace‑only queries are rejected.
        4. Enforce length limit (default 10k characters) to prevent memory/DoS attacks.
        5. Reject ASCII control characters (0x00‑0x08, 0x0B, 0x0C, 0x0E‑0x1F, 0x7F) that
           could break log formatting or terminal output.

    Args:
        query: Raw user input (any type, but should be string).
        max_length: Maximum allowed length after stripping. Default 10000.

    Returns:
        Stripped, validated query string.

    Raises:
        ValueError: If any validation fails, with a descriptive message.

    Invariants:
        - Returned string is non‑empty, stripped, and contains no control characters.
        - `len(returned) <= max_length`.

    Edge Cases & Risks:
        - Unicode characters (e.g., emoji, Chinese) are allowed and not checked for
          length in bytes – `len()` counts Unicode code points. This is acceptable
          because 10k code points is still a reasonable limit. For strict byte‑size
          control, use `len(query.encode('utf-8'))`.
        - Control character detection uses regex `[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]`.
          This does not catch zero‑width joiners or other Unicode control characters
          (e.g., U+200B). If needed, extend with `re.UNICODE` and a broader set.
        - The function does **not** escape or sanitise HTML/SQL – that's the
          responsibility of the components that output the query elsewhere.

    Performance:
        - O(n) where n = len(query). For max 10k, this is negligible (< 0.1ms).
        - Regex compilation happens once per call (not cached). For high‑throughput
          validation, consider compiling the pattern as a module constant.
    """
    if query is None:
        raise ValueError("Query tidak boleh None")
    if not isinstance(query, str):
        raise ValueError(f"Query harus string, bukan {type(query).__name__}")
    query = query.strip()
    if len(query) == 0:
        raise ValueError("Query tidak boleh kosong")
    if len(query) > max_length:
        raise ValueError(f"Query terlalu panjang: {len(query)} > {max_length} karakter")
    # Cek control characters (non-printable)
    if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', query):
        raise ValueError("Query mengandung karakter kontrol yang tidak valid")
    return query


def validate_context(context: Dict, required_keys: Optional[list] = None) -> Dict:
    """
    Validate that a context object is a dictionary and contains required keys.

    Why this exists:
    - Many agents expect a `dict`‑shaped context (e.g., for LLM input or memory).
    - Pipeline orchestration may require certain keys to be present before an agent runs.
    - Early validation prevents obscure `KeyError` failures deep inside agent logic.

    Design philosophy:
    - Very lightweight – only type checking and key presence. No deep validation
      of values (e.g., types of values under those keys). That is left to agents.
    - The input `context` is returned unchanged (not copied). This avoids unnecessary
      memory overhead for large contexts.

    Args:
        context: Any object (expected to be a dict).
        required_keys: Optional list of keys that must exist in the dict.
                       If `None`, only type checking is performed.

    Returns:
        The same `context` dictionary (for chaining or inline use).

    Raises:
        ValueError: If `context` is not a dict, or if any required key is missing.

    Invariants:
        - The returned dictionary is exactly the same object as the input (no copy).
        - If `required_keys` is given, all those keys are present in the returned dict.

    Edge Cases & Risks:
        - This function does **not** check that keys map to non‑`None` values.
          A required key with value `None` will pass validation, which may cause
          errors later. Add a separate `validate_context_values` if needed.
        - The `required_keys` list is used as‑is; if it contains duplicates or
          non‑string keys, the behaviour is still correct (duplicates are redundant,
          non‑string keys will never match dict keys). To be strict, callers should
          ensure `required_keys` contains only strings.
        - No recursion – nested dictionaries inside `context` are not validated.

    Performance:
        - O(k) where k = len(required_keys) (typical <10). Negligible overhead.
    """
    if not isinstance(context, dict):
        raise ValueError(f"Context harus dict, bukan {type(context).__name__}")
    if required_keys:
        missing = [k for k in required_keys if k not in context]
        if missing:
            raise ValueError(f"Missing required keys in context: {missing}")
    return context


def validate_agent_function(func: Callable) -> bool:
    """
    Ensure an agent function has the correct async signature: `async def (context, llm)`.

    Why this exists:
    - The agent orchestration engine expects a uniform interface for all agents.
    - Mistyped signatures (e.g., missing async, wrong number of parameters) would
      cause runtime crashes that are hard to debug.
    - Validating at registration time gives early feedback to developers.

    Signature requirement:
        - Must be a coroutine function (`async def`).
        - Must accept exactly two positional parameters.
        - Parameter names are irrelevant; only count matters.

    Args:
        func: The function object to validate (typically an agent's `process` method).

    Returns:
        True if validation passes (function is suitable as an agent).

    Raises:
        TypeError: With a descriptive message if the function does not meet requirements.

    Invariants:
        - If this function returns `True`, the function can be safely called as
          `await func(context, llm)`.

    Design notes:
        - Uses `inspect.signature` to examine parameters. Does **not** validate
          default values or type annotations – those are optional.
        - The function may be a method bound to an instance? Typically agents are
          free functions or static methods. The signature check works regardless
          because bound methods still have `self` as the first parameter. That would
          be counted as one of the two required parameters, which would break the
          intended signature (self, context, llm). Therefore this validator assumes
          the function is not a bound method. In practice, agents are registered as
          module‑level functions or class methods decorated with `@staticmethod`.

    Edge Cases & Risks:
        - Does **not** detect if the function uses `*args` or `**kwargs` to accept
          variable arguments. Such functions would pass the parameter count check
          (since they have 0 or 1 explicit parameters) but would actually be callable.
          The validator conservatively rejects them because the orchestration code
          does not expect to handle `*args`. If needed, enhance to allow functions
          with `*args` as long as they can accept two arguments.
        - Coroutine detection uses `asyncio.iscoroutinefunction`. This returns
          `False` for generators (`async def` with `yield`) – they are not callable
          in the same way and are correctly rejected.

    Performance:
        - Uses `inspect.signature` which parses the function's bytecode. This is
          moderately expensive (microseconds). Called only once per agent registration,
          so acceptable.
    """
    import inspect
    if not asyncio.iscoroutinefunction(func):
        raise TypeError(f"Agent function harus async, bukan {type(func).__name__}")
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if len(params) != 2:
        raise TypeError(f"Agent function harus menerima 2 parameter (context, llm), mendapat {len(params)}")
    return True

def validate_interval(interval: float) -> float:
    """
    Validate a loop interval (in seconds) for periodic agent execution.

    Why this exists:
        - Continuous loops (e.g., polling, background refresh) need a sane interval.
        - Zero or negative values would cause busy‑waiting (100% CPU) or weird behaviour.
        - Extremely large intervals (>24h) might be unintentional and should warn.

    Args:
        interval: The proposed interval in seconds (float or int).

    Returns:
        The same interval (unchanged) if valid.

    Raises:
        ValueError: If `interval <= 0`.

    Side Effects:
        - Logs a warning if `interval > 86400` (one day).

    Invariants:
        - Returned value is > 0.

    Edge Cases & Risks:
        - No upper bound enforced except a warning. The caller is free to use
          very large intervals (e.g., weekly run) – that's acceptable.
        - The function does not check for `NaN` or `inf` because `interval <= 0`
          will raise `ValueError` for those (since `NaN <= 0` is false and `inf > 0`
          is true, but `inf` is not a useful interval). It may be worth adding
          `if not math.isfinite(interval): raise ValueError(...)`.

    Performance:
        - Trivial O(1).
    """
    if interval <= 0:
        raise ValueError(f"Interval harus > 0, mendapat {interval}")
    if interval > 86400:  # > 24 jam
        logger.warning(f"Interval sangat besar ({interval} detik), mungkin tidak diinginkan")
    return interval

def validate_max_iterations(max_iterations: Optional[int]) -> Optional[int]:
    """
    Validate the maximum number of iterations for a loop (e.g., agent pipeline steps).

    Why this exists:
        - Prevents infinite loops in cases where termination conditions may never be met.
        - Ensures the value is a positive integer (or `None` for no limit).

    Args:
        max_iterations: Either `None` (meaning no limit) or a positive integer.

    Returns:
        The same value (unchanged) if valid.

    Raises:
        ValueError: If `max_iterations` is not `None` and not a positive integer.

    Invariants:
        - If return value is not `None`, it is an `int > 0`.

    Edge Cases & Risks:
        - Does **not** check for `float` values that are whole numbers (e.g., `3.0`).
          It raises `ValueError` because they are not `int`. This is intentional –
          using a float for iterations is a likely bug. If you need to accept `3.0`,
          convert to `int` before calling.
        - `max_iterations = 0` is rejected; zero iterations would mean no execution.
          For a loop that should run at least once, the caller should handle that logic.

    Performance:
        - Trivial O(1).
    """
    if max_iterations is not None:
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError(f"max_iterations harus integer positif, mendapat {max_iterations}")
    return max_iterations

async def llm_with_timeout(llm_func: Callable, prompt: str, timeout_seconds: float = 30.0,
                           retries: int = 1) -> str:
    """
    Call an LLM function with timeout and automatic retries.

    Why this exists:
        - LLM calls are inherently unreliable: network timeouts, rate limits, transient errors.
        - Without timeout, a hanging LLM could block an agent forever.
        - Retries improve robustness under temporary failures.
        - Centralising the pattern avoids repetitive code in every agent.

    How it works:
        1. For `retries+1` attempts:
            a. Call `llm_func(prompt)` wrapped with `asyncio.wait_for` using `timeout_seconds`.
            b. If the call returns a non‑string result, convert it to string.
            c. If successful, return the string.
            d. On `asyncio.TimeoutError`: log and retry (up to `retries` times).
            e. On any other `Exception`: log and retry.
        2. If all attempts fail, raise the last exception (or a `TimeoutError` if only timeouts occurred).

    Args:
        llm_func: An async callable that takes a `prompt` (string) and returns
                  an awaitable that yields a string (or convertible to string).
        prompt: The input string to pass to the LLM.
        timeout_seconds: Maximum time to wait for a single attempt (float).
        retries: Number of **additional** attempts after the first. Total attempts = `retries + 1`.

    Returns:
        The LLM's response as a string.

    Raises:
        TimeoutError: If all attempts timed out (and no other exception occurred).
        Exception: Any non‑timeout exception raised by `llm_func` that persists after retries.

    Invariants:
        - The function always returns a string (no `None`).
        - Exactly `retries+1` attempts are made, unless a non‑timeout exception is raised
          on the first attempt? Actually the code catches all exceptions and retries.
          So it will retry even on non‑timeout errors. That may be undesirable for
          certain errors (e.g., `ValueError` from malformed prompt) because retrying
          will not help. The design prioritises robustness over efficiency.

    Edge Cases & Risks:
        - If `llm_func` raises a `ValueError` due to an invalid prompt, retrying is
          useless and wastes time. Consider distinguishing between retryable errors
          (network, rate‑limit) and fatal errors (validation). The current code does
          not do this – it retries everything.
        - The conversion `if not isinstance(result, str): result = str(result)` is
          lossy for complex objects (e.g., dicts become `"{...}"`). Callers should
          ensure their LLM function returns a string.
        - `timeout_seconds` is applied per attempt, not total. If timeout=30s and
          retries=3, the total maximum wait is 120s. This is usually fine.
        - The function does **not** add exponential backoff between retries. This can
          exacerbate rate‑limit issues. For production, implement backoff.
        - Logging uses `logger.error` for each attempt failure. This can flood logs
          if many retries happen often. Consider logging at `warning` for first few attempts.

    Performance:
        - Each attempt may block for up to `timeout_seconds`. Use reasonable timeouts.
        - No overhead besides the async wait wrapper.
    """
    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                llm_func(prompt),
                timeout=timeout_seconds
            )
            if not isinstance(result, str):
                result = str(result)
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM timeout after {timeout_seconds}s (attempt {attempt+1})")
            if attempt == retries:
                raise TimeoutError(f"LLM tidak merespon setelah {retries+1} percobaan")
        except Exception as e:
            logger.error(f"LLM error: {e} (attempt {attempt+1})")
            if attempt == retries:
                raise
    raise RuntimeError("Unreachable")

def agent_error_handler(agent_name: str, fallback_context: Optional[Dict] = None):
    """
    Decorator that catches exceptions in an agent function and returns a fallback context.

    Why this exists:
        - In a pipeline of agents, a single failing agent should not crash the whole process.
        - The orchestrator can continue with a degraded context (e.g., marking the error).
        - Centralises error logging and context repair, avoiding try/except in every agent.

    How it works:
        - Wraps an async agent function `func(context, llm_func)`.
        - If `func` raises any exception:
            - Logs an error with agent name and exception details.
            - Creates a new context by merging the original context with `fallback_context`
              (if provided) and adds a special key `"_error"` containing the exception string.
        - The fallback context is shallow‑merged (using `{**context, **fallback_context}`).
          This overwrites top‑level keys from `fallback_context`.

    Args:
        agent_name: Human‑readable name for logging (e.g., "RetrieverAgent").
        fallback_context: Optional dict with default values to use when the agent fails.
                          Keys in this dict will override keys in the original context.

    Returns:
        A decorator that transforms the agent function into a resilient version.

    Design rationale:
        - The error is injected into the context under `"_error"` so that downstream
          agents (or the orchestrator) can detect that something went wrong.
        - The fallback context allows the system to provide sensible defaults
          (e.g., empty list for `retrieved_docs`) instead of propagating `None`.
        - The original context is never mutated; a new dict is returned. This avoids
          side effects that might surprise other agents.

    Invariants:
        - The wrapped function always returns a dict (never raises).
        - The returned dict always contains the original context's keys plus any
          keys from `fallback_context` and the `"_error"` key if an error occurred.
        - If no error, the returned dict is exactly the return value of the original
          function (no `_error` key added).

    Edge Cases & Risks:
        - Shallow merge: If the context contains nested dicts, `fallback_context` will
          not recursively merge; it will replace entire top‑level keys. This is simple
          and sufficient for most pipelines. For deep merging, use a utility like
          `deepmerge`.
        - The `_error` key may already exist in the context from previous agents.
          The decorator overwrites it only on failure. If you need a chain of errors,
          consider storing a list at `context['_errors']`.
        - The decorator does **not** handle the case where `func` returns `None` or
          a non‑dict. If that happens, the wrapper will still return `None`/non‑dict,
          which may break later validations. Agent functions should be defined to
          return a dict. Consider adding a check: `if result is not None and not isinstance(result, dict): ...`.
        - The decorator swallows `KeyboardInterrupt` and `SystemExit` because it
          catches `Exception`. To allow graceful shutdown, catch `BaseException` and
          re‑raise those two. For simplicity, this version does not; it's acceptable
          for a validation/error‑handling layer.

    Performance:
        - Minimal overhead: one function call and a try/except block.
        - Creating the fallback context dict `{**context, **fallback_context, "_error": str(e)}`
          copies all top‑level keys (O(n) where n = number of top‑level keys). This is
          acceptable because context size is typically small (<50 keys).
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(context: Dict, llm_func: Optional[Callable]) -> Dict:
            try:
                return await func(context, llm_func)
            except Exception as e:
                logger.error(f"Agent '{agent_name}' failed: {type(e).__name__}: {e}")
                if fallback_context is not None:
                    return {**context, **fallback_context, "_error": str(e)}
                return {**context, "_error": str(e)}
        return wrapper
    return decorator