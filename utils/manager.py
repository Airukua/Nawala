import asyncio
from typing import Dict, List, Callable, Awaitable, Optional, Union
from .logger import get_logger
from .validators import (
    validate_context,
    validate_agent_function,
    validate_interval,
    validate_max_iterations,
    agent_error_handler,
)

logger = get_logger("AgentManager")

class AgentManager:
    """
    Orchestrates execution of registered agents in single or looping modes.

    Why this exists:
    - Agents need to be composed into pipelines (sequential or parallel) without
      hardcoding orchestration logic inside each agent.
    - Supports both one‑shot execution (`run_single`) and continuous periodic
      execution (`run_loop_*`).
    - Provides background loops that run as separate asyncio tasks, allowing the
      main event loop to handle other requests concurrently.

    Design philosophy:
    - Agents are registered as async functions with signature `(context, llm_func) -> context`.
      This uniform interface enables mix‑and‑match composition.
    - The manager does not know agent internals; it only calls them with the current
      context and the shared LLM function (if any).
    - Context copying: for parallel execution, each agent gets a shallow copy of the
      base context to avoid cross‑agent interference. Sequential execution passes
      the same context object, allowing agents to mutate it progressively.
    - Validation is applied at registration (agent signature) and at runtime
      (context, interval, max_iterations) to catch errors early.

    Lifecycle:
    1. Instantiate with an optional LLM function that all agents can use.
    2. Register agents via `register_agent`.
    3. Run one or more pipelines using `run_single` (blocking) or start background
       loops with `start_loop_background` (non‑blocking).
    4. For background loops, call `stop_background()` to cleanly cancel tasks.

    State:
    - `agents`: dict mapping agent names to callables (wrapped with error handler).
    - `_background_tasks`: list of asyncio.Tasks for active background loops.
    - `_running`: boolean flag controlling loop termination.

    Invariants:
    - Registered agent names must be unique and not in the reserved set.
    - An agent function must accept exactly two arguments: `context: Dict` and
      `llm_func: Optional[Callable]`, and return an `Awaitable[Dict]`. This is
      enforced by `validate_agent_function` at registration time.
    - The manager never modifies the registered agent functions after wrapping.
    - For sequential runs, the same context dict is mutated along the chain.
    - For parallel runs, each agent receives a shallow copy of the base context.
      Deep copying is not performed – agents must not mutate nested structures
      if that would affect other parallel branches.

    Edge Cases & Risks:
    - If an agent raises an exception during execution:
        * Sequential run: the exception is caught by the error handler decorator,
          which adds an `_error` field to the returned context. The pipeline continues.
        * Parallel run: `asyncio.gather` with `return_exceptions=True` captures
          exceptions and returns error dicts in the result list.
    - Shallow copy in parallel mode: modifications to mutable nested objects in
      one agent's context affect other agents because they share references.
    - Background loops: only one background loop can run at a time. The manager does
      not automatically restart crashed loops.
    - If agents take longer than `interval_seconds`, the next iteration starts
      immediately after the previous finishes (no overlapping). This may cause drift.

    Performance:
    - Registration: O(1) plus signature inspection (moderate, one‑time cost).
    - Sequential: O(n) agent calls, each awaited.
    - Parallel: uses `asyncio.gather` – total time ≈ max(agent times).
    - Background loops: `asyncio.sleep` yields control, no busy‑wait.

    Example Usage:
        manager = AgentManager(llm_func=my_llm)
        manager.register_agent("retriever", retrieve_agent)
        manager.register_agent("ranker", rank_agent)

        final_context = await manager.run_single("sequential", ["retriever", "ranker"], {"query": "hello"})
        results = await manager.run_single("parallel", ["retriever", "ranker"], {"query": "hello"})

        await manager.start_loop_background("sequential", ["retriever"], {"query": "hello"}, interval=5.0)
        await manager.stop_background()
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        """
        Initialise the agent manager.

        Args:
            llm_func: Optional async callable that agents can use to invoke an LLM.
                      Should accept a prompt string and return a string (or awaitable).
                      Passed as second argument to every agent.

        Side Effects:
            - Initialises empty agent registry and background task list.
            - Sets `_running = False`.
        """
        self.llm_func = llm_func
        self.agents: Dict[str, Callable] = {}  # name -> async (context, llm) -> context
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

    def register_agent(self, name: str, func: Callable[[Dict, Optional[Callable]], Awaitable[Dict]]):
        """
        Register an agent function under a unique name.

        Why this method validates and wraps:
        - Prevents duplicate or reserved names, which would cause ambiguous pipelines.
        - Validates function signature early to avoid runtime crashes.
        - Wraps agent with error handler so that failures produce error context
          instead of crashing the entire pipeline.

        Args:
            name: Unique identifier. Must not be already registered and not in
                  reserved set `{"sequential", "parallel", "stop", "start", "run_agent"}`.
            func: Async function with signature `(context: Dict, llm_func: Optional[Callable]) -> Awaitable[Dict]`.

        Raises:
            ValueError: If name already exists, is reserved, or function signature is invalid.

        Side Effects:
            - Stores a wrapped version of `func` in `self.agents`.
            - Logs debug message.

        Implementation notes:
            - The error handler decorator catches any exception from the agent,
              logs it, and returns `{**context, "_error": str(e)}`. This ensures
              the manager always receives a dict, never an exception.
        """
        # Validate name
        if name in self.agents:
            raise ValueError(f"Agent dengan nama '{name}' sudah terdaftar")
        reserved = {"sequential", "parallel", "stop", "start", "run_agent"}
        if name in reserved:
            raise ValueError(f"'{name}' adalah reserved name, tidak boleh digunakan sebagai nama agen")

        # Validate function signature (must be async, 2 parameters)
        validate_agent_function(func)

        # Wrap with error handler to catch exceptions and add _error field
        wrapped_func = agent_error_handler(name)(func)

        self.agents[name] = wrapped_func
        logger.debug(f"Agent '{name}' terdaftar")

    async def run_single(self, mode: str, agent_names: List[str], context: Dict) -> Union[Dict, List[Dict]]:
        """
        Execute a list of agents exactly once, either sequentially or in parallel.

        Why this method:
            - Provides a simple, blocking API for request‑response pipelines.
            - Validates context type before execution.
            - Returns a single final context (sequential) or list of results (parallel).

        Args:
            mode: 'sequential' or 'parallel'.
            agent_names: List of registered agent names. Order matters for sequential,
                         ignored for parallel except for result ordering.
            context: Initial context dictionary (must be a dict).

        Returns:
            - Sequential: final context dict after all agents.
            - Parallel: list of results (each result is a dict or error dict).

        Raises:
            ValueError: If mode is invalid or context is not a dict.

        Edge Cases:
            - Missing agent names are skipped and logged.
            - Parallel mode with all invalid agents returns empty list.
            - Errors in agents are caught by the error handler; they appear as
              `_error` keys in the returned context (sequential) or as error dicts
              in the list (parallel). The method itself never raises agent exceptions.
        """
        # Validate context
        context = validate_context(context)  # raises if not dict

        if mode == 'sequential':
            return await self._run_sequential(agent_names, context)
        elif mode == 'parallel':
            return await self._run_parallel(agent_names, context)
        else:
            raise ValueError(f"Mode '{mode}' tidak dikenal. Gunakan 'sequential' atau 'parallel'.")

    async def _run_sequential(self, agent_names: List[str], initial_context: Dict) -> Dict:
        """
        Internal sequential runner.

        Design:
            - Copies initial context to avoid mutating caller's dict.
            - Agents receive the same context object, mutating it in place.
            - Missing agents are skipped (logged error).
            - No additional try/except because agents are wrapped with error handler.

        Args:
            agent_names: List of agent names to run in order.
            initial_context: Starting context (copied before first agent).

        Returns:
            Final context after all agents have executed.
        """
        context = initial_context.copy()
        for name in agent_names:
            if name not in self.agents:
                logger.error(f"Agent '{name}' tidak ditemukan, skip")
                continue
            logger.info(f"[Sequential] Running: {name}")
            context = await self.agents[name](context, self.llm_func)
        return context

    async def _run_parallel(self, agent_names: List[str], base_context: Dict) -> List[Dict]:
        """
        Internal parallel runner.

        Design:
            - Each agent receives a shallow copy of `base_context` to reduce interference.
            - Uses `asyncio.gather` with `return_exceptions=True` so one failure does not cancel others.
            - Exceptions are converted to error dicts in the output list.

        Why shallow copy:
            - Avoids serialisation overhead of deep copy.
            - Agents that only read nested structures work safely.
            - Agents that modify nested structures will affect other agents – this is a
              documented trade‑off. For full isolation, callers should deep copy manually.

        Args:
            agent_names: List of agent names to run concurrently.
            base_context: Context to copy for each agent.

        Returns:
            List of results in the same order as valid agent names. Each result is
            either a dict (agent output) or `{"error": str(e), "name": name}`.
        """
        tasks = []
        valid_names = []
        for name in agent_names:
            if name not in self.agents:
                logger.error(f"Agent '{name}' tidak ditemukan, skip")
                continue
            tasks.append(self.agents[name](base_context.copy(), self.llm_func))
            valid_names.append(name)
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for name, res in zip(valid_names, results):
            if isinstance(res, Exception):
                logger.error(f"Agent {name} error: {res}")
                outputs.append({"error": str(res), "name": name})
            else:
                outputs.append(res)
        return outputs

    async def run_loop_sequential(self, agent_names: List[str], initial_context: Dict,
                                  interval_seconds: float, max_iterations: Optional[int] = None):
        """
        Run a sequential pipeline repeatedly with a fixed interval between iterations.

        Why:
            - Useful for monitoring, polling, or periodic updates.
            - Each iteration starts after the previous iteration completes plus `interval_seconds`.
            - If agents take longer than the interval, the next iteration starts immediately
              (no overlap, but effective frequency increases).

        Args:
            agent_names: List of agents to run each iteration (in order).
            initial_context: Starting context. Mutated across iterations (final context
                             of iteration N becomes initial context of iteration N+1).
            interval_seconds: Sleep time between iterations. Must be >0.
            max_iterations: Optional limit. If None, runs until `self._running` becomes False.

        Returns:
            Final context after the last iteration (or when stopped).

        Important:
            - This method blocks the current coroutine until the loop finishes.
            - Use `start_loop_background` for non‑blocking execution.
            - The loop checks `self._running` before each iteration.
        """
        interval_seconds = validate_interval(interval_seconds)
        max_iterations = validate_max_iterations(max_iterations)
        initial_context = validate_context(initial_context)

        iteration = 0
        context = initial_context.copy()
        while self._running and (max_iterations is None or iteration < max_iterations):
            logger.info(f"[Loop Sequential] Iteration {iteration+1}")
            context = await self._run_sequential(agent_names, context)
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                await asyncio.sleep(interval_seconds)
        return context

    async def run_loop_parallel(self, agent_names: List[str], base_context: Dict,
                                interval_seconds: float, max_iterations: Optional[int] = None):
        """
        Run a parallel pipeline repeatedly with a fixed interval between iterations.

        How it differs from sequential loop:
            - Each iteration runs agents concurrently on **shallow copies** of `base_context`.
              The base context is not mutated across iterations.
            - The result of each iteration is a list of contexts (one per agent).
            - All iteration results are collected into a list of lists.

        Use cases:
            - Comparing outputs of read‑only agents over time.
            - Building a time series of agent results without cross‑iteration state.

        Args:
            agent_names: List of agents to run in parallel each iteration.
            base_context: Context to copy for each agent every iteration (unchanged across iterations).
            interval_seconds: Sleep between iterations.
            max_iterations: Optional limit.

        Returns:
            List of iteration results. Each element is a list of agent results
            (order matching valid agent names). Example:
                [
                    [context_agent1_iter1, context_agent2_iter1],
                    [context_agent1_iter2, context_agent2_iter2],
                ]
        """
        interval_seconds = validate_interval(interval_seconds)
        max_iterations = validate_max_iterations(max_iterations)
        base_context = validate_context(base_context)

        iteration = 0
        all_results = []
        while self._running and (max_iterations is None or iteration < max_iterations):
            logger.info(f"[Loop Parallel] Iteration {iteration+1}")
            results = await self._run_parallel(agent_names, base_context)
            all_results.append(results)
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                await asyncio.sleep(interval_seconds)
        return all_results

    async def start_loop_background(self, mode: str, agent_names: List[str], context: Dict,
                                    interval: float = 1.0, max_iterations: Optional[int] = None):
        """
        Start a background loop that runs independently of the caller.

        Why:
            - Allows the main event loop to continue processing other requests
              while the agent pipeline runs periodically.
            - Typical use: long‑lived monitoring agents that update shared state.

        How it works:
            - Creates an asyncio Task for either `run_loop_sequential` or `run_loop_parallel`.
            - Stores the task in `self._background_tasks`.
            - Sets `self._running = True`; the loop checks this flag each iteration.
            - Returns immediately (does not await the loop).

        Args:
            mode: 'sequential' or 'parallel'.
            agent_names: List of agent names.
            context: Initial context (sequential) or base context (parallel).
            interval: Sleep seconds between iterations. Must be >0.
            max_iterations: Optional iteration limit. If None, runs until stopped.

        Raises:
            ValueError: If mode is invalid.
            RuntimeWarning: If a background loop is already running (logged, not raised).

        Important:
            - Only one background loop can run at a time per manager instance.
            - To stop the loop, call `stop_background()`.
            - If the loop crashes (uncaught exception), the task completes and
              `self._running` may remain `True`. The manager does not auto‑restart.
        """
        if self._running:
            logger.warning("Manager already running a background loop. Stop it first.")
            return

        # Validate parameters
        interval = validate_interval(interval)
        max_iterations = validate_max_iterations(max_iterations)
        context = validate_context(context)

        self._running = True
        if mode == 'sequential':
            coro = self.run_loop_sequential(agent_names, context, interval, max_iterations)
        elif mode == 'parallel':
            coro = self.run_loop_parallel(agent_names, context, interval, max_iterations)
        else:
            raise ValueError(f"Mode '{mode}' tidak dikenal")

        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        logger.info(f"Background loop started with mode '{mode}', interval {interval}s")

    async def stop_background(self):
        """
        Stop all background loops and wait for them to finish.

        Why:
            - Clean shutdown is essential to avoid orphaned tasks that hold references.
            - Idempotent: calling when no loops are running does nothing.

        How it works:
            - Sets `self._running = False` – loops exit before the next iteration.
            - Cancels all background tasks (in case they are stuck in I/O or sleep).
            - Uses `asyncio.gather(..., return_exceptions=True)` to wait for termination,
              suppressing cancellation errors.
            - Clears the task list.

        Side Effects:
            - All background tasks are cancelled and removed.
            - `self._running` becomes False.
        """
        self._running = False
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        logger.info("All background loops stopped")

    def is_running(self) -> bool:
        """Return True if a background loop is currently active."""
        return self._running

    def list_agents(self) -> List[str]:
        """Return a list of registered agent names."""
        return list(self.agents.keys())

    async def run_agent(self, name: str, context: Dict) -> Dict:
        """
        Shortcut to run a single agent sequentially.

        Why:
            - Convenience for simple use cases where only one agent is needed.
            - Equivalent to `run_single("sequential", [name], context)`.

        Args:
            name: Registered agent name.
            context: Initial context.

        Returns:
            Final context after the agent runs.
        """
        return await self.run_single("sequential", [name], context)