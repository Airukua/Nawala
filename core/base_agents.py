from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class AgentContext:
    """
    Mutable data container that flows through a chain of agents in a pipeline.

    Why this exists:
    - Centralises all information an agent might need: original query, extracted
      anchors/keywords/entities, retrieved memories, graph structures, and metadata.
    - Allows agents to read from and write to a shared context without tight coupling
      to each other's signatures.
    - Simplifies adding new fields (e.g., `temporal_data`, `topic_graph`) without
      refactoring every agent.

    Invariants:
    - `query` is immutable from the agent's perspective (once set, should not be
      modified by agents – it represents the original user input).
    - `anchor`, `keywords`, `entities` are order‑preserving lists; agents may append
      or reorder them, but should not treat them as sets (duplicates may exist).
    - `reranked_memories` is a list of dicts, each dict must contain at least a
      `score` and `content` key (or whatever the retrieval pipeline guarantees).
      The order is from highest to lowest relevance after reranking.
    - Graph fields (`short_term_graph`, `long_term_tree`, `temporal_data`, `topic_graph`)
      are opaque; agents that need them must know the expected type (e.g., a custom
      graph class). `None` means the data is not available or not yet computed.
    - `summary` is a free‑form textual summarisation of the context; may be updated
      incrementally.
    - `metadata` is a catch‑all for any extra data (e.g., timestamps, user IDs,
      conversation history). Keys are strings, values are arbitrary.

    Edge Cases / Risks:
    - Large lists (e.g., thousands of `reranked_memories`) can cause high memory
      usage and slow down serialisation. Agents should consider truncating early.
    - Graph fields are weakly typed – an agent expecting a `networkx.Graph` may
      crash if another agent stored a different object. Define clear contracts
      in your pipeline documentation.
    - The context is mutable; sharing the same instance across multiple agents
      means side effects are visible downstream. This is intentional but can
      make debugging harder if an agent corrupts a field.
    - Not thread‑safe. If you run agents concurrently (e.g., in a parallel pipeline),
      you must deep‑copy the context per thread or implement locking.

    Performance:
    - Using `dataclass` with `field(default_factory=list)` ensures each instance
      gets its own list copies, avoiding accidental sharing of mutable defaults.
    - Assigning large objects to fields (e.g., a 100MB graph) does not copy them;
      only the reference is stored. This is efficient but can lead to unintended
      sharing across contexts if you're not careful.
    """
    query: str = ""
    anchor: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    reranked_memories: List[Dict] = field(default_factory=list)
    short_term_graph: Optional[Any] = None
    long_term_tree: Optional[Any] = None
    temporal_data: Optional[Any] = None
    topic_graph: Optional[Any] = None
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in a multi‑agent memory system.

    Design philosophy:
    - Each agent performs a specific, composable transformation on an `AgentContext`.
    - Agents are strung together into a pipeline (e.g., retrieval → reranking → fusion).
    - The base class provides common lifecycle fields: status, error tracking, and metrics.
    - Subclasses must implement `process`, `get_name`, and `get_capabilities`.

    Why an abstract class instead of a protocol?
    - We need to enforce a common interface for the pipeline orchestrator.
    - Shared functionality (`update_status`, `learn`) reduces boilerplate.
    - `memory_lake` provides a centralised, cross‑agent persistent storage
      (e.g., for saving/loading memories or sharing intermediate results).

    Lifecycle & State:
    - An agent is instantiated once with an `agent_id` and `config`.
    - `status` can be "idle", "running", "completed", "failed". The orchestrator
      may use it for monitoring.
    - `last_error` holds the most recent exception message (if any). Not cleared
      automatically – the orchestrator should reset it if needed.
    - `metrics` is a dict for performance counters (e.g., latency_ms, tokens_processed).
      Subclasses should update it during `process`.

    Invariants:
    - `agent_id` must be unique within a pipeline (the orchestrator enforces this).
    - `config` is read‑only after initialisation; subclasses may validate required keys
      in their `__init__`.
    - `memory_lake` can be `None` if the agent doesn't need persistent storage.
    - `logger` is initially `None` – the orchestrator should inject a logger (e.g., via
      a `set_logger` method) before the pipeline runs. Subclasses should check
      `self.logger is not None` before logging.

    Edge Cases / Risks:
    - `process` is called exactly once per agent per pipeline run. It may mutate the
      context in place. If an exception is raised, the orchestrator should catch it,
      set `status = "failed"`, and decide whether to continue or abort.
    - `learn` is called asynchronously after pipeline execution, with feedback signals
      (e.g., user clicks, relevance scores). It is not required to be thread‑safe;
      the orchestrator should call it from a single thread.
    - `get_capabilities` returns a static list of strings (e.g., ["retrieval", "rerank"]).
      This allows dynamic agent selection or pipeline validation.
    - The base class does not implement `__repr__`; subclasses may add it for debugging.

    Performance:
    - `process` should be as efficient as possible – it may be on the critical path
      of a user request. Avoid heavy I/O inside `process` unless it's necessary (e.g.,
      database lookups). Consider caching in `memory_lake`.
    - `learn` can be heavier (e.g., updating model weights) because it runs offline.
    - Metrics collection using `self.metrics` adds minimal overhead; use `time.perf_counter()`
      for high‑resolution timing.

    External Dependencies:
    - `memory_lake` is an abstract interface (e.g., a key‑value store). The base class
      does not define it; the pipeline injects a concrete implementation.
    - Subclasses may rely on external services (APIs, databases). Configuration should
      be passed via `config` (e.g., API keys, endpoints).
    """
    def __init__(self, agent_id: str, config: Dict[str, Any], memory_lake=None):
        """
        Initialise the agent with a unique ID, configuration, and optional storage.

        Args:
            agent_id: Unique identifier within the pipeline. Used for logging and metrics.
            config: A dict that may contain model paths, thresholds, API keys, etc.
            memory_lake: A shared object that implements `get(key)` and `set(key, value)`
                         (or a more sophisticated interface). May be `None`.

        Side Effects:
            - Sets initial `status = None` (orchestrator should set to "idle" later).
            - Does not validate `config`; subclasses should override `__init__` to do so.
        """
        self.agent_id = agent_id
        self.config = config
        self.memory_lake = memory_lake
        self.status = None
        self.logger = None  
        self.last_error: Optional[str] = None
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def process(self, context: AgentContext) -> AgentContext:
        """
        Execute the agent's core logic on the given context.

        Args:
            context: The mutable context object. The agent may read from and write to
                     any of its fields. It must not replace the context instance
                     (i.e., `id(context)` should stay the same), but it may modify
                     fields in place.

        Returns:
            The same (mutated) `AgentContext` instance. Returning a different instance
            breaks the pipeline contract – the orchestrator expects the original object.

        Raises:
            Any exception (e.g., `ValueError`, `RuntimeError`). The orchestrator should
            catch it, set `self.status = "failed"`, and propagate or log accordingly.

        Design notes:
            - The method is intentionally not async. For I/O‑bound agents, the caller
              (orchestrator) can run them in a thread pool.
            - Do not store the context beyond the lifetime of this call (no caching)
              unless you manage thread safety explicitly.
            - Update `self.metrics` with timing, counts, etc. before returning.
            - Use `self.logger` for structured logging if available.

        Example:
            >>> ctx = AgentContext(query="What is RAG?")
            >>> agent.process(ctx)
            >>> print(ctx.keywords)   # e.g., ["RAG", "retrieval", "generation"]
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return a human‑readable name for this agent.

        Why separate from `agent_id`?
            - `agent_id` is for programmatic uniqueness (e.g., "retriever_01").
            - `get_name()` is for UI or logs (e.g., "BM25 Retriever").

        Returns:
            A short string, e.g., "QueryExpander", "MemoryReranker".

        Note:
            This method must be idempotent and cheap (no I/O). The orchestrator may
            call it multiple times.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        List the capabilities this agent provides, used for pipeline orchestration.

        Returns:
            A list of strings, e.g., ["retrieval", "reranking", "fusion", "query_expansion"].

        Invariants:
            - The list should be static (same for all instances of a class).
            - The orchestrator may use these strings to decide whether to include
              the agent in a given pipeline, or to validate dependencies.

        Example:
            >>> agent.get_capabilities()
            ['retrieval', 'sparse_vector']

        Note:
            This method is not expected to change at runtime. Subclasses may implement
            it as a classmethod returning a constant list.
        """
        pass

    def update_status(self, status: str, error_msg: Optional[str] = None):
        """
        Update the agent's runtime status and optionally record an error.

        Args:
            status: One of a predetermined set of strings, e.g., "idle", "running",
                    "completed", "failed". The base class does not validate this.
            error_msg: If provided, sets `self.last_error` to this message. Otherwise,
                       `last_error` is left unchanged (caller may set it to None to clear).

        Side Effects:
            - Overwrites `self.status`.
            - If `error_msg` is not None, `self.last_error` is set; otherwise it stays
              as is (including possibly a previous error). To clear errors, call
              `update_status("idle", error_msg=None)` after explicitly setting
              `self.last_error = None`.

        Design rationale:
            - Centralises status changes, allowing subclasses to add hooks later
              (e.g., emit metrics when status becomes "failed").
            - Not abstract because most agents can use this default implementation.
        """
        self.status = status
        if error_msg:
            self.last_error = error_msg

    def learn(self, feedback: Dict[str, float]):
        """
        Incorporate feedback to improve future agent behaviour.

        When is this called?
            - After a pipeline completes, the orchestrator may invoke `learn` on each
              agent with feedback signals (e.g., user rating, clickthrough data).
            - This method is **not** called during normal inference – only during
              offline learning or online adaptation.

        Args:
            feedback: A dictionary mapping signal names to numeric values.
                      Example: {"relevance": 0.8, "clicked": 1.0, "latency_penalty": -0.1}.

        Default implementation:
            Does nothing. Subclasses that need learning (e.g., bandit algorithms,
            online gradient descent) must override this.

        Design considerations:
            - The method is synchronous. Long‑running learning (e.g., retraining a
              model) should be offloaded to a background task; `learn` could enqueue it.
            - The agent should use its `config` to determine learning rates, bounds, etc.
            - The feedback dictionary may be sparse; agents should ignore unknown keys.
            - Because `learn` may be called concurrently with `process` (if the
              orchestrator uses threading), implement thread‑safe state updates.
        """
        pass