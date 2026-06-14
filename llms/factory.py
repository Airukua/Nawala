from typing import Type, Dict
from .exceptions import ConfigurationError
from .base import BaseLLMProvider
from .models import ProviderConfig

class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Design pattern: Factory Method.
    Why it's used:
    - Centralizes provider creation logic.
    - Handles different configuration requirements per provider.
    - Makes adding new providers easy without modifying existing code.
    - Enables runtime provider selection.

    Based on the factory pattern used in llm-provider-abstraction and litellm.

    Key responsibilities:
        1. Maintain a registry of provider types to their corresponding classes.
        2. Instantiate the correct provider class based on a `ProviderConfig`.
        3. Allow dynamic registration of new providers at runtime.

    Why a class with class methods instead of a standalone function:
        - Provides a clear, extensible namespace for provider registration.
        - Allows subclasses (if needed) to override registration or creation logic.
        - The registry (`_providers`) is shared across all code, ensuring a single
          source of truth for available providers.

    Thread safety:
        - The `_providers` dict is mutable; registering a provider after the factory
          has been used in multiple threads may cause race conditions. In practice,
          registration is expected to happen once at application startup, before
          any concurrent requests. If dynamic runtime registration is required,
          consider adding a lock.

    Example usage:
        >>> # Register a provider (usually done once at startup).
        >>> LLMProviderFactory.register("openai", OpenAIProvider)
        >>>
        >>> # Create a config and instantiate the provider.
        >>> config = ProviderConfig(provider_type="openai", model="gpt-4", api_key="...")
        >>> provider = LLMProviderFactory.create(config)
        >>> response = await provider.generate(messages)
    """

    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    """
    Registry mapping provider_type strings to provider classes.

    Why a class variable:
        - Shared across all instances and callers – provides global access.
        - Registration and creation are stateless; no need to instantiate the factory.

    Invariants:
        - Keys are lowercased provider type strings (e.g., "openai", "anthropic").
        - Values are subclasses of `BaseLLMProvider`.
        - The dictionary may be empty initially; providers must be registered before
          `create()` is called for that type.
    """

    @classmethod
    def register(cls, provider_type: str, provider_class: Type[BaseLLMProvider]) -> None:
        """
        Register a provider class for a given type.

        Why this method:
            - Allows dynamic addition of providers without modifying the factory.
            - Third-party code can extend the system by registering their own providers.
            - Registration is idempotent: later registrations overwrite earlier ones
              for the same `provider_type`.

        Args:
            provider_type: String identifier (e.g., "openai", "anthropic", "ollama").
                           Case‑insensitive in lookups (stored in lowercase).
            provider_class: Provider class (must be a subclass of `BaseLLMProvider`).

        Raises:
            TypeError: If `provider_class` is not a subclass of `BaseLLMProvider`.
            (The current implementation does not perform this check – it's left to
             the caller. However, the method signature indicates the expected type.)

        Side Effects:
            - Adds or replaces an entry in `cls._providers`.

        Important:
            - Registering a provider after some code has already called `create()`
              for that type will affect subsequent calls, but not already created
              provider instances. That is usually fine.
            - Overwriting an existing provider type may break assumptions in other
              parts of the code that expect a particular implementation. Use with
              care, or restrict registration to application startup.

        Example:
            >>> LLMProviderFactory.register("my_provider", MyCustomProvider)
        """
        # Note: Consider adding runtime validation:
        # if not issubclass(provider_class, BaseLLMProvider):
        #     raise TypeError(f"{provider_class} must be a subclass of BaseLLMProvider")
        cls._providers[provider_type.lower()] = provider_class

    @classmethod
    def create(cls, config: ProviderConfig) -> BaseLLMProvider:
        """
        Create and return a provider instance based on configuration.

        Why this method:
            - Hides the conditional logic of instantiating the correct provider.
            - Ensures that the provider type string is normalized (lowercase) before
              registry lookup.
            - Provides a single point of creation, making it easy to add hooks
              (e.g., caching, logging) in the future.

        Args:
            config: Provider configuration containing at least `provider_type`
                    and any other required fields for that provider.

        Returns:
            An instance of the appropriate provider class (subclass of `BaseLLMProvider`).

        Raises:
            ConfigurationError: If `config.provider_type` is not registered.

        Important:
            - This method does not validate that `config` contains all necessary
              fields for the provider; that responsibility lies in the provider's
              `_validate_config()` method.
            - The returned provider instance is unstarted (no requests made yet).
              Callers must call `generate()` or `stream()` to use it.
            - Each call to `create()` returns a **new** provider instance. If you need
              a singleton, manage caching at the caller level.

        Thread safety:
            - Reading `cls._providers` is safe as long as no concurrent writes
              (registrations) happen during creation. In typical usage, registration
              is done at startup and creation happens later, so safe.
            - If registration can happen concurrently, add a lock around both the
              registration and creation methods.

        Example:
            >>> config = ProviderConfig(provider_type="openai", model="gpt-4", api_key="...")
            >>> provider = LLMProviderFactory.create(config)
        """
        provider_type = config.provider_type.lower()
        if provider_type not in cls._providers:
            raise ConfigurationError(
                f"Unknown provider type: {provider_type}. "
                f"Registered providers: {list(cls._providers.keys())}"
            )
        return cls._providers[provider_type](config)