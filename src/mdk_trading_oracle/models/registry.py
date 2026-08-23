"""Model Registry: Centralized management and discovery for Gold Layer predictive models."""

from typing import Callable, Dict, Type

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.models.base import BaseForecaster

logger = get_logger("mdk_oracle.models.registry")


class ModelRegistry:
    """Registry pattern allowing dynamic registration, listing, and instantiation of models.

    Supports scaling from Model 1 (Day-Start) to 10+ future forecasting models.
    """

    _registry: Dict[str, Type[BaseForecaster]] = {}

    @classmethod
    def register(cls, model_id: str) -> Callable:
        """Decorator to register a forecaster model class with a unique model identifier."""

        def decorator(subclass: Type[BaseForecaster]) -> Type[BaseForecaster]:
            if model_id in cls._registry:
                logger.warning(f"Model ID '{model_id}' is already registered. Overwriting with {subclass.__name__}.")
            cls._registry[model_id] = subclass
            logger.debug(f"Registered model '{model_id}' -> {subclass.__name__}")
            return subclass

        return decorator

    @classmethod
    def get(cls, model_id: str, *args, **kwargs) -> BaseForecaster:
        """Instantiate a registered model by its identifier."""
        if model_id not in cls._registry:
            raise KeyError(f"Model '{model_id}' not found in registry. Available models: {list(cls._registry.keys())}")
        return cls._registry[model_id](*args, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return list of all registered model identifiers."""
        return list(cls._registry.keys())
