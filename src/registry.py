from typing import Any, Dict

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str):
        def decorator(obj: Any):
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in {self.name}")
            self._registry[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

# Global registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
PROMPTS = Registry("prompts")
PARSERS = Registry("parsers")
METHODS = Registry("methods")
