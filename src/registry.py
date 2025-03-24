from typing import Dict, Type, Any
import importlib
from src.interfaces import *


class ComponentRegistry:
    _instance = None
    _components: Dict[str, Type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, component_type: str, component_class: Type):
        cls._components[component_type] = component_class

    @classmethod
    def create(cls, component_type: str, config: dict) -> Any:
        if component_type not in cls._components:
            raise ValueError(f"Unknown component type: {component_type}")
        return cls._components[component_type](config)

    @classmethod
    def load_from_config(cls, config: dict) -> Dict[str, Any]:
        components = {}
        for name, spec in config.items():
            if "class_path" not in spec:
                raise ValueError(f"Component {name} missing 'class_path'")

            module_path, class_name = spec["class_path"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            cls.register(name, component_class)
            components[name] = cls.create(name, spec.get("params", {}))

        return components
