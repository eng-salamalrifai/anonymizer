"""Registries + decorators: the heart of the "add one file" design.

A class registers itself by name when its module is imported. The pipeline
and the UI only look names up here, so they never import a concrete class.
"""
MODELS: dict = {}
HANDLERS: dict = {}     # key = file extension, e.g. ".txt"
STRATEGIES: dict = {}


def _add(registry: dict, key: str, cls):
    # Registering two different classes under one name is always a bug
    # (usually a copy-paste typo), so fail loudly instead of overwriting.
    if key in registry and registry[key] is not cls:
        raise ValueError(f"'{key}' is already registered by {registry[key].__name__}")
    registry[key] = cls


def register_model(name: str):
    def deco(cls):
        _add(MODELS, name, cls)
        return cls
    return deco


def register_handler(*extensions: str):
    def deco(cls):
        for ext in extensions:
            _add(HANDLERS, ext.lower(), cls)
        return cls
    return deco


def register_strategy(name: str):
    def deco(cls):
        _add(STRATEGIES, name, cls)
        return cls
    return deco
