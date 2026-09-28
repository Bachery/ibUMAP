from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


def filtered_kwargs(callable_obj: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return kwargs accepted by callable_obj unless it accepts **kwargs."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def filtered_call(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call func with only the supported keyword arguments."""
    return func(**filtered_kwargs(func, kwargs))

