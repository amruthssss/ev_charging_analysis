"""Model utilities for the EV charging analytics project."""

from importlib import import_module
from typing import Any

__all__ = [
	"inspect_daily_sessions",
	"load_daily_sessions",
]


def __getattr__(name: str) -> Any:
	"""Lazily import model helpers to avoid premature module loading."""
	if name in __all__:
		module = import_module("src.phase5_models.demand_forecasting")
		value = getattr(module, name)
		globals()[name] = value  # cache attribute for future access
		return value
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
