"""Unified exception hierarchy for the aria model quantization toolkit.

No silent failures: every except must explicitly convert or re-raise.
"""


class ModelError(Exception):
    """Base exception for the model quantization toolkit."""


class QuantError(ModelError):
    """Quantization failure: invalid bit width, packing failure, non-finite weights."""


class FormatError(ModelError):
    """Corrupted bundle / missing file / bad offsets / format mismatch."""


class ShapeMismatchError(ModelError):
    """Tensor shape vs data length mismatch."""


class UnsupportedError(ModelError):
    """Unimplemented operation."""


class ModelFetchError(ModelError):
    """HuggingFace fetch failure (network / auth / missing) or missing deps."""

    def __init__(self, repo: str, reason: str, kind: str | None = None):
        self.repo = repo
        self.kind = kind
        super().__init__(f"failed to fetch model '{repo}': {reason}")


class ConfigError(ModelError):
    """Config missing required fields or inconsistent values."""
