from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetDefinition:
    """Immutable description of a machine-learning prediction target.

    Attributes:
        name: Canonical name for this target (e.g. "forward_return").
            Leading/trailing whitespace is stripped; the result must be non-empty.
        horizon_bars: Number of trading bars into the future over which the
            target is realised.  Must be a positive integer; booleans are rejected
            because ``bool`` is a subclass of ``int`` in Python.
        description: Human-readable description of the target. Defaults to "".
        parameters: Optional key/value pairs for additional target configuration.
            Defaults to an empty dictionary.
    """

    name: str
    horizon_bars: int
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # --- name -----------------------------------------------------------
        if not isinstance(self.name, str):
            raise TypeError(f"TargetDefinition.name must be a str, got {type(self.name).__name__}")
        # Strip whitespace on the frozen dataclass using object.__setattr__
        object.__setattr__(self, "name", self.name.strip())
        if not self.name:
            raise ValueError("TargetDefinition.name cannot be empty or whitespace-only")

        # --- horizon_bars ---------------------------------------------------
        # Reject bool first (bool is a subclass of int in Python)
        if isinstance(self.horizon_bars, bool):
            raise TypeError("TargetDefinition.horizon_bars must be an integer, got bool")
        if not isinstance(self.horizon_bars, int):
            raise TypeError(
                f"TargetDefinition.horizon_bars must be an integer, "
                f"got {type(self.horizon_bars).__name__}"
            )
        if self.horizon_bars <= 0:
            raise ValueError(
                f"TargetDefinition.horizon_bars must be a positive integer, got {self.horizon_bars}"
            )

        # --- parameters -----------------------------------------------------
        if not isinstance(self.parameters, dict):
            raise TypeError(
                f"TargetDefinition.parameters must be a dict, got {type(self.parameters).__name__}"
            )
        # Defensive copy so callers cannot mutate the frozen dataclass's dict
        object.__setattr__(self, "parameters", dict(self.parameters))
