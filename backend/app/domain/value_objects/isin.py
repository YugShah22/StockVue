import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ISIN:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().upper()

        if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", normalized):
            raise ValueError(f"Invalid ISIN format: {self.value!r}")

        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value
