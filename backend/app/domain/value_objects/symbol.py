from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().upper()

        if not normalized:
            raise ValueError("Symbol cannot be empty.")

        object.__setattr__(self,"value",normalized)

    def __str__(self) -> str:
        return self.value
