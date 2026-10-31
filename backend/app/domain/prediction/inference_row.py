from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

_RESERVED_ROW_KEYS: Final[frozenset[str]] = frozenset({"instrument_id", "observation_date"})


@dataclass(frozen=True)
class InferenceRow:
    """Point-in-time unlabeled observation row for model inference.

    Contains exclusively historical feature values computed as of observation_date.
    Explicitly omits target_value and realization_date to prevent look-ahead bias
    and support prediction on recent market bars where future outcomes are unrealized.
    """

    instrument_id: UUID
    observation_date: date
    feature_values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"InferenceRow.instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if isinstance(self.observation_date, datetime):
            raise TypeError("InferenceRow.observation_date must be a date, not a datetime")
        if not isinstance(self.observation_date, date):
            raise TypeError(
                f"InferenceRow.observation_date must be a date instance, got {type(self.observation_date).__name__}"
            )

        if not isinstance(self.feature_values, (tuple, list)):
            raise TypeError(
                f"InferenceRow.feature_values must be a tuple or list of Decimal, got {type(self.feature_values).__name__}"
            )
        f_tuple = tuple(self.feature_values)
        for i, val in enumerate(f_tuple):
            if not isinstance(val, Decimal):
                raise TypeError(
                    f"InferenceRow.feature_values[{i}] must be a Decimal instance, got {type(val).__name__}"
                )
            if not val.is_finite():
                raise ValueError(
                    f"InferenceRow.feature_values[{i}] must be a finite Decimal, got {val}"
                )
        object.__setattr__(self, "feature_values", f_tuple)

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, Any]:
        """Convert row to a name-keyed dictionary for serialization or inspection."""
        if not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes)):
            raise TypeError("feature_names must be a sequence of str")

        if len(feature_names) != len(self.feature_values):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"feature_values length ({len(self.feature_values)})"
            )

        f_names_list: list[str] = []
        for i, name in enumerate(feature_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"feature_names[{i}] must be a non-empty string, got {name!r}")
            if name in _RESERVED_ROW_KEYS:
                raise ValueError(
                    f"feature_names[{i}] ('{name}') collides with reserved row attribute"
                )
            f_names_list.append(name)

        if len(set(f_names_list)) != len(f_names_list):
            raise ValueError(f"feature_names must be unique; found duplicates in {feature_names}")

        record: dict[str, Any] = {
            "instrument_id": self.instrument_id,
            "observation_date": self.observation_date,
        }
        for name, val in zip(f_names_list, self.feature_values, strict=True):
            record[name] = val
        return record
