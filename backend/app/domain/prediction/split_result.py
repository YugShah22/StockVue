"""Immutable result type for a leakage-safe dataset split (Phase 5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.domain.prediction.dataset import DatasetRow
from app.domain.prediction.split_config import SplitConfig


@dataclass(frozen=True)
class SplitBoundary:
    """Records the chronological boundary between two consecutive splits.

    Attributes
    ----------
    last_train_observation_date:
        The `observation_date` of the last row *kept* in the training split
        (after purging of the validation split).
    first_val_observation_date:
        The `observation_date` of the first row *kept* in the validation split
        (after purging and embargo).  `None` when val split is empty.
    last_val_observation_date:
        The `observation_date` of the last row *kept* in the validation split.
        `None` when val split is empty.
    first_test_observation_date:
        The `observation_date` of the first row *kept* in the test split
        (after purging and embargo).  `None` when test split is empty.
    """

    last_train_observation_date: date
    first_val_observation_date: date | None
    last_val_observation_date: date | None
    first_test_observation_date: date | None


@dataclass(frozen=True)
class SplitResult:
    """Immutable, leakage-safe result of a chronological dataset split.

    Contains the three data partitions together with full audit metadata so
    that callers can trace exactly which rows were kept, purged, and embargoed.

    Attributes
    ----------
    train_rows:
        Tuple of :class:DatasetRow objects in the training partition, in
        chronological order.
    val_rows:
        Tuple of :class:DatasetRow objects in the validation partition, in
        chronological order (after purging and embargo).
    test_rows:
        Tuple of :class:DatasetRow objects in the test partition, in
        chronological order (after purging and embargo).
    config:
        The :class:SplitConfig that produced this split.
    boundary:
        Chronological boundary metadata for auditing.
    val_rows_purged:
        Number of rows removed from the head of the raw validation segment due
        to target-interval overlap with the training split.  Zero when
        `config.purge_overlap` is `False`.
    val_rows_embargoed:
        Number of rows removed from the head of the validation segment after
        purging to enforce the embargo period.  Zero when
        `config.embargo_bars` is `0`.
    test_rows_purged:
        Number of rows removed from the head of the raw test segment due to
        target-interval overlap with the validation split.  Zero when
        `config.purge_overlap` is `False`.
    test_rows_embargoed:
        Number of rows removed from the head of the test segment after purging
        to enforce the embargo period.  Zero when `config.embargo_bars` is
        `0`.
    total_rows:
        Total number of rows in the source :class:FeatureMatrix before
        splitting.
    """

    train_rows: tuple[DatasetRow, ...]
    val_rows: tuple[DatasetRow, ...]
    test_rows: tuple[DatasetRow, ...]
    config: SplitConfig
    boundary: SplitBoundary
    val_rows_purged: int
    val_rows_embargoed: int
    test_rows_purged: int
    test_rows_embargoed: int
    total_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.config, SplitConfig):
            raise TypeError(
                f"SplitResult.config must be a SplitConfig, got {type(self.config).__name__}"
            )
        if not isinstance(self.boundary, SplitBoundary):
            raise TypeError(
                f"SplitResult.boundary must be a SplitBoundary, got {type(self.boundary).__name__}"
            )
        for attr_name in ("train_rows", "val_rows", "test_rows"):
            val = getattr(self, attr_name)
            if not isinstance(val, (tuple, list)):
                raise TypeError(
                    f"SplitResult.{attr_name} must be a tuple, got {type(val).__name__}"
                )
        # Normalize to tuples on frozen dataclass
        object.__setattr__(self, "train_rows", tuple(self.train_rows))
        object.__setattr__(self, "val_rows", tuple(self.val_rows))
        object.__setattr__(self, "test_rows", tuple(self.test_rows))

        for attr_name in (
            "val_rows_purged",
            "val_rows_embargoed",
            "test_rows_purged",
            "test_rows_embargoed",
            "total_rows",
        ):
            v = getattr(self, attr_name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise TypeError(f"SplitResult.{attr_name} must be an int, got {type(v).__name__}")
            if v < 0:
                raise ValueError(f"SplitResult.{attr_name} must be non-negative, got {v}")

    @property
    def train_size(self) -> int:
        """Number of rows in the training partition."""
        return len(self.train_rows)

    @property
    def val_size(self) -> int:
        """Number of rows in the validation partition."""
        return len(self.val_rows)

    @property
    def test_size(self) -> int:
        """Number of rows in the test partition."""
        return len(self.test_rows)

    @property
    def retained_rows(self) -> int:
        """Total rows kept across all three partitions."""
        return self.train_size + self.val_size + self.test_size

    @property
    def total_removed_rows(self) -> int:
        """Total rows removed by purging and embargo across both later splits."""
        return (
            self.val_rows_purged
            + self.val_rows_embargoed
            + self.test_rows_purged
            + self.test_rows_embargoed
        )
