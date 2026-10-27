"""Split configuration domain model for Phase 5.3 dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    """Immutable configuration for a chronological train/validation/test split.

    Allocation Rules
    ----------------
    Fractions are used to divide *n* total rows into three contiguous segments:

    1. `train_size = floor(train_fraction * n)`
    2. `val_size   = floor(val_fraction   * n)`
    3. `test_size  = n - train_size - val_size`

    This ensures every row is allocated exactly once and the test split absorbs
    any integer rounding remainder.  All three sizes must be strictly positive
    after allocation when validated against a concrete dataset.

    Purge Convention
    ----------------
    After the train/validation boundary, rows in the validation split whose
    target interval overlaps with the last training observation are purged from
    the *head* of the validation split.  The same applies at the
    validation/test boundary for the test split.

    An observation row's target interval is `(observation_date, realization_date]`.
    Overlap occurs when a later-split row's `observation_date` is strictly
    before the `realization_date` of the last retained row in the preceding
    split.  Concretely, a row at the start of a later split is purged when::

        row.observation_date < last_prior_realization_date

    where `last_prior_realization_date` is the `realization_date` of the
    last row *kept* in the preceding split (after its own purging).

    Endpoint inclusion: the realization date is *inclusive* - it is the date of
    the future bar P(t+h) whose close price is used to compute the return.

    Embargo Convention
    ------------------
    After purging, an additional `embargo_bars` rows are removed from the head
    of validation and test splits.  Embargo rows are counted separately from
    purged rows and are never double-counted.

    Embargo is measured in dataset rows (observations), not calendar days,
    because datasets may have irregular observation spacing.

    Attributes
    ----------
    train_fraction:
        Fraction of rows allocated to training.  Must be in `(0, 1)`.
    val_fraction:
        Fraction of rows allocated to validation.  Must be in `(0, 1)`.
    test_fraction:
        Fraction of rows allocated to testing.  Must be in `(0, 1)`.
        fractions must sum to 1.0 within tolerance 1e-9.
    purge_overlap:
        If True (default), purge rows at split boundaries whose target
        intervals overlap with the last row of the preceding split.
    embargo_bars:
        Number of additional rows to remove from the head of validation and test
        splits after purging.  Must be a non-negative integer.  Defaults to 0.
    """

    train_fraction: float
    val_fraction: float
    test_fraction: float
    purge_overlap: bool = True
    embargo_bars: int = 0

    _FRACTION_SUM_TOLERANCE: float = 1e-9

    def __post_init__(self) -> None:
        for attr_name, val in (
            ("train_fraction", self.train_fraction),
            ("val_fraction", self.val_fraction),
            ("test_fraction", self.test_fraction),
        ):
            if isinstance(val, bool) or not isinstance(val, float):
                raise TypeError(
                    f"SplitConfig.{attr_name} must be a float, got {type(val).__name__}"
                )
            if not (0.0 < val < 1.0):
                raise ValueError(f"SplitConfig.{attr_name} must be in (0, 1), got {val}")

        frac_sum = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(frac_sum - 1.0) > self._FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"SplitConfig fractions must sum to 1.0 within tolerance "
                f"{self._FRACTION_SUM_TOLERANCE}; got "
                f"{self.train_fraction} + {self.val_fraction} + {self.test_fraction} = {frac_sum}"
            )

        if isinstance(self.embargo_bars, bool) or not isinstance(self.embargo_bars, int):
            raise TypeError(
                f"SplitConfig.embargo_bars must be an int, got {type(self.embargo_bars).__name__}"
            )
        if self.embargo_bars < 0:
            raise ValueError(
                f"SplitConfig.embargo_bars must be non-negative, got {self.embargo_bars}"
            )

        if not isinstance(self.purge_overlap, bool):
            raise TypeError(
                f"SplitConfig.purge_overlap must be a bool, got {type(self.purge_overlap).__name__}"
            )

    @classmethod
    def default(cls) -> SplitConfig:
        """Return a standard 70/15/15 split with purging enabled and no embargo."""
        return cls(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
        )
