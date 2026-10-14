import math
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

from app.domain.entities.market_bar import MarketBar
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.calculators.forward_return_calculator import ForwardReturnCalculator
from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.target_definition import TargetDefinition
from app.services.features.feature_engine import FeatureEngine

_SCALAR_RESERVED_WORDS: Final[frozenset[str]] = frozenset({"True", "False", "None"})


def _is_simple_scalar_identifier(val: str) -> bool:
    """Return True if val can be represented unquoted without delimiter or keyword ambiguity."""
    return val.isidentifier() and "_" not in val and val not in _SCALAR_RESERVED_WORDS


def _canonicalize_parameter_value(val: Any, is_top_level: bool = False) -> str:
    """Recursively produce a stable, collision-resistant string for a parameter value.

    - Booleans: 'True' or 'False' (distinguished from integers).
    - Integers, floats, Decimals: exact numeric string. Note: integer and Decimal
      values with equivalent textual representations (e.g. 2 and Decimal("2"))
      intentionally share a canonical representation to preserve standard names.
    - None: 'None'.
    - Dates, datetimes: ISO 8601 string.
    - Strings: unquoted only if top-level and a simple identifier without underscores;
      otherwise single-quoted with escaping to avoid delimiter and type collisions.
    - Dictionaries: keys sorted alphabetically so insertion order does not affect column names.
    - Lists: '[v1, v2, ...]'.
    - Tuples: '(v1, v2, ...)'.
    - Sets: 'set{v1, v2, ...}' with sorted elements.
    - Frozensets: 'frozenset{v1, v2, ...}' with sorted elements.
    - Unsupported types: raises TypeError.
    """
    if isinstance(val, bool):
        return "True" if val else "False"

    if isinstance(val, (int, float, Decimal)):
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError(f"Non-finite float parameter value is not allowed: {val}")
        if isinstance(val, Decimal) and not val.is_finite():
            raise ValueError(f"Non-finite Decimal parameter value is not allowed: {val}")
        return str(val)

    if val is None:
        return "None"

    if isinstance(val, (date, datetime)):
        return val.isoformat()

    if isinstance(val, str):
        if is_top_level and _is_simple_scalar_identifier(val):
            return val
        escaped = val.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    if isinstance(val, dict):
        items: list[str] = []
        for k in sorted(val.keys(), key=str):
            if not isinstance(k, (str, int)):
                raise TypeError(
                    f"Unsupported parameter dictionary key type: {type(k).__name__} ({k!r})"
                )
            if isinstance(k, str) and not k:
                raise ValueError("Nested parameter dictionary key cannot be empty")
            k_repr = k if (isinstance(k, str) and _is_simple_scalar_identifier(k)) else repr(k)
            v_repr = _canonicalize_parameter_value(val[k], is_top_level=False)
            items.append(f"{k_repr}:{v_repr}")
        return "{" + ", ".join(items) + "}"

    if isinstance(val, (list, tuple)):
        items = [_canonicalize_parameter_value(x, is_top_level=False) for x in val]
        return (
            "[" + ", ".join(items) + "]" if isinstance(val, list) else "(" + ", ".join(items) + ")"
        )

    if isinstance(val, (set, frozenset)):
        items = sorted(_canonicalize_parameter_value(x, is_top_level=False) for x in val)
        prefix = "set" if isinstance(val, set) else "frozenset"
        return prefix + "{" + ", ".join(items) + "}"

    raise TypeError(
        f"Unsupported parameter value type for feature naming: {type(val).__name__} ({val!r})"
    )


def generate_feature_name(definition: FeatureDefinition) -> str:
    """Generate a deterministic, unique column name for a FeatureDefinition.

    When parameters are present, parameter keys are sorted alphabetically
    and parameter values are canonically normalized to ensure stable,
    collision-resistant column naming regardless of dictionary insertion order.

    Supported Contract & Constraints:
    - Top-level parameter keys must be non-empty, single-word Python identifiers
      without underscores (e.g. 'window', 'band', 'k'). This completely eliminates
      delimiter collisions with the parameter boundary separator ('_').
    - Integer and Decimal values with equivalent textual representations (such as
      `2` and `Decimal("2")`) intentionally share a column name. This preserves
      backward compatibility for standard feature names such as
      `bollinger_bands_band_upper_k_2_window_20`.
    - Strings with underscores or keywords, nested containers, lists, tuples, sets,
      and frozensets are uniquely and unambiguously encoded.
    """
    if not definition.parameters:
        return definition.name
    parts = [definition.name]
    for k in sorted(definition.parameters.keys(), key=str):
        if not isinstance(k, str):
            raise TypeError(f"Parameter key must be a string, got {type(k).__name__}")
        if not k:
            raise ValueError("Parameter key cannot be empty")
        if not k.isidentifier() or "_" in k or k in _SCALAR_RESERVED_WORDS:
            raise ValueError(
                f"Top-level parameter key must be a single-word identifier without underscores, got {k!r}"
            )
        val = definition.parameters[k]
        val_str = _canonicalize_parameter_value(val, is_top_level=True)
        parts.append(f"{k}_{val_str}")
    return "_".join(parts)


class DatasetBuilder:
    """Constructs point-in-time machine-learning training datasets.

    Coordinates FeatureEngine and ForwardReturnCalculator while guaranteeing:
    1. Zero future-information leakage into feature calculations.
    2. Exact alignment between observation dates, realization dates, and labels.
    3. Complete, rectangular matrix output without missing or fabricated values.
    4. Deterministic feature column ordering.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        target_calculator: ForwardReturnCalculator | None = None,
    ) -> None:
        if not isinstance(feature_engine, FeatureEngine):
            raise TypeError(
                f"feature_engine must be a FeatureEngine instance, got {type(feature_engine).__name__}"
            )
        if target_calculator is not None and not isinstance(
            target_calculator, ForwardReturnCalculator
        ):
            raise TypeError(
                f"target_calculator must be a ForwardReturnCalculator instance, got {type(target_calculator).__name__}"
            )
        self._feature_engine = feature_engine
        self._target_calculator = target_calculator or ForwardReturnCalculator()

    @property
    def feature_engine(self) -> FeatureEngine:
        return self._feature_engine

    @property
    def target_calculator(self) -> ForwardReturnCalculator:
        return self._target_calculator

    def build_dataset(
        self,
        instrument_id: UUID,
        bars: Sequence[MarketBar],
        features: Sequence[FeatureDefinition],
        target: TargetDefinition,
        observation_dates: Sequence[date] | None = None,
    ) -> FeatureMatrix:
        """Construct a complete, point-in-time FeatureMatrix for an instrument.

        Parameters
        ----------
        instrument_id:
            UUID of the instrument to build the dataset for.
        bars:
            Available market bars. Bars for other instruments are ignored.
        features:
            Sequence of feature definitions. Preserves caller's column ordering.
        target:
            The prediction target definition.
        observation_dates:
            Optional sequence of observation dates to evaluate. If None, all
            eligible dates in the instrument's bars are evaluated.

        Returns
        -------
        FeatureMatrix
            A validated dataset containing only rows where all features and the
            target label were fully realizable.
        """
        # 1. Validate scalar / container types
        if not isinstance(instrument_id, UUID):
            raise TypeError(
                f"instrument_id must be a UUID instance, got {type(instrument_id).__name__}"
            )
        if not isinstance(target, TargetDefinition):
            raise TypeError(
                f"target must be a TargetDefinition instance, got {type(target).__name__}"
            )
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
            raise TypeError("features must be a sequence of FeatureDefinition")
        if len(features) == 0:
            raise ValueError("At least one FeatureDefinition must be supplied")
        for i, defn in enumerate(features):
            if not isinstance(defn, FeatureDefinition):
                raise TypeError(
                    f"features[{i}] must be a FeatureDefinition instance, got {type(defn).__name__}"
                )

        if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
            raise TypeError("bars must be a sequence of MarketBar")
        for i, b in enumerate(bars):
            if not isinstance(b, MarketBar):
                raise TypeError(f"bars[{i}] must be a MarketBar instance, got {type(b).__name__}")

        # 2. Generate deterministic column names & detect collisions
        feature_names: list[str] = []
        seen_names: set[str] = set()
        for defn in features:
            col_name = generate_feature_name(defn)
            if col_name in seen_names:
                raise ValueError(
                    f"Duplicate feature configuration or column name collision detected: '{col_name}'"
                )
            seen_names.add(col_name)
            feature_names.append(col_name)

        # 3. Validate observation_dates if provided
        req_dates: list[date] | None = None
        if observation_dates is not None:
            if not isinstance(observation_dates, Sequence) or isinstance(
                observation_dates, (str, bytes)
            ):
                raise TypeError("observation_dates must be a sequence of date")
            req_dates = []
            for i, d in enumerate(observation_dates):
                if isinstance(d, datetime):
                    raise TypeError(f"observation_dates[{i}] must be a date, not a datetime")
                if not isinstance(d, date):
                    raise TypeError(
                        f"observation_dates[{i}] must be a date instance, got {type(d).__name__}"
                    )
                req_dates.append(d)
            if len(req_dates) != len(set(req_dates)):
                raise ValueError("Duplicate observation dates requested")

        # 4. Filter bars by instrument_id FIRST, then sort chronologically
        inst_bars = [b for b in bars if b.instrument_id == instrument_id]
        sorted_bars = sorted(inst_bars, key=lambda b: b.timestamp)

        # 5. Check for duplicate observation dates for this instrument
        bar_dates = [b.timestamp.date() for b in sorted_bars]
        if len(bar_dates) != len(set(bar_dates)):
            raise ValueError(
                f"Duplicate market observation dates found for instrument {instrument_id}. "
                f"Deduplicate the input bars before constructing a dataset."
            )

        # 6. Determine candidate observation dates
        if req_dates is None:
            candidate_dates = bar_dates
        else:
            bar_date_set = set(bar_dates)
            sorted_req_dates = sorted(req_dates)
            candidate_dates = [d for d in sorted_req_dates if d in bar_date_set]

        # 7. Map calendar dates to chronological bar indices
        date_to_idx = {b.timestamp.date(): i for i, b in enumerate(sorted_bars)}

        # 8. Point-in-time calculation loop
        rows: list[DatasetRow] = []
        dropped_incomplete_feature_count = 0
        dropped_incomplete_target_count = 0

        for obs_date in candidate_dates:
            obs_idx = date_to_idx[obs_date]
            # STRICT POINT-IN-TIME SLICE: only bars up to and including obs_idx
            history_bars = sorted_bars[: obs_idx + 1]

            ctx = FeatureCalculationContext(
                instrument_id=instrument_id,
                as_of_date=obs_date,
                market_bars=history_bars,
            )

            row_features: list[Decimal] = []
            has_missing_feature = False

            for defn in features:
                f_val = self._feature_engine.calculate(defn, ctx)
                if f_val is None or not f_val.value.is_finite():
                    has_missing_feature = True
                    break
                row_features.append(f_val.value)

            if has_missing_feature:
                dropped_incomplete_feature_count += 1
                continue

            # Forward target calculation (future bars allowed strictly for label computation)
            target_val = self._target_calculator.calculate(
                definition=target,
                instrument_id=instrument_id,
                bars=sorted_bars,
                observation_date=obs_date,
            )

            if target_val is None or not target_val.value.is_finite():
                dropped_incomplete_target_count += 1
                continue

            row = DatasetRow(
                instrument_id=instrument_id,
                observation_date=obs_date,
                realization_date=target_val.realization_date,
                feature_values=tuple(row_features),
                target_value=target_val.value,
            )
            rows.append(row)

        return FeatureMatrix(
            instrument_id=instrument_id,
            feature_names=tuple(feature_names),
            feature_definitions=tuple(features),
            target_definition=target,
            rows=tuple(rows),
            dropped_incomplete_feature_count=dropped_incomplete_feature_count,
            dropped_incomplete_target_count=dropped_incomplete_target_count,
        )
