from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue

# ===========================================================================
# FeatureCategory Tests
# ===========================================================================


class TestFeatureCategory:
    def test_all_five_categories_exist(self) -> None:
        expected = {"TECHNICAL", "FUNDAMENTAL", "MARKET", "SENTIMENT", "MACRO"}
        actual = {c.name for c in FeatureCategory}
        assert actual == expected

    def test_enum_values_are_stable_strings(self) -> None:
        assert FeatureCategory.TECHNICAL.value == "technical"
        assert FeatureCategory.FUNDAMENTAL.value == "fundamental"
        assert FeatureCategory.MARKET.value == "market"
        assert FeatureCategory.SENTIMENT.value == "sentiment"
        assert FeatureCategory.MACRO.value == "macro"

    def test_string_membership_and_coercion(self) -> None:
        assert FeatureCategory("technical") == FeatureCategory.TECHNICAL
        assert FeatureCategory("fundamental") == FeatureCategory.FUNDAMENTAL
        assert FeatureCategory("market") == FeatureCategory.MARKET
        assert FeatureCategory("sentiment") == FeatureCategory.SENTIMENT
        assert FeatureCategory("macro") == FeatureCategory.MACRO


# ===========================================================================
# FeatureDefinition Tests
# ===========================================================================


class TestFeatureDefinition:
    def test_valid_definition_creation(self) -> None:
        definition = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Historical percentage return over a given window",
            parameters={"window": 20},
        )
        assert definition.name == "return"
        assert definition.category == FeatureCategory.TECHNICAL
        assert definition.description == "Historical percentage return over a given window"
        assert definition.parameters == {"window": 20}

    def test_name_is_normalized(self) -> None:
        definition = FeatureDefinition(
            name="  Return_20D  ",
            category=FeatureCategory.TECHNICAL,
            description="20-day return",
        )
        assert definition.name == "return_20d"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Feature name cannot be empty"):
            FeatureDefinition(
                name="   ",
                category=FeatureCategory.TECHNICAL,
                description="Some feature",
            )

    def test_description_is_normalized(self) -> None:
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="   Simple Moving Average   ",
        )
        assert definition.description == "Simple Moving Average"

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="Feature description cannot be empty"):
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="   ",
            )

    def test_category_string_coercion(self) -> None:
        definition = FeatureDefinition(
            name="pe_ratio",
            category="fundamental",  # type: ignore[arg-type]
            description="Price to earnings ratio",
        )
        assert definition.category == FeatureCategory.FUNDAMENTAL

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid feature category"):
            FeatureDefinition(
                name="invalid",
                category="nonexistent",  # type: ignore[arg-type]
                description="Invalid feature",
            )

        with pytest.raises(ValueError, match="Invalid feature category"):
            FeatureDefinition(
                name="invalid",
                category=123,  # type: ignore[arg-type]
                description="Invalid feature",
            )

    def test_valid_parameters_retained_and_copied(self) -> None:
        params = {"window": 50, "price_field": "close"}
        definition = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="50-day SMA",
            parameters=params,
        )
        assert definition.parameters == {"window": 50, "price_field": "close"}
        # Ensure defensive copy was made
        params["window"] = 100
        assert definition.parameters["window"] == 50

    def test_invalid_parameters_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a dictionary"):
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="Invalid parameters",
                parameters=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_default_parameters_empty_dict(self) -> None:
        definition = FeatureDefinition(
            name="volume",
            category=FeatureCategory.MARKET,
            description="Raw trading volume",
        )
        assert definition.parameters == {}

    def test_parameterized_features_represent_same_family(self) -> None:
        return_5d = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="5-day return",
            parameters={"window": 5},
        )
        return_20d = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="20-day return",
            parameters={"window": 20},
        )
        return_60d = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="60-day return",
            parameters={"window": 60},
        )

        assert return_5d.name == return_20d.name == return_60d.name == "return"
        assert return_5d.category == return_20d.category == return_60d.category == FeatureCategory.TECHNICAL
        assert return_5d.parameters["window"] == 5
        assert return_20d.parameters["window"] == 20
        assert return_60d.parameters["window"] == 60
        assert return_5d != return_20d
        assert return_20d != return_60d

        sma_20 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="20-day SMA",
            parameters={"window": 20},
        )
        sma_50 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="50-day SMA",
            parameters={"window": 50},
        )
        sma_200 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="200-day SMA",
            parameters={"window": 200},
        )

        assert sma_20.name == sma_50.name == sma_200.name == "sma"
        assert sma_20.parameters["window"] == 20
        assert sma_50.parameters["window"] == 50
        assert sma_200.parameters["window"] == 200


# ===========================================================================
# FeatureValue Tests
# ===========================================================================


class TestFeatureValue:
    def test_valid_feature_value_creation(self) -> None:
        feature_def = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="20-day return",
            parameters={"window": 20},
        )
        inst_id = uuid4()
        obs_date = date(2026, 9, 1)
        val = Decimal("0.052300")

        feature_val = FeatureValue(
            feature=feature_def,
            instrument_id=inst_id,
            observation_date=obs_date,
            value=val,
        )

        assert feature_val.feature == feature_def
        assert feature_val.instrument_id == inst_id
        assert feature_val.observation_date == obs_date
        assert feature_val.value == val

    def test_invalid_feature_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a FeatureDefinition instance"):
            FeatureValue(
                feature="return_20d",  # type: ignore[arg-type]
                instrument_id=uuid4(),
                observation_date=date(2026, 9, 1),
                value=Decimal("10.0"),
            )

    def test_invalid_instrument_id_type_raises(self) -> None:
        feature_def = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
        )
        with pytest.raises(TypeError, match="must be a UUID instance"):
            FeatureValue(
                feature=feature_def,
                instrument_id="not-a-uuid",  # type: ignore[arg-type]
                observation_date=date(2026, 9, 1),
                value=Decimal("10.0"),
            )

    def test_datetime_observation_date_raises(self) -> None:
        feature_def = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
        )
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            FeatureValue(
                feature=feature_def,
                instrument_id=uuid4(),
                observation_date=datetime(2026, 9, 1, 10, 0),
                value=Decimal("10.0"),
            )

    def test_non_date_observation_date_raises(self) -> None:
        feature_def = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
        )
        with pytest.raises(TypeError, match="must be a date instance"):
            FeatureValue(
                feature=feature_def,
                instrument_id=uuid4(),
                observation_date="2026-09-01",  # type: ignore[arg-type]
                value=Decimal("10.0"),
            )

    def test_non_decimal_value_raises(self) -> None:
        feature_def = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="Return",
        )
        with pytest.raises(TypeError, match="must be a Decimal instance"):
            FeatureValue(
                feature=feature_def,
                instrument_id=uuid4(),
                observation_date=date(2026, 9, 1),
                value=12.5,  # type: ignore[arg-type] # float rejected!
            )

        with pytest.raises(TypeError, match="must be a Decimal instance"):
            FeatureValue(
                feature=feature_def,
                instrument_id=uuid4(),
                observation_date=date(2026, 9, 1),
                value=100,  # type: ignore[arg-type] # int rejected!
            )
