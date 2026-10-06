from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.entities.corporate_action import CorporateAction
from app.domain.enums.corporate_action_type import CorporateActionType


class TestCorporateActionEntity:
    def test_valid_split_creation(self) -> None:
        inst_id = uuid4()
        action = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 6, 1),
            value=Decimal("2.0"),
            description="2:1 Stock Split",
        )
        assert action.instrument_id == inst_id
        assert action.action_type == CorporateActionType.SPLIT
        assert action.execution_date == date(2026, 6, 1)
        assert action.value == Decimal("2.0")
        assert action.currency is None
        assert action.description == "2:1 Stock Split"

    def test_valid_dividend_creation(self) -> None:
        inst_id = uuid4()
        action = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 8, 15),
            value=Decimal("12.50"),
            currency="inr",
            description="Interim Dividend",
        )
        assert action.action_type == CorporateActionType.DIVIDEND
        assert action.currency == "INR"
        assert action.value == Decimal("12.50")

    def test_string_action_type_coercion(self) -> None:
        inst_id = uuid4()
        action = CorporateAction(
            instrument_id=inst_id,
            action_type="split",  # type: ignore[arg-type]
            execution_date=date(2026, 6, 1),
            value=Decimal("2.0"),
        )
        assert action.action_type == CorporateActionType.SPLIT

    def test_invalid_action_type_raises(self) -> None:
        inst_id = uuid4()
        with pytest.raises(ValueError, match="Invalid corporate action type"):
            CorporateAction(
                instrument_id=inst_id,
                action_type="merger",  # type: ignore[arg-type]
                execution_date=date(2026, 6, 1),
                value=Decimal("1.0"),
            )

    def test_non_positive_value_raises(self) -> None:
        inst_id = uuid4()
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=inst_id,
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 6, 1),
                value=Decimal("0"),
            )

        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=inst_id,
                action_type=CorporateActionType.DIVIDEND,
                execution_date=date(2026, 6, 1),
                value=Decimal("-5.0"),
            )

    def test_empty_currency_raises(self) -> None:
        inst_id = uuid4()
        with pytest.raises(ValueError, match="Currency cannot be empty string"):
            CorporateAction(
                instrument_id=inst_id,
                action_type=CorporateActionType.DIVIDEND,
                execution_date=date(2026, 6, 1),
                value=Decimal("5.0"),
                currency="   ",
            )
