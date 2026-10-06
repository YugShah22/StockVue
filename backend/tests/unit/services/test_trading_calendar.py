from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from app.core.exceptions import DataNotFoundError
from app.domain.entities.exchange import Exchange
from app.domain.entities.market_holiday import MarketHoliday
from app.domain.value_objects.exchange_code import ExchangeCode
from app.services.trading_calendar import TradingCalendarService


@pytest.fixture
def exchange() -> Exchange:
    return Exchange(
        exchange_id=uuid4(),
        code=ExchangeCode("NSE"),
        name="National Stock Exchange of India",
        country="India",
        timezone=ZoneInfo("Asia/Kolkata"),
    )


@pytest.fixture
def mock_holiday_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_exchange_repo(exchange: Exchange) -> MagicMock:
    repo = MagicMock()
    repo.get_by_code.side_effect = lambda code: (
        exchange if str(code) == "NSE" else None
    )
    return repo


@pytest.fixture
def calendar_service(
    mock_holiday_repo: MagicMock,
    mock_exchange_repo: MagicMock,
) -> TradingCalendarService:
    return TradingCalendarService(
        holiday_repository=mock_holiday_repo,
        exchange_repository=mock_exchange_repo,
    )


class TestTradingCalendarService:
    def test_is_holiday_true(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        target_date = date(2026, 8, 15)
        mock_holiday_repo.get_by_date.return_value = MarketHoliday(
            holiday_id=uuid4(),
            exchange_id=exchange.exchange_id,
            holiday_date=target_date,
            name="Independence Day",
            session_type="full_day",
        )

        assert calendar_service.is_holiday(exchange.exchange_id, target_date) is True
        assert calendar_service.is_holiday(ExchangeCode("NSE"), target_date) is True

    def test_is_holiday_false(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        mock_holiday_repo.get_by_date.return_value = None
        assert calendar_service.is_holiday(exchange.exchange_id, date(2026, 8, 17)) is False

    def test_is_trading_day_regular_weekday(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        # Monday
        target_date = date(2026, 8, 17)
        mock_holiday_repo.get_by_date.return_value = None

        assert calendar_service.is_trading_day(exchange.exchange_id, target_date) is True

    def test_is_trading_day_full_day_holiday(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        # Monday but holiday
        target_date = date(2026, 8, 17)
        mock_holiday_repo.get_by_date.return_value = MarketHoliday(
            holiday_id=uuid4(),
            exchange_id=exchange.exchange_id,
            holiday_date=target_date,
            name="Holiday",
            session_type="full_day",
        )

        assert calendar_service.is_trading_day(exchange.exchange_id, target_date) is False

    def test_is_trading_day_half_day_weekday(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        target_date = date(2026, 8, 17)
        mock_holiday_repo.get_by_date.return_value = MarketHoliday(
            holiday_id=uuid4(),
            exchange_id=exchange.exchange_id,
            holiday_date=target_date,
            name="Early Close",
            session_type="early_close",
        )

        assert calendar_service.is_trading_day(exchange.exchange_id, target_date) is True

    def test_is_trading_day_weekend_no_session(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        # Saturday
        saturday = date(2026, 8, 15)
        mock_holiday_repo.get_by_date.return_value = None

        assert calendar_service.is_trading_day(exchange.exchange_id, saturday) is False

    def test_is_trading_day_weekend_special_session(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        # Sunday with Muhurat Trading
        sunday = date(2026, 11, 1)
        mock_holiday_repo.get_by_date.return_value = MarketHoliday(
            holiday_id=uuid4(),
            exchange_id=exchange.exchange_id,
            holiday_date=sunday,
            name="Diwali Muhurat Trading",
            session_type="early_close",
        )

        assert calendar_service.is_trading_day(exchange.exchange_id, sunday) is True

    def test_get_trading_days_range(
        self,
        calendar_service: TradingCalendarService,
        mock_holiday_repo: MagicMock,
        exchange: Exchange,
    ) -> None:
        # 2026-08-10 (Mon) to 2026-08-16 (Sun)
        # Aug 10: Mon (trading)
        # Aug 11: Tue (trading)
        # Aug 12: Wed (trading)
        # Aug 13: Thu (trading)
        # Aug 14: Fri (holiday)
        # Aug 15: Sat (weekend)
        # Aug 16: Sun (weekend)
        start = date(2026, 8, 10)
        end = date(2026, 8, 16)

        mock_holiday_repo.get_by_exchange.return_value = [
            MarketHoliday(
                holiday_id=uuid4(),
                exchange_id=exchange.exchange_id,
                holiday_date=date(2026, 8, 14),
                name="Holiday",
                session_type="full_day",
            )
        ]

        trading_days = calendar_service.get_trading_days(
            exchange.exchange_id, start, end
        )
        assert trading_days == [
            date(2026, 8, 10),
            date(2026, 8, 11),
            date(2026, 8, 12),
            date(2026, 8, 13),
        ]

    def test_invalid_date_range_raises(
        self,
        calendar_service: TradingCalendarService,
        exchange: Exchange,
    ) -> None:
        with pytest.raises(ValueError, match="cannot be after"):
            calendar_service.get_trading_days(
                exchange.exchange_id,
                date(2026, 8, 20),
                date(2026, 8, 10),
            )

    def test_unknown_exchange_code_raises(
        self,
        calendar_service: TradingCalendarService,
    ) -> None:
        with pytest.raises(DataNotFoundError):
            calendar_service.is_trading_day(
                ExchangeCode("BSE"),
                date(2026, 8, 17),
            )
