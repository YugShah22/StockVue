from datetime import date, timedelta
from uuid import UUID

from app.core.exceptions import DataNotFoundError
from app.domain.entities.market_holiday import MarketHoliday
from app.domain.repositories.exchange_repository import ExchangeRepository
from app.domain.repositories.market_holiday_repository import (
    MarketHolidayRepository,
)
from app.domain.value_objects.exchange_code import ExchangeCode


class TradingCalendarService:
    """Service providing trading calendar logic backed by MarketHoliday records."""

    def __init__(
        self,
        holiday_repository: MarketHolidayRepository,
        exchange_repository: ExchangeRepository | None = None,
    ) -> None:
        self._holiday_repo = holiday_repository
        self._exchange_repo = exchange_repository

    def _resolve_exchange_id(self, exchange_ref: UUID | ExchangeCode | str) -> UUID:
        if isinstance(exchange_ref, UUID):
            return exchange_ref

        if self._exchange_repo is None:
            raise ValueError(
                "ExchangeRepository is required to resolve exchange by code"
            )

        code = (
            exchange_ref
            if isinstance(exchange_ref, ExchangeCode)
            else ExchangeCode(str(exchange_ref))
        )
        exchange = self._exchange_repo.get_by_code(code)
        if exchange is None:
            raise DataNotFoundError(f"Exchange not found: {code}")
        return exchange.exchange_id

    def is_holiday(
        self,
        exchange_ref: UUID | ExchangeCode | str,
        check_date: date,
    ) -> bool:
        """Return True if check_date is recorded as a market holiday for the exchange."""
        exchange_id = self._resolve_exchange_id(exchange_ref)
        holiday = self._holiday_repo.get_by_date(exchange_id, check_date)
        return holiday is not None

    def is_trading_day(
        self,
        exchange_ref: UUID | ExchangeCode | str,
        check_date: date,
    ) -> bool:
        """
        Return True if check_date is a trading day for the exchange.

        Weekends (Saturday, Sunday) are non-trading days unless a special
        session (e.g. half_day / early_close Muhurat trading) is scheduled.
        Full-day holidays are non-trading days.
        """
        exchange_id = self._resolve_exchange_id(exchange_ref)
        holiday = self._holiday_repo.get_by_date(exchange_id, check_date)

        is_weekend = check_date.weekday() in (5, 6)

        if is_weekend:
            # Only trading if explicitly a half-day/early-close special session
            if holiday is not None and holiday.session_type in (
                "half_day",
                "early_close",
            ):
                return True
            return False

        if holiday is not None and holiday.session_type == "full_day":
            return False

        return True

    def get_holidays(
        self,
        exchange_ref: UUID | ExchangeCode | str,
        start_date: date,
        end_date: date,
    ) -> list[MarketHoliday]:
        """Return all market holidays in [start_date, end_date]."""
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) cannot be after end_date ({end_date})"
            )
        exchange_id = self._resolve_exchange_id(exchange_ref)

        holidays: list[MarketHoliday] = []
        for year in range(start_date.year, end_date.year + 1):
            holidays.extend(
                self._holiday_repo.get_by_exchange(exchange_id, year=year)
            )

        return [
            h for h in holidays if start_date <= h.holiday_date <= end_date
        ]

    def get_trading_days(
        self,
        exchange_ref: UUID | ExchangeCode | str,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """
        Return all calendar dates in [start_date, end_date] that are trading days.

        Fetches holidays for the entire range to avoid N round-trips.
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) cannot be after end_date ({end_date})"
            )

        holidays = self.get_holidays(exchange_ref, start_date, end_date)
        holiday_by_date = {h.holiday_date: h for h in holidays}

        trading_days: list[date] = []
        current = start_date
        while current <= end_date:
            is_weekend = current.weekday() in (5, 6)
            h = holiday_by_date.get(current)

            if is_weekend:
                if h is not None and h.session_type in ("half_day", "early_close"):
                    trading_days.append(current)
            else:
                if h is None or h.session_type != "full_day":
                    trading_days.append(current)

            current += timedelta(days=1)

        return trading_days
