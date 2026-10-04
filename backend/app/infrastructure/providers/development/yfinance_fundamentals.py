from datetime import date
from decimal import Decimal
from uuid import UUID

import yfinance as yf

from app.core.exceptions import DataProviderError
from app.domain.entities.fundamental_record import FundamentalRecord
from app.domain.providers.fundamentals import FundamentalsProvider
from app.domain.repositories.instrument_repository import InstrumentRepository


class YFinanceFundamentalsProvider(FundamentalsProvider):
    def __init__(
        self,
        instrument_repository: InstrumentRepository,
    ) -> None:
        self._instrument_repository = instrument_repository

    def get_fundamentals(
        self,
        instrument_id: UUID,
        start_period: date,
        end_period: date,
    ) -> list[FundamentalRecord]:
        instrument = self._instrument_repository.get_by_id(instrument_id)

        if instrument is None:
            raise DataProviderError(
                f"Instrument not found: {instrument_id}"
            )

        ticker_symbol = self._resolve_ticker_symbol(
            str(instrument.exchange_code),
            str(instrument.symbol),
        )

        try:
            ticker = yf.Ticker(ticker_symbol)
            income_statement = ticker.get_income_stmt(
                freq="yearly",
            )
        except Exception as exc:
            raise DataProviderError(
                f"Failed to fetch fundamentals for {ticker_symbol}"
            ) from exc

        if income_statement.empty:
            return []

        records: list[FundamentalRecord] = []

        for metric_name, row in income_statement.iterrows():
            for period in income_statement.columns:
                period_date = period.date()

                if not start_period <= period_date <= end_period:
                    continue

                value = row[period]

                if value is None:
                    continue

                try:
                    decimal_value = Decimal(str(value))
                except Exception as exc:
                    raise DataProviderError(
                        f"Invalid fundamental value for {metric_name}"
                    ) from exc

                records.append(
                    FundamentalRecord(
                        instrument_id=instrument_id,
                        period_end=period_date,
                        metric_name=str(metric_name),
                        value=decimal_value,
                    )
                )

        return records

    @staticmethod
    def _resolve_ticker_symbol(
        exchange_code: str,
        symbol: str,
    ) -> str:
        exchange = exchange_code.upper()

        if exchange == "NSE":
            return f"{symbol}.NS"

        if exchange == "BSE":
            return f"{symbol}.BO"

        raise DataProviderError(
            f"Unsupported exchange for yfinance: {exchange_code}"
        )
