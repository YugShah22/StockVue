from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import yfinance as yf

from app.core.exceptions import (
    DataProviderError,
    InstrumentNotFoundError,
    ProviderRateLimitError,
)
from app.domain.entities.market_bar import MarketBar
from app.domain.providers.market_data import MarketDataProvider, MarketQuote
from app.domain.repositories.instrument_repository import InstrumentRepository


class YFinanceMarketDataProvider(MarketDataProvider):

    def __init__(
        self,
        instrument_repository: InstrumentRepository,
    ) -> None:
        self._instrument_repository = instrument_repository

    def _resolve_ticker(self, instrument_id: UUID) -> str:

        instrument = self._instrument_repository.get_by_id(instrument_id)

        if instrument is None:
            raise InstrumentNotFoundError(
                f"Instrument {instrument_id} was not found."
            )

        symbol = str(instrument.symbol).upper()
        exchange = str(instrument.exchange_code).upper()

        if exchange == "NSE":
            return f"{symbol}.NS"

        if exchange == "BSE":
            return f"{symbol}.BO"

        raise DataProviderError(
            f"yfinance symbol mapping is not supported for exchange '{exchange}'."
        )

    @staticmethod
    def _normalize_timestamp(timestamp: datetime) -> datetime:

        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=UTC)

        return timestamp

    @staticmethod
    def _handle_provider_error(exc: Exception, ticker: str) -> None:

        message = str(exc).lower()

        if (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        ):
            raise ProviderRateLimitError(
                f"yfinance rate limit reached for {ticker}."
            ) from exc

        raise DataProviderError(
            f"Failed to retrieve market data for {ticker}."
        ) from exc

    def get_bars(
        self,
        instrument_id: UUID,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[MarketBar]:
        ticker_symbol = self._resolve_ticker(instrument_id)

        try:
            ticker = yf.Ticker(ticker_symbol)

            data = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
            )

        except Exception as exc:
            self._handle_provider_error(exc, ticker_symbol)

        if data.empty:
            return []

        bars: list[MarketBar] = []

        for timestamp, row in data.iterrows():
            normalized_timestamp = self._normalize_timestamp(
                timestamp.to_pydatetime()
            )

            bars.append(
                MarketBar(
                    instrument_id=instrument_id,
                    timestamp=normalized_timestamp,
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=Decimal(str(row["Volume"])),
                )
            )

        return bars

    def get_latest_bar(
        self,
        instrument_id: UUID,
        interval: str,
    ) -> MarketBar:
        ticker_symbol = self._resolve_ticker(instrument_id)

        try:
            ticker = yf.Ticker(ticker_symbol)

            data = ticker.history(
                period="5d",
                interval=interval,
                auto_adjust=False,
            )

        except Exception as exc:
            self._handle_provider_error(exc, ticker_symbol)

        if data.empty:
            raise DataProviderError(
                f"No market data available for {ticker_symbol}."
            )

        timestamp, row = data.iloc[-1].name, data.iloc[-1]

        return MarketBar(
            instrument_id=instrument_id,
            timestamp=self._normalize_timestamp(
                timestamp.to_pydatetime()
            ),
            open=Decimal(str(row["Open"])),
            high=Decimal(str(row["High"])),
            low=Decimal(str(row["Low"])),
            close=Decimal(str(row["Close"])),
            volume=Decimal(str(row["Volume"])),
        )

    def get_quote(
        self,
        instrument_id: UUID,
    ) -> MarketQuote:
        ticker_symbol = self._resolve_ticker(instrument_id)

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info

            price = info.last_price

        except Exception as exc:
            self._handle_provider_error(exc, ticker_symbol)

        if price is None:
            raise DataProviderError(
                f"yfinance returned no price for {ticker_symbol}."
            )

        return MarketQuote(
            instrument_id=instrument_id,
            timestamp=datetime.now(UTC),
            price=float(price),
        )
