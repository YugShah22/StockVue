from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import InstrumentNotFoundError
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.services.market_data import MarketDataService


@pytest.fixture
def mock_bar_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_inst_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def market_data_service(
    mock_bar_repo: MagicMock,
    mock_inst_repo: MagicMock,
) -> MarketDataService:
    return MarketDataService(
        market_bar_repository=mock_bar_repo,
        instrument_repository=mock_inst_repo,
    )


@pytest.fixture
def sample_instrument() -> Instrument:
    return Instrument(
        instrument_id=uuid4(),
        company_id=uuid4(),
        symbol=Symbol("RELIANCE"),
        isin=ISIN("INE002A01018"),
        exchange_code=ExchangeCode("NSE"),
        asset_type=AssetType.EQUITY,
        status=InstrumentStatus.ACTIVE,
    )


def make_bar(instrument_id, ts, close="100.0") -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=ts,
        open=Decimal(close),
        high=Decimal(close) + Decimal("2"),
        low=Decimal(close) - Decimal("2"),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


class TestMarketDataService:
    def test_get_historical_bars_success(
        self,
        market_data_service: MarketDataService,
        mock_bar_repo: MagicMock,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        start = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
        end = datetime(2026, 9, 2, 0, 0, tzinfo=UTC)

        bar1 = make_bar(sample_instrument.instrument_id, start)
        bar2 = make_bar(sample_instrument.instrument_id, end)
        mock_bar_repo.get_bars.return_value = [bar1, bar2]

        bars = market_data_service.get_historical_bars(
            sample_instrument.instrument_id, start, end
        )
        assert bars == [bar1, bar2]
        mock_bar_repo.get_bars.assert_called_once_with(
            sample_instrument.instrument_id, start, end
        )

    def test_get_historical_bars_instrument_not_found(
        self,
        market_data_service: MarketDataService,
        mock_inst_repo: MagicMock,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = None
        with pytest.raises(InstrumentNotFoundError):
            market_data_service.get_historical_bars(
                uuid4(),
                datetime(2026, 9, 1, tzinfo=UTC),
                datetime(2026, 9, 2, tzinfo=UTC),
            )

    def test_get_historical_bars_naive_start_raises(
        self,
        market_data_service: MarketDataService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        with pytest.raises(ValueError, match="start timestamp must be timezone-aware"):
            market_data_service.get_historical_bars(
                sample_instrument.instrument_id,
                datetime(2026, 9, 1),
                datetime(2026, 9, 2, tzinfo=UTC),
            )

    def test_get_historical_bars_naive_end_raises(
        self,
        market_data_service: MarketDataService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        with pytest.raises(ValueError, match="end timestamp must be timezone-aware"):
            market_data_service.get_historical_bars(
                sample_instrument.instrument_id,
                datetime(2026, 9, 1, tzinfo=UTC),
                datetime(2026, 9, 2),
            )

    def test_get_historical_bars_start_after_end_raises(
        self,
        market_data_service: MarketDataService,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        with pytest.raises(ValueError, match="cannot be after"):
            market_data_service.get_historical_bars(
                sample_instrument.instrument_id,
                datetime(2026, 9, 5, tzinfo=UTC),
                datetime(2026, 9, 1, tzinfo=UTC),
            )

    def test_get_historical_bars_empty_range(
        self,
        market_data_service: MarketDataService,
        mock_bar_repo: MagicMock,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        mock_bar_repo.get_bars.return_value = []

        result = market_data_service.get_historical_bars(
            sample_instrument.instrument_id,
            datetime(2026, 9, 1, tzinfo=UTC),
            datetime(2026, 9, 2, tzinfo=UTC),
        )
        assert result == []

    def test_get_latest_bar(
        self,
        market_data_service: MarketDataService,
        mock_bar_repo: MagicMock,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_id.return_value = sample_instrument
        bar = make_bar(
            sample_instrument.instrument_id,
            datetime(2026, 9, 5, tzinfo=UTC),
        )
        mock_bar_repo.get_latest_bar.return_value = bar

        result = market_data_service.get_latest_bar(sample_instrument.instrument_id)
        assert result == bar

    def test_get_bars_by_symbol(
        self,
        market_data_service: MarketDataService,
        mock_bar_repo: MagicMock,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_symbol.return_value = sample_instrument
        mock_inst_repo.get_by_id.return_value = sample_instrument
        start = datetime(2026, 9, 1, tzinfo=UTC)
        end = datetime(2026, 9, 2, tzinfo=UTC)
        bar = make_bar(sample_instrument.instrument_id, start)
        mock_bar_repo.get_bars.return_value = [bar]

        result = market_data_service.get_bars_by_symbol(
            "RELIANCE", "NSE", start, end
        )
        assert result == [bar]

    def test_get_latest_bar_by_symbol(
        self,
        market_data_service: MarketDataService,
        mock_bar_repo: MagicMock,
        mock_inst_repo: MagicMock,
        sample_instrument: Instrument,
    ) -> None:
        mock_inst_repo.get_by_symbol.return_value = sample_instrument
        mock_inst_repo.get_by_id.return_value = sample_instrument
        bar = make_bar(sample_instrument.instrument_id, datetime(2026, 9, 5, tzinfo=UTC))
        mock_bar_repo.get_latest_bar.return_value = bar

        result = market_data_service.get_latest_bar_by_symbol("RELIANCE", "NSE")
        assert result == bar
