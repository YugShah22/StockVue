from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.entities.company import Company
from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.exchange import Exchange
from app.domain.entities.instrument import Instrument
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.asset_type import AssetType
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.enums.instrument_status import InstrumentStatus
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol
from app.infrastructure.database.repositories.company import (
    PostgresCompanyRepository,
)
from app.infrastructure.database.repositories.corporate_action import (
    PostgresCorporateActionRepository,
)
from app.infrastructure.database.repositories.exchange import (
    PostgresExchangeRepository,
)
from app.infrastructure.database.repositories.instrument import (
    PostgresInstrumentRepository,
)
from app.services.features.feature_calculator_registry import (
    create_default_feature_registry,
)
from app.services.features.feature_engine import FeatureEngine
from app.services.prediction.dataset_builder import DatasetBuilder


def _setup_test_instrument(
    db_session: Session,
    symbol: str = "BUILDERTEST",
    isin: str = "INE123456789",
) -> Instrument:
    """Helper to persist prerequisite company, exchange, and instrument records in PostgreSQL."""
    company_repo = PostgresCompanyRepository(db_session)
    exchange_repo = PostgresExchangeRepository(db_session)
    instrument_repo = PostgresInstrumentRepository(db_session)

    company = Company(
        company_id=uuid4(),
        name=f"Test Corp {symbol}",
        legal_name=f"Test Corp {symbol} Limited",
        country="India",
    )
    exchange = exchange_repo.get_by_code(ExchangeCode("NSE"))
    if exchange is None:
        exchange = Exchange(
            exchange_id=uuid4(),
            code=ExchangeCode("NSE"),
            name="National Stock Exchange",
            country="India",
            timezone=ZoneInfo("Asia/Kolkata"),
        )
        exchange_repo.save(exchange)

    instrument = Instrument(
        instrument_id=uuid4(),
        company_id=company.company_id,
        isin=ISIN(isin),
        symbol=Symbol(symbol),
        asset_type=AssetType.EQUITY,
        exchange_code=exchange.code,
        status=InstrumentStatus.ACTIVE,
    )

    company_repo.save(company)
    instrument_repo.save(instrument)
    return instrument


def _make_bar(
    instrument_id,
    timestamp: datetime,
    close: Decimal,
) -> MarketBar:
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=Decimal("1000"),
    )


def test_postgres_split_does_not_create_false_loss(db_session: Session) -> None:
    """A 2-for-1 split persisted in PostgreSQL adjusts endpoint prices to prevent a false loss."""
    ca_repo = PostgresCorporateActionRepository(db_session)
    instrument = _setup_test_instrument(db_session, symbol="SPLITTEST", isin="INE111111111")
    inst_id = instrument.instrument_id

    # 1. Persist a 2-for-1 split in PostgreSQL
    split = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 1, 6),
        value=Decimal("2"),
        description="2:1 Stock Split",
    )
    ca_repo.save(split)

    # Verify retrieval through real repository
    retrieved = ca_repo.get_by_id(split.corporate_action_id)
    assert retrieved is not None
    assert retrieved.action_type == CorporateActionType.SPLIT
    assert retrieved.value == Decimal("2.000000")

    # 2. Market bars: price drops 200 -> 100 due to 2:1 split
    bars = [
        _make_bar(inst_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200")),
        _make_bar(inst_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100")),
    ]

    engine = FeatureEngine(create_default_feature_registry())
    builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=ca_repo)

    features = [
        FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="1d SMA",
            parameters={"window": 1},
        )
    ]
    target = TargetDefinition(name="total_return", horizon_bars=1)

    matrix = builder.build_dataset(
        instrument_id=inst_id,
        bars=bars,
        features=features,
        target=target,
    )

    assert len(matrix) == 1
    # 2-for-1 split: 100 / (200 / 2) - 1 = 0.00 total return (not -0.50 false loss)
    assert matrix.target_vector[0] == Decimal("0")


def test_postgres_dividend_handled_with_reinvestment_convention(
    db_session: Session,
) -> None:
    """A cash dividend persisted in PostgreSQL is reinvested at the raw ex-date closing price."""
    ca_repo = PostgresCorporateActionRepository(db_session)
    instrument = _setup_test_instrument(db_session, symbol="DIVTEST", isin="INE222222222")
    inst_id = instrument.instrument_id

    # 1. Persist a 5.00 INR cash dividend in PostgreSQL
    dividend = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 1, 6),
        value=Decimal("5.00"),
        currency="INR",
        description="Interim Dividend",
    )
    ca_repo.save(dividend)

    retrieved = ca_repo.get_by_id(dividend.corporate_action_id)
    assert retrieved is not None
    assert retrieved.action_type == CorporateActionType.DIVIDEND
    assert retrieved.value == Decimal("5.000000")

    # 2. Market bars: price drops 100 -> 95 on ex-dividend date
    bars = [
        _make_bar(inst_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100")),
        _make_bar(inst_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("95")),
    ]

    engine = FeatureEngine(create_default_feature_registry())
    builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=ca_repo)

    features = [
        FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="1d SMA",
            parameters={"window": 1},
        )
    ]
    target = TargetDefinition(name="total_return", horizon_bars=1)

    matrix = builder.build_dataset(
        instrument_id=inst_id,
        bars=bars,
        features=features,
        target=target,
    )

    assert len(matrix) == 1
    # Reinvestment: (95 / 100) * (1 + 5 / 95) - 1 = 0.95 * (100 / 95) - 1 = 0.00
    assert matrix.target_vector[0] == Decimal("0")


def test_postgres_actions_after_realization_do_not_affect_label(
    db_session: Session,
) -> None:
    """Corporate actions executed after the realization date do not leak into target labels."""
    ca_repo = PostgresCorporateActionRepository(db_session)
    instrument = _setup_test_instrument(db_session, symbol="FUTACTTEST", isin="INE333333333")
    inst_id = instrument.instrument_id

    # Persist a future split in PostgreSQL well after realization date
    future_split = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 2, 1),
        value=Decimal("2"),
    )
    ca_repo.save(future_split)

    bars = [
        _make_bar(inst_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100")),
        _make_bar(inst_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105")),
    ]

    engine = FeatureEngine(create_default_feature_registry())
    builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=ca_repo)

    features = [
        FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="1d SMA",
            parameters={"window": 1},
        )
    ]
    target = TargetDefinition(name="total_return", horizon_bars=1)

    matrix = builder.build_dataset(
        instrument_id=inst_id,
        bars=bars,
        features=features,
        target=target,
    )

    assert len(matrix) == 1
    # Target from t0 to t1 is 105 / 100 - 1 = 0.05. Future split at Feb 1 has zero effect.
    assert matrix.target_vector[0] == Decimal("0.05")


def test_postgres_multi_instrument_isolation(db_session: Session) -> None:
    """Corporate actions for different instruments in PostgreSQL remain strictly isolated."""
    ca_repo = PostgresCorporateActionRepository(db_session)
    inst_a = _setup_test_instrument(db_session, symbol="ISOLATIONA", isin="INE444444444")
    inst_b = _setup_test_instrument(db_session, symbol="ISOLATIONB", isin="INE555555555")

    # Split for A only
    split_a = CorporateAction(
        instrument_id=inst_a.instrument_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 1, 6),
        value=Decimal("2"),
    )
    # Dividend for B only: price drops 100 -> 80, div = 20
    # Reinvestment: (80 / 100) * (1 + 20 / 80) - 1 = 0.8 * 1.25 - 1 = 0.00 exactly
    div_b = CorporateAction(
        instrument_id=inst_b.instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 1, 6),
        value=Decimal("20"),
    )
    ca_repo.save(split_a)
    ca_repo.save(div_b)

    bars_a = [
        _make_bar(inst_a.instrument_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200")),
        _make_bar(inst_a.instrument_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100")),
    ]
    bars_b = [
        _make_bar(inst_b.instrument_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100")),
        _make_bar(inst_b.instrument_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("80")),
    ]

    engine = FeatureEngine(create_default_feature_registry())
    builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=ca_repo)

    features = [
        FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="1d SMA",
            parameters={"window": 1},
        )
    ]
    target = TargetDefinition(name="total_return", horizon_bars=1)

    matrix_a = builder.build_dataset(
        instrument_id=inst_a.instrument_id,
        bars=bars_a,
        features=features,
        target=target,
    )
    matrix_b = builder.build_dataset(
        instrument_id=inst_b.instrument_id,
        bars=bars_b,
        features=features,
        target=target,
    )

    # Instrument A receives 2:1 split -> return = 0.00
    assert len(matrix_a) == 1
    assert matrix_a.target_vector[0] == Decimal("0")

    # Instrument B receives 10 dividend on 90 close -> (90/100)*(1 + 10/90) - 1 = 0.00
    assert len(matrix_b) == 1
    assert matrix_b.target_vector[0] == Decimal("0")


def test_postgres_sequential_actions_and_query_bounding(db_session: Session) -> None:
    """Sequential split and dividend inside bar range are applied; actions after last bar are bounded out."""
    ca_repo = PostgresCorporateActionRepository(db_session)
    instrument = _setup_test_instrument(db_session, symbol="SEQTEST", isin="INE666666666")
    inst_id = instrument.instrument_id

    # Action 1: Split 2:1 on Jan 6 (inside horizon t0->t1)
    split = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 1, 6),
        value=Decimal("2"),
    )
    # Action 2: Dividend 5.00 on Jan 7 (inside horizon t1->t2)
    dividend = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.DIVIDEND,
        execution_date=date(2026, 1, 7),
        value=Decimal("5.00"),
        currency="INR",
    )
    # Action 3: Split 3:1 on Feb 1 (after the last available bar on Jan 7)
    future_split = CorporateAction(
        instrument_id=inst_id,
        action_type=CorporateActionType.SPLIT,
        execution_date=date(2026, 2, 1),
        value=Decimal("3"),
    )
    ca_repo.save(split)
    ca_repo.save(dividend)
    ca_repo.save(future_split)

    bars = [
        _make_bar(inst_id, datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200")),
        _make_bar(inst_id, datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100")),
        _make_bar(inst_id, datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("95")),
    ]

    engine = FeatureEngine(create_default_feature_registry())
    builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=ca_repo)

    features = [
        FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="1d SMA",
            parameters={"window": 1},
        )
    ]
    target = TargetDefinition(name="total_return", horizon_bars=1)

    matrix = builder.build_dataset(
        instrument_id=inst_id,
        bars=bars,
        features=features,
        target=target,
    )

    assert len(matrix) == 2
    # Row 0: obs=Jan 5 (200), real=Jan 6 (100) -> 2:1 split -> return = 0.00
    assert matrix.rows[0].observation_date == date(2026, 1, 5)
    assert matrix.rows[0].realization_date == date(2026, 1, 6)
    assert matrix.rows[0].target_value == Decimal("0")

    # Row 1: obs=Jan 6 (100), real=Jan 7 (95) -> 5.00 div on 95 -> return = 0.00
    assert matrix.rows[1].observation_date == date(2026, 1, 6)
    assert matrix.rows[1].realization_date == date(2026, 1, 7)
    assert matrix.rows[1].target_value == Decimal("0")
