from app.domain.enums.asset_type import AssetType
from app.domain.enums.instrument_status import InstrumentStatus


def test_asset_type_values() -> None:
    assert AssetType.EQUITY == "equity"
    assert AssetType.ETF == "etf"
    assert AssetType.INDEX == "index"
    assert AssetType.BOND == "bond"


def test_instrument_status_values() -> None:
    assert InstrumentStatus.ACTIVE == "active"
    assert InstrumentStatus.INACTIVE == "inactive"
    assert InstrumentStatus.SUSPENDED == "suspended"
    assert InstrumentStatus.DELISTED == "delisted"


def test_asset_type_is_str_enum() -> None:
    assert isinstance(AssetType.EQUITY, str)


def test_instrument_status_is_str_enum() -> None:
    assert isinstance(InstrumentStatus.ACTIVE, str)
