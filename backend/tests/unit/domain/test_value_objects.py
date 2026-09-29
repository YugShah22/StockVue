import pytest

from app.domain.value_objects.exchange_code import ExchangeCode
from app.domain.value_objects.isin import ISIN
from app.domain.value_objects.symbol import Symbol

# -------------------------
# Symbol
# -------------------------


def test_symbol_normalizes_to_uppercase() -> None:
    symbol = Symbol("reliance")

    assert symbol.value == "RELIANCE"


def test_symbol_str_returns_value() -> None:
    symbol = Symbol("TCS")

    assert str(symbol) == "TCS"


def test_symbol_rejects_empty_value() -> None:
    with pytest.raises(ValueError):
        Symbol("")


def test_symbol_rejects_whitespace_only_value() -> None:
    with pytest.raises(ValueError):
        Symbol("   ")


def test_symbol_strips_whitespace() -> None:
    symbol = Symbol("  TCS  ")

    assert symbol.value == "TCS"


# -------------------------
# ISIN
# -------------------------


def test_isin_normalizes_to_uppercase() -> None:
    isin = ISIN("ine467b01029")

    assert isin.value == "INE467B01029"


def test_isin_str_returns_value() -> None:
    isin = ISIN("INE467B01029")

    assert str(isin) == "INE467B01029"


def test_isin_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        ISIN("INVALID")


def test_isin_rejects_empty_value() -> None:
    with pytest.raises(ValueError):
        ISIN("")


# -------------------------
# ExchangeCode
# -------------------------


def test_exchange_code_normalizes_to_uppercase() -> None:
    exchange = ExchangeCode("nse")

    assert exchange.value == "NSE"


def test_exchange_code_str_returns_value() -> None:
    exchange = ExchangeCode("NSE")

    assert str(exchange) == "NSE"


def test_exchange_code_rejects_empty_value() -> None:
    with pytest.raises(ValueError):
        ExchangeCode("")


def test_exchange_code_rejects_whitespace_only_value() -> None:
    with pytest.raises(ValueError):
        ExchangeCode("   ")


def test_exchange_code_strips_whitespace() -> None:
    exchange = ExchangeCode("  nse  ")

    assert exchange.value == "NSE"
