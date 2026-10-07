"""Guard against accidental live yfinance network access in pytest."""

import pytest
import yfinance as yf


def test_yfinance_ticker_is_blocked_by_default():
    with pytest.raises(RuntimeError, match="External network access is disabled"):
        yf.Ticker("VOO")
