# Checks black_scholes_price against known textbook reference values
# (S=100, K=100, T=1yr, r=5%, sigma=20% -> call ~10.4506, put ~5.5735),
# verified by hand earlier this project. Also checks put-call parity holds
# and that an expired option collapses to intrinsic value.
from __future__ import annotations

import math

from lavish_core.reporting.option_estimator import black_scholes_price


def test_call_matches_textbook_reference():
    price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert round(price, 4) == 10.4506


def test_put_matches_textbook_reference():
    price = black_scholes_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert round(price, 4) == 5.5735


def test_put_call_parity_holds():
    S, K, T, r, sigma = 150.0, 140.0, 0.5, 0.04, 0.35
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    # C - P = S - K*e^(-rT)
    assert math.isclose(call - put, S - K * math.exp(-r * T), abs_tol=1e-9)


def test_expired_call_collapses_to_intrinsic_value():
    price = black_scholes_price(S=120, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
    assert price == 20.0


def test_expired_otm_put_is_worthless():
    price = black_scholes_price(S=120, K=100, T=0, r=0.05, sigma=0.2, option_type="put")
    assert price == 0.0
