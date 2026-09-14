"""Tests du pilier momentum 12-1."""

import numpy as np
import pandas as pd
import pytest

from taurus_core.config import ValuationConfig
from taurus_core.momentum import compute_momentum

CFG = ValuationConfig()


def price_series(monthly_returns, start: float = 100.0) -> pd.Series:
    """Série de cours mensuels engendrée par une suite de rendements."""
    index = pd.date_range("2020-01-31", periods=len(monthly_returns) + 1, freq="ME")
    prices = [start]
    for r in monthly_returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices, index=index)


def flat_market(index, rate: float = 0.0) -> pd.Series:
    return pd.Series(rate, index=index)


# ── Fenêtre de mesure ────────────────────────────────────────────────────

def test_skips_the_most_recent_month():
    """Le mois le plus récent doit être exclu de la mesure.

    C'est le cœur du 12-1 : les titres qui viennent de bondir refluent le mois
    suivant (retournement de court terme), et les inclure pollue le signal.
    """
    # 24 mois calmes, puis un dernier mois à +50 % qui doit être ignoré.
    returns = [0.01] * 24 + [0.50]
    prices = price_series(returns)
    result = compute_momentum(prices, flat_market(prices.index), CFG)

    assert result is not None
    # La fenêtre 12-1 couvre 12 relevés de cours, soit 11 rendements mensuels
    # — la convention de Jegadeesh & Titman, reprise telle quelle de l'algo.
    assert result.raw == pytest.approx(1.01 ** 11 - 1, rel=1e-9)
    assert result.window_end == str(prices.index[-2].date())


def test_measures_exactly_twelve_months():
    returns = [0.0] * 10 + [0.02] * 12 + [0.0] * 2
    prices = price_series(returns)
    result = compute_momentum(prices, flat_market(prices.index), CFG)

    assert result is not None
    assert result.n_months == 12
    # La fenêtre couvre les 12 mois s'achevant un mois avant la fin.
    span = (prices.index.get_loc(pd.Timestamp(result.window_end))
            - prices.index.get_loc(pd.Timestamp(result.window_start)))
    assert span == 11


# ── Ajustement de la volatilité ──────────────────────────────────────────

def test_volatility_adjustment_favours_the_steadier_stock():
    """Un titre régulier doit primer sur un titre erratique à rendement égal.

    Barroso & Santa-Clara (2015) : diviser par la volatilité réalisée réduit
    de moitié la sévérité des krachs de momentum.
    """
    # Deux titres qui gagnent autant, avec des volatilités très différentes :
    # ±3 % autour de +2 % par mois (≈10 % annualisé) contre ±15 % (≈52 %).
    steady = price_series([0.050, -0.010] * 7)
    erratic = price_series([0.175, -0.125] * 7)

    market = flat_market(steady.index)
    steady_result = compute_momentum(steady, market, CFG)
    erratic_result = compute_momentum(erratic, market, CFG)

    assert steady_result is not None and erratic_result is not None
    # Le titre erratique gagne davantage en brut…
    assert erratic_result.raw > steady_result.raw
    # …mais perd une fois le risque pris en compte.
    assert steady_result.sharpe > erratic_result.sharpe


def test_raw_momentum_used_when_adjustment_disabled():
    cfg = ValuationConfig(vol_adjust_momentum=False)
    prices = price_series([0.015] * 14)
    result = compute_momentum(prices, flat_market(prices.index), cfg)

    assert result is not None
    assert result.sharpe == pytest.approx(result.raw)


# ── Comparaison au marché ────────────────────────────────────────────────

def test_excess_is_positive_when_stock_beats_market():
    # Le marché suit la même trajectoire, diminuée de 2 points par mois :
    # volatilité identique, rendement inférieur, donc momentum inférieur.
    rng = np.random.default_rng(21)
    steps = list(rng.normal(0.030, 0.035, 14))
    prices = price_series(steps)
    market = pd.Series([0.0] + [s - 0.02 for s in steps], index=prices.index)

    result = compute_momentum(prices, market, CFG)
    assert result is not None
    assert result.excess_sharpe > 0
    assert result.direction == 1


def test_excess_is_negative_when_stock_lags_market():
    rng = np.random.default_rng(22)
    steps = list(rng.normal(0.005, 0.035, 14))
    prices = price_series(steps)
    market = pd.Series([0.0] + [s + 0.02 for s in steps], index=prices.index)

    result = compute_momentum(prices, market, CFG)
    assert result is not None
    assert result.excess_sharpe < 0
    assert result.direction == -1


def test_lagging_factors_compare_identical_windows():
    """Les facteurs Fama-French ont un à deux mois de retard sur les cours.

    Comparer un momentum du titre sur 12 mois à un momentum de marché sur 11
    fausserait l'écart d'environ un mois de performance de marché.
    """
    rng = np.random.default_rng(33)
    steps = list(rng.normal(0.02, 0.03, 14))
    prices = price_series(steps)
    # Le titre et le marché suivent exactement la même trajectoire.
    market = pd.Series([0.0] + steps, index=prices.index)
    truncated = market.iloc[:-2]     # les facteurs s'arrêtent deux mois plus tôt

    result = compute_momentum(prices, truncated, CFG)

    assert result is not None
    # Mesurés sur les mêmes mois, les deux momentums doivent coïncider.
    assert result.excess_sharpe == pytest.approx(0.0, abs=1e-6)


# ── Régime de krach ──────────────────────────────────────────────────────

def test_crash_regime_detected_on_volatility_spike():
    prices = price_series([0.01] * 14)
    market = pd.Series(0.004, index=prices.index)
    market.iloc[-1] = -0.28            # choc brutal sur le dernier mois

    result = compute_momentum(prices, market, CFG)
    assert result is not None
    assert result.crash_regime is True


def test_no_crash_regime_in_calm_market():
    prices = price_series([0.01] * 14)
    rng = np.random.default_rng(5)
    market = pd.Series(rng.normal(0.006, 0.03, len(prices)), index=prices.index)

    result = compute_momentum(prices, market, CFG)
    assert result is not None
    assert result.crash_regime is False


def test_crash_detection_can_be_disabled():
    cfg = ValuationConfig(momentum_crash_dampen=False)
    prices = price_series([0.01] * 14)
    market = pd.Series(0.004, index=prices.index)
    market.iloc[-1] = -0.28

    result = compute_momentum(prices, market, cfg)
    assert result is not None
    assert result.crash_regime is False


# ── Données dégradées ────────────────────────────────────────────────────

def test_short_history_is_refused():
    prices = price_series([0.01] * 8)
    assert compute_momentum(prices, flat_market(prices.index), CFG) is None


def test_empty_series_is_refused():
    assert compute_momentum(pd.Series(dtype=float), pd.Series(dtype=float), CFG) is None
    assert compute_momentum(None, pd.Series(dtype=float), CFG) is None


def test_unavailable_market_leaves_excess_undefined():
    prices = price_series([0.02] * 14)
    empty_market = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    result = compute_momentum(prices, empty_market, CFG)
    assert result is not None
    assert np.isnan(result.excess_sharpe)
    assert result.direction == 0


def test_frozen_price_series_does_not_explode():
    """Un cours figé donnerait un momentum-Sharpe de l'ordre de 1e16.

    La volatilité calculée sur une série quasi constante n'est que du bruit
    d'arrondi ; diviser par elle sature le verdict à lui seul.
    """
    prices = price_series([0.01] * 14)   # rendements constants → volatilité ~0
    market = pd.Series(0.01, index=prices.index)

    result = compute_momentum(prices, market, CFG)

    assert result is not None
    assert abs(result.sharpe) < 100
    assert abs(result.excess_sharpe) < 100
