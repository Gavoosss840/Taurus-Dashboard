"""
Tests de la reconstitution du rendement total.

Une série de cours non réajustée mesure un rendement en capital. Le manque
décale le t-stat de l'alpha de 0,07 (Apple) à 0,94 (Altria) et peut en inverser
le signe : il pénalise systématiquement les valeurs de rendement.
"""

import numpy as np
import pandas as pd
import pytest

from taurus_core.providers.fundamentals import (
    MIN_DIVIDEND_QUARTERS,
    _complete_fourth_quarter,
    _quarterly_dividends,
)
from taurus_core.total_return import (
    MAX_ANNUAL_YIELD,
    MIN_QUARTERS_IN_WINDOW,
    reconstruct,
)


def price_series(n_months: int = 72, start: float = 100.0,
                 monthly_growth: float = 0.005) -> pd.Series:
    index = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    return pd.Series(start * (1 + monthly_growth) ** np.arange(n_months), index=index)


def dividend_series(prices: pd.Series, quarterly_yield: float = 0.015) -> pd.Series:
    """Un versement tous les trois mois, à rendement constant."""
    quarters = prices.index[2::3]
    return pd.Series(prices.reindex(quarters) * quarterly_yield, index=quarters)


# --------------------------------------------------------------------------- #
#  Reconstitution                                                              #
# --------------------------------------------------------------------------- #

def test_total_return_exceeds_capital_return():
    prices = price_series()
    result = reconstruct(prices, dividend_series(prices))

    assert result is not None
    capital = prices.iloc[-1] / prices.iloc[0]
    total = result.prices.iloc[-1] / result.prices.iloc[0]
    assert total > capital


def test_reconstructed_yield_matches_the_dividends_paid():
    prices = price_series()
    result = reconstruct(prices, dividend_series(prices, quarterly_yield=0.015))

    assert result is not None
    assert result.annual_yield == pytest.approx(0.06, abs=0.005)


def test_added_performance_matches_the_yield():
    """Sur une série régulière, l'écart annualisé doit valoir le rendement."""
    prices = price_series(n_months=61)
    result = reconstruct(prices, dividend_series(prices, quarterly_yield=0.0125))

    assert result is not None
    years = (len(prices) - 1) / 12
    capital = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1
    total = (result.prices.iloc[-1] / result.prices.iloc[0]) ** (1 / years) - 1
    assert total - capital == pytest.approx(0.05, abs=0.01)


def test_first_price_is_preserved():
    """L'indice part du même niveau : seule sa pente change."""
    prices = price_series()
    result = reconstruct(prices, dividend_series(prices))
    assert result is not None
    assert result.prices.iloc[0] == pytest.approx(prices.iloc[0])


# --------------------------------------------------------------------------- #
#  Garde-fous                                                                  #
# --------------------------------------------------------------------------- #

def test_thin_coverage_is_refused():
    """Mieux vaut garder l'avertissement que reconstituer sur quatre versements."""
    prices = price_series()
    few = dividend_series(prices).iloc[: MIN_QUARTERS_IN_WINDOW - 1]
    assert reconstruct(prices, few) is None


def test_implausible_yield_is_refused():
    """Un « dividende » à 40 % par an n'en est pas un."""
    prices = price_series()
    assert reconstruct(prices, dividend_series(prices, quarterly_yield=0.10)) is None


def test_yield_at_the_ceiling_is_still_refused():
    prices = price_series()
    excessive = dividend_series(prices, quarterly_yield=MAX_ANNUAL_YIELD / 4 + 0.01)
    assert reconstruct(prices, excessive) is None


def test_missing_dividends_yield_nothing():
    prices = price_series()
    assert reconstruct(prices, None) is None
    assert reconstruct(prices, pd.Series(dtype=float)) is None
    assert reconstruct(None, dividend_series(prices)) is None


def test_foreign_currency_dividends_are_refused():
    """Pour un certificat de dépôt, le rapport à l'action ordinaire est inconnu.

    ASML publie en euros et cote en dollars : appliquer ses dividendes tels
    quels à son cours mêlerait deux devises, et un ratio ADR inconnu.
    """
    prices = price_series()
    assert reconstruct(prices, dividend_series(prices), currency_matches=False) is None


def test_a_split_artefact_is_clipped():
    """Les cours sont ajustés des divisions, les dividendes déclarés ne le sont pas.

    Une division dans la fenêtre quadruple le rapport D/P sur sa partie
    ancienne ; sans bornage, ce seul artefact dominerait le rendement.
    """
    prices = price_series()
    dividends = dividend_series(prices)
    dividends.iloc[:4] = dividends.iloc[:4] * 4.0     # période pré-division

    result = reconstruct(prices, dividends)
    assert result is not None
    assert result.clipped_periods == 4
    assert result.annual_yield == pytest.approx(0.06, abs=0.005)
    assert any("division" in note.lower() for note in result.notes)


def test_ordinary_variation_is_not_clipped():
    """Une hausse ordinaire du dividende doit passer sans être bornée."""
    prices = price_series()
    dividends = dividend_series(prices)
    dividends.iloc[-4:] = dividends.iloc[-4:] * 1.15

    result = reconstruct(prices, dividends)
    assert result is not None
    assert result.clipped_periods == 0


# --------------------------------------------------------------------------- #
#  Quatrième trimestre, absorbé par le rapport annuel                          #
# --------------------------------------------------------------------------- #

def fact(start: str, end: str, value: float) -> dict:
    return {"start": start, "end": end, "val": value, "filed": end}


def test_fourth_quarter_is_restored_from_the_annual_figure():
    """Trois trimestres en 10-Q, l'exercice entier en 10-K.

    Le quatrième versement n'a donc pas de période de 90 jours propre : il
    manquait un dividende sur quatre, soit un quart du rendement.
    """
    quarters = {
        ("2025-01-01", "2025-03-31"): fact("2025-01-01", "2025-03-31", 0.68),
        ("2025-04-01", "2025-06-30"): fact("2025-04-01", "2025-06-30", 0.68),
        ("2025-07-01", "2025-09-30"): fact("2025-07-01", "2025-09-30", 0.68),
    }
    annuals = {
        ("2025-01-01", "2025-12-31"): fact("2025-01-01", "2025-12-31", 2.735),
    }
    completed = _complete_fourth_quarter(quarters, annuals)

    assert len(completed) == 4
    fourth = max(completed, key=lambda f: f["end"])
    assert fourth["val"] == pytest.approx(2.735 - 3 * 0.68)
    assert fourth["end"] == "2025-12-31"


def test_a_complete_year_is_left_alone():
    quarters = {
        (f"2025-{m:02d}-01", f"2025-{m + 2:02d}-28"): fact(
            f"2025-{m:02d}-01", f"2025-{m + 2:02d}-28", 0.68,
        )
        for m in (1, 4, 7, 10)
    }
    annuals = {("2025-01-01", "2025-12-31"): fact("2025-01-01", "2025-12-31", 2.72)}
    assert len(_complete_fourth_quarter(quarters, annuals)) == 4


def test_an_implausible_residual_is_discarded():
    """Un résidu hors de proportion trahit un rapprochement douteux."""
    quarters = {
        ("2025-01-01", "2025-03-31"): fact("2025-01-01", "2025-03-31", 0.68),
        ("2025-04-01", "2025-06-30"): fact("2025-04-01", "2025-06-30", 0.68),
        ("2025-07-01", "2025-09-30"): fact("2025-07-01", "2025-09-30", 0.68),
    }
    for annual_value in (2.04, 1.50, 10.0):      # résidu nul, négatif, démesuré
        annuals = {("2025-01-01", "2025-12-31"): fact("2025-01-01", "2025-12-31", annual_value)}
        assert len(_complete_fourth_quarter(quarters, annuals)) == 3


# --------------------------------------------------------------------------- #
#  Extraction depuis EDGAR                                                     #
# --------------------------------------------------------------------------- #

def edgar_book(concept: str, quarters: int = 12, unit: str = "USD/shares") -> dict:
    entries = []
    for i in range(quarters):
        start = pd.Timestamp("2021-01-01") + pd.DateOffset(months=3 * i)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        entries.append(fact(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), 0.5))
    return {concept: {"units": {unit: entries}}}


def test_dividends_are_read_from_the_filing():
    series = _quarterly_dividends(edgar_book("CommonStockDividendsPerShareDeclared"))
    assert series is not None
    assert len(series) == 12
    assert (series == 0.5).all()


def test_too_few_quarters_yield_nothing():
    book = edgar_book("CommonStockDividendsPerShareDeclared",
                      quarters=MIN_DIVIDEND_QUARTERS - 1)
    assert _quarterly_dividends(book) is None


def test_reporting_currency_selects_the_unit():
    """ASML déclare ses dividendes en EUR/shares."""
    book = edgar_book("CommonStockDividendsPerShareDeclared", unit="EUR/shares")
    assert _quarterly_dividends(book, "USD") is None
    assert _quarterly_dividends(book, "EUR") is not None


def test_a_stale_concept_does_not_win():
    """Coca-Cola a cessé d'alimenter « Declared » en 2018.

    Suivre l'ordre de priorité des concepts renverrait un chiffre figé depuis
    des années.
    """
    stale = edgar_book("CommonStockDividendsPerShareDeclared", quarters=12)
    for entry in stale["CommonStockDividendsPerShareDeclared"]["units"]["USD/shares"]:
        entry["start"] = entry["start"].replace("202", "201")
        entry["end"] = entry["end"].replace("202", "201")
    fresh = edgar_book("CommonStockDividendsPerShareCashPaid", quarters=12)
    fresh["CommonStockDividendsPerShareCashPaid"]["units"]["USD/shares"] = [
        {**f, "val": 0.9}
        for f in fresh["CommonStockDividendsPerShareCashPaid"]["units"]["USD/shares"]
    ]

    series = _quarterly_dividends({**stale, **fresh})
    assert series is not None
    assert (series == 0.9).all()
