"""
Tests de la couverture internationale.

Trois pièges spécifiques aux titres étrangers, chacun capable de produire un
verdict faux sans rien signaler :

  • une société dépose auprès de la SEC mais publie dans une autre devise ;
  • un certificat de dépôt (ADR) ne représente pas une action ordinaire ;
  • un titre européen ou japonais ne se régresse pas sur les facteurs
    américains.
"""

import numpy as np
import pandas as pd
import pytest

from taurus_core.config import ValuationConfig
from taurus_core.providers import fx, quotes, regions
from taurus_core.providers.factors import FACTOR_FILES, FACTOR_LABELS, factor_label
from taurus_core.providers.fundamentals import reporting_currency
from taurus_core.providers.sectors import sector_from_sic

CFG = ValuationConfig()


# --------------------------------------------------------------------------- #
#  Devise de publication                                                       #
# --------------------------------------------------------------------------- #

def book(**units_by_concept) -> dict:
    """Construit un bloc de faits XBRL à partir de {concept: {unité: n}}."""
    return {
        concept: {"units": {unit: [{"val": 1, "end": "2026-03-31"}] * count
                            for unit, count in units.items()}}
        for concept, units in units_by_concept.items()
    }


def test_reporting_currency_detects_a_foreign_filer():
    """ASML dépose auprès de la SEC mais tient ses comptes en euros.

    Lire USD en dur — ce que faisait le module — renvoyait des comptes vides
    pour tout émetteur privé étranger, alors que ses chiffres étaient là.
    """
    facts = book(Assets={"EUR": 40}, Revenue={"EUR": 30}, Equity={"EUR": 25})
    assert reporting_currency(facts) == "EUR"


def test_reporting_currency_defaults_to_usd():
    assert reporting_currency(book(Assets={"USD": 10})) == "USD"
    assert reporting_currency({}) == "USD"


def test_reporting_currency_ignores_non_monetary_units():
    """« shares » et « pure » ne sont pas des devises."""
    facts = book(
        Shares={"shares": 90},
        Ratio={"pure": 80},
        Assets={"JPY": 10},
    )
    assert reporting_currency(facts) == "JPY"


def test_reporting_currency_ignores_per_share_units():
    facts = book(Eps={"EUR/shares": 50}, Assets={"EUR": 10})
    assert reporting_currency(facts) == "EUR"


# --------------------------------------------------------------------------- #
#  Sous-unités de cotation                                                     #
# --------------------------------------------------------------------------- #

def test_london_prices_are_quoted_in_pence():
    """Yahoo cote Londres en « GBp ». Ignorer ce point divise tout par cent."""
    currency, factor = fx.normalise_currency("GBp")
    assert currency == "GBP"
    assert factor == 0.01


def test_ordinary_currency_is_untouched():
    assert fx.normalise_currency("EUR") == ("EUR", 1.0)
    assert fx.normalise_currency("") == ("USD", 1.0)


# --------------------------------------------------------------------------- #
#  Conversion de change                                                        #
# --------------------------------------------------------------------------- #

def test_same_currency_needs_no_conversion():
    series = pd.Series([1.0, 2.0])
    assert fx.convert_series(series, "USD", "USD", CFG) is series
    assert fx.monthly_rates("USD", "USD", cfg=CFG) is None
    assert fx.latest_rate("EUR", "EUR", CFG) == 1.0


def test_unsupported_pair_returns_none_rather_than_assuming_parity():
    """La BCE ne publie pas le dollar de Taïwan.

    Supposer 1,0 comparerait une capitalisation en dollars à des comptes en
    TWD — un facteur trente d'écart, silencieusement.
    """
    assert fx.monthly_rates("TWD", "USD", cfg=CFG) is None
    assert fx.latest_rate("TWD", "USD", CFG) is None


def test_conversion_applies_the_rate(monkeypatch):
    index = pd.date_range("2025-01-31", periods=4, freq="ME")
    prices = pd.Series([100.0, 110.0, 120.0, 130.0], index=index)
    rates = pd.Series([1.1, 1.2, 1.1, 1.05], index=index)
    monkeypatch.setattr(fx, "monthly_rates", lambda *a, **k: rates)

    converted = fx.convert_series(prices, "EUR", "USD", CFG)
    assert converted is not None
    assert list(converted) == pytest.approx([110.0, 132.0, 132.0, 136.5])


def test_conversion_carries_rates_over_gaps(monkeypatch):
    """Un trou ponctuel dans la série BCE ne doit pas amputer l'historique."""
    index = pd.date_range("2025-01-31", periods=4, freq="ME")
    prices = pd.Series([100.0] * 4, index=index)
    rates = pd.Series([1.1, np.nan, np.nan, 1.2], index=index)
    monkeypatch.setattr(fx, "monthly_rates", lambda *a, **k: rates)

    converted = fx.convert_series(prices, "EUR", "USD", CFG)
    assert converted is not None
    assert len(converted) == 4


# --------------------------------------------------------------------------- #
#  Région et facteurs                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ticker,expected", [
    ("MC.PA", regions.EUROPE),
    ("ASML.AS", regions.EUROPE),
    ("SAP.DE", regions.EUROPE),
    ("SHEL.L", regions.EUROPE),
    ("7203.T", regions.JAPAN),
    ("0700.HK", regions.ASIA_PACIFIC),
    ("BHP.AX", regions.ASIA_PACIFIC),
    ("005930.KS", regions.ASIA_PACIFIC),
    ("RELIANCE.NS", regions.EMERGING),
    ("PETR4.SA", regions.EMERGING),
    ("SHOP.TO", regions.NORTH_AMERICA),
])
def test_suffix_determines_the_region(ticker, expected):
    guess = regions.detect_region(ticker)
    assert guess.region == expected
    assert guess.evidence == "suffixe"
    assert guess.confident


def test_filing_country_is_used_without_a_suffix():
    guess = regions.detect_region("ASML", country="Netherlands", currency="EUR")
    assert guess.region == regions.EUROPE
    assert guess.evidence == "pays"


def test_currency_is_the_last_resort():
    """TSMC n'a pas de pays renseigné chez EDGAR ; sa devise le trahit."""
    guess = regions.detect_region("TSM", country="", currency="TWD")
    assert guess.region == regions.ASIA_PACIFIC
    assert guess.evidence == "devise"


def test_a_us_listed_dollar_filer_is_a_confident_north_american():
    guess = regions.detect_region("AAPL", country="", currency="USD")
    assert guess.region == regions.NORTH_AMERICA
    assert guess.confident


def test_a_foreign_currency_guess_is_flagged_uncertain():
    """Une région devinée sur la seule devise mérite d'être signalée."""
    guess = regions.detect_region("SOMEADR", country="", currency="EUR")
    assert guess.region == regions.EUROPE
    assert not guess.confident


def test_unknown_signals_fall_back_to_north_america():
    guess = regions.detect_region("XYZ")
    assert guess.region == regions.NORTH_AMERICA
    assert guess.evidence == "défaut"


def test_every_region_has_a_factor_file_and_a_label():
    for region in FACTOR_FILES:
        assert region in FACTOR_LABELS
        assert factor_label(region)
    assert set(FACTOR_FILES) == set(regions.REGION_LABELS)


def test_unknown_region_label_falls_back():
    assert factor_label("mars") == FACTOR_LABELS["north_america"]


# --------------------------------------------------------------------------- #
#  Capitalisation et secteur du titre coté                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("250,650,592,821", 250650592821.0),
    ("$1,625.42", 1625.42),
    ("1000", 1000.0),
    ("N/A", None),
    ("", None),
    (None, None),
    ("0", None),
    ("abc", None),
])
def test_amount_parsing(raw, expected):
    assert quotes._parse_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Technology", "Information Technology"),
    ("Health Care", "Health Care"),
    ("Consumer Non-Durables", "Consumer Staples"),
    ("Public Utilities", "Utilities"),
    ("Finance", "Financials"),
    ("Basic Industries", "Materials"),
    (None, "Unknown"),
    ("something else", "Unknown"),
])
def test_sector_labels_are_mapped_to_gics(raw, expected):
    assert quotes.normalise_sector(raw) == expected


def test_semiconductor_equipment_is_technology():
    """Le code SIC d'ASML (3559) le rangeait dans les machines industrielles."""
    assert sector_from_sic(3559) == "Information Technology"


# --------------------------------------------------------------------------- #
#  Dividendes : le drapeau de rendement total ne doit jamais être optimiste    #
# --------------------------------------------------------------------------- #
# Un cours sans dividendes décale le t-stat de l'alpha de 0,04 (Alphabet) à
# 0,81 (Altria). Le biais va toujours dans le même sens et croît avec le
# rendement du titre : il pénalise systématiquement les valeurs de rendement.
# Un drapeau optimiste supprimerait l'avertissement sans supprimer le biais.

from datetime import datetime, timezone   # noqa: E402

from taurus_core.providers import prices as prices_provider   # noqa: E402

WINDOW_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 1, 1, tzinfo=timezone.utc)


def yahoo_payload(with_adjclose: bool) -> dict:
    timestamps = [int(pd.Timestamp(f"2020-{m:02d}-28").timestamp())
                  for m in range(1, 13)]
    timestamps += [int(pd.Timestamp(f"2021-{m:02d}-28").timestamp())
                   for m in range(1, 13)]
    timestamps += [int(pd.Timestamp(f"2022-{m:02d}-28").timestamp())
                   for m in range(1, 13)]
    closes = [100.0 + i for i in range(len(timestamps))]
    indicators = {"quote": [{"close": closes}]}
    if with_adjclose:
        indicators["adjclose"] = [{"adjclose": closes}]
    return {
        "chart": {"result": [{
            "timestamp": timestamps,
            "indicators": indicators,
            "meta": {"currency": "USD", "regularMarketPrice": closes[-1]},
        }]}
    }


def test_yahoo_with_adjusted_closes_is_total_return(monkeypatch):
    monkeypatch.setattr(prices_provider.http, "get_json",
                        lambda *a, **k: yahoo_payload(True))
    history = prices_provider._from_yahoo("TEST", WINDOW_START, WINDOW_END)
    assert history is not None
    assert history.total_return is True


def test_yahoo_falling_back_to_raw_closes_is_flagged(monkeypatch):
    """Sans série `adjclose`, Yahoo ne fournit qu'un rendement en capital."""
    monkeypatch.setattr(prices_provider.http, "get_json",
                        lambda *a, **k: yahoo_payload(False))
    history = prices_provider._from_yahoo("TEST", WINDOW_START, WINDOW_END)
    assert history is not None
    assert history.total_return is False
    assert "bruts" in history.source


def fmp_payload(with_adjclose: bool) -> dict:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    rows = []
    for i, day in enumerate(dates):
        row = {"date": day.strftime("%Y-%m-%d"), "close": 100.0 + i}
        if with_adjclose:
            row["adjClose"] = 100.0 + i
        rows.append(row)
    return {"historical": rows}


def test_fmp_with_adjusted_closes_is_total_return(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "clé-de-test")
    monkeypatch.setattr(prices_provider.http, "get_json",
                        lambda *a, **k: fmp_payload(True))
    history = prices_provider._from_fmp("TEST", WINDOW_START, WINDOW_END)
    assert history is not None
    assert history.total_return is True


def test_fmp_falling_back_to_raw_closes_is_flagged(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "clé-de-test")
    monkeypatch.setattr(prices_provider.http, "get_json",
                        lambda *a, **k: fmp_payload(False))
    history = prices_provider._from_fmp("TEST", WINDOW_START, WINDOW_END)
    assert history is not None
    assert history.total_return is False


def test_sources_without_documented_adjustment_are_flagged(monkeypatch):
    """marketdata.app ne documente aucun ajustement : on suppose des cours bruts.

    Mieux vaut un avertissement de trop qu'un biais silencieux sur l'alpha.
    """
    timestamps = [int(pd.Timestamp(f"2021-{m:02d}-28").timestamp())
                  for m in range(1, 13)]
    timestamps += [int(pd.Timestamp(f"2022-{m:02d}-28").timestamp())
                   for m in range(1, 13)]
    timestamps += [int(pd.Timestamp(f"2023-{m:02d}-28").timestamp())
                   for m in range(1, 13)]
    monkeypatch.setattr(
        prices_provider.http, "get_json",
        lambda *a, **k: {"s": "ok", "t": timestamps,
                         "c": [100.0 + i for i in range(len(timestamps))]},
    )
    history = prices_provider._from_marketdata("TEST", WINDOW_START, WINDOW_END)
    assert history is not None
    assert history.total_return is False


# --------------------------------------------------------------------------- #
#  En-têtes HTTP : deux refus déterministes, pris pour des quotas              #
# --------------------------------------------------------------------------- #

def test_browser_user_agent_is_not_the_blocked_one():
    """Yahoo refuse par un HTTP 429 certaines chaînes de User-Agent répandues.

    Celle par défaut de nombreux scripts — « Macintosh; Intel Mac OS X
    10_15_7 … Chrome/124.0.0.0 » — en fait partie. Le refus est déterministe,
    malgré le code renvoyé : la même requête, au même instant, passe avec une
    autre chaîne. La conséquence était lourde, Yahoo étant le seul fournisseur
    couvrant les places locales.
    """
    from taurus_core.providers.http import _BROWSER_UA

    assert "Macintosh" not in _BROWSER_UA
    assert "Mozilla/5.0" in _BROWSER_UA


def test_binary_downloads_do_not_ask_for_json(monkeypatch):
    """L'en-tête `Accept` de la session vaut un 406 sur un fichier ZIP."""
    captured = {}

    class Response:
        status_code = 200
        content = b"zip"

        def raise_for_status(self):
            pass

    def capture(url, headers=None, timeout=None, **kw):
        captured["headers"] = headers or {}
        return Response()

    monkeypatch.setattr(prices_provider.http.session(), "get", capture)
    prices_provider.http.get_bytes("https://example.invalid/fichier.zip")
    assert captured["headers"].get("Accept") == "*/*"


def test_the_yahoo_token_is_requested_as_plain_text(monkeypatch):
    """Le jeton est du texte brut : demander du JSON vaut un 406."""
    from taurus_core.providers import http as http_module

    seen = []

    class Response:
        status_code = 200
        text = "abc123"

    def capture(url, headers=None, timeout=None, **kw):
        seen.append((url, headers or {}))
        return Response()

    monkeypatch.setattr(http_module.session(), "get", capture)
    monkeypatch.setattr(http_module, "_YAHOO_CRUMB", None)
    monkeypatch.setattr(http_module, "_YAHOO_CRUMB_TRIED", False)

    assert http_module.yahoo_crumb(force=True) == "abc123"
    crumb_call = next(h for url, h in seen if "getcrumb" in url)
    assert crumb_call.get("Accept") == "*/*"


def test_an_html_answer_is_not_mistaken_for_a_token(monkeypatch):
    """Une page de refus ne doit pas être prise pour un jeton valide."""
    from taurus_core.providers import http as http_module

    class Response:
        status_code = 200
        text = "<!DOCTYPE html><html>refus</html>"

    monkeypatch.setattr(http_module.session(), "get",
                        lambda *a, **k: Response())
    assert http_module.yahoo_crumb(force=True) is None


def test_quotes_give_up_without_a_token(monkeypatch):
    """`quoteSummary` exige le jeton : sans lui, inutile d'appeler."""
    from taurus_core.providers import quotes

    monkeypatch.setattr(quotes.http, "yahoo_crumb", lambda: None)
    monkeypatch.setattr(quotes.http, "get_json",
                        lambda *a, **k: pytest.fail("appel inutile"))
    assert quotes._from_yahoo("AAPL") is None
