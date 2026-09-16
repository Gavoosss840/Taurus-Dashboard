"""Tests de l'extraction des fondamentaux SEC EDGAR.

Le point délicat : EDGAR mélange trimestres et exercices dans un même tableau,
et les entreprises abandonnent des concepts XBRL en cours de route sans cesser
de les exposer, figés sur leur dernière valeur.
"""

import pytest

from taurus_core.providers.fundamentals import (
    Fundamentals,
    _latest_instant,
    _pick_freshest,
    _ttm,
    _money_facts,
)
from taurus_core.providers.sectors import sector_from_sic


def quarter(start, end, val, filed="2026-05-01", form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def instant(end, val, filed="2026-05-01", form="10-Q"):
    return {"end": end, "val": val, "filed": filed, "form": form}


def facts(**concepts):
    return {
        name: {"units": {"USD": entries}} for name, entries in concepts.items()
    }


# ── Déduplication ────────────────────────────────────────────────────────

def test_latest_filing_wins_for_a_restated_period():
    """Un 10-K/A republie un trimestre : c'est le chiffre retraité qui compte."""
    data = facts(Assets=[
        instant("2026-03-31", 100.0, filed="2026-05-01"),
        instant("2026-03-31", 115.0, filed="2026-08-15"),   # retraitement
    ])
    value, end = _latest_instant(data, "Assets")
    assert value == 115.0
    assert end == "2026-03-31"


# ── Fraîcheur contre priorité ────────────────────────────────────────────

def test_freshest_concept_wins_over_priority():
    """Coca-Cola a cessé d'alimenter `LongTermDebt` en 2024.

    Suivre l'ordre de priorité renverrait un chiffre vieux de deux ans.
    """
    data = facts(
        LongTermDebt=[instant("2024-03-29", 36.5e9)],                    # figé
        LongTermDebtAndCapitalLeaseObligations=[instant("2026-04-03", 39.1e9)],
    )
    value, end = _latest_instant(
        data, "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations",
    )
    assert value == 39.1e9
    assert end == "2026-04-03"


def test_priority_wins_at_comparable_freshness():
    """À arrêtés proches, la définition comptable la plus pertinente l'emporte."""
    data = facts(
        StockholdersEquity=[instant("2026-03-31", 50.0e9)],
        OtherEquity=[instant("2026-04-03", 80.0e9)],       # 3 jours plus tard
    )
    value, _ = _latest_instant(data, "StockholdersEquity", "OtherEquity")
    assert value == 50.0e9


def test_pick_freshest_returns_none_without_candidates():
    assert _pick_freshest([]) is None


# ── Flux sur 12 mois glissants ───────────────────────────────────────────

def test_ttm_sums_four_consecutive_quarters():
    data = facts(Revenues=[
        quarter("2026-01-01", "2026-03-31", 25.0),
        quarter("2025-10-01", "2025-12-31", 30.0),
        quarter("2025-07-01", "2025-09-30", 22.0),
        quarter("2025-04-01", "2025-06-30", 23.0),
        quarter("2025-01-01", "2025-03-31", 20.0),   # 5e trimestre, à ignorer
    ])
    value, end = _ttm(data, "Revenues")
    assert value == pytest.approx(100.0)
    assert end == "2026-03-31"


def test_ttm_ignores_overlapping_ytd_periods():
    """Les 10-Q publient aussi des cumuls : les additionner doublerait les flux."""
    data = facts(Revenues=[
        quarter("2026-01-01", "2026-03-31", 25.0),
        quarter("2025-10-01", "2025-12-31", 30.0),
        quarter("2025-07-01", "2025-09-30", 22.0),
        quarter("2025-04-01", "2025-06-30", 23.0),
        quarter("2025-04-01", "2025-12-31", 75.0),   # cumul 9 mois
    ])
    value, _ = _ttm(data, "Revenues")
    assert value == pytest.approx(100.0)


def test_ttm_falls_back_to_the_annual_figure():
    data = facts(Revenues=[
        quarter("2025-01-01", "2025-12-31", 98.0, form="10-K"),
    ])
    value, end = _ttm(data, "Revenues")
    assert value == pytest.approx(98.0)
    assert end == "2025-12-31"


def test_ttm_annualises_a_partial_year_as_last_resort():
    data = facts(Revenues=[
        quarter("2026-01-01", "2026-06-30", 50.0),   # 180 jours
    ])
    value, _ = _ttm(data, "Revenues")
    assert value == pytest.approx(50.0 * 365 / 180, rel=1e-6)


def test_ttm_prefers_the_concept_with_recent_data():
    """NextEra a abandonné `Revenues` en 2013 au profit d'un autre concept."""
    data = facts(
        Revenues=[quarter("2013-07-01", "2013-09-30", 4.0, filed="2013-11-01")],
        RevenueFromContractWithCustomerIncludingAssessedTax=[
            quarter("2026-01-01", "2026-03-31", 7.0),
            quarter("2025-10-01", "2025-12-31", 6.0),
            quarter("2025-07-01", "2025-09-30", 8.0),
            quarter("2025-04-01", "2025-06-30", 5.0),
        ],
    )
    value, end = _ttm(
        data, "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
    )
    assert value == pytest.approx(26.0)
    assert end == "2026-03-31"


def test_ttm_returns_nan_without_any_usable_period():
    value, end = _ttm(facts(Revenues=[]), "Revenues")
    assert value != value          # NaN
    assert end == ""


def test_usd_facts_skips_entries_without_value():
    data = facts(Assets=[
        instant("2026-03-31", None),
        instant("2025-12-31", 90.0),
    ])
    entries = _money_facts(data, "Assets")
    assert len(entries) == 1
    assert entries[0]["val"] == 90.0


# ── Champs manquants ─────────────────────────────────────────────────────

def test_missing_fields_are_reported():
    data = Fundamentals(ticker="TEST")
    missing = data.missing_fields()
    assert set(missing) == {"total_debt", "total_equity", "ebit", "cash"}


def test_complete_fundamentals_report_nothing_missing():
    data = Fundamentals(
        ticker="TEST", total_debt=1.0, total_equity=2.0, ebit=3.0, cash=4.0,
    )
    assert data.missing_fields() == []


def test_net_debt_is_debt_minus_cash():
    data = Fundamentals(ticker="TEST", total_debt=30.0, cash=10.0)
    assert data.net_debt == 20.0


# ── Secteurs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sic,expected", [
    (3571, "Information Technology"),    # ordinateurs
    (7372, "Information Technology"),    # logiciels
    (2834, "Health Care"),               # pharmacie
    (4911, "Utilities"),                 # électricité
    (6022, "Financials"),                # banques
    (6798, "Real Estate"),               # REIT
    (1311, "Energy"),                    # pétrole et gaz
    (5812, "Consumer Discretionary"),    # restauration
    (2011, "Consumer Staples"),          # alimentaire
    (2840, "Consumer Staples"),          # savons et cosmétiques (Procter & Gamble)
    (2844, "Consumer Staples"),          # produits de toilette
    (2860, "Materials"),                 # chimie organique industrielle
    (3711, "Consumer Discretionary"),    # automobile
    (3721, "Industrials"),               # aéronautique
])
def test_sic_maps_to_gics_sector(sic, expected):
    assert sector_from_sic(sic) == expected


@pytest.mark.parametrize("sic", [None, "", "abc", 0, 9999])
def test_unmappable_sic_is_unknown(sic):
    assert sector_from_sic(sic) == "Unknown"
