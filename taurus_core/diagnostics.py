"""
Taurus Dashboard – Diagnostic des sources de données.

Quand une analyse retombe sur une source dégradée, l'utilisateur voit le
repli — « prix : Nasdaq Data » — sans savoir pourquoi la source préférée a été
écartée. Quota atteint, ticker inconnu, réseau coupé et clé d'API absente
produisent le même symptôme et appellent des réponses opposées.

Ce module interroge chaque source et rapporte ce qu'elle répond. Il ne
diagnostique rien de lui-même : il rend visible ce qui, sinon, se perd dans
les journaux du serveur.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from .config import DEFAULT_CONFIG, ValuationConfig
from .providers import http
from .providers.fundamentals import ticker_to_cik

logger = logging.getLogger(__name__)

# Yahoo limite le débit par adresse IP et la limite fluctue à la seconde : une
# tentative isolée ne prouve rien. Plusieurs essais espacés distinguent un
# quota transitoire d'une indisponibilité réelle.
YAHOO_ATTEMPTS = 3


def _interpretation(name: str, result: http.Probe, has_key: bool = True) -> str:
    """Traduit un code de retour en conseil actionnable."""
    if result.ok:
        return "Disponible."
    if not has_key:
        return "Aucune clé d'API configurée : source ignorée, ce qui est normal."
    if result.status is None:
        return (
            f"Pas de connexion ({result.error}) : réseau, pare-feu ou proxy "
            "d'entreprise."
        )
    if result.status == 429:
        return (
            "Quota de débit atteint pour votre adresse IP. La limite fluctue : "
            "réessayez dans quelques minutes, ou configurez une clé Financial "
            "Modeling Prep pour ne plus en dépendre."
        )
    if result.status in (401, 403):
        return "Accès refusé : clé d'API absente, expirée ou insuffisante."
    if result.status == 404:
        return "Ticker inconnu de cette source."
    if result.status >= 500:
        return "Panne du fournisseur : réessayez plus tard."
    return f"Réponse inattendue (HTTP {result.status})."


def run(ticker: str = "AAPL", cfg: ValuationConfig = DEFAULT_CONFIG) -> Dict:
    """Interroge toutes les sources et renvoie un rapport sérialisable."""
    symbol = (ticker or "AAPL").upper().strip()
    fmp_key = os.environ.get("FMP_API_KEY", "").strip()
    checks: List[Dict] = []

    def record(name: str, role: str, result: http.Probe, has_key: bool = True) -> None:
        checks.append({
            "name": name,
            "role": role,
            "ok": result.ok,
            "status": result.status,
            "latency_ms": result.latency_ms,
            "attempts": result.attempts,
            "error": result.error,
            "excerpt": result.excerpt,
            "interpretation": _interpretation(name, result, has_key),
        })

    # ── Cours ──────────────────────────────────────────────────────────── #
    record(
        "Yahoo Finance", "cours et places locales",
        http.probe(
            "yahoo",
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "5y", "interval": "1mo"},
            attempts=YAHOO_ATTEMPTS,
        ),
    )
    if fmp_key:
        record(
            "Financial Modeling Prep", "cours, fondamentaux, capitalisation",
            http.probe(
                "fmp",
                f"https://financialmodelingprep.com/api/v3/profile/{symbol}",
                params={"apikey": fmp_key},
            ),
        )
    else:
        checks.append({
            "name": "Financial Modeling Prep",
            "role": "cours, fondamentaux, capitalisation",
            "ok": False, "status": None, "latency_ms": None, "attempts": 0,
            "error": "", "excerpt": "",
            "interpretation": (
                "Aucune clé configurée. Facultatif, mais c'est la seule source "
                "de fondamentaux pour les sociétés ne déposant pas auprès de "
                "la SEC, et elle affranchit des quotas de Yahoo."
            ),
        })
    record(
        "Nasdaq Data", "cours de repli (sans dividendes)",
        http.probe(
            "nasdaq",
            f"https://api.nasdaq.com/api/quote/{symbol}/historical",
            params={"assetclass": "stocks", "fromdate": "2024-01-01",
                    "todate": "2026-01-01", "limit": 10},
        ),
    )

    # ── Comptes ────────────────────────────────────────────────────────── #
    cik = ticker_to_cik(symbol, cfg)
    record(
        "SEC EDGAR", "fondamentaux et dividendes",
        http.probe(
            "edgar",
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            if cik else "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": http.sec_user_agent()},
        ),
    )

    # ── Facteurs et change ─────────────────────────────────────────────── #
    record(
        "Kenneth R. French Data Library", "facteurs Fama-French",
        http.probe(
            "french",
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_CSV.zip",
            headers={"Accept": "*/*"},
        ),
    )
    record(
        "Banque centrale européenne", "taux de change",
        http.probe("bce", "https://api.frankfurter.app/latest",
                   params={"from": "EUR", "to": "USD"}),
    )

    essential = [c for c in checks if c["name"] in
                 ("SEC EDGAR", "Kenneth R. French Data Library")]
    price_sources = [c for c in checks if c["role"].startswith("cours")]

    return {
        "ticker": symbol,
        "checks": checks,
        "summary": {
            "essential_ok": all(c["ok"] for c in essential),
            "price_sources_ok": sum(1 for c in price_sources if c["ok"]),
            "price_sources_total": len(price_sources),
            "fmp_key_configured": bool(fmp_key),
        },
    }
