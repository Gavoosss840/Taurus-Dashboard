"""
Taurus Dashboard – Capitalisation boursière et secteur du titre coté.

Pourquoi ne pas reconstituer la capitalisation
──────────────────────────────────────────────
Le calcul naturel — actions en circulation × dernier cours — est faux pour un
certificat de dépôt américain (ADR). Un ADR Toyota représente dix actions
ordinaires : le nombre d'actions publié à la SEC est celui des ordinaires,
tandis que le cours est celui de l'ADR. Le produit des deux surestime la
capitalisation d'un facteur dix. Toyota ressortirait à 2 500 milliards de
dollars au lieu de 250, et l'écran Modigliani-Miller le déclarerait
massivement sur-évalué sans que rien ne le signale.

Le rapport ADR / action ordinaire n'est publié nulle part de façon
exploitable. On interroge donc directement une source qui connaît le titre
coté, ce qui règle aussi le cas des sociétés dont les comptes sont libellés
dans une autre devise que leur cotation.

Chaîne de repli : Financial Modeling Prep (si clé) → Yahoo Finance →
Nasdaq Data. Aucune n'est indispensable ; en dernier recours, l'appelant
reconstitue la capitalisation et signale son incertitude.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from .. import cache
from ..config import DEFAULT_CONFIG, ValuationConfig
from . import http

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
NASDAQ_BASE = "https://api.nasdaq.com/api"

# Les fournisseurs emploient leurs propres libellés sectoriels ; le moteur
# raisonne dans la nomenclature GICS, qui pilote les coûts de détresse.
_SECTOR_ALIASES = {
    "technology": "Information Technology",
    "information technology": "Information Technology",
    "telecommunications": "Communication Services",
    "communication services": "Communication Services",
    "health care": "Health Care",
    "healthcare": "Health Care",
    "consumer discretionary": "Consumer Discretionary",
    "consumer services": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "consumer non-durables": "Consumer Staples",
    "consumer durables": "Consumer Discretionary",
    "industrials": "Industrials",
    "industrial goods": "Industrials",
    "capital goods": "Industrials",
    "basic materials": "Materials",
    "basic industries": "Materials",
    "materials": "Materials",
    "energy": "Energy",
    "finance": "Financials",
    "financials": "Financials",
    "financial services": "Financials",
    "real estate": "Real Estate",
    "utilities": "Utilities",
    "public utilities": "Utilities",
    "miscellaneous": "Unknown",
}


def normalise_sector(label: Optional[str]) -> str:
    """Traduit un libellé sectoriel de fournisseur en nomenclature GICS."""
    if not label:
        return "Unknown"
    return _SECTOR_ALIASES.get(str(label).strip().lower(), "Unknown")


@dataclass
class Quote:
    """Données du titre tel qu'il est coté."""

    market_cap: float
    currency: str = "USD"
    sector: str = "Unknown"
    company_name: str = ""
    source: str = ""


def _parse_amount(value: object) -> Optional[float]:
    """Convertit "250,650,592,821" ou "$1,625.42" en flottant."""
    if value is None:
        return None
    text = str(value).replace("$", "").replace(",", "").replace(" ", "").strip()
    if not text or text.upper() in {"N/A", "NA", "--"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


# --------------------------------------------------------------------------- #
#  Fournisseur 1 : Financial Modeling Prep                                     #
# --------------------------------------------------------------------------- #

def _from_fmp(ticker: str) -> Optional[Quote]:
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        return None

    payload = http.get_json(f"{FMP_BASE}/profile/{ticker}", params={"apikey": api_key})
    if not isinstance(payload, list) or not payload:
        return None

    info = payload[0]
    market_cap = _parse_amount(info.get("mktCap"))
    if market_cap is None:
        return None
    return Quote(
        market_cap=market_cap,
        currency=str(info.get("currency") or "USD").upper(),
        sector=normalise_sector(info.get("sector")),
        company_name=str(info.get("companyName") or "").strip(),
        source="Financial Modeling Prep",
    )


# --------------------------------------------------------------------------- #
#  Fournisseur 2 : Yahoo Finance                                               #
# --------------------------------------------------------------------------- #

def _from_yahoo(ticker: str) -> Optional[Quote]:
    # Contrairement à `chart`, ce point d'entrée exige un couple cookie/jeton.
    crumb = http.yahoo_crumb()
    if not crumb:
        return None

    for host in YAHOO_HOSTS:
        payload = http.get_json(
            f"{host}/v10/finance/quoteSummary/{ticker}",
            params={"modules": "price,assetProfile", "crumb": crumb},
            retries=2,
        )
        if not isinstance(payload, dict):
            continue
        results = ((payload.get("quoteSummary") or {}).get("result")) or []
        if not results:
            continue

        block = results[0]
        price_block = block.get("price") or {}
        market_cap = _parse_amount((price_block.get("marketCap") or {}).get("raw"))
        if market_cap is None:
            continue

        profile = block.get("assetProfile") or {}
        return Quote(
            market_cap=market_cap,
            currency=str(price_block.get("currency") or "USD").upper(),
            sector=normalise_sector(profile.get("sector")),
            company_name=str(
                price_block.get("longName") or price_block.get("shortName") or ""
            ).strip(),
            source="Yahoo Finance",
        )
    return None


# --------------------------------------------------------------------------- #
#  Fournisseur 3 : Nasdaq Data                                                 #
# --------------------------------------------------------------------------- #

def _from_nasdaq(ticker: str) -> Optional[Quote]:
    payload = http.get_json(
        f"{NASDAQ_BASE}/quote/{ticker}/summary",
        params={"assetclass": "stocks"},
        retries=2,
    )
    if not isinstance(payload, dict):
        return None
    summary = ((payload.get("data") or {}).get("summaryData")) or {}
    if not summary:
        return None

    market_cap = _parse_amount((summary.get("MarketCap") or {}).get("value"))
    if market_cap is None:
        return None
    return Quote(
        market_cap=market_cap,
        currency="USD",                      # Nasdaq ne couvre que les cotations américaines
        sector=normalise_sector((summary.get("Sector") or {}).get("value")),
        source="Nasdaq Data",
    )


_PROVIDERS = (
    ("fmp", _from_fmp),
    ("yahoo", _from_yahoo),
    ("nasdaq", _from_nasdaq),
)


def get_quote(ticker: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> Optional[Quote]:
    """Capitalisation boursière et secteur du titre, ou None si introuvable."""
    ticker = ticker.upper().strip()
    cache_key = f"quote_v1_{ticker}"
    cached = cache.load(cache_key, cfg)
    if cached is not None:
        return cached

    for name, provider in _PROVIDERS:
        try:
            quote = provider(ticker)
        except Exception as exc:
            logger.debug("Cotation %s en erreur pour %s : %s", name, ticker, exc)
            continue
        if quote is not None:
            logger.info(
                "Capitalisation de %s : %.3e %s via %s.",
                ticker, quote.market_cap, quote.currency, quote.source,
            )
            cache.save(cache_key, quote, cfg)
            return quote

    logger.info("Capitalisation de %s introuvable chez les fournisseurs.", ticker)
    return None
