"""
Taurus Dashboard – Taux de change mensuels.

Deux besoins distincts, qui n'appellent pas la même conversion :

  • L'écran Modigliani-Miller compare des fondamentaux comptables à une
    capitalisation boursière. Les deux doivent être dans la MÊME devise, sans
    quoi l'écart mesuré serait un taux de change. ASML publie en euros et cote
    en dollars à New York : le rapprochement direct donnerait une divergence
    d'environ +16 %, qui n'est que la parité EUR/USD.

  • La régression Fama-French compare les rendements du titre aux facteurs de
    Kenneth French, qui sont libellés EN DOLLARS, y compris pour l'Europe et
    l'Asie. Un titre coté en euros doit donc être converti en dollars avant la
    régression, faute de quoi son alpha absorberait la variation de l'euro.

Source : api.frankfurter.app, qui republie les taux de référence quotidiens de
la Banque centrale européenne. Gratuit, sans clé, 30 devises.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .. import cache
from ..config import DEFAULT_CONFIG, ValuationConfig
from . import http

logger = logging.getLogger(__name__)

FRANKFURTER_BASE = "https://api.frankfurter.app"

# Devises publiées par la BCE. Hors de cette liste, aucune conversion n'est
# possible et l'appelant doit le signaler plutôt que de supposer la parité.
SUPPORTED = {
    "AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP",
    "HKD", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR",
    "NOK", "NZD", "PHP", "PLN", "RON", "SEK", "SGD", "THB", "TRY", "USD",
    "ZAR",
}

# Certaines places cotent en sous-unité : Londres en pence, Tel-Aviv en agorot.
# Le suffixe de devise renvoyé par le fournisseur de cours le signale.
_SUBUNITS = {
    "GBP": ("GBp", 0.01),      # pence sterling
    "ZAR": ("ZAc", 0.01),      # cents sud-africains
    "ILS": ("ILA", 0.01),      # agorot
}


def normalise_currency(currency: str) -> tuple[str, float]:
    """Ramène une sous-unité à sa devise principale.

    Renvoie (devise, facteur) : Yahoo cote Londres en « GBp », donc en pence,
    et un cours de 2 500 vaut 25 livres. Ignorer ce point diviserait la
    capitalisation par cent.
    """
    code = (currency or "USD").strip()
    for main, (subunit, factor) in _SUBUNITS.items():
        if code == subunit:
            return main, factor
    return code.upper(), 1.0


def _fetch_series(base: str, quote: str, start: date, end: date) -> Optional[pd.Series]:
    """Série quotidienne du taux base → quote, sur la période demandée."""
    payload = http.get_json(
        f"{FRANKFURTER_BASE}/{start.isoformat()}..{end.isoformat()}",
        params={"from": base, "to": quote},
    )
    if not isinstance(payload, dict):
        return None
    rates = payload.get("rates") or {}
    if not rates:
        return None

    points = {
        pd.Timestamp(day): float(values[quote])
        for day, values in rates.items()
        if isinstance(values, dict) and quote in values
    }
    if not points:
        return None
    return pd.Series(points).sort_index()


def monthly_rates(
    base: str,
    quote: str = "USD",
    months: int = 96,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[pd.Series]:
    """Taux de change de fin de mois, indexés en fin de mois.

    Renvoie None si la paire n'est pas couverte par la BCE — l'appelant doit
    alors renoncer à la conversion et le dire, plutôt que de supposer 1,0.
    """
    base, quote = base.upper(), quote.upper()
    if base == quote:
        return None
    if base not in SUPPORTED or quote not in SUPPORTED:
        logger.info("Paire %s/%s non couverte par la BCE.", base, quote)
        return None

    cache_key = f"fx_v1_{base}_{quote}_{months}"
    cached = cache.load(cache_key, cfg)
    if cached is not None:
        return cached

    end = date.today()
    start = end - timedelta(days=int(months * 31))

    daily = _fetch_series(base, quote, start, end)
    if daily is None or daily.empty:
        logger.warning("Taux %s/%s indisponibles.", base, quote)
        return None

    # Dernière cotation de chaque mois : la BCE ne publie pas les jours fériés.
    monthly = daily.resample("ME").last().dropna()
    if monthly.empty:
        return None

    logger.info(
        "Taux %s/%s : %d mois (%s → %s).",
        base, quote, len(monthly), monthly.index[0].date(), monthly.index[-1].date(),
    )
    cache.save(cache_key, monthly, cfg)
    return monthly


def latest_rate(
    base: str,
    quote: str = "USD",
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[float]:
    """Taux de change courant, pour convertir un bilan ponctuel."""
    base, quote = base.upper(), quote.upper()
    if base == quote:
        return 1.0
    if base not in SUPPORTED or quote not in SUPPORTED:
        return None

    cache_key = f"fx_latest_v1_{base}_{quote}"
    cached = cache.load(cache_key, cfg)
    if cached is not None:
        return cached

    payload = http.get_json(f"{FRANKFURTER_BASE}/latest", params={"from": base, "to": quote})
    if not isinstance(payload, dict):
        return None
    rate = (payload.get("rates") or {}).get(quote)
    if rate is None:
        return None

    value = float(rate)
    cache.save(cache_key, value, cfg)
    return value


def convert_series(
    prices: pd.Series,
    from_currency: str,
    to_currency: str = "USD",
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[pd.Series]:
    """Convertit une série de cours mensuels dans une autre devise.

    Les mois sans taux publié sont reportés depuis le mois précédent : un trou
    ponctuel dans la série de la BCE ne doit pas amputer l'historique de cours.
    """
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    if from_currency == to_currency:
        return prices

    rates = monthly_rates(from_currency, to_currency, cfg=cfg)
    if rates is None:
        return None

    aligned = rates.reindex(prices.index).ffill().bfill()
    if aligned.isna().all():
        return None
    return (prices * aligned).dropna()
