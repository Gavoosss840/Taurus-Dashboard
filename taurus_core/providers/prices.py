"""
Taurus Dashboard – Prix mensuels ajustés.

Trois fournisseurs sont essayés dans l'ordre ; le premier qui renvoie un
historique exploitable gagne.  Aucun n'est indispensable, ce qui évite qu'une
panne ou un quota atteint chez l'un rende le dashboard inutilisable :

  1. Financial Modeling Prep  — si FMP_API_KEY est défini (historique complet,
     cours ajustés des splits et dividendes) ;
  2. Yahoo Finance (API chart publique) — gratuit, sans clé ;
  3. marketdata.app — gratuit et sans clé ;
  4. Nasdaq Data (API publique) — dernier recours.

Dividendes
──────────
`PriceHistory.total_return` n'est vrai que lorsque la source réintègre
DÉMONTRABLEMENT les dividendes : la série `adjClose` de Financial Modeling
Prep, la série `adjclose` de Yahoo. Dès qu'un fournisseur retombe sur le cours
brut — ou n'en dit rien, comme marketdata.app et Nasdaq Data — le drapeau
passe à faux et l'utilisateur est averti.

Ce n'est pas un détail de présentation. Un cours sans dividendes mesure un
rendement en capital, inférieur au rendement total du montant du dividende :
le t-stat de l'alpha s'en trouve décalé de 0,04 pour Alphabet à 0,81 pour
Altria. Le biais va toujours dans le même sens et croît avec le rendement du
titre, donc il pénalise systématiquement les valeurs de rendement — là même
où la comparaison entre titres a le plus d'importance.

Sortie commune : une `pd.Series` de cours mensuels ajustés, indexée en fin de
mois — exactement le format attendu par `taurus/data.py:get_monthly_prices`
dans l'algorithme de production.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from .. import cache
from ..config import DEFAULT_CONFIG, ValuationConfig
from . import http

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
YAHOO_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
MARKETDATA_BASE = "https://api.marketdata.app/v1"
NASDAQ_BASE = "https://api.nasdaq.com/api"

# Historique minimal pour que la régression FF5 ait un sens (cf. hard_min_obs).
MIN_MONTHS = 25


class PriceHistory:
    """Historique de cours d'un titre, accompagné de sa provenance."""

    def __init__(self, monthly: pd.Series, source: str, currency: str = "USD",
                 last_price: Optional[float] = None, total_return: bool = True):
        self.monthly = monthly           # Series indexée fin de mois
        self.source = source             # nom du fournisseur retenu
        self.currency = currency
        # True seulement si la source réintègre DÉMONTRABLEMENT les dividendes.
        # Le drapeau ne doit jamais être optimiste : il pilote un avertissement
        # à l'utilisateur, et un cours sans dividendes décale le t-stat de
        # l'alpha de 0,04 (GOOGL) à 0,81 (Altria) — un biais systématique qui
        # pénalise les titres à fort rendement, exactement là où la
        # comparaison transversale compte.
        self.total_return = total_return
        # Le dernier cours connu : le point intra-mois si le fournisseur le
        # donne, sinon la dernière clôture mensuelle.
        self.last_price = float(last_price) if last_price is not None else (
            float(monthly.iloc[-1]) if len(monthly) else float("nan")
        )

    @property
    def returns(self) -> pd.Series:
        """Rendements mensuels simples (sans remplissage des trous)."""
        return self.monthly.pct_change(fill_method=None).dropna()

    def __len__(self) -> int:
        return len(self.monthly)


# --------------------------------------------------------------------------- #
#  Normalisation                                                               #
# --------------------------------------------------------------------------- #

def _to_month_end(series: pd.Series) -> pd.Series:
    """Aligne un index de dates sur la fin de mois et déduplique."""
    if series.empty:
        return series
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    series.index = series.index.to_period("M").to_timestamp("M")
    # Un fournisseur peut livrer plusieurs points pour un même mois : on garde
    # le dernier (la clôture du mois).
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index().astype(float).dropna()


# --------------------------------------------------------------------------- #
#  Fournisseur 1 : Financial Modeling Prep                                     #
# --------------------------------------------------------------------------- #

def _from_fmp(ticker: str, start: datetime, end: datetime) -> Optional[PriceHistory]:
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        return None

    data = http.get_json(
        f"{FMP_BASE}/historical-price-full/{ticker}",
        params={
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "apikey": api_key,
        },
    )
    if not isinstance(data, dict):
        return None
    rows = data.get("historical") or []
    if not rows:
        return None

    # `adjClose` intègre splits et dividendes ; `close` ne couvre que les
    # splits, et le repli sur lui doit être signalé, pas masqué.
    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        return None
    dividends_included = "adjClose" in frame.columns
    price_col = "adjClose" if dividends_included else "close"
    daily = pd.Series(
        pd.to_numeric(frame[price_col], errors="coerce").values,
        index=pd.to_datetime(frame["date"]),
    ).sort_index().dropna()
    if daily.empty:
        return None

    monthly = _to_month_end(daily.resample("ME").last())
    if len(monthly) < MIN_MONTHS:
        return None
    return PriceHistory(
        monthly,
        source="Financial Modeling Prep" if dividends_included
               else "Financial Modeling Prep (cours bruts)",
        last_price=float(daily.iloc[-1]),
        total_return=dividends_included,
    )


# --------------------------------------------------------------------------- #
#  Fournisseur 2 : Yahoo Finance                                               #
# --------------------------------------------------------------------------- #

def _from_yahoo(ticker: str, start: datetime, end: datetime) -> Optional[PriceHistory]:
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1mo",
        "events": "div|split",
        # Yahoo n'applique les ajustements que si on les demande explicitement.
        "includeAdjustedClose": "true",
    }
    for host in YAHOO_HOSTS:
        data = http.get_json(f"{host}/v8/finance/chart/{ticker}", params=params, retries=2)
        if not isinstance(data, dict):
            continue
        chart = data.get("chart") or {}
        results = chart.get("result") or []
        if not results:
            continue

        result = results[0]
        timestamps = result.get("timestamp") or []
        if not timestamps:
            continue

        indicators = result.get("indicators") or {}
        adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
        raw = (indicators.get("quote") or [{}])[0].get("close")
        dividends_included = bool(adj)
        values = adj if dividends_included else raw
        if not values:
            continue

        series = pd.Series(
            values,
            index=pd.to_datetime(timestamps, unit="s", utc=True),
        )
        monthly = _to_month_end(series)
        if len(monthly) < MIN_MONTHS:
            continue

        meta = result.get("meta") or {}
        return PriceHistory(
            monthly,
            source="Yahoo Finance" if dividends_included
                   else "Yahoo Finance (cours bruts)",
            currency=meta.get("currency") or "USD",
            last_price=meta.get("regularMarketPrice"),
            total_return=dividends_included,
        )
    return None


# --------------------------------------------------------------------------- #
#  Fournisseur 3 : marketdata.app                                              #
# --------------------------------------------------------------------------- #

def _from_marketdata(ticker: str, start: datetime, end: datetime) -> Optional[PriceHistory]:
    data = http.get_json(
        f"{MARKETDATA_BASE}/stocks/candles/M/{ticker}/",
        params={
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        },
        retries=2,
    )
    if not isinstance(data, dict) or data.get("s") != "ok":
        return None
    timestamps, closes = data.get("t") or [], data.get("c") or []
    if not timestamps or len(timestamps) != len(closes):
        return None

    series = pd.Series(closes, index=pd.to_datetime(timestamps, unit="s", utc=True))
    monthly = _to_month_end(series)
    if len(monthly) < MIN_MONTHS:
        return None
    # Ce point d'entrée ne documente aucun ajustement des dividendes, et rien
    # dans les séries renvoyées ne permet de l'établir. Par défaut on suppose
    # donc des cours bruts : mieux vaut un avertissement de trop qu'un biais
    # silencieux sur l'alpha.
    return PriceHistory(monthly, source="marketdata.app", total_return=False)


# --------------------------------------------------------------------------- #
#  Fournisseur 4 : Nasdaq Data (sans dividendes)                               #
# --------------------------------------------------------------------------- #

def _parse_money(value: object) -> float:
    """Convertit "$88.29" ou "1,234.5" en flottant."""
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _from_nasdaq(ticker: str, start: datetime, end: datetime) -> Optional[PriceHistory]:
    data = http.get_json(
        f"{NASDAQ_BASE}/quote/{ticker}/historical",
        params={
            "assetclass": "stocks",
            "fromdate": start.strftime("%Y-%m-%d"),
            "todate": end.strftime("%Y-%m-%d"),
            "limit": 9999,
        },
        retries=2,
    )
    if not isinstance(data, dict):
        return None
    rows = (((data.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns or "close" not in frame.columns:
        return None

    daily = pd.Series(
        [_parse_money(v) for v in frame["close"]],
        index=pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="coerce"),
    ).dropna().sort_index()
    daily = daily[daily > 0]
    if daily.empty:
        return None

    monthly = _to_month_end(daily.resample("ME").last())
    if len(monthly) < MIN_MONTHS:
        return None
    return PriceHistory(
        monthly,
        source="Nasdaq Data",
        last_price=float(daily.iloc[-1]),
        total_return=False,
    )


# --------------------------------------------------------------------------- #
#  API publique                                                                #
# --------------------------------------------------------------------------- #

_PROVIDERS = (
    ("fmp", _from_fmp),
    ("yahoo", _from_yahoo),
    ("marketdata", _from_marketdata),
    ("nasdaq", _from_nasdaq),
)


def get_monthly_prices(
    ticker: str,
    cfg: ValuationConfig = DEFAULT_CONFIG,
    failures: Optional[dict] = None,
) -> Optional[PriceHistory]:
    """Historique mensuel ajusté du titre, ou None si aucune source ne répond.

    `failures`, si fourni, reçoit {nom du fournisseur: raison de l'échec}.
    L'appelant peut alors expliquer POURQUOI aucune source n'a répondu :
    « ticker inconnu » et « quota atteint chez le seul fournisseur couvrant
    les places locales » appellent des réponses opposées, et renvoyer le même
    « vérifiez le ticker » dans les deux cas envoie corriger une saisie
    correcte.
    """
    ticker = ticker.upper().strip()
    end = datetime.now(timezone.utc)
    # On demande 18 mois de plus que le strict nécessaire : les fournisseurs
    # tronquent parfois le début de fenêtre, et une régression sur 60 mois
    # pleins vaut mieux qu'une régression sur 52.
    start = end - timedelta(days=int((cfg.history_months_needed + 18) * 30.5))

    cache_key = f"prices_v1_{ticker}_{cfg.history_months_needed}"
    cached = cache.load(cache_key, cfg)
    if cached is not None:
        return cached

    errors: List[str] = []
    recorded = failures if failures is not None else {}

    for name, provider in _PROVIDERS:
        try:
            history = provider(ticker, start, end)
        except Exception as exc:
            logger.debug("Fournisseur de prix %s en erreur pour %s : %s", name, ticker, exc)
            errors.append(f"{name}: {exc}")
            recorded[name] = type(exc).__name__
            continue
        if history is not None and len(history) >= MIN_MONTHS:
            logger.info(
                "Prix de %s : %d mois via %s.", ticker, len(history), history.source,
            )
            cache.save(cache_key, history, cfg)
            return history
        errors.append(f"{name}: aucune donnée exploitable")
        recorded[name] = "aucune donnée exploitable"

    logger.warning("Aucun cours trouvé pour %s (%s).", ticker, " | ".join(errors))
    return None
