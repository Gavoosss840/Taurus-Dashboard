"""
Taurus Dashboard – Pilier 3 : momentum 12-1 ajusté de la volatilité.

Transposition mono-titre de `taurus/momentum.py`.

  • Jegadeesh & Titman (1993) : rendement cumulé sur 12 mois, en sautant le
    mois le plus récent.  Ce saut évite la contamination par le retournement
    de court terme (les titres qui viennent de monter fort ont tendance à
    refluer le mois suivant).

  • Barroso & Santa-Clara (2015) : diviser le momentum brut par la volatilité
    réalisée du titre.  Un titre à +30 % avec 15 % de volatilité porte un
    signal plus fort qu'un titre à +40 % avec 60 % de volatilité.

Différence avec l'algorithme de production : celui-ci classe les titres les
uns par rapport aux autres (terciles de l'univers).  Un dashboard mono-titre
n'a pas d'univers ; la référence devient donc le MARCHÉ, mesuré sur le même
horizon à partir du facteur de marché Fama-French.  Le signal exploité est
l'ÉCART de momentum-Sharpe entre le titre et le marché.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, ValuationConfig

logger = logging.getLogger(__name__)

# Volatilité annualisée en dessous de laquelle l'estimation n'est pas crédible.
# Aucune action ordinaire ne varie de moins de 0,5 % par an ; une telle valeur
# trahit un cours figé (titre suspendu, cotation répétée, série synthétique).
# Sans ce plancher, diviser le momentum par une volatilité de 1e-17 produirait
# un score de l'ordre de 1e16 qui saturerait le verdict à lui seul.
MIN_ANNUAL_VOL = 0.005


@dataclass
class MomentumResult:
    """Momentum du titre, situé par rapport au marché."""

    raw: float                 # rendement cumulé sur la fenêtre 12-1
    volatility: float          # volatilité annualisée sur la même fenêtre
    sharpe: float              # momentum ajusté de la volatilité
    market_raw: float          # même mesure pour le marché
    market_sharpe: float
    excess_sharpe: float       # titre − marché : le signal retenu
    crash_regime: bool         # régime de krach de momentum détecté
    window_start: str
    window_end: str
    n_months: int

    @property
    def direction(self) -> int:
        """+1 momentum supérieur au marché, −1 inférieur, 0 indéterminé."""
        if not math.isfinite(self.excess_sharpe) or self.excess_sharpe == 0:
            return 0
        return 1 if self.excess_sharpe > 0 else -1


def _annualised_vol(returns: pd.Series) -> float:
    """Volatilité annualisée à partir de rendements mensuels."""
    clean = returns.dropna()
    if len(clean) < 6:      # moins de six mois : estimation non fiable
        return float("nan")
    value = float(clean.std() * np.sqrt(12))
    return value if value >= MIN_ANNUAL_VOL else float("nan")


def _cumulative_return(prices: pd.Series, start_loc: int, end_loc: int) -> float:
    start_price = float(prices.iloc[start_loc])
    end_price = float(prices.iloc[end_loc])
    if not (math.isfinite(start_price) and math.isfinite(end_price)) or start_price <= 0:
        return float("nan")
    return end_price / start_price - 1.0


def _detect_crash_regime(market_returns: pd.Series, cfg: ValuationConfig) -> bool:
    """Détecte un régime propice aux krachs de momentum.

    Déclenche lorsque la volatilité du dernier mois dépasse le double de la
    moyenne des douze mois précédents.  Dans ce régime, la prime de momentum
    s'inverse brutalement : l'algorithme de production réduit alors de moitié
    le poids du momentum.
    """
    if not cfg.momentum_crash_dampen:
        return False

    clean = market_returns.dropna()
    if len(clean) < 13:
        return False

    # Sous l'hypothèse de normalité, E|r| = σ·√(2/π) ≈ 0,80σ : le facteur
    # √(π/2) rétablit l'échelle pour que le seuil de 2× joue bien son rôle.
    vol_last_month = float(abs(clean.iloc[-1]) * np.sqrt(np.pi / 2.0) * np.sqrt(12))
    # La base de comparaison EXCLUT le mois testé : l'y inclure gonflerait le
    # dénominateur précisément lorsque le choc survient.
    vol_trailing = float(clean.iloc[-13:-1].std() * np.sqrt(12))

    if vol_trailing > 0 and vol_last_month > 2.0 * vol_trailing:
        logger.info(
            "Régime de krach de momentum : volatilité du mois %.1f %% contre "
            "%.1f %% en moyenne sur 12 mois.",
            vol_last_month * 100, vol_trailing * 100,
        )
        return True
    return False


def compute_momentum(
    monthly_prices: pd.Series,
    market_returns: pd.Series,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[MomentumResult]:
    """Momentum 12-1 du titre, comparé à celui du marché.

    Parameters
    ----------
    monthly_prices : cours mensuels ajustés, indexés en fin de mois
    market_returns : rendements mensuels du marché (Mkt-RF + RF)
    """
    if monthly_prices is None or monthly_prices.empty:
        return None

    prices = monthly_prices.dropna().sort_index()
    needed = cfg.momentum_months + cfg.momentum_skip + 1
    if len(prices) < needed:
        logger.info(
            "Historique insuffisant pour le momentum : %d mois (minimum %d).",
            len(prices), needed,
        )
        return None

    # Fenêtre 12-1 : du mois t−12 au mois t−1, le mois courant étant écarté.
    end_loc = len(prices) - 1 - cfg.momentum_skip
    start_loc = end_loc - cfg.momentum_months + 1
    if start_loc < 0:
        return None

    raw = _cumulative_return(prices, start_loc, end_loc)
    if not math.isfinite(raw):
        return None

    window_returns = prices.iloc[start_loc:end_loc + 1].pct_change(fill_method=None)
    volatility = _annualised_vol(window_returns)

    # ── Même mesure pour le marché, sur exactement la même fenêtre ─────── #
    # Kenneth French publie ses facteurs avec un à deux mois de décalage sur
    # les cours de bourse. Comparer un momentum du titre sur 12 mois à un
    # momentum de marché sur 11 fausserait l'écart d'à peu près un mois de
    # performance de marché. On restreint donc les DEUX mesures aux mois
    # effectivement communs.
    # Le momentum du titre est P[fin]/P[début] − 1 : il couvre les rendements
    # des mois start_loc+1 à end_loc, soit un mois de MOINS que le nombre de
    # relevés de cours. Le marché doit couvrir exactement ces mois-là, sans
    # quoi on comparerait 12 mois de marché à 11 mois de titre.
    window_index = prices.index[start_loc + 1:end_loc + 1]
    market_window = market_returns.reindex(window_index).dropna()

    market_available = len(market_window) >= cfg.momentum_months - 2
    if not market_available:
        logger.info(
            "Momentum du marché indisponible sur la fenêtre (%d mois alignés).",
            len(market_window),
        )

    market_raw = float("nan")
    market_vol = float("nan")
    comparable_raw = raw
    comparable_vol = volatility

    if market_available:
        market_raw = float((1.0 + market_window).prod() - 1.0)
        market_vol = _annualised_vol(market_window)

        if len(market_window) < len(window_index):
            # Recalcul du momentum du titre sur les seuls mois communs.
            aligned = window_returns.reindex(market_window.index).dropna()
            if len(aligned) >= cfg.momentum_months - 3:
                comparable_raw = float((1.0 + aligned).prod() - 1.0)
                comparable_vol = _annualised_vol(aligned)

    # L'ajustement de volatilité n'est appliqué que si les DEUX volatilités
    # sont estimables : comparer un ratio rendement/risque du titre à un
    # rendement brut du marché n'aurait aucun sens dimensionnel.
    both_vols = (
        cfg.vol_adjust_momentum
        and math.isfinite(comparable_vol)
        and (not market_available or math.isfinite(market_vol))
    )

    if cfg.vol_adjust_momentum and math.isfinite(volatility):
        sharpe = raw / volatility
    else:
        sharpe = raw

    if both_vols:
        comparable_sharpe = comparable_raw / comparable_vol
        market_sharpe = market_raw / market_vol if market_available else float("nan")
    else:
        comparable_sharpe = comparable_raw
        market_sharpe = market_raw if market_available else float("nan")
        if cfg.vol_adjust_momentum and market_available:
            logger.info(
                "Volatilité non estimable : comparaison au marché en rendement brut."
            )

    # Le signal retenu : l'écart de momentum entre le titre et le marché,
    # les deux mesurés sur les mêmes mois et sur la même base.
    excess = (
        comparable_sharpe - market_sharpe
        if math.isfinite(market_sharpe) else float("nan")
    )

    # Régime de krach : évalué sur l'historique de marché jusqu'au mois courant.
    market_to_date = market_returns.reindex(
        market_returns.index[market_returns.index <= prices.index[-1]]
    )
    crash_regime = _detect_crash_regime(market_to_date, cfg)

    return MomentumResult(
        raw=float(raw),
        volatility=float(volatility),
        sharpe=float(sharpe),
        market_raw=float(market_raw),
        market_sharpe=float(market_sharpe),
        excess_sharpe=float(excess),
        crash_regime=crash_regime,
        window_start=str(prices.index[start_loc].date()),
        window_end=str(prices.index[end_loc].date()),
        n_months=cfg.momentum_months,
    )
