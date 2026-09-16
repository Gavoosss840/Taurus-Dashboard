"""
Taurus Dashboard – Reconstitution du rendement total.

Pourquoi
────────
Les fournisseurs de cours gratuits ne réintègrent pas tous les dividendes.
Une série non réajustée mesure un rendement en CAPITAL, inférieur au rendement
TOTAL du montant du dividende. Le biais va toujours dans le même sens et croît
avec le rendement du titre : mesuré sur un échantillon de grandes
capitalisations, il décale le t-stat de l'alpha de 0,04 pour Alphabet à 0,81
pour Altria, et peut inverser le signe de l'alpha d'une valeur de rendement.
Il pénalise donc systématiquement le segment où la comparaison entre titres a
le plus d'importance.

Comment
───────
Les dividendes par action figurent dans le fichier `companyfacts` de SEC EDGAR
que le moteur télécharge déjà pour les fondamentaux : les reconstituer ne
coûte aucun appel réseau supplémentaire.

    rendement total du mois t = (P_t + D_t) / P_{t-1} − 1

Deux précautions.

**Les divisions d'actions.** Les cours sont ajustés des divisions, les
dividendes par action déclarés à la SEC ne le sont pas. Une division survenue
dans la fenêtre décale donc le rapport D/P d'un facteur entier sur sa partie
ancienne. Le rendement de chaque période est pour cette raison ramené dans une
bande autour de sa médiane : un artefact de division en sort, une variation
réelle non.

**Le calendrier.** EDGAR date un dividende par la fin de la période comptable
où il est déclaré, pas par sa date de détachement. Le décalage peut atteindre
un trimestre. Sur une moyenne de soixante mois — ce que mesure l'alpha — il est
du second ordre, mais il interdit d'employer cette série pour dater précisément
un flux.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nombre de trimestres à dividende exigés DANS la fenêtre de cours. En dessous,
# la série est trop lacunaire : mieux vaut garder l'avertissement que
# reconstituer un rendement total sur quatre versements.
MIN_QUARTERS_IN_WINDOW = 8

# Bande de tolérance autour du rendement médian de période. Une division
# d'actions décale le rapport D/P d'un facteur 2, 4 ou 10 ; une variation
# ordinaire du rendement reste bien à l'intérieur.
YIELD_BAND = (0.40, 2.50)

# Au-delà, le « dividende » n'en est pas un : versement exceptionnel, erreur de
# déclaration, ou unité incohérente.
MAX_ANNUAL_YIELD = 0.15


@dataclass
class TotalReturnResult:
    """Série reconstituée et de quoi en juger."""

    prices: pd.Series                  # indice de rendement total
    annual_yield: float                # rendement médian annualisé
    quarters_used: int
    clipped_periods: int               # périodes ramenées dans la bande
    notes: list = field(default_factory=list)


def reconstruct(
    prices: pd.Series,
    dividends: Optional[pd.Series],
    currency_matches: bool = True,
) -> Optional[TotalReturnResult]:
    """Transforme une série de cours en indice de rendement total.

    Parameters
    ----------
    prices           : cours mensuels, indexés en fin de mois
    dividends        : dividendes trimestriels par action (SEC EDGAR)
    currency_matches : les dividendes sont-ils libellés dans la devise de
                       cotation ?  Pour un certificat de dépôt, ils ne le sont
                       pas, et le rapport certificat / action ordinaire est
                       inconnu : on renonce plutôt que de deviner.

    Renvoie None lorsque la reconstitution n'est pas fiable — l'appelant
    conserve alors ses cours bruts et son avertissement.
    """
    if prices is None or prices.empty or dividends is None or dividends.empty:
        return None

    if not currency_matches:
        logger.info(
            "Dividendes libellés dans une autre devise que la cotation — "
            "reconstitution abandonnée."
        )
        return None

    prices = prices.dropna().sort_index()
    if len(prices) < 2:
        return None

    # Dividendes tombant dans la fenêtre de cours.
    in_window = dividends.reindex(prices.index).dropna()
    in_window = in_window[in_window > 0]
    if len(in_window) < MIN_QUARTERS_IN_WINDOW:
        logger.info(
            "Seulement %d trimestres à dividende dans la fenêtre (minimum %d).",
            len(in_window), MIN_QUARTERS_IN_WINDOW,
        )
        return None

    # Rendement de période, puis bornage autour de la médiane.
    period_yield = in_window / prices.reindex(in_window.index)
    period_yield = period_yield.replace([np.inf, -np.inf], np.nan).dropna()
    if period_yield.empty:
        return None

    median_yield = float(period_yield.median())
    if median_yield <= 0:
        return None

    annual_yield = median_yield * 4.0
    if annual_yield > MAX_ANNUAL_YIELD:
        logger.info(
            "Rendement reconstitué implausible (%.1f %% par an) — abandon.",
            annual_yield * 100,
        )
        return None

    low, high = median_yield * YIELD_BAND[0], median_yield * YIELD_BAND[1]
    clipped = period_yield.clip(low, high)
    n_clipped = int((clipped != period_yield).sum())

    notes = []
    if n_clipped:
        notes.append(
            f"{n_clipped} versement(s) ramené(s) dans la bande de rendement : "
            "une division d'actions décale le dividende par action déclaré à "
            "la SEC par rapport au cours, qui en est ajusté."
        )

    # Montant du dividende exprimé sur la base des cours ajustés.
    adjusted = clipped * prices.reindex(clipped.index)
    dividend_flow = pd.Series(0.0, index=prices.index)
    dividend_flow.loc[adjusted.index] = adjusted.values

    # Indice de rendement total : (P_t + D_t) / P_{t-1}.
    growth = (prices + dividend_flow) / prices.shift(1)
    growth.iloc[0] = 1.0
    total_return_index = float(prices.iloc[0]) * growth.cumprod()

    logger.info(
        "Rendement total reconstitué : %d trimestres, rendement médian "
        "%.2f %% par an%s.",
        len(clipped), annual_yield * 100,
        f", {n_clipped} borné(s)" if n_clipped else "",
    )

    return TotalReturnResult(
        prices=total_return_index,
        annual_yield=annual_yield,
        quarters_used=len(clipped),
        clipped_periods=n_clipped,
        notes=notes,
    )
