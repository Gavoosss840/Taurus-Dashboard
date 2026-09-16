"""
Taurus Dashboard – Orchestrateur : du ticker au verdict.

Enchaîne les trois piliers de la stratégie Taurus sur un titre unique, puis
les agrège en un score composite et un verdict lisible.

Passage du cross-sectionnel à l'absolu
───────────────────────────────────────
L'algorithme de production est un long/short : il classe ~500 titres les uns
par rapport aux autres et retient les 25 meilleurs et les 25 pires.  Ce
classement n'a aucun sens sur un titre isolé.  Chaque pilier est donc rapporté
au SEUIL DE DÉCLENCHEMENT que l'algorithme utilise déjà en absolu :

    score = signal / seuil,  borné à ±2

Un score de +1 signifie « ce pilier déclenche exactement son signal d'achat » ;
+2 et au-delà, « il le déclenche deux fois plus fort ».  Le bornage empêche
une valeur aberrante — un t-stat de 40 sur un titre au cours figé — d'écraser
les deux autres piliers.

    • Alpha FF5   : t-stat / valeur critique de Student (typiquement 2,00)
    • Structure MM: divergence / 25 % (cfg.leverage_gap_threshold)
    • Momentum    : écart de momentum-Sharpe au marché / 0,50

Le score composite reprend la pondération de `TaurusConfig` :
0,40 alpha + 0,30 structure du capital + 0,30 momentum.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .alpha import AlphaResult, compute_alpha
from .capital_structure import MMResult, leverage_alert, mm_valuation
from .config import DEFAULT_CONFIG, ValuationConfig
from .momentum import MomentumResult, compute_momentum
from .total_return import reconstruct as reconstruct_total_return
from .providers import factors as factors_provider
from .providers import fundamentals as fundamentals_provider
from .providers import fx as fx_provider
from .providers import prices as prices_provider
from .providers import quotes as quotes_provider
from .providers import regions as regions_provider

logger = logging.getLogger(__name__)

# Un ticker boursier, cotation américaine ou place locale : lettres, chiffres,
# point, tiret. Le suffixe de place peut allonger la chaîne — « RELIANCE.NS »
# fait onze caractères, « BRK-B » cinq, « 7203.T » six.
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,15}$")

VERDICT_UNDERVALUED = "SOUS-ÉVALUÉE"
VERDICT_OVERVALUED = "SUR-ÉVALUÉE"
VERDICT_FAIR = "AU JUSTE PRIX"


class TickerError(ValueError):
    """Ticker introuvable chez tous les fournisseurs de données."""


class InvalidTickerError(TickerError):
    """Ticker mal formé : la saisie elle-même est en cause, pas les données."""


@dataclass
class Pillar:
    """Un pilier d'analyse, normalisé pour l'affichage."""

    key: str
    name: str
    score: float                 # borné à ±cfg.score_clip
    weight: float
    available: bool
    headline: str                # valeur principale, déjà formatée
    verdict: str                 # SOUS-ÉVALUÉE / SUR-ÉVALUÉE / NEUTRE
    explanation: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class Analysis:
    """Résultat complet de l'analyse d'un titre."""

    ticker: str
    company_name: str
    sector: str
    currency: str            # devise de cotation
    region: str              # région Fama-French retenue
    region_label: str

    verdict: str
    verdict_label: str           # libellé nuancé (« fortement sous-évaluée »)
    composite_score: float
    confidence: float            # 0 → 1
    summary: str

    price: float
    fair_value: float
    upside_pct: float
    market_cap: float
    # Seuils du verdict traduits en COURS, tous piliers confondus : sous
    # `buy_below` le score composite passe en « sous-évaluée », au-dessus de
    # `sell_above` en « sur-évaluée ». NaN quand aucun cours n'y suffit —
    # le pilier Modigliani-Miller sature et les deux autres s'y opposent.
    buy_below: float
    sell_above: float
    # Pourquoi la zone n'est pas calculable, le cas échéant :
    # « pilier_absent » (aucun pilier ne dépend du cours) ou « sature »
    # (le pilier de valorisation plafonne avant d'emporter le verdict).
    buy_below_reason: str

    pillars: List[Pillar]
    warnings: List[str]
    data_sources: Dict[str, str]
    computed_at: str
    elapsed_seconds: float

    # Objets bruts, exposés pour les tests et un éventuel export
    alpha: Optional[AlphaResult] = None
    mm: Optional[MMResult] = None
    momentum: Optional[MomentumResult] = None


# --------------------------------------------------------------------------- #
#  Normalisation d'un signal en score borné                                    #
# --------------------------------------------------------------------------- #

def _normalise(value: float, threshold: float, cfg: ValuationConfig) -> float:
    """Rapporte un signal à son seuil de déclenchement, borné à ±score_clip."""
    if not math.isfinite(value) or threshold <= 0:
        return 0.0
    return float(np.clip(value / threshold, -cfg.score_clip, cfg.score_clip))


def _pillar_verdict(score: float) -> str:
    if score >= 0.5:
        return VERDICT_UNDERVALUED
    if score <= -0.5:
        return VERDICT_OVERVALUED
    return "NEUTRE"


def _format_amount(value: float, currency: str = "USD") -> str:
    """Formate un montant en notation courte (Md, M)."""
    if not math.isfinite(value):
        return "n/d"
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    absolute = abs(value)
    if absolute >= 1e12:
        return f"{symbol}{value / 1e12:.2f} Bn"
    if absolute >= 1e9:
        return f"{symbol}{value / 1e9:.1f} Md"
    if absolute >= 1e6:
        return f"{symbol}{value / 1e6:.1f} M"
    return f"{symbol}{value:,.0f}".replace(",", " ")


# --------------------------------------------------------------------------- #
#  Construction des piliers                                                    #
# --------------------------------------------------------------------------- #

def _build_alpha_pillar(result: Optional[AlphaResult], cfg: ValuationConfig) -> Pillar:
    if result is None:
        return Pillar(
            key="alpha", name="Alpha Fama-French 5 facteurs", score=0.0,
            weight=cfg.w_alpha, available=False, headline="n/d", verdict="NEUTRE",
            explanation=(
                "Historique de cours trop court pour estimer l'alpha : il faut "
                f"au moins {cfg.hard_min_obs} mois de données."
            ),
        )

    score = _normalise(result.alpha_tstat, result.t_critical, cfg)

    # Le verdict du pilier suit le score (seuil 0,5), c'est-à-dire la moitié du
    # seuil de significativité. La formulation doit refléter ces trois bandes,
    # faute de quoi un t de 1,17 serait étiqueté « sous-évaluée » tout en étant
    # décrit comme non significatif — deux affirmations contradictoires.
    leaning_threshold = result.t_critical * cfg.verdict_threshold
    magnitude = abs(result.alpha_tstat) if math.isfinite(result.alpha_tstat) else 0.0

    # Nombre de mois qu'il faudrait pour que ce même alpha devienne
    # significatif : le t-stat croît comme la racine du nombre d'observations.
    months_needed = 0
    if 0 < magnitude < result.t_critical:
        months_needed = int(round(result.n_obs * (result.t_critical / magnitude) ** 2))

    if result.significant:
        sense = "supérieure" if result.direction > 0 else "inférieure"
        explanation = (
            f"Sur {result.n_obs} mois, le titre dégage une performance {sense} "
            f"de {abs(result.alpha_annual) * 100:.1f} % par an à ce que son "
            f"exposition aux cinq facteurs de risque justifie. L'écart est "
            f"statistiquement significatif (t = {result.alpha_tstat:.2f}, "
            f"p = {result.p_value:.3f}), donc peu susceptible d'être dû au hasard."
        )
    elif magnitude >= leaning_threshold:
        sense = "au-dessus" if result.direction > 0 else "en dessous"
        explanation = (
            f"L'alpha ressort à {result.alpha_annual * 100:+.1f} % par an, soit "
            f"{sense} de ce que l'exposition aux cinq facteurs de risque "
            f"justifie. Le signe est net mais l'écart n'atteint pas le seuil de "
            f"significativité (t = {result.alpha_tstat:.2f}, seuil "
            f"{result.t_critical:.2f}) : la tendance penche dans ce sens sans "
            "être démontrée. Ce pilier ne compte donc que pour une fraction de "
            "son poids."
        )
        if months_needed:
            explanation += (
                f" Au même rythme, il faudrait environ {months_needed} mois "
                f"({months_needed / 12:.0f} ans) d'historique pour conclure."
            )
    else:
        explanation = (
            f"L'alpha ressort à {result.alpha_annual * 100:+.1f} % par an mais "
            f"n'est pas significatif (t = {result.alpha_tstat:.2f}, seuil "
            f"{result.t_critical:.2f}) : sur {result.n_obs} mois, on ne peut pas "
            "le distinguer du bruit."
        )

    explanation += (
        f" Les cinq facteurs de risque expliquent "
        f"{result.r_squared * 100:.0f} % de la variance des rendements."
    )

    return Pillar(
        key="alpha",
        name="Alpha Fama-French 5 facteurs",
        score=score,
        weight=cfg.w_alpha,
        available=True,
        headline=f"{result.alpha_annual * 100:+.1f} % / an",
        verdict=_pillar_verdict(score),
        explanation=explanation,
        details={
            "alpha_annual": result.alpha_annual,
            "alpha_monthly": result.alpha_monthly,
            "t_stat": result.alpha_tstat,
            "t_critical": result.t_critical,
            "p_value": result.p_value,
            "significant": result.significant,
            "months_for_significance": months_needed or None,
            "r_squared": result.r_squared,
            "n_obs": result.n_obs,
            "betas": result.betas,
            "window": f"{result.window_start} → {result.window_end}",
        },
    )


def _build_mm_pillar(result: Optional[MMResult], cfg: ValuationConfig) -> Pillar:
    if result is None:
        return Pillar(
            key="capital_structure", name="Structure du capital (Modigliani-Miller)",
            score=0.0, weight=cfg.w_mm, available=False, headline="n/d",
            verdict="NEUTRE",
            explanation=(
                "Fondamentaux comptables indisponibles : la juste valeur "
                "Modigliani-Miller n'a pas pu être calculée."
            ),
        )

    threshold_pct = cfg.leverage_gap_threshold * 100
    score = _normalise(result.divergence_pct, threshold_pct, cfg)

    # Le verdict du pilier suit le score (seuil 0,5), c'est-à-dire la moitié
    # du seuil de déclenchement plein de l'algorithme. La formulation doit
    # refléter ces trois bandes, faute de quoi une divergence de +17 % serait
    # étiquetée « sous-évaluée » tout en étant décrite comme neutre.
    partial_pct = threshold_pct * cfg.verdict_threshold
    divergence = result.divergence_pct

    if divergence > threshold_pct:
        sense = (
            f"La juste valeur théorique dépasse la capitalisation de "
            f"{divergence:.1f} %, au-delà du seuil de {threshold_pct:.0f} % "
            "qui déclenche le signal d'achat de la stratégie."
        )
    elif divergence > partial_pct:
        sense = (
            f"La juste valeur théorique dépasse la capitalisation de "
            f"{divergence:.1f} %. L'écart n'atteint pas le seuil de "
            f"{threshold_pct:.0f} % du signal plein, mais il penche nettement "
            "du côté de la décote."
        )
    elif divergence < -threshold_pct:
        sense = (
            f"La capitalisation dépasse la juste valeur théorique de "
            f"{abs(divergence):.1f} %, au-delà du seuil de {threshold_pct:.0f} % "
            "qui déclenche le signal de vente de la stratégie."
        )
    elif divergence < -partial_pct:
        sense = (
            f"La capitalisation dépasse la juste valeur théorique de "
            f"{abs(divergence):.1f} %. L'écart n'atteint pas le seuil de "
            f"{threshold_pct:.0f} % du signal plein, mais il penche nettement "
            "du côté de la surcote."
        )
    else:
        sense = (
            f"L'écart de {divergence:+.1f} % reste dans la zone neutre de "
            f"±{partial_pct:.0f} % : le marché valorise la société à peu près "
            "comme le modèle."
        )

    explanation = (
        f"{sense} La valeur de la firme non endettée ressort à "
        f"{_format_amount(result.unlevered_value)}, en actualisant un résultat "
        f"d'exploitation net d'impôt de {_format_amount(result.nopat)} au taux "
        f"de {result.discount_rate * 100:.1f} % (bêta dé-leviérisé de "
        f"{result.unlevered_beta:.2f}), avec une croissance de "
        f"{result.growth_start * 100:.1f} % par an convergeant vers "
        f"{result.growth_rate * 100:.1f} % en dix ans. S'y ajoute le bouclier fiscal de la "
        f"dette pour {_format_amount(result.pv_tax_shield)} ; s'en retranchent "
        f"les coûts de détresse financière pour {_format_amount(result.pv_distress)} "
        f"(probabilité de défaut à un an : {result.prob_default * 100:.2f} %, "
        f"taux de destruction sectoriel : {result.distress_rate * 100:.0f} %), les "
        f"coûts d'agence pour {_format_amount(result.pv_agency)}, et la dette "
        f"nette pour {_format_amount(result.net_debt)}."
    )

    # Le chiffre le plus discutable du modèle, donc le plus utile à montrer :
    # l'écart de valorisation devient une hypothèse de croissance, et non un
    # verdict à prendre ou à laisser.
    if math.isfinite(result.implied_growth):
        gap = result.implied_growth - result.growth_start
        if abs(gap) >= 0.005:
            sense = "au-delà de" if gap > 0 else "en deçà de"
            explanation += (
                f" Autrement dit, le cours actuel suppose une croissance de "
                f"{result.implied_growth * 100:.1f} % par an, soit "
                f"{abs(gap) * 100:.1f} points {sense} ce que la société a "
                "réalisé."
            )

    return Pillar(
        key="capital_structure",
        name="Structure du capital (Modigliani-Miller)",
        score=score,
        weight=cfg.w_mm,
        available=True,
        headline=f"{result.divergence_pct:+.1f} %",
        verdict=_pillar_verdict(score),
        explanation=explanation,
        details={
            "divergence_pct": result.divergence_pct,
            "threshold_pct": threshold_pct,
            "fair_equity_value": result.fair_equity_value,
            "nopat": result.nopat,
            "unlevered_value": result.unlevered_value,
            "unlevered_beta": result.unlevered_beta,
            "discount_rate": result.discount_rate,
            "growth_rate": result.growth_rate,
            "growth_start": result.growth_start,
            "implied_growth": (
                result.implied_growth
                if math.isfinite(result.implied_growth) else None
            ),
            "sensitivity_grid": result.sensitivity_grid,
            "market_cap": result.market_cap,
            "net_debt": result.net_debt,
            "pv_tax_shield": result.pv_tax_shield,
            "pv_distress": result.pv_distress,
            "pv_agency": result.pv_agency,
            "prob_default": result.prob_default,
            "credit_spread": result.credit_spread,
            "distress_rate": result.distress_rate,
            "leverage_ratio": result.leverage_ratio,
            "interest_coverage": (
                result.interest_coverage
                if math.isfinite(result.interest_coverage) else None
            ),
            "sector": result.sector,
        },
    )


def _build_momentum_pillar(
    result: Optional[MomentumResult], cfg: ValuationConfig,
) -> Pillar:
    if result is None:
        return Pillar(
            key="momentum", name="Momentum 12-1 ajusté de la volatilité",
            score=0.0, weight=cfg.w_momentum, available=False, headline="n/d",
            verdict="NEUTRE",
            explanation=(
                "Historique insuffisant : le momentum 12-1 exige au moins "
                f"{cfg.momentum_months + cfg.momentum_skip + 1} mois de cours."
            ),
        )

    score = _normalise(result.excess_sharpe, cfg.momentum_scale, cfg)

    if math.isfinite(result.excess_sharpe):
        comparison = (
            "au-dessus" if result.excess_sharpe > 0 else "en dessous"
        )
        explanation = (
            f"Sur les 12 mois arrêtés il y a un mois, le titre progresse de "
            f"{result.raw * 100:+.1f} % pour une volatilité de "
            f"{result.volatility * 100:.0f} %, soit un momentum ajusté du risque "
            f"de {result.sharpe:.2f} — {comparison} du marché ({result.market_sharpe:.2f}). "
            "Le mois le plus récent est écarté pour neutraliser le retournement "
            "de court terme."
        )
    else:
        explanation = (
            f"Le titre progresse de {result.raw * 100:+.1f} % sur la fenêtre "
            "12-1, mais la comparaison au marché n'a pas pu être établie."
        )

    if result.crash_regime:
        explanation += (
            " Attention : la volatilité du marché signale un régime de krach de "
            "momentum, où ce signal s'inverse brutalement — son poids est réduit."
        )

    return Pillar(
        key="momentum",
        name="Momentum 12-1 ajusté de la volatilité",
        score=score,
        weight=cfg.w_momentum,
        available=True,
        headline=f"{result.raw * 100:+.1f} %",
        verdict=_pillar_verdict(score),
        explanation=explanation,
        details={
            "raw": result.raw,
            "volatility": result.volatility,
            "sharpe": result.sharpe,
            "market_raw": result.market_raw,
            "market_sharpe": result.market_sharpe,
            "excess_sharpe": result.excess_sharpe,
            "crash_regime": result.crash_regime,
            "window": f"{result.window_start} → {result.window_end}",
        },
    )


# --------------------------------------------------------------------------- #
#  Agrégation                                                                  #
# --------------------------------------------------------------------------- #

def _combine(pillars: List[Pillar], crash_regime: bool, cfg: ValuationConfig) -> float:
    """Score composite pondéré, renormalisé sur les piliers disponibles.

    Si un pilier manque (fondamentaux introuvables, historique trop court), son
    poids est redistribué sur les autres plutôt que compté comme un zéro : un
    signal absent n'est pas un signal neutre.
    """
    weights = {pillar.key: pillar.weight for pillar in pillars}

    if crash_regime and cfg.momentum_crash_dampen:
        # Comme dans l'algorithme de production : la moitié du poids du
        # momentum bascule sur l'alpha en régime de krach.
        transfer = weights.get("momentum", 0.0) * 0.5
        weights["momentum"] = weights.get("momentum", 0.0) - transfer
        weights["alpha"] = weights.get("alpha", 0.0) + transfer

    available = [p for p in pillars if p.available]
    total_weight = sum(weights[p.key] for p in available)
    if total_weight <= 0:
        return 0.0

    return float(sum(p.score * weights[p.key] for p in available) / total_weight)


def _verdict_from_score(score: float, cfg: ValuationConfig) -> tuple[str, str]:
    """Traduit le score composite en verdict et en libellé nuancé."""
    if score >= cfg.verdict_strong_threshold:
        return VERDICT_UNDERVALUED, "Fortement sous-évaluée"
    if score >= cfg.verdict_threshold:
        return VERDICT_UNDERVALUED, "Sous-évaluée"
    if score <= -cfg.verdict_strong_threshold:
        return VERDICT_OVERVALUED, "Fortement sur-évaluée"
    if score <= -cfg.verdict_threshold:
        return VERDICT_OVERVALUED, "Sur-évaluée"
    return VERDICT_FAIR, "Au juste prix"


def _confidence(pillars: List[Pillar], alpha: Optional[AlphaResult],
                warnings: List[str], cfg: ValuationConfig) -> float:
    """Indice de fiabilité du verdict, entre 0 et 1.

    Agrège la couverture des piliers, la longueur de l'historique de
    régression et le nombre d'alertes sur la qualité des données.
    """
    available = [p for p in pillars if p.available]
    coverage = sum(p.weight for p in available) / max(
        sum(p.weight for p in pillars), 1e-9
    )

    history = 1.0
    if alpha is not None:
        # Plein crédit à partir de cfg.lookback_months mois de régression.
        history = min(alpha.n_obs / cfg.lookback_months, 1.0)
    else:
        history = 0.4

    penalty = min(len(warnings) * 0.08, 0.30)
    return float(np.clip(0.55 * coverage + 0.45 * history - penalty, 0.0, 1.0))


def _short_name(pillar: Pillar) -> str:
    """Nom du pilier ramené à sa partie utile, pour une énumération."""
    return pillar.name.split("(")[0].strip().lower()


def _summary(analysis_ticker: str, verdict: str, label: str, score: float,
             upside: float, pillars: List[Pillar]) -> str:
    """Phrase de synthèse affichée en tête du dashboard."""
    available = [p for p in pillars if p.available]
    leaning_under = [p for p in available if p.verdict == VERDICT_UNDERVALUED]
    leaning_over = [p for p in available if p.verdict == VERDICT_OVERVALUED]

    if verdict == VERDICT_FAIR:
        opening = (
            f"{analysis_ticker} ressort au juste prix : le score composite de "
            f"{score:+.2f} reste dans la zone neutre."
        )
    else:
        sense = "décote" if verdict == VERDICT_UNDERVALUED else "surcote"
        opening = (
            f"{analysis_ticker} ressort {label.lower()} : score composite de "
            f"{score:+.2f}, soit une {sense} par rapport à la valeur que "
            "le modèle Taurus justifie."
        )

    if math.isfinite(upside) and abs(upside) > 0.5:
        opening += (
            f" La juste valeur Modigliani-Miller implique un potentiel de "
            f"{upside:+.0f} % par rapport au cours actuel."
        )

    # Le désaccord entre piliers est l'information la plus utile : c'est lui
    # qui explique un score composite modéré, et c'est à l'analyste de le
    # trancher. Un pilier plus optimiste que le verdict d'ensemble n'est pas
    # « en sens inverse » pour autant — seule une opposition franche entre
    # sous-évaluation et surévaluation mérite ce terme.
    if leaning_under and leaning_over:
        opening += (
            f" Les piliers divergent : {', '.join(_short_name(p) for p in leaning_under)} "
            f"vers la sous-évaluation, {', '.join(_short_name(p) for p in leaning_over)} "
            "vers la surévaluation."
        )
    elif verdict == VERDICT_FAIR and (leaning_under or leaning_over):
        leaning = leaning_under or leaning_over
        direction = "la sous-évaluation" if leaning_under else "la surévaluation"
        plural = "s" if len(leaning) > 1 else ""
        opening += (
            f" Le{plural} pilier{plural} {', '.join(_short_name(p) for p in leaning)} "
            f"penche{'nt' if plural else ''} vers {direction}, sans entraîner les "
            "autres : le verdict d'ensemble reste au juste prix."
        )
    elif verdict != VERDICT_FAIR:
        agreeing = leaning_under if verdict == VERDICT_UNDERVALUED else leaning_over
        neutral = [p for p in available if p.verdict == "NEUTRE"]
        if agreeing:
            opening += f" Piliers concordants : {', '.join(_short_name(p) for p in agreeing)}."
        if neutral:
            opening += f" Sans opinion : {', '.join(_short_name(p) for p in neutral)}."

    return opening


# --------------------------------------------------------------------------- #
#  Point d'entrée                                                              #
# --------------------------------------------------------------------------- #

def normalise_ticker(ticker: str) -> str:
    """Valide et normalise un ticker saisi par l'utilisateur."""
    candidate = (ticker or "").strip().upper()
    if not candidate:
        raise InvalidTickerError("Aucun ticker fourni.")
    if not TICKER_PATTERN.match(candidate):
        raise InvalidTickerError(
            f"« {ticker} » n'est pas un ticker valide : attendu 1 à 10 "
            "caractères alphanumériques (par exemple AAPL, MSFT, BRK-B)."
        )
    return candidate


# --------------------------------------------------------------------------- #
#  Pilier Modigliani-Miller : rapprochement des devises                        #
# --------------------------------------------------------------------------- #

def _valuation_pillar(
    accounting,
    quote,
    history,
    price: float,
    quote_currency: str,
    monthly_prices: pd.Series,
    alpha_result: Optional[AlphaResult],
    cfg: ValuationConfig,
) -> tuple[float, float, float, Optional[MMResult], List[str], Optional[Callable[[float], float]]]:
    """Calcule la juste valeur MM en ramenant tout dans la devise des comptes.

    Deux pièges spécifiques aux titres étrangers :

      • Les comptes et la cotation ne sont pas toujours dans la même devise.
        ASML publie en euros et cote en dollars à New York : comparer
        directement ses fondamentaux à sa capitalisation mesurerait la parité
        EUR/USD, pas une décote.

      • Un certificat de dépôt américain ne vaut pas une action. Un ADR Toyota
        représente dix actions ordinaires, or SEC EDGAR publie le nombre
        d'ACTIONS ORDINAIRES. Reconstituer la capitalisation par « actions ×
        cours » la multiplierait par dix. On interroge donc un fournisseur qui
        connaît le titre coté, et la reconstitution n'intervient qu'à défaut,
        avec un avertissement.

    Renvoie (capitalisation dans la devise de COTATION — celle que l'utilisateur
    voit à côté du cours —, juste valeur par action, capitalisation dans la
    devise des comptes, résultat MM, messages).
    """
    notes: List[str] = []
    accounting_currency = (accounting.currency or quote_currency).upper()

    # ── Capitalisation, dans la devise des comptes ─────────────────────── #
    market_cap = float("nan")
    market_cap_currency = accounting_currency

    if quote is not None and math.isfinite(quote.market_cap):
        market_cap = quote.market_cap
        market_cap_currency = (quote.currency or quote_currency).upper()
    else:
        shares = accounting.shares_outstanding
        if math.isfinite(shares) and shares > 0 and math.isfinite(price):
            market_cap = shares * price
            market_cap_currency = quote_currency
            if accounting_currency != quote_currency:
                notes.append(
                    "Capitalisation reconstituée à partir du nombre d'actions "
                    "publié à la SEC et du cours. S'il s'agit d'un certificat "
                    "de dépôt (ADR) représentant plusieurs actions ordinaires, "
                    "elle est surestimée d'autant et la juste valeur est à "
                    "lire avec prudence."
                )
        else:
            notes.append(
                "Capitalisation boursière indisponible : la juste valeur "
                "Modigliani-Miller n'a pas pu être calculée."
            )
            return float("nan"), float("nan"), float("nan"), None, notes, None

    # Capitalisation telle qu'elle sera AFFICHÉE, dans la devise de cotation :
    # la montrer en euros sous un symbole dollar induirait en erreur.
    display_cap = market_cap
    if market_cap_currency != quote_currency:
        display_rate = fx_provider.latest_rate(market_cap_currency, quote_currency, cfg)
        display_cap = market_cap * display_rate if display_rate is not None else float("nan")

    # Conversion vers la devise des comptes, pour l'écran Modigliani-Miller.
    if market_cap_currency != accounting_currency:
        rate = fx_provider.latest_rate(market_cap_currency, accounting_currency, cfg)
        if rate is None:
            notes.append(
                f"Comptes en {accounting_currency}, capitalisation en "
                f"{market_cap_currency}, et taux de change indisponible : "
                "la juste valeur Modigliani-Miller n'a pas pu être calculée."
            )
            return display_cap, float("nan"), float("nan"), None, notes, None
        market_cap = market_cap * rate

    if not math.isfinite(market_cap) or market_cap <= 0:
        notes.append(
            "Capitalisation boursière inexploitable : la juste valeur "
            "Modigliani-Miller n'a pas pu être calculée."
        )
        return display_cap, float("nan"), float("nan"), None, notes, None

    # ── Volatilité des capitaux propres, pour le modèle de Merton ──────── #
    recent = monthly_prices.pct_change(fill_method=None).dropna().tail(cfg.lookback_months)
    equity_vol = float(recent.std() * np.sqrt(12)) if len(recent) >= 12 else 0.30

    # Le bêta vient de la régression Fama-French : même titre, même fenêtre,
    # donc un bêta cohérent avec l'alpha affiché à côté.
    levered_beta = (
        alpha_result.betas.get("Mkt-RF", float("nan"))
        if alpha_result is not None else float("nan")
    )

    # Le nombre d'actions sert à ramener la juste valeur à un prix. Il doit
    # correspondre au titre COTÉ : on le déduit de la capitalisation et du
    # cours, tous deux relatifs au même titre, plutôt que du chiffre SEC qui
    # porte sur les actions ordinaires.
    implied_shares = float("nan")
    if math.isfinite(price) and price > 0:
        price_in_accounting = price
        if quote_currency != accounting_currency:
            rate = fx_provider.latest_rate(quote_currency, accounting_currency, cfg)
            price_in_accounting = price * rate if rate is not None else float("nan")
        if math.isfinite(price_in_accounting) and price_in_accounting > 0:
            implied_shares = market_cap / price_in_accounting

    fundamentals_row = accounting.to_dict()

    def revalue(candidate_price: float) -> float:
        """Juste valeur par action si le titre cotait `candidate_price`.

        La juste valeur n'est pas tout à fait indépendante du cours : la
        dé-leviérisation de Hamada prend D/E en valeur de marché, et le modèle
        de Merton une valeur de firme qui contient la capitalisation. Résoudre
        le prix d'équilibre demande donc d'itérer, pas de diviser une fois.
        """
        if not (math.isfinite(candidate_price) and candidate_price > 0
                and math.isfinite(implied_shares) and implied_shares > 0):
            return float("nan")
        candidate_cap = implied_shares * candidate_price
        candidate = mm_valuation(
            fundamentals_row, candidate_cap, equity_vol,
            implied_shares, levered_beta, cfg,
        )
        return candidate.fair_value_per_share if candidate is not None else float("nan")

    result = mm_valuation(
        fundamentals_row, market_cap, equity_vol,
        implied_shares, levered_beta, cfg,
    )

    if result is None:
        notes.append(
            "Juste valeur Modigliani-Miller non calculable : le résultat "
            "d'exploitation sur 12 mois glissants est négatif ou nul, une "
            "valorisation par perpétuité n'aurait pas de sens."
        )
        return display_cap, float("nan"), market_cap, None, notes, None

    notes.extend(result.notes)
    alert = leverage_alert(result, accounting.ebit, accounting.total_debt, cfg)
    if alert:
        notes.append(alert)

    # La juste valeur par action revient dans la devise de cotation, pour être
    # comparable au cours affiché.
    fair_value = result.fair_value_per_share
    if math.isfinite(fair_value) and accounting_currency != quote_currency:
        rate = fx_provider.latest_rate(accounting_currency, quote_currency, cfg)
        fair_value = fair_value * rate if rate is not None else float("nan")

    return display_cap, fair_value, market_cap, result, notes, revalue


def _no_prices_message(symbol: str, failures: Dict[str, str]) -> str:
    """Explique pourquoi aucune source de cours n'a répondu.

    Renvoyer « vérifiez le ticker » quand le ticker est correct envoie
    l'utilisateur corriger une saisie qui n'a rien à se reprocher. Une place
    locale n'est couverte que par Yahoo et Financial Modeling Prep : quand
    Yahoo est au quota et qu'aucune clé n'est configurée, le ticker n'y est
    pour rien et le dire évite une recherche inutile.
    """
    suffix = regions_provider.suffix_of(symbol)
    has_fmp_key = bool(os.environ.get("FMP_API_KEY", "").strip())

    if suffix:
        message = (
            f"« {symbol} » désigne une cotation sur une place locale. Seuls "
            "Yahoo Finance et Financial Modeling Prep couvrent ces places, et "
            "aucun des deux n'a répondu."
        )
        if not has_fmp_key:
            message += (
                " Aucune clé Financial Modeling Prep n'est configurée, et "
                "Yahoo limite le débit par adresse IP."
            )
        message += (
            " Deux issues : saisir la cotation américaine de la société "
            "lorsqu'elle existe — ASML, SAP, TM, TSM, SHEL se cherchent ainsi, "
            "sans suffixe — ou configurer une clé Financial Modeling Prep. "
            "Le bouton « Diagnostic des sources » indique l'état de chaque "
            "fournisseur."
        )
        return message

    if failures and all(reason == "aucune donnée exploitable"
                        for reason in failures.values()):
        return (
            f"Aucune source ne connaît « {symbol} ». Vérifiez l'orthographe du "
            "ticker ; une place locale s'écrit avec son suffixe, par exemple "
            "MC.PA, 7203.T ou 0700.HK."
        )

    return (
        f"Aucune donnée de marché trouvée pour « {symbol} ». Les fournisseurs "
        "gratuits limitent le débit des requêtes : réessayez dans quelques "
        "minutes, ou consultez le « Diagnostic des sources » pour savoir "
        "lequel fait défaut."
    )


def _pillar_weights(pillars: List[Pillar], crash_regime: bool,
                    cfg: ValuationConfig) -> Dict[str, float]:
    """Poids effectifs des piliers, amortissement de krach compris."""
    weights = {p.key: p.weight for p in pillars}
    if crash_regime and cfg.momentum_crash_dampen:
        transfer = weights.get("momentum", 0.0) * 0.5
        weights["momentum"] = weights.get("momentum", 0.0) - transfer
        weights["alpha"] = weights.get("alpha", 0.0) + transfer
    return weights


def _price_for_score(
    target: float,
    pillars: List[Pillar],
    weights: Dict[str, float],
    revalue: Optional[Callable[[float], float]],
    current_price: float,
    cfg: ValuationConfig,
) -> tuple[float, str]:
    """Cours auquel le score COMPOSITE atteindrait `target`.

    Les seuils de l'algorithme portent sur un score sans dimension, or c'est un
    cours que l'on regarde. Traduire l'un dans l'autre demande de répondre à :
    « à quel prix ce titre basculerait-il ? »

    Un seul pilier dépend du cours du jour. L'alpha mesure soixante mois de
    performance passée et le momentum douze mois arrêtés il y a un mois : un
    prix hypothétique aujourd'hui ne réécrit pas cette histoire. C'est donc la
    juste valeur Modigliani-Miller, seule à confronter l'entreprise à son cours,
    qui porte la variation — les deux autres piliers gardent leur contribution.

    Renvoie (cours, raison). Le cours vaut NaN dans deux cas bien distincts,
    que la raison sépare : soit aucun pilier ne dépend du cours — le pilier
    Modigliani-Miller est indisponible, faute de fondamentaux —, soit il est
    disponible mais sature avant d'emporter le verdict, les deux autres
    piliers s'y opposant. Les confondre sous un même « hors d'atteinte »
    laisserait croire à un jugement du modèle là où il n'y a qu'une donnée
    manquante.
    """
    available = [p for p in pillars if p.available]
    mm = next((p for p in available if p.key == "capital_structure"), None)
    if mm is None or revalue is None:
        return float("nan"), "pilier_absent"

    total_weight = sum(weights[p.key] for p in available)
    mm_weight = weights.get("capital_structure", 0.0)
    if total_weight <= 0 or mm_weight <= 0:
        return float("nan"), "pilier_absent"

    # Contribution figée des piliers insensibles au cours.
    fixed = sum(p.score * weights[p.key] for p in available if p.key != "capital_structure")

    needed = (target * total_weight - fixed) / mm_weight
    if abs(needed) > cfg.score_clip:
        # Au-delà du bornage, le pilier sature : aucun cours ne suffit.
        return float("nan"), "sature"

    # Score du pilier → écart de valorisation visé.
    divergence = needed * cfg.leverage_gap_threshold

    # La juste valeur dépend faiblement du cours (Hamada, Merton) : on itère.
    price = current_price
    for _ in range(8):
        fair = revalue(price)
        if not math.isfinite(fair) or fair <= 0:
            return float("nan"), "non_resolu"
        candidate = fair / (1.0 + divergence)
        if not math.isfinite(candidate) or candidate <= 0:
            return float("nan"), "non_resolu"
        converged = abs(candidate / price - 1.0) < 1e-4
        price = candidate
        if converged:
            break

    return float(price), "atteignable"


def analyze(ticker: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> Analysis:
    """Analyse complète d'un titre : les trois piliers, puis le verdict.

    Fonctionne sur une cotation américaine (« AAPL », « ASML ») comme sur une
    place locale (« MC.PA », « 7203.T », « 0700.HK »). Trois différences de
    traitement en découlent :

      • la régression utilise le jeu de facteurs Fama-French de la RÉGION du
        titre, et non systématiquement celui des États-Unis ;
      • les facteurs internationaux étant libellés en dollars, un titre coté
        dans une autre devise est converti avant la régression ;
      • l'écran Modigliani-Miller ramène fondamentaux et capitalisation dans
        une même devise — ASML publie en euros et cote en dollars.

    Raises
    ------
    InvalidTickerError : ticker mal formé.
    TickerError        : aucune donnée de marché disponible.
    """
    started = time.perf_counter()
    symbol = normalise_ticker(ticker)
    warnings: List[str] = []
    sources: Dict[str, str] = {}

    # ── 1. Cours ───────────────────────────────────────────────────────── #
    price_failures: Dict[str, str] = {}
    history = prices_provider.get_monthly_prices(symbol, cfg, failures=price_failures)
    if history is None:
        raise TickerError(_no_prices_message(symbol, price_failures))
    sources["prix"] = history.source

    # Londres cote en pence, pas en livres : sans cette normalisation la
    # capitalisation serait divisée par cent.
    quote_currency, subunit_factor = fx_provider.normalise_currency(history.currency)
    price = history.last_price * subunit_factor
    monthly_prices = history.monthly * subunit_factor

    # ── 2. Fondamentaux, capitalisation et secteur ─────────────────────── #
    accounting = fundamentals_provider.get_fundamentals(symbol, cfg)
    quote = quotes_provider.get_quote(symbol, cfg)

    # ── Rendement total ────────────────────────────────────────────────── #
    # Quand la source de cours ne réintègre pas les dividendes, ils sont
    # reconstitués depuis les comptes SEC EDGAR déjà téléchargés. Le biais
    # corrigé atteint 0,9 point de t-stat sur une valeur de rendement et peut
    # inverser le signe de son alpha.
    performance_prices = monthly_prices
    if not history.total_return:
        rebuilt = None
        if accounting is not None:
            rebuilt = reconstruct_total_return(
                monthly_prices,
                accounting.dividends_per_share,
                currency_matches=(accounting.currency or quote_currency).upper()
                                 == quote_currency,
            )
        if rebuilt is not None:
            performance_prices = rebuilt.prices
            sources["dividendes"] = "SEC EDGAR (reconstitués)"
            warnings.extend(rebuilt.notes)
            warnings.append(
                f"Les cours de {history.source} excluent les dividendes ; ils "
                f"ont été reconstitués depuis les comptes SEC EDGAR "
                f"({rebuilt.quarters_used} trimestres, rendement médian de "
                f"{rebuilt.annual_yield * 100:.1f} % par an). La SEC date un "
                "dividende par la fin de la période comptable où il est "
                "déclaré, pas par son détachement : l'alpha et le momentum sont "
                "corrigés en niveau, pas au mois près."
            )
        else:
            warnings.append(
                f"Les cours proviennent de {history.source}, qui ne réintègre "
                "pas les dividendes, et les comptes SEC EDGAR n'en donnent pas "
                "assez pour les reconstituer : l'alpha et le momentum sont "
                "sous-estimés à hauteur du rendement du dividende."
            )

    company_name, sector = symbol, "Unknown"
    accounting_currency = quote_currency
    country = ""

    if accounting is not None:
        sources["fondamentaux"] = accounting.source
        company_name = accounting.company_name or symbol
        sector = accounting.sector
        accounting_currency = (accounting.currency or quote_currency).upper()
        country = accounting.country
        warnings.extend(accounting.warnings)
    else:
        warnings.append(
            "Fondamentaux comptables introuvables : le pilier Modigliani-Miller "
            "est neutralisé et son poids reporté sur les deux autres. Hors des "
            "sociétés déposant auprès de la SEC, une clé Financial Modeling "
            "Prep est nécessaire."
        )

    # Secteur : le code SIC déposé à la SEC d'abord, le fournisseur de cotation
    # seulement en secours. Les deux se trompent, mais pas de la même façon —
    # le SIC, figé sur la nomenclature de 1987, range l'équipement de
    # semi-conducteurs dans les machines industrielles (corrigé dans
    # `sectors.py`), tandis que les taxonomies commerciales dérivent
    # franchement : Nasdaq classe Altria, cigarettier, en « Health Care ».
    # Le SIC est déposé légalement, stable et auditable, et la table de
    # correspondance est sous notre contrôle ; c'est donc lui qui prime.
    # Le secteur pilote le taux de destruction en faillite du modèle
    # Modigliani-Miller — 35 % pour la santé contre 20 % pour la consommation
    # de base — l'erreur n'est donc pas cosmétique.
    if quote is not None:
        sources["capitalisation"] = quote.source
        if sector in ("", "Unknown") and quote.sector != "Unknown":
            sector = quote.sector
        # Hors du périmètre SEC — une cotation locale, par exemple — le nom
        # vient du fournisseur de cotation, faute de quoi la carte afficherait
        # « MC.PA » en guise de raison sociale.
        if company_name in ("", symbol) and quote.company_name:
            company_name = quote.company_name

    # ── 3. Région et facteurs Fama-French ──────────────────────────────── #
    guess = regions_provider.detect_region(symbol, country, accounting_currency)
    factor_frame = factors_provider.get_ff5_factors(guess.region, cfg)

    if factor_frame is None:
        warnings.append(
            "Facteurs Fama-French indisponibles : l'alpha et la comparaison de "
            "momentum au marché n'ont pas pu être calculés."
        )
        alpha_result = None
        momentum_result = None
    else:
        sources["facteurs"] = factors_provider.factor_label(guess.region)
        if guess.evidence == "devise" and not guess.confident:
            warnings.append(
                f"Région déduite de la seule devise de publication "
                f"({accounting_currency}) : {guess.label}. Si le siège est "
                "ailleurs, le jeu de facteurs retenu n'est pas le bon."
            )

        # Les facteurs internationaux de Kenneth French sont libellés en
        # dollars. Un titre coté en euros ou en yens doit l'être aussi, sans
        # quoi son alpha absorberait la variation de sa devise.
        regression_prices = performance_prices
        if quote_currency != "USD":
            converted = fx_provider.convert_series(
                performance_prices, quote_currency, "USD", cfg,
            )
            if converted is not None and len(converted) >= cfg.hard_min_obs:
                regression_prices = converted
                sources["change"] = "Banque centrale européenne"
            else:
                warnings.append(
                    f"Conversion {quote_currency} → USD impossible : la "
                    "régression compare des rendements en "
                    f"{quote_currency} à des facteurs en dollars, et l'alpha "
                    "absorbe la variation de change."
                )

        stock_returns = regression_prices.pct_change(fill_method=None).dropna()
        alpha_result = compute_alpha(stock_returns, factor_frame, cfg)
        momentum_result = compute_momentum(
            regression_prices, factors_provider.market_returns(factor_frame), cfg,
        )

    # ── 4. Juste valeur Modigliani-Miller ──────────────────────────────── #
    mm_result: Optional[MMResult] = None
    fair_value = float("nan")
    revalue: Optional[Callable[[float], float]] = None

    # La capitalisation vient du titre coté : elle est connue même sans
    # comptes, ce qui est le cas d'une cotation locale hors périmètre SEC.
    # `_valuation_pillar` l'affinera si les fondamentaux sont disponibles.
    market_cap = quote.market_cap if quote is not None else float("nan")

    if accounting is not None:
        market_cap, fair_value, _mm_cap, mm_result, mm_notes, revalue = _valuation_pillar(
            accounting, quote, history, price, quote_currency,
            monthly_prices, alpha_result, cfg,
        )
        warnings.extend(mm_notes)

    # ── 5. Piliers, score composite, verdict ───────────────────────────── #
    pillars = [
        _build_alpha_pillar(alpha_result, cfg),
        _build_mm_pillar(mm_result, cfg),
        _build_momentum_pillar(momentum_result, cfg),
    ]

    crash_regime = bool(momentum_result and momentum_result.crash_regime)
    composite = _combine(pillars, crash_regime, cfg)
    verdict, label = _verdict_from_score(composite, cfg)

    upside = (
        (fair_value / price - 1.0) * 100.0
        if math.isfinite(fair_value) and math.isfinite(price) and price > 0
        else float("nan")
    )

    # Bornes en prix du verdict, tous piliers confondus : au-dessous de la
    # première le composite passe en « sous-évaluée », au-dessus de la seconde
    # en « sur-évaluée ».
    weights = _pillar_weights(pillars, crash_regime, cfg)
    buy_below_price, buy_reason = _price_for_score(
        cfg.verdict_threshold, pillars, weights, revalue, price, cfg,
    )
    sell_above_price, _sell_reason = _price_for_score(
        -cfg.verdict_threshold, pillars, weights, revalue, price, cfg,
    )

    if not [p for p in pillars if p.available]:
        raise TickerError(
            f"Aucun pilier d'analyse n'a pu être calculé pour « {symbol} »."
        )

    return Analysis(
        ticker=symbol,
        company_name=company_name,
        sector=sector,
        currency=quote_currency,
        region=guess.region,
        region_label=guess.label,
        verdict=verdict,
        verdict_label=label,
        composite_score=composite,
        confidence=_confidence(pillars, alpha_result, warnings, cfg),
        summary=_summary(symbol, verdict, label, composite, upside, pillars),
        price=price,
        fair_value=fair_value,
        upside_pct=upside,
        market_cap=market_cap,
        buy_below=buy_below_price,
        sell_above=sell_above_price,
        buy_below_reason=buy_reason,
        pillars=pillars,
        warnings=warnings,
        data_sources=sources,
        computed_at=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        alpha=alpha_result,
        mm=mm_result,
        momentum=momentum_result,
    )
