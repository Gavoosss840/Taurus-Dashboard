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
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .alpha import AlphaResult, compute_alpha
from .capital_structure import MMResult, leverage_alert, mm_valuation
from .config import DEFAULT_CONFIG, ValuationConfig
from .momentum import MomentumResult, compute_momentum
from .providers import factors as factors_provider
from .providers import fundamentals as fundamentals_provider
from .providers import prices as prices_provider

logger = logging.getLogger(__name__)

# Un ticker boursier : lettres, chiffres, point, tiret (BRK-B, BRK.B, RDS-A).
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

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
    currency: str

    verdict: str
    verdict_label: str           # libellé nuancé (« fortement sous-évaluée »)
    composite_score: float
    confidence: float            # 0 → 1
    summary: str

    price: float
    fair_value: float
    upside_pct: float
    market_cap: float

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

    if result.significant:
        sense = "supérieure" if result.direction > 0 else "inférieure"
        explanation = (
            f"Sur {result.n_obs} mois, le titre dégage une performance {sense} "
            f"de {abs(result.alpha_annual) * 100:.1f} % par an à ce que son "
            f"exposition aux cinq facteurs de risque justifie. L'écart est "
            f"statistiquement significatif (t = {result.alpha_tstat:.2f}, "
            f"p = {result.p_value:.3f}), donc peu susceptible d'être dû au hasard."
        )
    else:
        explanation = (
            f"L'alpha ressort à {result.alpha_annual * 100:+.1f} % par an mais "
            f"n'est pas significatif (t = {result.alpha_tstat:.2f}, seuil "
            f"{result.t_critical:.2f}) : sur {result.n_obs} mois, on ne peut pas "
            "le distinguer du bruit. Les cinq facteurs de risque expliquent "
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
        f"{result.unlevered_beta:.2f}) avec une croissance perpétuelle de "
        f"{result.growth_rate * 100:.1f} %. S'y ajoute le bouclier fiscal de la "
        f"dette pour {_format_amount(result.pv_tax_shield)} ; s'en retranchent "
        f"les coûts de détresse financière pour {_format_amount(result.pv_distress)} "
        f"(probabilité de défaut à un an : {result.prob_default * 100:.2f} %, "
        f"taux de destruction sectoriel : {result.distress_rate * 100:.0f} %), les "
        f"coûts d'agence pour {_format_amount(result.pv_agency)}, et la dette "
        f"nette pour {_format_amount(result.net_debt)}."
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


def analyze(ticker: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> Analysis:
    """Analyse complète d'un titre : les trois piliers, puis le verdict.

    Raises
    ------
    TickerError : ticker mal formé, ou aucune donnée de marché disponible.
    """
    started = time.perf_counter()
    symbol = normalise_ticker(ticker)
    warnings: List[str] = []
    sources: Dict[str, str] = {}

    # ── 1. Cours ───────────────────────────────────────────────────────── #
    history = prices_provider.get_monthly_prices(symbol, cfg)
    if history is None:
        raise TickerError(
            f"Aucune donnée de marché trouvée pour « {symbol} ». Vérifiez le "
            "ticker, ou réessayez : les fournisseurs gratuits limitent parfois "
            "le débit des requêtes."
        )
    sources["prix"] = history.source
    if not history.total_return:
        warnings.append(
            f"Les cours proviennent de {history.source}, qui ne réintègre pas "
            "les dividendes : l'alpha et le momentum sont sous-estimés à "
            "hauteur du rendement du dividende."
        )

    # ── 2. Facteurs Fama-French ────────────────────────────────────────── #
    factor_frame = factors_provider.get_ff5_factors(cfg)
    if factor_frame is None:
        warnings.append(
            "Facteurs Fama-French indisponibles : l'alpha et la comparaison de "
            "momentum au marché n'ont pas pu être calculés."
        )
        alpha_result = None
        momentum_result = None
    else:
        sources["facteurs"] = "Kenneth R. French Data Library"
        alpha_result = compute_alpha(history.returns, factor_frame, cfg)
        momentum_result = compute_momentum(
            history.monthly, factors_provider.market_returns(factor_frame), cfg,
        )

    # ── 3. Fondamentaux et juste valeur MM ─────────────────────────────── #
    accounting = fundamentals_provider.get_fundamentals(symbol, cfg)
    mm_result: Optional[MMResult] = None
    company_name, sector = symbol, "Unknown"
    market_cap = float("nan")

    if accounting is None:
        warnings.append(
            "Fondamentaux comptables introuvables : le pilier Modigliani-Miller "
            "est neutralisé et son poids reporté sur les deux autres."
        )
    else:
        sources["fondamentaux"] = accounting.source
        company_name = accounting.company_name or symbol
        sector = accounting.sector
        warnings.extend(accounting.warnings)

        # La capitalisation se déduit du dernier cours et du nombre d'actions :
        # deux chiffres frais, plutôt qu'une capitalisation publiée qui peut
        # dater de plusieurs semaines.
        shares = accounting.shares_outstanding
        if math.isfinite(shares) and shares > 0 and math.isfinite(history.last_price):
            market_cap = shares * history.last_price
        else:
            warnings.append(
                "Nombre d'actions en circulation indisponible : la "
                "capitalisation boursière n'a pas pu être reconstituée."
            )

        # Volatilité des capitaux propres pour le modèle de Merton, estimée
        # sur les 60 derniers mois comme dans l'écran de production.
        recent_returns = history.returns.tail(cfg.lookback_months)
        equity_vol = (
            float(recent_returns.std() * np.sqrt(12))
            if len(recent_returns) >= 12 else 0.30
        )

        # Le bêta de marché vient de la régression Fama-French : c'est le
        # même titre, la même fenêtre, donc un bêta cohérent avec l'alpha —
        # plutôt qu'un bêta publié par un fournisseur tiers sur un autre
        # horizon.
        levered_beta = (
            alpha_result.betas.get("Mkt-RF", float("nan"))
            if alpha_result is not None else float("nan")
        )

        if math.isfinite(market_cap) and market_cap > 0:
            mm_result = mm_valuation(
                accounting.to_dict(), market_cap, equity_vol,
                accounting.shares_outstanding, levered_beta, cfg,
            )
            if mm_result is None:
                warnings.append(
                    "Juste valeur Modigliani-Miller non calculable : le "
                    "résultat d'exploitation sur 12 mois glissants est négatif "
                    "ou nul, une valorisation par perpétuité n'aurait pas de "
                    "sens."
                )
            else:
                warnings.extend(mm_result.notes)
                alert = leverage_alert(
                    mm_result, accounting.ebit, accounting.total_debt, cfg,
                )
                if alert:
                    warnings.append(alert)

    # ── 4. Piliers, score composite, verdict ───────────────────────────── #
    pillars = [
        _build_alpha_pillar(alpha_result, cfg),
        _build_mm_pillar(mm_result, cfg),
        _build_momentum_pillar(momentum_result, cfg),
    ]

    crash_regime = bool(momentum_result and momentum_result.crash_regime)
    composite = _combine(pillars, crash_regime, cfg)
    verdict, label = _verdict_from_score(composite, cfg)

    fair_value = mm_result.fair_value_per_share if mm_result else float("nan")
    price = history.last_price
    upside = (
        (fair_value / price - 1.0) * 100.0
        if math.isfinite(fair_value) and math.isfinite(price) and price > 0
        else float("nan")
    )

    if not [p for p in pillars if p.available]:
        raise TickerError(
            f"Aucun pilier d'analyse n'a pu être calculé pour « {symbol} »."
        )

    return Analysis(
        ticker=symbol,
        company_name=company_name,
        sector=sector,
        currency=history.currency,
        verdict=verdict,
        verdict_label=label,
        composite_score=composite,
        confidence=_confidence(pillars, alpha_result, warnings, cfg),
        summary=_summary(symbol, verdict, label, composite, upside, pillars),
        price=price,
        fair_value=fair_value,
        upside_pct=upside,
        market_cap=market_cap,
        pillars=pillars,
        warnings=warnings,
        data_sources=sources,
        computed_at=pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        alpha=alpha_result,
        mm=mm_result,
        momentum=momentum_result,
    )
