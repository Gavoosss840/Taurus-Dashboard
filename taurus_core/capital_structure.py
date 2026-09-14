"""
Taurus Dashboard – Pilier 2 : juste valeur Modigliani-Miller (méthode APV).

Reprend `taurus/capital_structure.py` en corrigeant une circularité du modèle
d'origine.

Le modèle de production calcule la valeur de la firme non endettée par
    V_U = capitalisation + dette nette − VA(bouclier fiscal)
puis la valeur de la firme endettée par
    V_L = V_U + VA(bouclier fiscal) − VA(détresse) − coûts d'agence

Le bouclier fiscal s'annule entre les deux lignes, et il reste
    valeur théorique des capitaux propres = capitalisation − détresse − agence

c'est-à-dire la capitalisation boursière diminuée de deux frottements
positifs. La divergence qui en découle vaut −(détresse + agence)/capitalisation,
une quantité TOUJOURS négative ou nulle : le signal ne peut structurellement
jamais désigner une société comme sous-évaluée. Vérifié numériquement sur
l'algorithme de production (voir docs/METHODOLOGIE.md). En cross-sectionnel
la conséquence reste limitée — le classement relatif garde un sens et la
stratégie bascule sur ses branches de repli — mais un verdict mono-titre
exige une valeur indépendante du cours.

La valeur non endettée est donc estimée ici à partir des FONDAMENTAUX, selon
la valeur actuelle ajustée (APV), qui est la formulation canonique de
Modigliani-Miller :

    NOPAT = EBIT × (1 − τ)                       résultat d'exploitation net d'impôt
    β_U   = β_L / (1 + (1 − τ) · D/E)            dé-leviérisation de Hamada
    r_U   = rf + β_U × prime de risque           MEDAF sans effet de levier
    V_U   = NOPAT × (1 + g) / (r_U − g)          perpétuité croissante

    V_L   = V_U + VA(bouclier fiscal) − VA(détresse) − coûts d'agence
    capitaux propres théoriques = V_L − dette nette

Les trois frottements — bouclier fiscal, coûts de détresse de Merton et coûts
d'agence — sont calculés exactement comme dans l'algorithme de production.

Le modèle de Merton traite les capitaux propres comme une option d'achat sur
l'actif économique, dont le prix d'exercice est la dette. La loi de Student
(ν = 5) y remplace la loi normale pour tenir compte des queues épaisses des
rendements, qu'une loi normale sous-estime gravement.

Divergence = (capitaux propres théoriques − capitalisation) / capitalisation
    > +25 %  → sous-évaluée
    < −25 %  → sur-évaluée

Sensibilité : une valorisation par perpétuité dépend fortement de r_U et de g.
`sensitivity_grid` expose cette dépendance plutôt que de la masquer derrière
un chiffre unique.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm, t as student_t

from .config import DEFAULT_CONFIG, ValuationConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Coûts de détresse par secteur                                               #
# --------------------------------------------------------------------------- #
# Fraction de la valeur de la firme détruite en cas de faillite.  Une société
# technologique perd son capital humain, sa propriété intellectuelle et ses
# contrats ; un service public réglementé conserve des actifs physiques que le
# liquidateur revend près de leur valeur comptable.
# Sources : Altman (1984), Andrade & Kaplan (1998), Bris et al. (2006).

SECTOR_DISTRESS_RATE: Dict[str, float] = {
    "Information Technology": 0.40,
    "Communication Services": 0.35,
    "Health Care": 0.35,
    "Consumer Discretionary": 0.25,
    "Consumer Staples": 0.20,
    "Industrials": 0.18,
    "Materials": 0.15,
    "Energy": 0.15,
    "Financials": 0.10,
    "Real Estate": 0.08,
    "Utilities": 0.10,
    "Unknown": 0.20,
}

FLAT_DISTRESS_RATE = 0.20
FLAT_CREDIT_SPREAD = 0.02


@dataclass
class MMResult:
    """Décomposition de la juste valeur Modigliani-Miller."""

    market_cap: float
    net_debt: float
    unlevered_value: float
    nopat: float
    unlevered_beta: float
    discount_rate: float          # r_U, coût des fonds propres non endettés
    growth_rate: float            # g, croissance perpétuelle retenue
    pv_tax_shield: float
    pv_distress: float
    pv_agency: float
    levered_firm_value: float
    fair_equity_value: float
    divergence_pct: float
    prob_default: float
    credit_spread: float
    distress_rate: float
    leverage_ratio: float
    interest_coverage: float
    sector: str = "Unknown"
    fair_value_per_share: float = float("nan")
    interest_imputed: bool = False
    notes: List[str] = field(default_factory=list)
    # Juste valeur par action selon (r_U, g) — pour juger de la robustesse.
    sensitivity_grid: List[Dict[str, float]] = field(default_factory=list)

    @property
    def undervalued(self) -> bool:
        """Le seuil est celui de l'algorithme : 25 % d'écart."""
        return self.divergence_pct > DEFAULT_CONFIG.leverage_gap_threshold * 100

    @property
    def overvalued(self) -> bool:
        return self.divergence_pct < -DEFAULT_CONFIG.leverage_gap_threshold * 100


def distress_rate(sector: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> float:
    """Taux de destruction de valeur en faillite, propre au secteur."""
    if not cfg.industry_distress_costs:
        return FLAT_DISTRESS_RATE
    return SECTOR_DISTRESS_RATE.get(sector, SECTOR_DISTRESS_RATE["Unknown"])


def credit_spread(leverage_ratio: float, cfg: ValuationConfig = DEFAULT_CONFIG) -> float:
    """Spread de crédit estimé à partir du levier (dette / capitaux propres).

    Calage indicatif sur les spreads américains :
        D/E < 0,5 → ~100 pb (AAA/AA)      D/E = 2,0 → ~400 pb (BB)
        D/E = 1,0 → ~200 pb (A/BBB)       D/E > 5,0 → plafonné à 1 000 pb
    """
    if not cfg.variable_credit_spread:
        return FLAT_CREDIT_SPREAD
    spread = 0.005 + 0.03 * min(max(leverage_ratio, 0.0), 5.0)
    return float(np.clip(spread, 0.005, 0.10))


def debt_beta(spread: float, cfg: ValuationConfig = DEFAULT_CONFIG) -> float:
    """Risque systématique porté par la dette elle-même, déduit de son spread.

    La forme classique de Hamada suppose une dette SANS RISQUE (β_D = 0). Chez
    une société très endettée, cette hypothèse abaisse beaucoup trop β_U, donc
    r_U, donc gonfle la perpétuité : le modèle récompenserait l'endettement,
    précisément l'inversion contre laquelle la comparaison au niveau des
    capitaux propres met déjà en garde. Une société à D/E = 1,5 avec β_L = 0,95
    se dé-leviérise en β_U = 0,44 sous β_D = 0 — un bêta d'actif inférieur à
    celui d'un service public, pour une cyclique endettée.

    Une prime de crédit ne rémunère qu'en partie le risque systématique ; le
    reste couvre la perte attendue en cas de défaut et l'illiquidité. En
    retenir la moitié est l'approximation usuelle (Cooper & Davydenko, 2007),
    plafonnée à 0,4 : au-delà, la créance se comporte comme une action et la
    séparation dette / capitaux propres perd son sens.
    """
    if cfg.equity_risk_premium <= 0:
        return 0.0
    return float(np.clip(0.5 * spread / cfg.equity_risk_premium, 0.0, 0.4))


def unlever_beta(
    levered_beta: float,
    total_debt: float,
    equity_value: float,
    tax_rate: float,
    beta_debt: float = 0.0,
) -> float:
    """Dé-leviérisation d'un bêta de capitaux propres en bêta d'actif.

        β_U = (E·β_L + D(1 − τ)·β_D) / (E + D(1 − τ))

    qui se réduit à la forme de Hamada β_L / (1 + (1 − τ)·D/E) quand β_D = 0.

    Le bêta estimé par la régression Fama-French est celui des CAPITAUX
    PROPRES : il intègre le risque financier créé par la dette. Actualiser un
    flux d'exploitation à ce taux compterait le levier deux fois — une fois
    dans le taux, une fois dans la dette retranchée en fin de calcul.

    Le ratio D/E est pris en VALEUR DE MARCHÉ des capitaux propres, comme le
    veut la pratique : la valeur comptable est déformée par les rachats
    d'actions, au point de devenir négative. Il subsiste donc un couplage
    résiduel entre le cours et la juste valeur — un cours plus élevé abaisse
    D/E, relève β_U, relève r_U et abaisse la valeur actualisée. Ce couplage
    est d'un ordre de grandeur inférieur à celui du modèle d'origine (une
    hausse de 200 % du cours déplace la valeur non endettée de moins de 10 %,
    contre 100 % auparavant) et il joue dans le sens stabilisant : il renforce
    le signal au lieu de l'annuler.
    """
    if not math.isfinite(levered_beta) or equity_value <= 0:
        return float("nan")
    debt_value = max(total_debt, 0.0) * (1.0 - tax_rate)
    return float(
        (equity_value * levered_beta + debt_value * beta_debt)
        / (equity_value + debt_value)
    )


def unlevered_cost_of_capital(
    unlevered_beta_value: float,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> float:
    """MEDAF sans effet de levier : r_U = rf + β_U × prime de risque actions."""
    beta = (
        unlevered_beta_value if math.isfinite(unlevered_beta_value)
        else cfg.default_unlevered_beta
    )
    # Un bêta négatif ou nul donnerait un taux d'actualisation inférieur au
    # taux sans risque, donc une valeur de perpétuité aberrante.
    beta = max(beta, 0.2)
    return float(cfg.risk_free_rate_annual + beta * cfg.equity_risk_premium)


def _perpetuity_value(nopat: float, discount: float, growth: float) -> float:
    """Valeur d'une perpétuité croissante : NOPAT × (1 + g) / (r − g)."""
    if discount <= growth or not math.isfinite(nopat):
        return float("nan")
    return float(nopat * (1.0 + growth) / (discount - growth))


def _clean(value: object, default: float = 0.0) -> float:
    """Convertit en flottant fini, en retombant sur `default` sinon."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def mm_valuation(
    fundamentals: Dict,
    market_cap: float,
    equity_volatility: float,
    shares_outstanding: float = float("nan"),
    levered_beta: float = float("nan"),
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[MMResult]:
    """Juste valeur MM d'un titre, par la valeur actuelle ajustée (APV).

    Parameters
    ----------
    fundamentals      : dict avec total_debt, total_equity, cash, ebit,
                        interest_expense, tax_rate, fcf, sector
    market_cap        : capitalisation boursière courante
    equity_volatility : volatilité annualisée des capitaux propres (Merton)
    shares_outstanding: pour convertir la juste valeur en prix par action
    levered_beta      : bêta de marché issu de la régression Fama-French ;
                        à défaut, `cfg.default_unlevered_beta` est retenu

    Renvoie None si la capitalisation est inconnue ou si le résultat
    d'exploitation est négatif — une perpétuité de flux négatifs n'a pas de
    sens économique.
    """
    market_cap = _clean(market_cap, 0.0)
    if market_cap <= 0:
        logger.info("Capitalisation boursière indisponible — écran MM impossible.")
        return None

    total_debt = max(_clean(fundamentals.get("total_debt")), 0.0)
    cash = max(_clean(fundamentals.get("cash")), 0.0)
    book_equity = _clean(fundamentals.get("total_equity"))
    ebit = _clean(fundamentals.get("ebit"))
    interest_expense = _clean(fundamentals.get("interest_expense"))
    tax_rate = _clean(fundamentals.get("tax_rate"), cfg.default_tax_rate)
    free_cash_flow = _clean(fundamentals.get("fcf"))
    sector = str(fundamentals.get("sector") or "Unknown")
    sigma_equity = _clean(equity_volatility, 0.30) or 0.30

    risk_free = cfg.risk_free_rate_annual
    notes: List[str] = []

    net_debt = max(total_debt - cash, 0.0)

    # ── Levier ─────────────────────────────────────────────────────────── #
    # Les entreprises qui rachètent massivement leurs actions affichent des
    # capitaux propres comptables négatifs.  Diviser par ces capitaux propres
    # donnerait un levier absurde (de l'ordre de la dette en valeur absolue) et
    # un signal de survalorisation mécanique : on bascule alors sur la
    # capitalisation boursière comme assiette, et on plafonne le ratio.
    equity_base = book_equity if book_equity > 0 else market_cap
    if book_equity <= 0:
        notes.append(
            "Capitaux propres comptables négatifs ou nuls (rachats d'actions) : "
            "le levier est calculé sur la capitalisation boursière."
        )
    leverage_ratio = min(total_debt / max(equity_base, 1.0), 10.0)

    # ── 1. Valeur actuelle du bouclier fiscal ──────────────────────────── #
    spread = credit_spread(leverage_ratio, cfg)
    interest_imputed = False
    if interest_expense <= 0 and total_debt > 0:
        # Charge d'intérêts non publiée : on l'impute au coût de la dette
        # estimé, cohérent avec le taux d'actualisation du bouclier — la
        # valeur actuelle vaut alors τ·D, la perpétuité de Modigliani-Miller.
        interest_expense = total_debt * (risk_free + spread)
        interest_imputed = True
        notes.append(
            "Charge d'intérêts non publiée : imputée au coût de la dette estimé "
            f"({(risk_free + spread) * 100:.1f} %)."
        )

    shield_discount = risk_free + spread
    pv_tax_shield = (
        tax_rate * max(interest_expense, 0.0) / shield_discount
        if shield_discount > 0 else 0.0
    )

    # ── 2. Valeur de la firme non endettée, à partir des fondamentaux ──── #
    # C'est ici que ce module s'écarte de l'algorithme de production : celui-ci
    # posait V_U = capitalisation + dette nette − bouclier, ce qui rendait la
    # juste valeur dépendante du cours qu'elle est censée juger (voir le
    # docstring du module). La perpétuité ci-dessous n'utilise que le compte de
    # résultat, le bilan et le bêta.
    if ebit <= 0:
        logger.info(
            "Résultat d'exploitation négatif (%.3e) : valorisation par "
            "perpétuité impossible.", ebit,
        )
        return None

    nopat = ebit * (1.0 - tax_rate)

    # `spread` a été calculé à l'étape 1 à partir du levier de cette société.
    beta_unlevered = unlever_beta(
        levered_beta, total_debt, market_cap, tax_rate,
        beta_debt=debt_beta(spread, cfg),
    )
    if not math.isfinite(beta_unlevered):
        beta_unlevered = cfg.default_unlevered_beta
        notes.append(
            "Bêta de marché non estimable : bêta dé-leviérisé supposé égal à "
            f"{cfg.default_unlevered_beta:.1f}."
        )

    discount_rate = unlevered_cost_of_capital(beta_unlevered, cfg)

    # La croissance perpétuelle ne peut pas approcher le taux d'actualisation :
    # la perpétuité diverge et la juste valeur explose.
    growth_rate = min(cfg.terminal_growth, discount_rate - cfg.min_discount_spread)

    unlevered_value = _perpetuity_value(nopat, discount_rate, growth_rate)
    if not math.isfinite(unlevered_value) or unlevered_value <= 0:
        logger.info("Perpétuité non calculable (r_U=%.4f, g=%.4f).", discount_rate, growth_rate)
        return None

    # ── 3. Coûts de détresse financière (modèle de Merton) ─────────────── #
    # Convention en dette BRUTE de bout en bout : valeur de firme = E + D,
    # barrière de défaut = D, et volatilité d'actif dé-leviérisée par E/(E+D).
    equity_value = max(market_cap, 1.0)
    debt_barrier = max(total_debt, 1.0)
    firm_value = equity_value + debt_barrier

    sigma_assets = sigma_equity * (equity_value / (equity_value + debt_barrier))
    probability_default = 0.0
    try:
        if sigma_assets > 0:
            d2 = (
                math.log(firm_value / debt_barrier)
                + (risk_free - 0.5 * sigma_assets ** 2)
            ) / sigma_assets
            if cfg.return_df and cfg.return_df > 2:
                probability_default = float(student_t.cdf(-d2, df=cfg.return_df))
            else:
                probability_default = float(norm.cdf(-d2))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        logger.debug("Calcul de Merton impossible (%s) — probabilité de défaut nulle.", exc)
        notes.append("Probabilité de défaut non calculable : supposée nulle.")

    sector_rate = distress_rate(sector, cfg)
    pv_distress = probability_default * sector_rate * firm_value

    # ── 4. Coûts d'agence ──────────────────────────────────────────────── #
    agency_score = 0.0
    if leverage_ratio > 2.0:
        agency_score += (leverage_ratio - 2.0) * 0.05
    # Le coût d'agence du flux de trésorerie libre (Jensen, 1986) ne concerne
    # que les flux POSITIFS : une entreprise qui brûle du cash ne souffre pas
    # d'un excès de liquidités à mal employer.
    fcf_yield = max(free_cash_flow, 0.0) / market_cap
    if fcf_yield > 0.10:
        agency_score += (fcf_yield - 0.10) * 0.5
    pv_agency = agency_score * market_cap

    # ── 5. Valeur de la firme endettée, puis des capitaux propres ──────── #
    levered_firm_value = unlevered_value + pv_tax_shield - pv_distress - pv_agency

    # V_L est une valeur de FIRME (elle inclut la dette) tandis que la
    # capitalisation ne porte que sur les capitaux propres. Les comparer
    # directement ferait mesurer le LEVIER et non la valorisation : la
    # divergence vaudrait approximativement dette_nette / capitalisation et
    # désignerait les sociétés les plus endettées comme les plus attrayantes.
    fair_equity_value = levered_firm_value - net_debt
    divergence_pct = (fair_equity_value - market_cap) / market_cap * 100.0

    # ── Couverture des intérêts ────────────────────────────────────────── #
    coverage = float("inf")
    if interest_expense > 0:
        coverage = ebit / interest_expense

    fair_price = float("nan")
    shares = _clean(shares_outstanding, 0.0)
    if shares > 0:
        fair_price = fair_equity_value / shares

    # ── Grille de sensibilité ──────────────────────────────────────────── #
    # Une valorisation par perpétuité est très sensible à ses deux paramètres
    # exogènes. Plutôt que de livrer un chiffre unique faussement précis, on
    # expose la juste valeur sur un voisinage de (r_U, g).
    frictions = pv_tax_shield - pv_distress - pv_agency - net_debt
    grid: List[Dict[str, float]] = []
    for rate_shift in (-0.01, 0.0, 0.01):
        for growth in (0.015, 0.025, 0.035):
            rate = discount_rate + rate_shift
            capped_growth = min(growth, rate - cfg.min_discount_spread)
            scenario_value = _perpetuity_value(nopat, rate, capped_growth)
            if not math.isfinite(scenario_value):
                continue
            scenario_equity = scenario_value + frictions
            grid.append({
                "discount_rate": round(rate, 4),
                "growth_rate": round(capped_growth, 4),
                "equity_value": scenario_equity,
                "price_per_share": (
                    scenario_equity / shares if shares > 0 else float("nan")
                ),
                "divergence_pct": (
                    (scenario_equity - market_cap) / market_cap * 100.0
                ),
            })

    return MMResult(
        market_cap=market_cap,
        net_debt=net_debt,
        unlevered_value=unlevered_value,
        nopat=nopat,
        unlevered_beta=beta_unlevered,
        discount_rate=discount_rate,
        growth_rate=growth_rate,
        pv_tax_shield=pv_tax_shield,
        pv_distress=pv_distress,
        pv_agency=pv_agency,
        levered_firm_value=levered_firm_value,
        fair_equity_value=fair_equity_value,
        divergence_pct=divergence_pct,
        prob_default=probability_default,
        credit_spread=spread,
        distress_rate=sector_rate,
        leverage_ratio=leverage_ratio,
        interest_coverage=coverage,
        sector=sector,
        fair_value_per_share=fair_price,
        interest_imputed=interest_imputed,
        notes=notes,
        sensitivity_grid=grid,
    )


def leverage_alert(
    result: MMResult,
    ebit: float,
    total_debt: float,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[str]:
    """Garde-fou de solvabilité, repris de l'écran MM de production.

    Indépendamment de la divergence de valorisation, deux situations
    disqualifient un titre côté achat : une couverture des intérêts trop
    faible, et un résultat d'exploitation négatif chez une société endettée.
    """
    if result.interest_coverage < cfg.min_interest_coverage:
        return (
            f"Couverture des intérêts de {result.interest_coverage:.2f}× "
            f"(seuil {cfg.min_interest_coverage:.1f}×) : le résultat "
            "d'exploitation couvre à peine le service de la dette."
        )
    if ebit < 0 and total_debt > 0:
        return (
            "Résultat d'exploitation négatif alors que la société porte de la "
            "dette : risque de solvabilité."
        )
    return None
