"""
Taurus Dashboard – Fondamentaux comptables.

Source primaire : SEC EDGAR (XBRL companyfacts).  Gratuite, sans clé, et
alimentée directement par les dépôts 10-Q/10-K — donc à jour dans le quart
d'heure suivant la publication des résultats.  Financial Modeling Prep prend
le relais lorsqu'une clé est fournie et qu'EDGAR ne couvre pas le titre
(sociétés non cotées aux États-Unis, notamment).

Point de vigilance : EDGAR mélange dans un même tableau les valeurs
trimestrielles (10-Q) et annuelles (10-K).  Le modèle Modigliani-Miller
raisonne en flux ANNUELS (bouclier fiscal = τ × intérêts annuels ; couverture
des intérêts = EBIT annuel / intérêts annuels).  Prendre naïvement « la valeur
la plus récente » donnerait donc tantôt un trimestre, tantôt un exercice — un
facteur 4 d'écart selon le calendrier de publication.  Ce module reconstruit
explicitement des flux sur 12 mois glissants (TTM).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .. import cache
from ..config import DEFAULT_CONFIG, ValuationConfig
from . import http
from .sectors import sector_from_sic

logger = logging.getLogger(__name__)

SEC_BASE = "https://www.sec.gov"
EDGAR_BASE = "https://data.sec.gov"
FMP_BASE = "https://financialmodelingprep.com/api/v3"

NAN = float("nan")


@dataclass
class Fundamentals:
    """Fondamentaux normalisés, dans les unités attendues par le modèle MM."""

    ticker: str
    company_name: str = ""
    sector: str = "Unknown"
    source: str = ""
    # Devise de publication des comptes. ASML dépose auprès de la SEC mais
    # publie en euros tout en cotant en dollars à New York : comparer ses
    # fondamentaux à sa capitalisation sans conversion mesurerait la parité
    # EUR/USD, pas une décote.
    currency: str = "USD"
    # Pays du déposant, quand SEC EDGAR le renseigne : sert à choisir le jeu
    # de facteurs Fama-French régional.
    country: str = ""

    # Bilan (valeurs instantanées, dernier arrêté connu)
    total_debt: float = NAN
    total_equity: float = NAN
    total_assets: float = NAN
    cash: float = NAN
    shares_outstanding: float = NAN

    # Compte de résultat et flux de trésorerie (12 mois glissants)
    ebit: float = NAN
    interest_expense: float = NAN
    revenue: float = NAN
    net_income: float = NAN
    fcf: float = NAN
    tax_rate: float = 0.21

    # Dividendes trimestriels par action, indexés en fin de période.  Ils
    # proviennent du même fichier `companyfacts` que le reste : les extraire
    # ici ne coûte aucun appel réseau supplémentaire, et ils permettent de
    # reconstituer un rendement TOTAL quand la source de cours n'en fournit
    # qu'un rendement en capital.
    dividends_per_share: Optional["pd.Series"] = None

    # Traçabilité
    fiscal_period_end: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def net_debt(self) -> float:
        if math.isnan(self.total_debt):
            return NAN
        cash = 0.0 if math.isnan(self.cash) else self.cash
        return self.total_debt - cash

    def missing_fields(self) -> List[str]:
        """Champs indispensables au modèle MM qui restent indéterminés."""
        required = {
            "total_debt": self.total_debt,
            "total_equity": self.total_equity,
            "ebit": self.ebit,
            "cash": self.cash,
        }
        return [name for name, value in required.items()
                if value is None or (isinstance(value, float) and math.isnan(value))]

    def to_dict(self) -> Dict:
        """Ligne compatible avec `capital_structure.mm_valuation`."""
        return {
            "total_debt": self.total_debt,
            "total_equity": self.total_equity,
            "total_assets": self.total_assets,
            "cash": self.cash,
            "ebit": self.ebit,
            "interest_expense": self.interest_expense,
            "tax_rate": self.tax_rate,
            "fcf": self.fcf,
            "sector": self.sector,
        }


# --------------------------------------------------------------------------- #
#  SEC EDGAR : correspondance ticker → CIK                                     #
# --------------------------------------------------------------------------- #

_CIK_MAP: Optional[Dict[str, str]] = None


def _cik_map(cfg: ValuationConfig) -> Dict[str, str]:
    """Table ticker → CIK (mémoire + cache disque : ~10 000 entrées)."""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP

    def _fetch() -> Optional[Dict[str, str]]:
        data = http.get_json(
            f"{SEC_BASE}/files/company_tickers.json",
            headers={"User-Agent": http.sec_user_agent()},
        )
        if not isinstance(data, dict):
            return None
        return {
            str(entry["ticker"]).upper(): str(entry["cik_str"]).zfill(10)
            for entry in data.values()
            if entry.get("ticker") and entry.get("cik_str") is not None
        }

    _CIK_MAP = cache.memoize("sec_cik_map_v1", _fetch, cfg) or {}
    return _CIK_MAP


def ticker_to_cik(ticker: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> Optional[str]:
    return _cik_map(cfg).get(ticker.upper().strip())


# --------------------------------------------------------------------------- #
#  SEC EDGAR : extraction XBRL                                                 #
# --------------------------------------------------------------------------- #

# Unités XBRL qui ne sont pas des montants monétaires.
_NON_MONETARY_UNITS = {"shares", "pure", "Year", "Store", "Rate", "Y", "D"}


def reporting_currency(us_gaap: dict) -> str:
    """Devise dans laquelle l'entreprise publie ses comptes.

    SEC EDGAR indexe chaque fait par son unité : « USD » pour un déposant
    américain, « EUR » pour ASML, « JPY » pour Sony. Lire USD en dur — ce que
    faisait ce module — renvoyait des comptes vides pour tout émetteur privé
    étranger, alors même que ses chiffres étaient là.

    On retient l'unité monétaire la plus employée dans la taxonomie.
    """
    from collections import Counter

    tally: Counter = Counter()
    for concept in us_gaap.values():
        for unit, entries in ((concept.get("units") or {})).items():
            if unit in _NON_MONETARY_UNITS or "/" in unit:
                continue
            if len(unit) == 3 and unit.isalpha():
                tally[unit.upper()] += len(entries or [])

    if not tally:
        return "USD"
    return tally.most_common(1)[0][0]


def _money_facts(us_gaap: dict, concept: str, currency: str = "USD") -> List[dict]:
    """Faits monétaires d'un concept, dédupliqués sur la période déclarée.

    Une même période est souvent republiée (10-K reprenant un trimestre,
    amendement 10-K/A…).  On conserve le dépôt le plus récent, qui porte les
    chiffres retraités.
    """
    units = (us_gaap.get(concept) or {}).get("units") or {}
    raw = units.get(currency) or []
    if not raw and currency != "USD":
        # Un émetteur étranger publie parfois quelques agrégats en dollars.
        raw = units.get("USD") or []

    best: Dict[tuple, dict] = {}
    for fact in raw:
        if fact.get("val") is None or not fact.get("end"):
            continue
        key = (fact.get("start"), fact["end"])
        previous = best.get(key)
        if previous is None or str(fact.get("filed", "")) >= str(previous.get("filed", "")):
            best[key] = fact
    return sorted(best.values(), key=lambda f: f["end"], reverse=True)


# Une valeur de bilan publiée il y a plus de ~7 mois signale soit un déposant
# en retard, soit un concept comptable abandonné par l'entreprise.
_STALE_DAYS = 210

# Tolérance de fraîcheur entre concepts : deux concepts dont les arrêtés sont
# séparés de moins d'un trimestre décrivent la même situation comptable.
_FRESHNESS_WINDOW_DAYS = 100


def _days_between(earlier: str, later: str) -> Optional[int]:
    from datetime import date

    try:
        y1, m1, d1 = (int(part) for part in str(earlier).split("-"))
        y2, m2, d2 = (int(part) for part in str(later).split("-"))
        return (date(y2, m2, d2) - date(y1, m1, d1)).days
    except (ValueError, TypeError):
        return None


def _pick_freshest(candidates: List[tuple[int, dict]]) -> Optional[dict]:
    """Choisit un fait parmi des candidats (rang de priorité, fait).

    La fraîcheur prime sur la priorité : les entreprises changent de concept
    XBRL au fil des ans et continuent parfois d'exposer l'ancien, figé sur sa
    dernière valeur.  Coca-Cola a ainsi cessé d'alimenter `LongTermDebt` en
    2024 au profit de `LongTermDebtAndCapitalLeaseObligations` ; NextEra a
    abandonné `Revenues` en 2013.  Suivre l'ordre de priorité d'abord
    renverrait dans les deux cas un chiffre périmé de plusieurs années.

    À fraîcheur comparable (moins d'un trimestre d'écart), c'est la priorité
    qui tranche — elle encode la définition comptable la plus pertinente.
    """
    if not candidates:
        return None
    newest = max(str(fact["end"]) for _, fact in candidates)
    fresh = [
        (rank, fact) for rank, fact in candidates
        if (gap := _days_between(str(fact["end"]), newest)) is not None
        and gap <= _FRESHNESS_WINDOW_DAYS
    ]
    if not fresh:
        fresh = candidates
    return min(fresh, key=lambda item: item[0])[1]


def _latest_instant(
    us_gaap: dict, *concepts: str, currency: str = "USD",
) -> tuple[float, str]:
    """Dernière valeur de bilan (fait instantané : pas de date de début).

    Renvoie (valeur, date_de_cloture).
    """
    candidates: List[tuple[int, dict]] = []
    for rank, concept in enumerate(concepts):
        instants = [f for f in _money_facts(us_gaap, concept, currency)
                    if not f.get("start")]
        if instants:
            candidates.append((rank, instants[0]))

    chosen = _pick_freshest(candidates)
    if chosen is None:
        return NAN, ""
    return float(chosen["val"]), str(chosen["end"])


def _duration_days(fact: dict) -> Optional[int]:
    return _days_between(str(fact.get("start") or ""), str(fact.get("end") or ""))


def _ttm_from_facts(facts: List[dict]) -> Optional[tuple[float, str]]:
    """Reconstruit un flux sur 12 mois glissants à partir des faits d'un concept.

    Stratégie, du plus fiable au moins fiable :
      1. somme des 4 derniers trimestres consécutifs et disjoints ;
      2. à défaut, dernier exercice annuel complet ;
      3. à défaut, dernière période cumulée (YTD) annualisée.
    """
    quarters = [f for f in facts if (d := _duration_days(f)) and 80 <= d <= 100]
    annuals = [f for f in facts if (d := _duration_days(f)) and 350 <= d <= 380]

    # 1. Quatre trimestres consécutifs, sans chevauchement.
    if len(quarters) >= 4:
        selected, cursor = [], None
        for fact in quarters:               # déjà triés du plus récent au plus ancien
            if cursor is not None and str(fact["end"]) >= cursor:
                continue                    # chevauche la période déjà retenue
            selected.append(fact)
            cursor = str(fact["start"])
            if len(selected) == 4:
                break
        if len(selected) == 4:
            span = _days_between(str(selected[-1]["start"]), str(selected[0]["end"]))
            if span and 330 <= span <= 400:  # les 4 trimestres couvrent bien un an
                total = sum(float(f["val"]) for f in selected)
                return total, str(selected[0]["end"])

    # 2. Dernier exercice annuel.
    if annuals:
        return float(annuals[0]["val"]), str(annuals[0]["end"])

    # 3. Dernière période cumulée, ramenée à 12 mois.
    ytd = [f for f in facts if (d := _duration_days(f)) and d >= 150]
    if ytd:
        days = _duration_days(ytd[0]) or 365
        return float(ytd[0]["val"]) * 365.0 / days, str(ytd[0]["end"])

    return None


def _ttm(us_gaap: dict, *concepts: str, currency: str = "USD") -> tuple[float, str]:
    """Flux sur 12 mois glissants, en privilégiant le concept le plus à jour.

    Renvoie (valeur, date_de_fin_de_periode).
    """
    candidates: List[tuple[int, dict]] = []
    computed: Dict[int, tuple[float, str]] = {}

    for rank, concept in enumerate(concepts):
        facts = [f for f in _money_facts(us_gaap, concept, currency)
                 if f.get("start")]
        if not facts:
            continue
        result = _ttm_from_facts(facts)
        if result is None:
            continue
        computed[rank] = result
        candidates.append((rank, {"end": result[1], "val": result[0]}))

    chosen = _pick_freshest(candidates)
    if chosen is None:
        return NAN, ""
    for rank, (value, end) in computed.items():
        if end == str(chosen["end"]) and value == float(chosen["val"]):
            return value, end
    return float(chosen["val"]), str(chosen["end"])


def _shares_outstanding(facts: dict) -> float:
    """Actions en circulation : espace de noms `dei` d'abord, puis us-gaap."""
    for namespace, concept in (
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstandingBasic"),
    ):
        units = ((facts.get("facts", {}).get(namespace, {}).get(concept) or {})
                 .get("units") or {})
        entries = units.get("shares") or []
        valid = [e for e in entries if e.get("val")]
        if valid:
            valid.sort(key=lambda e: str(e.get("end", "")), reverse=True)
            return float(valid[0]["val"])

    # Repli : nombre moyen d'actions dilué du compte de résultat.
    units = ((facts.get("facts", {}).get("us-gaap", {})
              .get("WeightedAverageNumberOfDilutedSharesOutstanding") or {}).get("units") or {})
    entries = [e for e in (units.get("shares") or []) if e.get("val")]
    if entries:
        entries.sort(key=lambda e: str(e.get("end", "")), reverse=True)
        return float(entries[0]["val"])
    return NAN


# --------------------------------------------------------------------------- #
#  Dividendes par action                                                       #
# --------------------------------------------------------------------------- #
# Les concepts de dividende par action, par ordre de préférence.  Coca-Cola a
# cessé d'alimenter « Declared » en 2018 au profit de « CashPaid » : la
# sélection par fraîcheur (_pick_freshest) évite de renvoyer un chiffre figé.
_DPS_CONCEPTS = (
    "CommonStockDividendsPerShareDeclared",
    "CommonStockDividendsPerShareCashPaid",
    "CommonStockDividendsPerShareDeclaredButUnpaid",
    "DividendsPayableAmountPerShare",
)

# En dessous de ce nombre de trimestres, la série est trop lacunaire pour
# reconstituer un rendement total crédible.
MIN_DIVIDEND_QUARTERS = 8


def _dividend_facts(book: dict, concept: str, unit: str, span: tuple[int, int]) -> Dict[tuple, dict]:
    """Faits de dividende d'une durée donnée, dédupliqués sur la période."""
    entries = ((book.get(concept) or {}).get("units") or {}).get(unit) or []
    best: Dict[tuple, dict] = {}
    for fact in entries:
        if fact.get("val") is None or not fact.get("start") or not fact.get("end"):
            continue
        length = _days_between(str(fact["start"]), str(fact["end"]))
        if length is None or not (span[0] <= length <= span[1]):
            continue
        key = (fact["start"], fact["end"])
        previous = best.get(key)
        if previous is None or str(fact.get("filed", "")) >= str(previous.get("filed", "")):
            best[key] = fact
    return best


def _complete_fourth_quarter(
    quarters: Dict[tuple, dict], annuals: Dict[tuple, dict],
) -> List[dict]:
    """Reconstitue le quatrième trimestre, que le rapport annuel absorbe.

    Une société publie trois trimestres dans ses 10-Q puis son exercice entier
    dans son 10-K : le quatrième versement n'a donc pas de période de 90 jours
    propre. Il manquait un dividende sur quatre, soit un quart du rendement.

    Le résidu « annuel − somme des trois trimestres » le restitue, à condition
    qu'il soit positif et du même ordre que les trois autres — au-delà, le
    rapprochement des périodes est douteux et on renonce.
    """
    completed = list(quarters.values())

    for (start, end), annual in annuals.items():
        inside = [
            fact for (q_start, q_end), fact in quarters.items()
            if start <= q_start and q_end <= end
        ]
        if len(inside) != 3:
            continue

        paid = sum(float(f["val"]) for f in inside)
        residual = float(annual["val"]) - paid
        average = paid / 3.0
        if residual <= 0 or average <= 0:
            continue
        if not (0.3 * average <= residual <= 3.0 * average):
            continue

        completed.append({
            "start": max(f["end"] for f in inside),
            "end": end,
            "val": residual,
            "filed": annual.get("filed", ""),
        })

    return completed


def _quarterly_dividends(
    book: dict, currency: str = "USD",
) -> Optional["pd.Series"]:
    """Dividendes trimestriels par action, indexés en fin de mois.

    Seules les périodes d'environ un trimestre sont retenues : les cumuls
    depuis le début d'exercice feraient double emploi avec les trimestres
    qu'ils recouvrent. Le quatrième trimestre, absorbé par le rapport annuel,
    est reconstitué par différence.
    """
    unit = f"{currency}/shares"
    candidates: List[tuple[int, dict]] = []
    harvested: Dict[int, List[dict]] = {}

    for rank, concept in enumerate(_DPS_CONCEPTS):
        quarters = _dividend_facts(book, concept, unit, (80, 100))
        if not quarters:
            continue
        annuals = _dividend_facts(book, concept, unit, (350, 380))
        facts = _complete_fourth_quarter(quarters, annuals)

        harvested[rank] = facts
        candidates.append((rank, max(facts, key=lambda f: str(f["end"]))))

    chosen = _pick_freshest(candidates)
    if chosen is None:
        return None

    for rank, facts in harvested.items():
        if any(f["end"] == chosen["end"] and f["val"] == chosen["val"] for f in facts):
            series = pd.Series(
                {pd.Timestamp(f["end"]): float(f["val"]) for f in facts}
            ).sort_index()
            series.index = series.index.to_period("M").to_timestamp("M")
            series = series[~series.index.duplicated(keep="last")]
            return series if len(series) >= MIN_DIVIDEND_QUARTERS else None
    return None


# --------------------------------------------------------------------------- #
#  Concepts comptables, par taxonomie                                          #
# --------------------------------------------------------------------------- #
# Les déposants américains publient en US-GAAP ; les émetteurs privés étrangers
# (formulaire 20-F) publient le plus souvent en IFRS, avec des noms de concepts
# entièrement différents. Ne connaître que l'US-GAAP privait le dashboard de
# SAP, TSMC, Shell, Unilever et de la plupart des grandes capitalisations
# européennes et asiatiques cotées à New York.
#
# Chaque entrée est une liste de synonymes par ordre de préférence ; la
# sélection retient malgré tout le concept le plus RÉCENT (voir _pick_freshest).

_USGAAP_CONCEPTS: Dict[str, List[str]] = {
    "debt_aggregate": [
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "DebtLongtermAndShorttermCombinedAmount",
        "DebtAndCapitalLeaseObligations",
    ],
    "debt_noncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    ],
    "debt_current": [
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
        "ShortTermBorrowings",
        "OtherShortTermBorrowings",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "assets": ["Assets"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "interest": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
        "InterestExpenseBorrowings",
        "InterestIncomeExpenseNet",
    ],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpendituresIncurredButNotYetPaid",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "pretax": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
}

_IFRS_CONCEPTS: Dict[str, List[str]] = {
    # `Borrowings` est l'agrégat propre quand il existe (Shell, Unilever) ;
    # `FinancialLiabilities` prend le relais chez les déposants qui ne le
    # publient pas (SAP), au prix d'un périmètre un peu plus large.
    "debt_aggregate": ["Borrowings", "FinancialLiabilities"],
    "debt_noncurrent": [
        "LongtermBorrowings",
        "NoncurrentPortionOfNoncurrentBorrowings",
        "NoncurrentFinancialLiabilities",
    ],
    "debt_current": [
        "ShorttermBorrowings",
        "CurrentPortionOfLongtermBorrowings",
        "CurrentFinancialLiabilities",
    ],
    "equity": ["Equity", "EquityAttributableToOwnersOfParent"],
    "assets": ["Assets"],
    "cash": ["CashAndCashEquivalents"],
    "ebit": ["ProfitLossFromOperatingActivities", "OperatingProfitLoss"],
    "interest": [
        "FinanceCosts",
        "InterestExpense",
        "InterestExpenseOnBorrowings",
        "FinanceCostsPaidClassifiedAsOperatingActivities",
    ],
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "net_income": ["ProfitLoss", "ProfitLossAttributableToOwnersOfParent"],
    "operating_cf": ["CashFlowsFromUsedInOperatingActivities"],
    "capex": [
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
    ],
    "tax_expense": ["IncomeTaxExpenseContinuingOperations"],
    "pretax": ["ProfitLossBeforeTax"],
}

# Ordre d'essai : la taxonomie la plus fournie l'emporte (Toyota publie les
# deux, l'US-GAAP étant chez lui la plus complète).
_TAXONOMIES = (
    ("us-gaap", _USGAAP_CONCEPTS),
    ("ifrs-full", _IFRS_CONCEPTS),
)


def _select_taxonomy(facts: dict) -> tuple[str, Dict[str, List[str]], dict]:
    """Retient la taxonomie la mieux renseignée pour ce déposant.

    Renvoie (nom, carte des concepts, dictionnaire des faits).
    """
    available = facts.get("facts") or {}
    best_name, best_map, best_facts, best_size = "", {}, {}, 0
    for name, concept_map in _TAXONOMIES:
        block = available.get(name) or {}
        if len(block) > best_size:
            best_name, best_map, best_facts, best_size = name, concept_map, block, len(block)
    return best_name, best_map, best_facts


def _from_edgar(ticker: str, cfg: ValuationConfig) -> Optional[Fundamentals]:
    cik = ticker_to_cik(ticker, cfg)
    if not cik:
        return None

    headers = {"User-Agent": http.sec_user_agent()}

    facts = http.get_json(f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json", headers=headers)
    if not isinstance(facts, dict):
        return None

    taxonomy, concepts, book = _select_taxonomy(facts)
    if not book:
        return None

    result = Fundamentals(ticker=ticker, source="SEC EDGAR")
    result.company_name = str(facts.get("entityName") or "").strip()
    result.currency = reporting_currency(book)
    if taxonomy == "ifrs-full":
        result.source = "SEC EDGAR (IFRS)"

    money = result.currency

    def instant(key: str) -> tuple[float, str]:
        return _latest_instant(book, *concepts.get(key, ()), currency=money)

    def ttm(key: str) -> tuple[float, str]:
        return _ttm(book, *concepts.get(key, ()), currency=money)

    # ── Métadonnées : nom lisible et secteur (via le code SIC) ──────────── #
    meta = http.get_json(f"{EDGAR_BASE}/submissions/CIK{cik}.json", headers=headers)
    if isinstance(meta, dict):
        result.company_name = str(meta.get("name") or result.company_name).strip()
        result.sector = sector_from_sic(meta.get("sic"))
        addresses = meta.get("addresses") or {}
        address = addresses.get("business") or addresses.get("mailing") or {}
        result.country = str(address.get("stateOrCountryDescription") or "").strip()

    # ── Bilan : dette totale = part long terme + part courante ─────────── #
    # Certains déposants publient directement l'agrégat toutes échéances
    # confondues ; on le retient quand il est aussi frais que le détail.
    aggregate, aggregate_end = instant("debt_aggregate")
    long_term, long_term_end = instant("debt_noncurrent")
    short_term, _ = instant("debt_current")

    detail_total = NAN
    if not (math.isnan(long_term) and math.isnan(short_term)):
        detail_total = (0.0 if math.isnan(long_term) else long_term) + \
                       (0.0 if math.isnan(short_term) else short_term)

    if math.isnan(detail_total):
        result.total_debt, period_end = aggregate, aggregate_end
    elif math.isnan(aggregate):
        result.total_debt, period_end = detail_total, long_term_end
    else:
        # Les deux existent : on suit le plus récent des deux arrêtés.
        gap = _days_between(str(long_term_end), str(aggregate_end))
        if gap is not None and gap > _FRESHNESS_WINDOW_DAYS:
            result.total_debt, period_end = aggregate, aggregate_end
        else:
            result.total_debt, period_end = detail_total, long_term_end

    result.fiscal_period_end = period_end

    result.total_equity, _ = instant("equity")
    result.total_assets, _ = instant("assets")
    result.cash, _ = instant("cash")
    result.shares_outstanding = _shares_outstanding(facts)
    result.dividends_per_share = _quarterly_dividends(book, money)

    # ── Flux 12 mois glissants ─────────────────────────────────────────── #
    result.ebit, ebit_end = ttm("ebit")
    if ebit_end and not result.fiscal_period_end:
        result.fiscal_period_end = ebit_end

    interest, _ = ttm("interest")
    result.interest_expense = abs(interest) if not math.isnan(interest) else NAN

    result.revenue, _ = ttm("revenue")
    result.net_income, _ = ttm("net_income")

    # Flux de trésorerie disponible = flux opérationnel − investissements.
    operating_cf, _ = ttm("operating_cf")
    capex, _ = ttm("capex")
    if not math.isnan(operating_cf):
        result.fcf = operating_cf - (0.0 if math.isnan(capex) else abs(capex))
    else:
        result.fcf = result.net_income

    # Taux d'imposition retenu pour le bouclier fiscal.  La théorie MM
    # raisonne sur le taux MARGINAL ; le taux effectif ne s'y substitue que
    # lorsqu'il reste plausible.  Un exercice à 0 % (crédits d'impôt des
    # énergéticiens renouvelables) ou à 60 % (redressement exceptionnel)
    # reflète un accident comptable, pas la fiscalité structurelle de la dette
    # — on retombe alors sur le taux statutaire.
    tax_expense, _ = ttm("tax_expense")
    pretax, _ = ttm("pretax")
    result.tax_rate = cfg.default_tax_rate
    if not math.isnan(tax_expense) and not math.isnan(pretax) and pretax > 0:
        effective = tax_expense / pretax
        if 0.10 <= effective <= 0.40:
            result.tax_rate = float(effective)

    # ── Contrôle de fraîcheur ──────────────────────────────────────────── #
    if result.fiscal_period_end:
        from datetime import date
        today = date.today().isoformat()
        age = _days_between(result.fiscal_period_end, today)
        if age is not None and age > _STALE_DAYS:
            result.warnings.append(
                f"Dernier bilan disponible au {result.fiscal_period_end} "
                f"({age} jours) : les fondamentaux peuvent être périmés."
            )

    return result


# --------------------------------------------------------------------------- #
#  Financial Modeling Prep (repli / titres hors SEC)                           #
# --------------------------------------------------------------------------- #

def _from_fmp(ticker: str, cfg: ValuationConfig) -> Optional[Fundamentals]:
    api_key = os.environ.get("FMP_API_KEY", "").strip()
    if not api_key:
        return None

    key = {"apikey": api_key}
    balance = http.get_json(f"{FMP_BASE}/balance-sheet-statement/{ticker}",
                            params={**key, "period": "quarter", "limit": 1})
    income = http.get_json(f"{FMP_BASE}/income-statement/{ticker}",
                           params={**key, "period": "annual", "limit": 1})
    cash_flow = http.get_json(f"{FMP_BASE}/cash-flow-statement/{ticker}",
                              params={**key, "period": "annual", "limit": 1})
    profile = http.get_json(f"{FMP_BASE}/profile/{ticker}", params=key)

    if not isinstance(balance, list) or not balance:
        return None

    result = Fundamentals(ticker=ticker, source="Financial Modeling Prep")
    sheet = balance[0]
    result.total_debt = float(sheet.get("totalDebt") or NAN)
    result.total_equity = float(sheet.get("totalStockholdersEquity") or NAN)
    result.total_assets = float(sheet.get("totalAssets") or NAN)
    result.cash = float(sheet.get("cashAndCashEquivalents") or NAN)
    result.fiscal_period_end = str(sheet.get("date") or "")

    if isinstance(income, list) and income:
        statement = income[0]
        result.ebit = float(statement.get("operatingIncome") or NAN)
        result.interest_expense = abs(float(statement.get("interestExpense") or 0.0)) or NAN
        result.revenue = float(statement.get("revenue") or NAN)
        result.net_income = float(statement.get("netIncome") or NAN)
        pretax = float(statement.get("incomeBeforeTax") or 0.0)
        tax_expense = float(statement.get("incomeTaxExpense") or 0.0)
        if pretax > 0:
            result.tax_rate = float(min(max(tax_expense / pretax, 0.0), 0.40))
        else:
            result.tax_rate = cfg.default_tax_rate

    if isinstance(cash_flow, list) and cash_flow:
        flow = cash_flow[0]
        result.fcf = float(flow.get("freeCashFlow") or NAN)

    if isinstance(profile, list) and profile:
        info = profile[0]
        result.company_name = str(info.get("companyName") or "").strip()
        result.sector = str(info.get("sector") or "Unknown") or "Unknown"
        result.currency = str(info.get("currency") or "USD").upper()
        result.country = str(info.get("country") or "").strip()

    return result


# --------------------------------------------------------------------------- #
#  API publique                                                                #
# --------------------------------------------------------------------------- #

def get_fundamentals(
    ticker: str,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[Fundamentals]:
    """Fondamentaux normalisés du titre, ou None si aucune source ne répond."""
    ticker = ticker.upper().strip()
    cache_key = f"fundamentals_v2_{ticker}"
    cached = cache.load(cache_key, cfg)
    if cached is not None:
        return cached

    for name, provider in (("edgar", _from_edgar), ("fmp", _from_fmp)):
        try:
            result = provider(ticker, cfg)
        except Exception as exc:
            logger.debug("Fondamentaux %s en erreur pour %s : %s", name, ticker, exc)
            continue
        if result is not None and not result.missing_fields():
            logger.info("Fondamentaux de %s via %s.", ticker, result.source)
            cache.save(cache_key, result, cfg)
            return result
        if result is not None:
            # Source partielle : on garde la main pour essayer la suivante,
            # mais on la conserve comme repli si aucune n'est complète.
            logger.info(
                "Fondamentaux de %s incomplets via %s (manque : %s).",
                ticker, result.source, ", ".join(result.missing_fields()),
            )
            result.warnings.append(
                "Données incomplètes : " + ", ".join(result.missing_fields())
            )
            cache.save(cache_key, result, cfg)
            return result

    logger.warning("Aucun fondamental trouvé pour %s.", ticker)
    return None
