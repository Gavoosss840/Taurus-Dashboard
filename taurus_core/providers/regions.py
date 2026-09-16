"""
Taurus Dashboard – Région de rattachement d'un titre.

Kenneth French publie des facteurs distincts pour l'Amérique du Nord,
l'Europe, le Japon, l'Asie-Pacifique hors Japon et les marchés émergents.
Régresser un titre japonais sur les facteurs américains attribuerait à son
alpha tout ce qui n'est en réalité qu'un écart entre les deux marchés.

La région retenue est celle du SIÈGE, pas de la place de cotation. Un ADR
européen coté à New York reste exposé au risque européen, et les facteurs
européens de Kenneth French sont libellés en dollars comme lui : ils
constituent donc bien la bonne référence.

Trois signaux, du plus fiable au moins fiable :

  1. le suffixe du ticker, qui désigne sans ambiguïté la place de cotation
     locale (`MC.PA`, `7203.T`, `0700.HK`) ;
  2. le pays du déposant chez SEC EDGAR, renseigné pour une partie seulement
     des émetteurs étrangers ;
  3. la devise de publication des comptes, faute de mieux.

Le module expose aussi le degré de certitude, que le dashboard affiche : une
région devinée à partir de la seule devise mérite d'être signalée.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Régions au sens des jeux de données de Kenneth French.
NORTH_AMERICA = "north_america"
EUROPE = "europe"
JAPAN = "japan"
ASIA_PACIFIC = "asia_pacific"
EMERGING = "emerging"

REGION_LABELS = {
    NORTH_AMERICA: "Amérique du Nord",
    EUROPE: "Europe",
    JAPAN: "Japon",
    ASIA_PACIFIC: "Asie-Pacifique hors Japon",
    EMERGING: "Marchés émergents",
}

# Suffixe de ticker → région. Les suffixes suivent la convention Yahoo Finance.
_SUFFIX_REGIONS = {
    # ── Europe ──────────────────────────────────────────────────────────
    "PA": EUROPE, "AS": EUROPE, "BR": EUROPE, "LS": EUROPE, "MC": EUROPE,
    "MI": EUROPE, "DE": EUROPE, "F": EUROPE, "BE": EUROPE, "HM": EUROPE,
    "MU": EUROPE, "SG": EUROPE, "DU": EUROPE, "VI": EUROPE, "SW": EUROPE,
    "ST": EUROPE, "OL": EUROPE, "CO": EUROPE, "HE": EUROPE, "IC": EUROPE,
    "IR": EUROPE, "L": EUROPE, "IL": EUROPE, "AT": EUROPE, "LS4": EUROPE,
    "PR": EUROPE, "WA": EUROPE, "BD": EUROPE, "RG": EUROPE, "VS": EUROPE,
    "TL": EUROPE,
    # ── Japon ───────────────────────────────────────────────────────────
    "T": JAPAN, "JP": JAPAN,
    # ── Asie-Pacifique développée ───────────────────────────────────────
    "HK": ASIA_PACIFIC, "TW": ASIA_PACIFIC, "TWO": ASIA_PACIFIC,
    "KS": ASIA_PACIFIC, "KQ": ASIA_PACIFIC, "SI": ASIA_PACIFIC,
    "AX": ASIA_PACIFIC, "NZ": ASIA_PACIFIC,
    # ── Amérique du Nord ────────────────────────────────────────────────
    "TO": NORTH_AMERICA, "V": NORTH_AMERICA, "NE": NORTH_AMERICA,
    "CN": NORTH_AMERICA,
    # ── Émergents ───────────────────────────────────────────────────────
    "SS": EMERGING, "SZ": EMERGING, "NS": EMERGING, "BO": EMERGING,
    "SA": EMERGING, "MX": EMERGING, "JK": EMERGING, "BK": EMERGING,
    "KL": EMERGING, "IS": EMERGING, "JO": EMERGING, "SR": EMERGING,
    "QA": EMERGING, "AD": EMERGING, "DU2": EMERGING, "CR": EMERGING,
}

# Libellé de pays chez SEC EDGAR → région.
_COUNTRY_REGIONS = {
    # Europe
    "netherlands": EUROPE, "germany": EUROPE, "france": EUROPE,
    "united kingdom": EUROPE, "ireland": EUROPE, "switzerland": EUROPE,
    "denmark": EUROPE, "sweden": EUROPE, "norway": EUROPE, "finland": EUROPE,
    "belgium": EUROPE, "spain": EUROPE, "italy": EUROPE, "portugal": EUROPE,
    "austria": EUROPE, "luxembourg": EUROPE, "iceland": EUROPE,
    "jersey": EUROPE, "guernsey": EUROPE, "isle of man": EUROPE,
    "gibraltar": EUROPE, "monaco": EUROPE, "greece": EUROPE,
    # Japon
    "japan": JAPAN,
    # Asie-Pacifique développée
    "hong kong": ASIA_PACIFIC, "singapore": ASIA_PACIFIC,
    "australia": ASIA_PACIFIC, "new zealand": ASIA_PACIFIC,
    "taiwan": ASIA_PACIFIC, "korea, republic of": ASIA_PACIFIC,
    "south korea": ASIA_PACIFIC,
    # Amérique du Nord
    "united states": NORTH_AMERICA, "canada": NORTH_AMERICA,
    # Émergents
    "china": EMERGING, "india": EMERGING, "brazil": EMERGING,
    "mexico": EMERGING, "south africa": EMERGING, "turkey": EMERGING,
    "thailand": EMERGING, "malaysia": EMERGING, "indonesia": EMERGING,
    "philippines": EMERGING, "chile": EMERGING, "poland": EMERGING,
    "israel": EMERGING, "cayman islands": EMERGING, "bermuda": EMERGING,
}

# Devise de publication → région, en dernier recours.
_CURRENCY_REGIONS = {
    "EUR": EUROPE, "GBP": EUROPE, "CHF": EUROPE, "DKK": EUROPE,
    "SEK": EUROPE, "NOK": EUROPE, "ISK": EUROPE, "PLN": EUROPE,
    "CZK": EUROPE, "HUF": EUROPE, "RON": EUROPE,
    "JPY": JAPAN,
    "HKD": ASIA_PACIFIC, "TWD": ASIA_PACIFIC, "KRW": ASIA_PACIFIC,
    "SGD": ASIA_PACIFIC, "AUD": ASIA_PACIFIC, "NZD": ASIA_PACIFIC,
    "USD": NORTH_AMERICA, "CAD": NORTH_AMERICA,
    "CNY": EMERGING, "INR": EMERGING, "BRL": EMERGING, "MXN": EMERGING,
    "ZAR": EMERGING, "TRY": EMERGING, "THB": EMERGING, "MYR": EMERGING,
    "IDR": EMERGING, "PHP": EMERGING, "ILS": EMERGING,
}


@dataclass
class RegionGuess:
    """Région retenue, et sur quel signal elle repose."""

    region: str
    evidence: str            # "suffixe" | "pays" | "devise" | "défaut"
    confident: bool          # False quand seule la devise a tranché

    @property
    def label(self) -> str:
        return REGION_LABELS.get(self.region, self.region)


def suffix_of(ticker: str) -> Optional[str]:
    """Suffixe de place d'un ticker, ou None pour une cotation américaine."""
    if "." not in ticker:
        return None
    suffix = ticker.rsplit(".", 1)[-1].strip().upper()
    return suffix or None


def detect_region(
    ticker: str,
    country: Optional[str] = None,
    currency: Optional[str] = None,
) -> RegionGuess:
    """Détermine la région de rattachement à partir des signaux disponibles."""
    suffix = suffix_of(ticker.upper().strip())
    if suffix and suffix in _SUFFIX_REGIONS:
        return RegionGuess(_SUFFIX_REGIONS[suffix], "suffixe", True)

    if country:
        key = str(country).strip().lower()
        if key in _COUNTRY_REGIONS:
            return RegionGuess(_COUNTRY_REGIONS[key], "pays", True)

    if currency:
        key = str(currency).strip().upper()
        if key in _CURRENCY_REGIONS:
            region = _CURRENCY_REGIONS[key]
            # Une société sans suffixe de place qui publie en dollars est
            # américaine dans l'immense majorité des cas : l'inférence est
            # solide. Elle échoue sur les rares groupes étrangers qui tiennent
            # leurs comptes en dollars — Shell, britannique, en est l'exemple —
            # d'où le champ `evidence`, que le dashboard affiche.
            confident = not (suffix is None and key != "USD")
            return RegionGuess(region, "devise", confident)

    return RegionGuess(NORTH_AMERICA, "défaut", False)
