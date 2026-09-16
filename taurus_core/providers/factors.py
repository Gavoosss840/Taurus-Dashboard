"""
Taurus Dashboard – Facteurs Fama-French 5.

Téléchargement direct depuis la bibliothèque de données de Kenneth French
(Dartmouth) : gratuit, sans clé, et c'est la source de référence académique
utilisée par `taurus/data.py:get_ff5_factors` dans l'algorithme de production.

Colonnes renvoyées (en décimal, pas en pourcentage) :
    Mkt-RF, SMB, HML, RMW, CMA, RF
Index : fin de mois.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Optional

import pandas as pd

from .. import cache
from ..config import DEFAULT_CONFIG, ValuationConfig
from . import http

logger = logging.getLogger(__name__)

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

# Un jeu de facteurs par région. Régresser un titre japonais sur les facteurs
# américains attribuerait à son alpha tout ce qui n'est qu'un écart entre les
# deux marchés.
#
# Les fichiers internationaux de Kenneth French sont libellés EN DOLLARS : les
# rendements du titre doivent l'être aussi, d'où la conversion effectuée en
# amont par `providers.fx`.
FACTOR_FILES = {
    "north_america": "North_America_5_Factors_CSV.zip",
    "europe":        "Europe_5_Factors_CSV.zip",
    "japan":         "Japan_5_Factors_CSV.zip",
    "asia_pacific":  "Asia_Pacific_ex_Japan_5_Factors_CSV.zip",
    "emerging":      "Emerging_5_Factors_CSV.zip",
}

# Le jeu américain historique, plus profond que « North America » (1963 contre
# 1990) : on le garde pour les titres américains, où il fait référence.
US_FACTOR_FILE = "F-F_Research_Data_5_Factors_2x3_CSV.zip"

FACTOR_LABELS = {
    "north_america": "Kenneth R. French — États-Unis (5 facteurs)",
    "europe":        "Kenneth R. French — Europe (5 facteurs)",
    "japan":         "Kenneth R. French — Japon (5 facteurs)",
    "asia_pacific":  "Kenneth R. French — Asie-Pacifique hors Japon (5 facteurs)",
    "emerging":      "Kenneth R. French — marchés émergents (5 facteurs)",
}


def _parse_french_csv(payload: bytes) -> Optional[pd.DataFrame]:
    """Extrait le bloc mensuel de l'archive CSV de Kenneth French.

    Le fichier enchaîne un en-tête libre, le bloc mensuel (dates `AAAAMM`)
    puis un bloc annuel (dates `AAAA`).  On ne conserve que les lignes dont la
    première colonne compte exactement six chiffres, ce qui isole le mensuel
    sans dépendre du nombre de lignes d'en-tête, qui varie d'une publication
    à l'autre.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        logger.warning("Archive Fama-French illisible.")
        return None

    names = [n for n in archive.namelist() if n.upper().endswith(".CSV")]
    if not names:
        return None

    text = archive.read(names[0]).decode("utf-8", errors="replace")

    rows, header = [], None
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        if header is None and not parts[0].isdigit():
            header = parts          # ligne d'en-tête juste avant les données
            continue
        if len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[:7])

    if not rows:
        logger.warning("Aucune ligne mensuelle trouvée dans le fichier Fama-French.")
        return None

    frame = pd.DataFrame(rows, columns=["Date"] + FF5_COLUMNS)
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m")
    frame = frame.set_index("Date")
    frame.index = frame.index.to_period("M").to_timestamp("M")

    for column in FF5_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Kenneth French publie en pourcentage ; le moteur travaille en décimal.
    # Les valeurs manquantes sont codées -99.99 dans ces fichiers.
    frame = frame.mask(frame <= -99.0)
    frame = frame / 100.0
    return frame.dropna(how="all").sort_index()


def get_ff5_factors(
    region: str = "north_america",
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[pd.DataFrame]:
    """Série mensuelle des 5 facteurs Fama-French pour une région.

    Le fichier est mis à jour une fois par mois : un cache long est donc sans
    danger et évite de retélécharger l'archive à chaque analyse.

    Pour l'Amérique du Nord, le jeu américain historique est essayé d'abord :
    il remonte à 1963 contre 1990 pour le fichier « North America », ce qui
    donne une fenêtre de régression toujours pleine.
    """
    region = region if region in FACTOR_FILES else "north_america"

    candidates = (
        [US_FACTOR_FILE, FACTOR_FILES[region]] if region == "north_america"
        else [FACTOR_FILES[region]]
    )

    def _fetch() -> Optional[pd.DataFrame]:
        for filename in candidates:
            payload = http.get_bytes(f"{FRENCH_BASE}/{filename}")
            if payload is None:
                continue
            frame = _parse_french_csv(payload)
            if frame is None or frame.empty:
                continue
            logger.info(
                "Facteurs %s chargés : %d mois (%s → %s).",
                region, len(frame), frame.index[0].date(), frame.index[-1].date(),
            )
            return frame
        logger.warning("Aucun jeu de facteurs disponible pour la région %s.", region)
        return None

    return cache.memoize(f"ff5_factors_v2_{region}", _fetch, cfg)


def factor_label(region: str) -> str:
    """Libellé de la source de facteurs, affiché dans le dashboard."""
    return FACTOR_LABELS.get(region, FACTOR_LABELS["north_america"])


def market_returns(factors: pd.DataFrame) -> pd.Series:
    """Rendement total du marché = prime de marché + taux sans risque."""
    return (factors["Mkt-RF"] + factors["RF"]).rename("market")
