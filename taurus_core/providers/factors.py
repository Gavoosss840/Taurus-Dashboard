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
FF5_URL = f"{FRENCH_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip"

FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


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


def get_ff5_factors(cfg: ValuationConfig = DEFAULT_CONFIG) -> Optional[pd.DataFrame]:
    """Série mensuelle complète des 5 facteurs Fama-French (+ taux sans risque).

    Le fichier est mis à jour une fois par mois : un cache long est donc sans
    danger et évite de retélécharger 1,2 Mo à chaque analyse.
    """

    def _fetch() -> Optional[pd.DataFrame]:
        payload = http.get_bytes(FF5_URL)
        if payload is None:
            return None
        frame = _parse_french_csv(payload)
        if frame is None or frame.empty:
            return None
        logger.info(
            "Facteurs FF5 chargés : %d mois (%s → %s).",
            len(frame), frame.index[0].date(), frame.index[-1].date(),
        )
        return frame

    return cache.memoize("ff5_factors_v1", _fetch, cfg)


def market_returns(factors: pd.DataFrame) -> pd.Series:
    """Rendement total du marché = prime de marché + taux sans risque."""
    return (factors["Mkt-RF"] + factors["RF"]).rename("market")
