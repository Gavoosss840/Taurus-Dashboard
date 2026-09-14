"""
Taurus Dashboard – Cache disque avec durée de vie.

Reprend la mécanique de `taurus/data.py` (pickle + TTL) : les appels réseau
vers SEC EDGAR, Kenneth French ou les fournisseurs de prix sont lents et
limités en débit, alors qu'un même ticker est souvent consulté plusieurs fois
d'affilée depuis le dashboard.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .config import DEFAULT_CONFIG, ValuationConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Les écritures concurrentes (plusieurs requêtes HTTP en parallèle) doivent
# être sérialisées : deux processus écrivant le même fichier pickle
# produiraient un fichier tronqué illisible.
_WRITE_LOCK = threading.Lock()


def _cache_path(key: str, cfg: ValuationConfig) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    return Path(cfg.cache_dir) / f"{digest}.pkl"


def load(key: str, cfg: ValuationConfig = DEFAULT_CONFIG) -> Optional[Any]:
    """Renvoie l'entrée si elle existe et n'a pas expiré, sinon None."""
    path = _cache_path(key, cfg)
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > cfg.cache_ttl_hours:
        return None

    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:                      # pickle corrompu, version…
        logger.debug("Cache illisible pour %s (%s) — ignoré.", key, exc)
        return None


def save(key: str, value: Any, cfg: ValuationConfig = DEFAULT_CONFIG) -> None:
    """Écrit une entrée de cache (écriture atomique via fichier temporaire)."""
    path = _cache_path(key, cfg)
    try:
        with _WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(value, fh)
            tmp.replace(path)                     # atomique sur POSIX
    except Exception as exc:
        logger.debug("Écriture du cache impossible pour %s : %s", key, exc)


def memoize(key: str, producer: Callable[[], T], cfg: ValuationConfig = DEFAULT_CONFIG) -> T:
    """Renvoie la valeur en cache, sinon appelle `producer` et la met en cache.

    Une valeur « vide » (None) n'est jamais mise en cache : mémoriser un échec
    réseau pendant 12 h rendrait le dashboard durablement inutilisable.
    """
    cached = load(key, cfg)
    if cached is not None:
        return cached

    value = producer()
    if value is not None:
        save(key, value, cfg)
    return value


def clear(cfg: ValuationConfig = DEFAULT_CONFIG) -> int:
    """Vide le cache. Renvoie le nombre de fichiers supprimés."""
    directory = Path(cfg.cache_dir)
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.pkl"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
