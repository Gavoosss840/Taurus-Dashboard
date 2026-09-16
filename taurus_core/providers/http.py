"""
Taurus Dashboard – Client HTTP partagé.

Un seul endroit pour : la session `requests` réutilisée (connexions
persistantes), le retry avec back-off exponentiel, et le respect du proxy
sortant configuré dans l'environnement.
"""

from __future__ import annotations

import logging
import os
import time
import time as _time
from dataclasses import dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 20

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_session: Optional[requests.Session] = None


def session() -> requests.Session:
    """Session `requests` partagée (connexions HTTP persistantes)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": _BROWSER_UA, "Accept": "application/json"})
    return _session


def sec_user_agent() -> str:
    """User-Agent exigé par la SEC : elle refuse les clients anonymes."""
    return os.environ.get(
        "TAURUS_SEC_USER_AGENT",
        "Taurus Dashboard contact@example.com",
    )


def get_json(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Any]:
    """GET renvoyant du JSON, avec retry exponentiel.

    Renvoie None sur échec définitif — les appelants enchaînent alors sur le
    fournisseur suivant plutôt que de propager une exception.
    """
    for attempt in range(retries):
        try:
            resp = session().get(url, params=params, headers=headers, timeout=timeout)

            # 429/5xx : transitoire, on retente. 4xx : définitif, on abandonne
            # immédiatement (clé d'API absente, ticker inconnu…).
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                logger.debug("GET %s → HTTP %d (abandon)", url, resp.status_code)
                return None

            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                logger.debug("GET %s a échoué définitivement : %s", url, exc)
                return None
            wait = 2 ** attempt
            logger.debug("GET %s a échoué (%s), nouvelle tentative dans %ds", url, exc, wait)
            time.sleep(wait)
    return None


def get_bytes(
    url: str,
    headers: Optional[dict] = None,
    retries: int = 3,
    timeout: int = 30,
) -> Optional[bytes]:
    """GET renvoyant le contenu binaire (archives ZIP de Kenneth French).

    L'en-tête `Accept` de la session vise le JSON ; certains serveurs de
    fichiers — celui de Dartmouth notamment — répondent alors 406. On le
    remplace ici par un `Accept` universel.
    """
    merged = {"Accept": "*/*"}
    merged.update(headers or {})
    for attempt in range(retries):
        try:
            resp = session().get(url, headers=merged, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            if attempt == retries - 1:
                logger.debug("GET %s a échoué définitivement : %s", url, exc)
                return None
            time.sleep(2 ** attempt)
    return None


# --------------------------------------------------------------------------- #
#  Sondage d'un fournisseur                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Probe:
    """Ce qu'un fournisseur répond, sans interprétation."""

    name: str
    url: str
    status: Optional[int] = None      # code HTTP, None si la connexion a échoué
    latency_ms: Optional[int] = None
    error: str = ""                   # type d'exception, le cas échéant
    excerpt: str = ""                 # début de la réponse, pour diagnostic
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def probe(
    name: str,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 15,
    attempts: int = 1,
    backoff: float = 2.0,
) -> Probe:
    """Interroge une source et rapporte ce qu'elle répond, sans lever.

    Sert au diagnostic : quand une analyse retombe sur une source dégradée,
    l'utilisateur doit pouvoir savoir POURQUOI la source préférée a été
    écartée — quota atteint, ticker inconnu, réseau coupé — plutôt que de
    constater le repli sans explication.
    """
    result = Probe(name=name, url=url, attempts=0)

    for attempt in range(max(attempts, 1)):
        result.attempts = attempt + 1
        started = _time.perf_counter()
        try:
            response = session().get(url, params=params, headers=headers, timeout=timeout)
            result.latency_ms = int((_time.perf_counter() - started) * 1000)
            result.status = response.status_code
            result.error = ""
            result.excerpt = response.text[:160].replace("\n", " ").strip()
            if result.ok or response.status_code < 500 and response.status_code != 429:
                return result
        except Exception as exc:
            result.latency_ms = int((_time.perf_counter() - started) * 1000)
            result.status = None
            result.error = type(exc).__name__
            result.excerpt = str(exc)[:160]

        if attempt < attempts - 1:
            _time.sleep(backoff * (attempt + 1))

    return result
