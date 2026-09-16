"""
Taurus Dashboard – Application FastAPI.

Expose le moteur de valorisation et sert l'interface web.

Routes
──────
  GET /                      interface du dashboard
  GET /api/health            état du service
  GET /api/analyze/{ticker}  analyse complète d'un titre
  GET /api/config            paramètres du moteur (affichés dans « méthode »)
  POST /api/cache/clear      vide le cache disque
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from taurus_core import __version__, cache, diagnostics
from taurus_core.config import DEFAULT_CONFIG
from taurus_core.valuation import InvalidTickerError, TickerError, analyze

from .schemas import analysis_to_dict, error_to_dict

logging.basicConfig(
    level=os.environ.get("TAURUS_LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
logger = logging.getLogger("taurus.dashboard")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Taurus Dashboard",
    description=(
        "Valorisation d'une société par les trois piliers de la stratégie "
        "Taurus : alpha Fama-French, structure du capital Modigliani-Miller, "
        "momentum 12-1."
    ),
    version=__version__,
)

# Le dashboard est pensé pour tourner en local ; l'ouverture CORS permet de
# servir l'interface depuis un autre port pendant le développement.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Vérifie que le service répond et indique les sources configurées."""
    return {
        "status": "ok",
        "version": __version__,
        "fmp_key_configured": bool(os.environ.get("FMP_API_KEY", "").strip()),
        "cache_dir": DEFAULT_CONFIG.cache_dir,
        "cache_ttl_hours": DEFAULT_CONFIG.cache_ttl_hours,
    }


@app.get("/api/config")
def engine_config() -> dict:
    """Paramètres du moteur, affichés dans le volet « méthode »."""
    cfg = DEFAULT_CONFIG
    return {
        "lookback_months": cfg.lookback_months,
        "min_obs": cfg.min_obs,
        "momentum_months": cfg.momentum_months,
        "momentum_skip": cfg.momentum_skip,
        "leverage_gap_threshold_pct": cfg.leverage_gap_threshold * 100,
        "weights": {
            "alpha": cfg.w_alpha,
            "capital_structure": cfg.w_mm,
            "momentum": cfg.w_momentum,
        },
        "verdict_threshold": cfg.verdict_threshold,
        "verdict_strong_threshold": cfg.verdict_strong_threshold,
        "risk_free_rate_annual": cfg.risk_free_rate_annual,
        "equity_risk_premium": cfg.equity_risk_premium,
        "terminal_growth": cfg.terminal_growth,
        "return_df": cfg.return_df,
    }


@app.get("/api/analyze/{ticker}")
def analyze_ticker(
    ticker: str,
    refresh: bool = Query(
        False, description="Ignore le cache et retélécharge les données."
    ),
) -> JSONResponse:
    """Analyse un titre et renvoie le verdict et le détail des trois piliers."""
    if refresh:
        cache.clear(DEFAULT_CONFIG)

    try:
        result = analyze(ticker, DEFAULT_CONFIG)
    except InvalidTickerError as exc:
        # La saisie est fautive : 400, pas 404.
        logger.info("Ticker invalide « %s » : %s", ticker, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TickerError as exc:
        logger.info("Aucune donnée pour « %s » : %s", ticker, exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:                       # pragma: no cover
        logger.exception("Échec de l'analyse de %s", ticker)
        raise HTTPException(
            status_code=500,
            detail=(
                "Une erreur interne est survenue pendant l'analyse. "
                "Consultez les journaux du serveur pour le détail."
            ),
        ) from exc

    logger.info(
        "%s → %s (score %.3f) en %.2fs",
        result.ticker, result.verdict, result.composite_score, result.elapsed_seconds,
    )
    return JSONResponse(content=analysis_to_dict(result))


@app.get("/api/diagnostics")
def source_diagnostics(
    ticker: str = Query("AAPL", description="Ticker servant de sonde."),
) -> JSONResponse:
    """Interroge chaque source de données et rapporte ce qu'elle répond.

    Une analyse qui retombe sur une source dégradée n'en dit pas la raison :
    quota atteint, ticker inconnu, réseau coupé et clé absente produisent le
    même repli et appellent des réponses opposées.
    """
    report = diagnostics.run(ticker, DEFAULT_CONFIG)
    logger.info(
        "Diagnostic (%s) : %d/%d sources de cours disponibles.",
        report["ticker"],
        report["summary"]["price_sources_ok"],
        report["summary"]["price_sources_total"],
    )
    return JSONResponse(content=report)


@app.post("/api/cache/clear")
def clear_cache() -> dict:
    """Vide le cache disque des données de marché."""
    removed = cache.clear(DEFAULT_CONFIG)
    return {"status": "ok", "files_removed": removed}


@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Réponses d'erreur au même format que les réponses nominales."""
    return JSONResponse(
        status_code=exc.status_code,
        content=error_to_dict(str(exc.detail)),
    )


# ── Interface web ──────────────────────────────────────────────────────── #
# Montée en dernier : les routes /api/* ci-dessus ont la priorité.

if FRONTEND_DIR.is_dir():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
else:                                              # pragma: no cover
    logger.warning("Répertoire frontend introuvable : %s", FRONTEND_DIR)
