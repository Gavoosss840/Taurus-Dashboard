"""
Taurus Dashboard – Schémas de réponse de l'API.

Convertit les objets du moteur (`taurus_core`) en structures JSON sûres :
les flottants non finis (NaN, ±∞) produits par un calcul dégradé ne sont pas
représentables en JSON strict et feraient échouer la désérialisation côté
navigateur — ils sont convertis en `null`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from taurus_core.valuation import Analysis, Pillar


def safe_float(value: object) -> Optional[float]:
    """Renvoie un flottant sérialisable en JSON, ou None."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean(value: object) -> object:
    """Nettoie récursivement une structure destinée à JSON."""
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def pillar_to_dict(pillar: Pillar) -> Dict:
    return {
        "key": pillar.key,
        "name": pillar.name,
        "score": safe_float(pillar.score),
        "weight": safe_float(pillar.weight),
        "available": pillar.available,
        "headline": pillar.headline,
        "verdict": pillar.verdict,
        "explanation": pillar.explanation,
        "details": _clean(pillar.details),
    }


def analysis_to_dict(analysis: Analysis) -> Dict:
    return {
        "ticker": analysis.ticker,
        "company_name": analysis.company_name,
        "sector": analysis.sector,
        "currency": analysis.currency,
        "region": analysis.region,
        "region_label": analysis.region_label,
        "verdict": analysis.verdict,
        "verdict_label": analysis.verdict_label,
        "composite_score": safe_float(analysis.composite_score),
        "confidence": safe_float(analysis.confidence),
        "summary": analysis.summary,
        "price": safe_float(analysis.price),
        "fair_value": safe_float(analysis.fair_value),
        "upside_pct": safe_float(analysis.upside_pct),
        "fair_price_low": safe_float(analysis.fair_price_low),
        "fair_price_high": safe_float(analysis.fair_price_high),
        "market_cap": safe_float(analysis.market_cap),
        "pillars": [pillar_to_dict(p) for p in analysis.pillars],
        "warnings": list(analysis.warnings),
        "data_sources": dict(analysis.data_sources),
        "computed_at": analysis.computed_at,
        "elapsed_seconds": safe_float(analysis.elapsed_seconds),
    }


def error_to_dict(message: str, ticker: str = "") -> Dict:
    return {"error": message, "ticker": ticker}
