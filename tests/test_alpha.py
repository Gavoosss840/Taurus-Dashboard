"""Tests du pilier alpha Fama-French."""

import numpy as np
import pandas as pd
import pytest

from taurus_core.alpha import FACTOR_COLUMNS, compute_alpha
from taurus_core.config import ValuationConfig

CFG = ValuationConfig()


def synthetic_factors(n_months: int = 72, seed: int = 7) -> pd.DataFrame:
    """Série de facteurs reproductible, d'amplitude réaliste."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.007, 0.042, n_months),
            "SMB": rng.normal(0.001, 0.022, n_months),
            "HML": rng.normal(0.001, 0.025, n_months),
            "RMW": rng.normal(0.002, 0.018, n_months),
            "CMA": rng.normal(0.001, 0.016, n_months),
            "RF": np.full(n_months, 0.0035),
        },
        index=index,
    )


def returns_from(factors: pd.DataFrame, alpha: float, betas: dict,
                 noise: float = 0.0, seed: int = 11) -> pd.Series:
    """Rendements construits exactement selon le modèle, plus un bruit."""
    rng = np.random.default_rng(seed)
    excess = np.full(len(factors), alpha)
    for name, beta in betas.items():
        excess = excess + beta * factors[name].values
    if noise > 0:
        excess = excess + rng.normal(0.0, noise, len(factors))
    return pd.Series(excess + factors["RF"].values, index=factors.index)


BETAS = {"Mkt-RF": 1.15, "SMB": -0.30, "HML": 0.45, "RMW": 0.20, "CMA": -0.10}


# ── Exactitude de l'estimation ───────────────────────────────────────────

def test_recovers_known_alpha_and_betas():
    """Sans bruit, la régression doit retrouver les paramètres exacts."""
    factors = synthetic_factors()
    returns = returns_from(factors, alpha=0.004, betas=BETAS)

    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.alpha_monthly == pytest.approx(0.004, abs=1e-9)
    for name, beta in BETAS.items():
        assert result.betas[name] == pytest.approx(beta, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_annualisation_is_compounded_not_multiplied():
    factors = synthetic_factors()
    returns = returns_from(factors, alpha=0.005, betas=BETAS)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.alpha_annual == pytest.approx((1.005) ** 12 - 1, rel=1e-9)


def test_zero_alpha_is_not_significant():
    factors = synthetic_factors()
    returns = returns_from(factors, alpha=0.0, betas=BETAS, noise=0.03)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert not result.significant


def test_large_alpha_is_significant():
    factors = synthetic_factors()
    returns = returns_from(factors, alpha=0.02, betas=BETAS, noise=0.01)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.significant
    assert result.direction == 1


def test_negative_alpha_direction():
    factors = synthetic_factors()
    returns = returns_from(factors, alpha=-0.02, betas=BETAS, noise=0.01)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.direction == -1
    assert result.alpha_annual < 0


# ── Degrés de liberté et valeur critique ─────────────────────────────────

def test_critical_value_uses_residual_degrees_of_freedom():
    """df = n_obs − K, et non les degrés de liberté de la loi des rendements.

    L'algorithme de production utilisait min(df_résiduel, ν=5), ce qui portait
    le seuil à 2,57 et étouffait des signaux légitimes ; l'audit quantitatif
    du dépôt a corrigé ce point.
    """
    from scipy.stats import t as student_t

    factors = synthetic_factors(n_months=60)
    returns = returns_from(factors, alpha=0.003, betas=BETAS, noise=0.02)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    expected_df = result.n_obs - (len(FACTOR_COLUMNS) + 1)
    assert result.t_critical == pytest.approx(student_t.ppf(0.975, df=expected_df))
    assert 1.9 < result.t_critical < 2.1


# ── Fenêtre et données dégradées ─────────────────────────────────────────

def test_window_is_limited_to_lookback():
    factors = synthetic_factors(n_months=120)
    returns = returns_from(factors, alpha=0.003, betas=BETAS, noise=0.02)
    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.n_obs == CFG.lookback_months


def test_short_history_is_refused():
    factors = synthetic_factors(n_months=18)
    returns = returns_from(factors, alpha=0.003, betas=BETAS)
    assert compute_alpha(returns, factors, CFG) is None


def test_missing_months_are_dropped_not_imputed():
    """Une valeur manquante ne doit pas être remplacée par un zéro.

    Un rendement imputé à 0 % est presque parfaitement expliqué par le facteur
    de marché : il réduit artificiellement la variance résiduelle et gonfle
    le t-stat de l'alpha.
    """
    factors = synthetic_factors(n_months=72)
    returns = returns_from(factors, alpha=0.004, betas=BETAS)
    # Positions choisies à l'intérieur de la fenêtre de régression
    # (les 60 derniers mois sur 72, soit les positions 12 à 71).
    returns.iloc[40] = np.nan
    returns.iloc[55] = np.nan

    result = compute_alpha(returns, factors, CFG)

    assert result is not None
    assert result.n_obs == CFG.lookback_months - 2
    # L'estimation reste exacte sur les mois conservés.
    assert result.alpha_monthly == pytest.approx(0.004, abs=1e-9)


def test_frozen_price_series_yields_nan_tstat():
    """Un cours figé donne une erreur-type nulle : le t-stat exploserait."""
    factors = synthetic_factors()
    constant = pd.Series(0.0, index=factors.index)

    result = compute_alpha(constant, factors, CFG)

    assert result is not None
    # L'alpha vaut −RF exactement ; c'est le t-stat qui doit être neutralisé.
    assert not result.significant or np.isnan(result.alpha_tstat)


def test_missing_factor_column_is_refused():
    factors = synthetic_factors().drop(columns=["RMW"])
    returns = returns_from(synthetic_factors(), alpha=0.004, betas=BETAS)
    assert compute_alpha(returns, factors, CFG) is None


def test_empty_inputs_are_refused():
    assert compute_alpha(pd.Series(dtype=float), synthetic_factors(), CFG) is None
    assert compute_alpha(None, synthetic_factors(), CFG) is None


def test_hc1_stderr_matches_white_formula():
    """L'erreur-type de l'intercept doit suivre exactement la formule de White.

    HC1 n'est pas systématiquement supérieure à l'erreur-type MCO classique —
    elle peut être plus faible lorsque les observations très volatiles ont un
    faible levier. Sa vertu est d'être CONSISTANTE sous hétéroscédasticité,
    là où l'erreur-type classique ne l'est pas. On vérifie donc la formule.
    """
    rng = np.random.default_rng(3)
    factors = synthetic_factors(n_months=60)
    # Volatilité qui double à mi-parcours : hétéroscédasticité franche.
    scale = np.concatenate([np.full(30, 0.01), np.full(30, 0.08)])
    returns = returns_from(factors, alpha=0.003, betas=BETAS)
    returns = returns + rng.normal(0.0, 1.0, 60) * scale

    result = compute_alpha(returns, factors, CFG)
    assert result is not None

    # Reconstruction indépendante : V = (X'X)⁻¹ X' diag(e²) X (X'X)⁻¹,
    # corrigée du facteur n/(n−K) propre à HC1.
    design = np.column_stack(
        [np.ones(len(factors))] + [factors[c].values for c in FACTOR_COLUMNS]
    )
    n_obs, n_params = design.shape
    excess = returns.values - factors["RF"].values
    xtx_inv = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inv @ (design.T @ excess)
    residuals = excess - design @ coefficients

    meat = design.T @ np.diag(residuals ** 2) @ design
    covariance = xtx_inv @ meat @ xtx_inv * (n_obs / (n_obs - n_params))
    expected = np.sqrt(covariance[0, 0])

    assert result.alpha_stderr == pytest.approx(expected, rel=1e-9)


def test_heteroskedasticity_changes_the_stderr():
    """Sur des données hétéroscédastiques, HC1 s'écarte de l'erreur classique."""
    rng = np.random.default_rng(3)
    factors = synthetic_factors(n_months=60)
    scale = np.concatenate([np.full(30, 0.01), np.full(30, 0.08)])
    returns = returns_from(factors, alpha=0.003, betas=BETAS)
    returns = returns + rng.normal(0.0, 1.0, 60) * scale

    result = compute_alpha(returns, factors, CFG)
    assert result is not None

    design = np.column_stack(
        [np.ones(len(factors))] + [factors[c].values for c in FACTOR_COLUMNS]
    )
    excess = returns.values - factors["RF"].values
    xtx_inv = np.linalg.pinv(design.T @ design)
    residuals = excess - design @ (xtx_inv @ (design.T @ excess))
    sigma2 = (residuals ** 2).sum() / (len(factors) - design.shape[1])
    naive_stderr = float(np.sqrt(sigma2 * xtx_inv[0, 0]))

    assert abs(result.alpha_stderr / naive_stderr - 1.0) > 0.02
