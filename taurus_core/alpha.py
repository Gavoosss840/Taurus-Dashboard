"""
Taurus Dashboard – Pilier 1 : alpha de Jensen sur le modèle Fama-French 5.

Transposition mono-titre de `taurus/factors.py`.  L'algorithme de production
régresse les ~500 titres de l'univers en une seule inversion matricielle ; ici
un seul titre est concerné, mais la spécification économétrique est identique :

    r_i − rf = α + β_mkt·(Mkt−RF) + β_smb·SMB + β_hml·HML
                 + β_rmw·RMW + β_cma·CMA + ε

  • MCO sur les 60 derniers mois (cfg.lookback_months) ;
  • erreur-type de l'intercept robuste à l'hétéroscédasticité (HC1) ;
  • valeur critique de Student à 5 % bilatéral, avec df = n_obs − K
    — le nombre de degrés de liberté résiduels de la régression, pas les
    degrés de liberté de la loi des rendements (correctif appliqué dans
    l'algorithme de production après audit quantitatif).

Lecture économique : un alpha positif et significatif signifie que le titre a
dégagé un rendement que son exposition aux cinq facteurs de risque n'explique
pas — présomption de sous-évaluation.  Un alpha négatif significatif indique
l'inverse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .config import DEFAULT_CONFIG, ValuationConfig

logger = logging.getLogger(__name__)

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FACTOR_LABELS = {
    "Mkt-RF": "Marché",
    "SMB": "Taille (SMB)",
    "HML": "Valeur (HML)",
    "RMW": "Rentabilité (RMW)",
    "CMA": "Investissement (CMA)",
}


@dataclass
class AlphaResult:
    """Résultat de la régression factorielle pour un titre."""

    alpha_monthly: float
    alpha_annual: float
    alpha_tstat: float
    alpha_stderr: float
    t_critical: float
    p_value: float
    r_squared: float
    n_obs: int
    betas: Dict[str, float] = field(default_factory=dict)
    window_start: str = ""
    window_end: str = ""

    @property
    def significant(self) -> bool:
        """Vrai si |t| dépasse la valeur critique — l'alpha n'est pas du bruit."""
        return bool(np.isfinite(self.alpha_tstat) and abs(self.alpha_tstat) >= self.t_critical)

    @property
    def direction(self) -> int:
        """+1 alpha positif, −1 alpha négatif, 0 indéterminé."""
        if not np.isfinite(self.alpha_monthly) or self.alpha_monthly == 0:
            return 0
        return 1 if self.alpha_monthly > 0 else -1


def _hc1_stderr_intercept(
    design: np.ndarray,      # (T, K)
    residuals: np.ndarray,   # (T,)
    xtx_inv: np.ndarray,     # (K, K)
) -> float:
    """Erreur-type HC1 de l'intercept (White, corrigée des degrés de liberté).

    Les rendements boursiers sont hétéroscédastiques : l'erreur-type MCO
    classique sous-estime l'incertitude en période volatile et ferait passer
    pour significatif un alpha qui ne l'est pas.
    """
    n_obs, n_params = design.shape
    correction = n_obs / (n_obs - n_params)
    hat = design @ xtx_inv[:, 0]                     # (T,)
    variance = float((hat ** 2 * residuals ** 2).sum() * correction)
    return float(np.sqrt(max(variance, 1e-16)))


def compute_alpha(
    stock_returns: pd.Series,
    factors: pd.DataFrame,
    cfg: ValuationConfig = DEFAULT_CONFIG,
) -> Optional[AlphaResult]:
    """Régresse les rendements du titre sur les 5 facteurs Fama-French.

    Renvoie None si l'historique commun est trop court pour que la régression
    ait un sens (moins de `cfg.hard_min_obs` mois).
    """
    if stock_returns is None or stock_returns.empty or factors is None or factors.empty:
        return None

    missing = [c for c in FACTOR_COLUMNS + ["RF"] if c not in factors.columns]
    if missing:
        logger.warning("Facteurs manquants : %s", ", ".join(missing))
        return None

    # ── Alignement sur les mois communs ────────────────────────────────── #
    common = stock_returns.index.intersection(factors.index)
    if len(common) < cfg.hard_min_obs:
        logger.info(
            "Historique commun trop court (%d mois, minimum %d).",
            len(common), cfg.hard_min_obs,
        )
        return None

    # Fenêtre glissante : les 60 derniers mois communs.
    common = common.sort_values()[-cfg.lookback_months:]
    returns = stock_returns.loc[common].astype(float)
    factor_window = factors.loc[common]

    valid = returns.notna() & factor_window[FACTOR_COLUMNS + ["RF"]].notna().all(axis=1)
    returns = returns[valid]
    factor_window = factor_window[valid]
    n_obs = len(returns)

    if n_obs < cfg.hard_min_obs:
        logger.info("Trop d'observations manquantes (%d exploitables).", n_obs)
        return None
    if n_obs < cfg.min_obs:
        logger.info(
            "Régression sur %d mois seulement (seuil confortable : %d).",
            n_obs, cfg.min_obs,
        )

    # ── Matrice de régression : [1, Mkt-RF, SMB, HML, RMW, CMA] ────────── #
    excess = returns.values - factor_window["RF"].values
    design = np.column_stack(
        [np.ones(n_obs)] + [factor_window[c].values for c in FACTOR_COLUMNS]
    )
    n_params = design.shape[1]

    xtx_inv = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inv @ (design.T @ excess)
    residuals = excess - design @ coefficients

    stderr = _hc1_stderr_intercept(design, residuals, xtx_inv)
    alpha_monthly = float(coefficients[0])

    # Une erreur-type quasi nulle trahit une série de prix figée (titre
    # suspendu, cours répété) : le t-stat exploserait sans que l'information
    # soit réelle.  On préfère renvoyer NaN qu'un « t = 4 000 ».
    tstat = float(alpha_monthly / stderr) if stderr > 1e-7 else float("nan")

    degrees_freedom = max(n_obs - n_params, 1)
    t_critical = float(student_t.ppf(0.975, df=degrees_freedom))
    p_value = (
        float(2 * student_t.sf(abs(tstat), df=degrees_freedom))
        if np.isfinite(tstat) else float("nan")
    )

    ss_residual = float((residuals ** 2).sum())
    ss_total = float(((excess - excess.mean()) ** 2).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")

    return AlphaResult(
        alpha_monthly=alpha_monthly,
        alpha_annual=float((1 + alpha_monthly) ** 12 - 1),
        alpha_tstat=tstat,
        alpha_stderr=stderr,
        t_critical=t_critical,
        p_value=p_value,
        r_squared=float(r_squared),
        n_obs=n_obs,
        betas={
            name: float(coefficients[i + 1])
            for i, name in enumerate(FACTOR_COLUMNS)
        },
        window_start=str(returns.index[0].date()),
        window_end=str(returns.index[-1].date()),
    )
