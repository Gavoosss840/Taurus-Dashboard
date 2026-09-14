"""
Taurus Dashboard – Configuration du moteur de valorisation.

Les hyper-paramètres proviennent directement de `taurus/config.py` du dépôt
Trading-strategy-Taurus (TaurusConfig) : ils sont reproduits ici à l'identique
afin que le verdict du dashboard soit cohérent avec ce que l'algorithme de
production calculerait pour ce même titre.

Différence de périmètre : l'algo Taurus est un long/short cross-sectionnel
(il classe ~500 titres les uns par rapport aux autres).  Le dashboard répond à
une question mono-titre ("cette société est-elle sous-évaluée ?"), il s'appuie
donc sur les *seuils absolus* déjà présents dans l'algo :

  • Modigliani-Miller : leverage_gap_threshold = 25 %  → sous/sur-évaluation
  • Alpha FF5         : |t| > valeur critique de Student à 5 %  → significatif
  • Momentum 12-1     : comparé au momentum du marché (facteur Mkt de FF5)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


@dataclass
class ValuationConfig:
    """Paramètres du moteur de valorisation mono-titre."""

    # ------------------------------------------------------------------ #
    #  Modèle factoriel Fama-French 5 facteurs                            #
    # ------------------------------------------------------------------ #
    lookback_months: int = 60          # fenêtre de régression roulante
    min_obs: int = 36                  # seuil « confortable » d'observations
    hard_min_obs: int = 24             # en dessous : régression refusée
    alpha_tstat_threshold: float = 2.0 # repli si le t critique est indisponible

    # ------------------------------------------------------------------ #
    #  Structure du capital (écran Modigliani-Miller)                     #
    # ------------------------------------------------------------------ #
    leverage_gap_threshold: float = 0.25   # 25 % d'écart → signal
    min_interest_coverage: float = 1.5     # EBIT / intérêts en dessous → alerte
    industry_distress_costs: bool = True   # coûts de faillite sectoriels
    variable_credit_spread: bool = True    # spread fonction du levier
    return_df: float = 5.0                 # degrés de liberté Student-t (Merton)
    default_tax_rate: float = 0.21         # taux d'IS par défaut (US)

    # Valorisation de la firme non endettée (méthode APV).
    # Le coût des fonds propres non endettés est obtenu par le MEDAF à partir
    # du bêta dé-leviérisé issu de la régression Fama-French ; la prime de
    # risque des actions est le paramètre exogène le plus structurant.
    equity_risk_premium: float = 0.05      # prime de risque actions (US, long terme)
    terminal_growth: float = 0.025         # croissance perpétuelle du résultat
    min_discount_spread: float = 0.02      # écart plancher entre r_U et g
    default_unlevered_beta: float = 1.0    # si le bêta n'est pas estimable

    # ------------------------------------------------------------------ #
    #  Momentum (Jegadeesh & Titman 12-1, ajusté de la volatilité)        #
    # ------------------------------------------------------------------ #
    momentum_months: int = 12
    momentum_skip: int = 1
    vol_adjust_momentum: bool = True       # momentum « Sharpe » (Barroso 2015)
    momentum_crash_dampen: bool = True     # amortir en régime de krach

    # ------------------------------------------------------------------ #
    #  Combinaison des signaux (poids identiques à TaurusConfig)          #
    # ------------------------------------------------------------------ #
    w_alpha: float = 0.40
    w_mm: float = 0.30
    w_momentum: float = 0.30

    # Bornes de chaque score élémentaire : ±2 signifie « deux fois le seuil
    # de déclenchement de l'algo ».  Au-delà l'information est saturée, ce qui
    # empêche une valeur aberrante (t-stat de 40 sur un titre au prix figé)
    # de dominer le score composite.
    score_clip: float = 2.0

    # Seuils du verdict, exprimés sur le score composite.
    verdict_threshold: float = 0.50        # au-delà : sous/sur-évaluée
    verdict_strong_threshold: float = 1.00 # au-delà : fortement

    # Échelle de normalisation du momentum : écart de momentum-Sharpe
    # (titre − marché) considéré comme « un cran complet » de signal.
    momentum_scale: float = 0.50

    # ------------------------------------------------------------------ #
    #  Taux et divers                                                     #
    # ------------------------------------------------------------------ #
    risk_free_rate_annual: float = 0.045

    # ------------------------------------------------------------------ #
    #  Cache disque                                                       #
    # ------------------------------------------------------------------ #
    cache_dir: str = field(default_factory=lambda: os.environ.get("TAURUS_CACHE_DIR", ".cache"))
    cache_ttl_hours: float = field(default_factory=lambda: _env_float("TAURUS_CACHE_TTL_HOURS", 12.0))

    # ------------------------------------------------------------------ #
    #  Propriétés utilitaires                                             #
    # ------------------------------------------------------------------ #
    @property
    def history_months_needed(self) -> int:
        """Nombre de mois d'historique à télécharger pour tout calculer."""
        return max(
            self.lookback_months,
            self.momentum_months + self.momentum_skip,
        ) + 2   # marge pour le calcul des rendements et l'alignement mensuel


DEFAULT_CONFIG = ValuationConfig()
