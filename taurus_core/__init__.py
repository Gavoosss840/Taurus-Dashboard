"""
Taurus Dashboard – moteur de valorisation mono-titre.

Reprend les trois piliers analytiques de la stratégie Taurus
(Trading-strategy-Taurus) et les applique à un seul ticker :

  • `alpha`             — alpha de Jensen sur le modèle Fama-French 5 facteurs
  • `capital_structure` — juste valeur Modigliani-Miller (bouclier fiscal,
                          coûts de détresse de Merton, coûts d'agence)
  • `momentum`          — momentum 12-1 ajusté de la volatilité

`valuation.analyze(ticker)` orchestre les trois et rend le verdict.
`diagnostics.run(ticker)` interroge les sources et rapporte leur état.
"""

__version__ = "1.0.0"
