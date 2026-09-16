# Taurus Dashboard

Saisissez le ticker d'une société : le moteur rejoue sur ce seul titre les
trois piliers de la stratégie **Taurus** et rend un verdict — **sous-évaluée**,
**sur-évaluée** ou **au juste prix** — accompagné d'une juste valeur par
action et du détail de chaque calcul.

![Aperçu](docs/apercu.png)

---

## Démarrage

**macOS / Linux**

```bash
pip install -r requirements.txt
cp .env.example .env        # facultatif : clé d'API et contact SEC
./run.sh                    # → http://127.0.0.1:8000
```

**Windows (PowerShell)**

```powershell
pip install -r requirements.txt
copy .env.example .env      # facultatif : clé d'API et contact SEC
.\run.ps1                   # → http://127.0.0.1:8000
```

`run.sh` est un script bash : PowerShell ne l'exécute pas, d'où `run.ps1`.
Si la stratégie d'exécution de PowerShell bloque le script, lancez le serveur
directement :

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Les deux lanceurs acceptent un port en argument (`./run.sh 8080`,
`.\run.ps1 8080`).

Aucune clé d'API n'est requise. Le dashboard fonctionne sur des sources
gratuites (SEC EDGAR, bibliothèque de Kenneth French, Yahoo Finance). Une clé
Financial Modeling Prep dans `.env` améliore la couverture et la fiabilité.

```bash
python -m pytest              # 191 tests, sans accès réseau
```

---

## Ce que fait le moteur

L'algorithme de production (dépôt `Trading-strategy-Taurus`) est un long/short
**cross-sectionnel** : il classe environ 500 titres les uns par rapport aux
autres et retient les 25 meilleurs et les 25 pires. Ce classement n'a aucun
sens sur un titre isolé.

Le dashboard conserve les trois piliers analytiques et les rapporte aux
**seuils absolus** que l'algorithme utilise déjà :

| Pilier | Mesure | Seuil de déclenchement | Poids |
|---|---|---|---|
| Alpha Fama-French 5 facteurs | statistique *t* de l'intercept | valeur critique de Student (≈ 2,00) | 40 % |
| Structure du capital (Modigliani-Miller) | divergence juste valeur / capitalisation | ± 25 % | 30 % |
| Momentum 12-1 ajusté de la volatilité | écart de momentum-Sharpe au marché | 0,50 | 30 % |

Chaque pilier produit un score `signal / seuil`, **borné à ± 2** : un score de
+1 signifie « ce pilier déclenche exactement son signal d'achat ». Le bornage
empêche une valeur aberrante — un *t* de 40 sur un titre au cours figé —
d'écraser les deux autres.

Le score composite est la moyenne pondérée des piliers **disponibles** : si un
pilier manque, son poids est redistribué plutôt que compté comme un zéro. Un
signal absent n'est pas un signal neutre.

| Score composite | Verdict |
|---|---|
| ≥ +1,00 | Fortement sous-évaluée |
| ≥ +0,50 | Sous-évaluée |
| entre −0,50 et +0,50 | Au juste prix |
| ≤ −0,50 | Sur-évaluée |
| ≤ −1,00 | Fortement sur-évaluée |

Le verdict exige donc la **concordance d'au moins deux piliers** : un pilier
seul, même saturé, ne pèse que 0,60 et ne franchit pas le seuil.

En régime de krach de momentum — volatilité du dernier mois supérieure au
double de la moyenne des douze précédents — la moitié du poids du momentum
bascule sur l'alpha, comme dans l'algorithme de production.

---

## Une correction apportée au modèle Modigliani-Miller

Le module `taurus/capital_structure.py` de l'algorithme de production calcule

```
V_U = capitalisation + dette nette − VA(bouclier fiscal)
V_L = V_U + VA(bouclier fiscal) − VA(détresse) − coûts d'agence
```

Le bouclier fiscal s'annule entre les deux lignes. Il reste :

```
capitaux propres théoriques = capitalisation − détresse − agence
divergence = −(détresse + agence) / capitalisation  ≤ 0  toujours
```

La juste valeur est donc **définie à partir du cours qu'elle est censée
juger**, et le signal ne peut structurellement jamais désigner une société
comme sous-évaluée. Vérification numérique sur l'algorithme lui-même :

| Cas de figure | Divergence |
|---|---|
| Technologie peu endettée | −0,001 % |
| Service public très endetté | −0,011 % |
| Valeur délaissée | −2,79 % |
| Société en détresse | −53,56 % |

Le dashboard estime la valeur non endettée **à partir des fondamentaux**,
selon la valeur actuelle ajustée (APV), formulation canonique de
Modigliani-Miller :

```
NOPAT = EBIT × (1 − τ)
β_U   = β_L / (1 + (1 − τ)·D/E)          dé-leviérisation de Hamada
r_U   = rf + β_U × prime de risque       MEDAF sans effet de levier
V_U   = NOPAT × (1 + g) / (r_U − g)      perpétuité croissante
```

Les trois frottements — bouclier fiscal, coûts de détresse de Merton, coûts
d'agence — sont calculés exactement comme dans l'algorithme de production.

Le détail de l'analyse, ses conséquences sur la stratégie de production et les
limites de la correction figurent dans [`docs/METHODOLOGIE.md`](docs/METHODOLOGIE.md).

---

## Couverture internationale

Le dashboard accepte une cotation américaine comme une place locale :

```
AAPL        ASML        TM              cotations américaines, ADR compris
MC.PA       SAP.DE      SHEL.L          Paris, Francfort, Londres
7203.T      0700.HK     005930.KS       Tokyo, Hong Kong, Séoul
RELIANCE.NS BHP.AX      PETR4.SA        Bombay, Sydney, São Paulo
```

Trois traitements en découlent.

**Le jeu de facteurs suit la région du siège.** Kenneth French publie des
facteurs distincts pour l'Amérique du Nord, l'Europe, le Japon,
l'Asie-Pacifique hors Japon et les marchés émergents. Régresser un titre
japonais sur les facteurs américains attribuerait à son alpha tout ce qui
n'est qu'un écart entre les deux marchés. La région se déduit du suffixe de
place, à défaut du pays déclaré à la SEC, à défaut de la devise de
publication — le dashboard indique laquelle de ces pistes a tranché.

**Les rendements sont convertis en dollars avant la régression.** Les facteurs
internationaux de Kenneth French sont libellés en dollars ; un titre coté en
euros ou en yens doit l'être aussi, faute de quoi son alpha absorberait la
variation de sa devise. Les taux viennent de la Banque centrale européenne.

**L'écran Modigliani-Miller raisonne dans la devise des comptes.** ASML publie
en euros et cote en dollars : rapprocher directement ses fondamentaux de sa
capitalisation mesurerait la parité EUR/USD, pas une décote. La capitalisation
reste affichée dans la devise de cotation, celle du cours.

Deux limites à connaître :

- **Les ADR.** Un certificat Toyota représente dix actions ordinaires, alors
  que la SEC publie le nombre d'ordinaires. La capitalisation est donc
  demandée à un fournisseur qui connaît le titre coté ; à défaut seulement,
  elle est reconstituée — et le dashboard signale alors qu'elle peut être
  surestimée.
- **Les devises hors BCE.** Le dollar de Taïwan n'est pas publié par la
  Banque centrale européenne : pour TSMC, le pilier Modigliani-Miller est
  neutralisé plutôt que calculé sur une parité supposée.

---

## Sources de données

Chaque famille de données passe par une chaîne de repli : aucune source n'est
indispensable, et l'interface indique toujours celle qui a servi.

| Donnée | Ordre de priorité |
|---|---|
| Cours mensuels | Financial Modeling Prep (si clé) → Yahoo Finance → marketdata.app → Nasdaq Data |
| Facteurs FF5 | Kenneth R. French Data Library, jeu régional |
| Fondamentaux | SEC EDGAR (XBRL, US-GAAP et IFRS) → Financial Modeling Prep (si clé) |
| Capitalisation et secteur | Financial Modeling Prep (si clé) → Yahoo Finance → Nasdaq Data |
| Taux de change | Banque centrale européenne, via api.frankfurter.app |

SEC EDGAR couvre les déposants américains **et** les émetteurs privés
étrangers déposant un formulaire 20-F — ASML, SAP, TSMC, Toyota, Shell,
Unilever, Novo Nordisk… — dont les comptes sont lus dans leur taxonomie
(US-GAAP ou IFRS) et leur devise. Hors de ce périmètre, une clé Financial
Modeling Prep est nécessaire ; sans elle, le pilier Modigliani-Miller est
neutralisé et son poids reporté sur les deux autres.

Nasdaq Data ne réintègre pas les dividendes : lorsque cette source est
utilisée, l'alpha et le momentum sont sous-estimés à hauteur du rendement du
dividende, et le dashboard le signale. Elle ne couvre par ailleurs que les
cotations américaines — une place locale passe nécessairement par Yahoo ou
Financial Modeling Prep.

Les réponses sont mises en cache sur disque (`.cache/`, 12 h par défaut).

---

## API

| Route | Description |
|---|---|
| `GET /api/analyze/{ticker}` | Analyse complète. `?refresh=true` ignore le cache. |
| `GET /api/config` | Paramètres du moteur. |
| `GET /api/health` | État du service. |
| `POST /api/cache/clear` | Vide le cache disque. |

```bash
curl -s http://127.0.0.1:8000/api/analyze/JNJ | jq '{verdict_label, composite_score, fair_value, upside_pct}'
```

```json
{
  "verdict_label": "Sous-évaluée",
  "composite_score": 0.548,
  "fair_value": 310.33,
  "upside_pct": 16.85
}
```

---

## Organisation du dépôt

```
taurus_core/              moteur de valorisation
├── config.py             hyper-paramètres (repris de TaurusConfig)
├── alpha.py              pilier 1 — régression Fama-French, erreur-type HC1
├── capital_structure.py  pilier 2 — juste valeur Modigliani-Miller (APV)
├── momentum.py           pilier 3 — momentum 12-1 ajusté de la volatilité
├── valuation.py          orchestration, score composite, verdict
├── cache.py              cache disque avec durée de vie
└── providers/            accès aux données, avec repli entre fournisseurs
    ├── prices.py         cours mensuels ajustés
    ├── fundamentals.py   comptes SEC EDGAR (US-GAAP et IFRS, toutes devises)
    ├── quotes.py         capitalisation et secteur du titre coté
    ├── factors.py        facteurs Fama-French régionaux
    ├── fx.py             taux de change de la BCE
    ├── regions.py        région de rattachement d'un titre
    └── sectors.py        code SIC → secteur GICS
backend/                  API FastAPI et sérialisation JSON
frontend/                 interface web (HTML/CSS/JS, sans compilation)
tests/                    191 tests, sans accès réseau
docs/METHODOLOGIE.md      justification des choix et limites du modèle
```

---

## Limites

- La valorisation par perpétuité est **très sensible** au taux d'actualisation
  et à la croissance retenue. Le pilier Modigliani-Miller affiche une grille de
  sensibilité : elle est à lire avant toute décision.
- Un modèle sans prime de croissance sous-valorise structurellement les
  sociétés en forte expansion. Un verdict « sur-évaluée » sur une valeur de
  croissance dit que le marché anticipe mieux que la perpétuité, pas
  nécessairement qu'il a tort.
- **Hors des déposants SEC**, les fondamentaux exigent une clé Financial
  Modeling Prep : SEC EDGAR ne référence que les sociétés déposant auprès
  d'elle, émetteurs étrangers en 20-F compris.
- Les facteurs de Kenneth French sont publiés avec un à deux mois de décalage.

Cet outil est un support d'analyse quantitative. Ce n'est pas un conseil en
investissement.
