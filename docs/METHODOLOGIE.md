# Méthodologie

Ce document justifie les choix du moteur de valorisation et expose ses limites.
Il s'adresse à quelqu'un qui doit décider s'il peut faire confiance à un
verdict du dashboard.

---

## 1. Du cross-sectionnel à l'absolu

L'algorithme de production (`Trading-strategy-Taurus`) construit un portefeuille
long/short : il régresse ~500 titres, les classe, achète les 25 premiers et
vend les 25 derniers. Tous ses signaux sont **relatifs** — z-scores,
quantiles, terciles — et un titre isolé n'a pas de rang.

Le dashboard s'appuie donc sur les **seuils absolus déjà présents** dans
l'algorithme, plutôt que d'inventer une calibration :

| Pilier | Seuil, et son origine dans l'algorithme |
|---|---|
| Alpha | valeur critique de Student à 5 % bilatéral, `df = n_obs − K` (`factors.py`) |
| Structure du capital | `leverage_gap_threshold = 0,25` (`config.py`) |
| Momentum | comparaison au marché, au lieu des terciles de l'univers |

Chaque pilier rend `score = signal / seuil`, borné à ± 2.

**Pourquoi borner ?** Sans bornage, un titre au cours figé (suspension,
cotation répétée) produit une erreur-type quasi nulle et un *t* de plusieurs
dizaines : ce seul pilier emporterait le verdict. Le bornage à ± 2 signifie
« deux fois le seuil de déclenchement, au-delà l'information est saturée ».

**Pourquoi redistribuer le poids d'un pilier absent ?** Compter un pilier
indisponible comme un score de zéro le ferait voter « au juste prix ». Or
l'absence de données n'est pas une opinion. Le poids est donc réparti sur les
piliers disponibles, et l'indice de fiabilité affiché baisse en conséquence.

---

## 2. La circularité du modèle Modigliani-Miller d'origine

### Le constat

`taurus/capital_structure.py` calcule, dans cet ordre :

```python
pv_tax_shield = tax_rate * interest / (rf + spread)
VU  = market_cap + net_debt - pv_tax_shield
VL  = VU + pv_tax_shield - pv_distress - pv_agency
VL_equity  = VL - net_debt
divergence = (VL_equity - market_cap) / market_cap
```

En substituant :

```
VL          = market_cap + net_debt − détresse − agence
VL_equity   = market_cap − détresse − agence
divergence  = −(détresse + agence) / market_cap
```

Le bouclier fiscal s'annule : il est retranché dans `VU` puis rajouté dans
`VL`. La divergence ne dépend donc **que de deux frottements positifs**, et
elle est **toujours négative ou nulle**.

### Vérification numérique

Exécution de `taurus.capital_structure._mm_valuation` sur quatre profils :

| Cas | Divergence | VA détresse | VA agence | P(défaut) |
|---|---|---|---|---|
| Technologie peu endettée | −0,001 % | 1,7 × 10⁷ | 0 | 0,0000 |
| Service public très endetté | −0,011 % | 1,7 × 10⁷ | 0 | 0,0004 |
| Valeur délaissée | −2,785 % | 5,7 × 10⁷ | 5,0 × 10⁸ | 0,0057 |
| Société en détresse | −53,559 % | 2,7 × 10⁸ | 8,0 × 10⁸ | 0,0798 |

Le seuil d'achat (`divergence > +25 %`) est **inatteignable**.

### Conséquence sur la stratégie de production

Dans `strategy.py:_binary_signal` :

```python
under_tickers = mm_df[mm_df["underleveraged"]].index   # toujours vide
mm_underval_ratio = len(under_tickers) / max(len(alpha_df), 1)   # toujours 0
if mm_underval_ratio >= MM_MIN_RATIO:   # 0 >= 0,10 → jamais vrai
    ...
else:
    q70 = alpha_df["alpha_tstat"].quantile(0.70)
    long_candidates = alpha_df[alpha_df["alpha_tstat"] >= q70].index
```

La branche de repli est donc systématiquement empruntée : **la jambe longue de
la stratégie est sélectionnée sur le seul alpha**, le pilier Modigliani-Miller
n'y contribuant pas.

Côté court, l'écran reste opérant : `overleveraged` se déclenche quand
détresse + agence dépassent 25 % de la capitalisation, ainsi que par les deux
garde-fous de solvabilité (couverture des intérêts inférieure à 1,5× ; EBIT
négatif avec dette). Le pilier fonctionne donc comme un **filtre de risque de
faillite**, pas comme une mesure de valorisation.

La même correction a été portée dans l'algorithme de production
(`Trading-strategy-Taurus`, branche `claude/confident-brahmagupta-7uh32s`), où
elle change la composition de la jambe longue et les niveaux de take-profit :
les back-tests publiés doivent y être relancés.

### La correction retenue ici

La valeur non endettée est estimée à partir des **fondamentaux seuls**, par la
valeur actuelle ajustée :

```
NOPAT = EBIT₁₂ₘ × (1 − τ)
β_U   = β_L / (1 + (1 − τ) · D/E)
r_U   = rf + β_U × prime de risque actions
V_U   = NOPAT × (1 + g) / (r_U − g)

V_L                        = V_U + VA(bouclier) − VA(détresse) − agence
capitaux propres théoriques = V_L − dette nette
```

Le bêta `β_L` provient de la régression Fama-French du pilier 1 : même titre,
même fenêtre, donc un bêta cohérent avec l'alpha affiché à côté.

Les trois frottements sont repris **à l'identique** de l'algorithme de
production, y compris les taux de destruction sectoriels et la probabilité de
défaut de Merton sous loi de Student.

### Dette risquée : pourquoi Hamada seul ne suffit pas

La forme classique de Hamada, `β_U = β_L / (1 + (1 − τ)·D/E)`, suppose une dette
**sans risque**. Chez une société très endettée, cette hypothèse abaisse
beaucoup trop `β_U`, donc `r_U`, et gonfle la perpétuité : le modèle
récompenserait l'endettement — précisément l'inversion que la comparaison au
niveau des capitaux propres cherche à éviter. Une société à `D/E = 1,5` avec
`β_L = 0,95` se dé-leviérise en `β_U = 0,44`, un bêta d'actif inférieur à celui
d'un service public pour une cyclique endettée.

Le moteur utilise donc la forme complète, qui attribue à la dette son propre
risque systématique :

```
β_U = (E·β_L + D(1 − τ)·β_D) / (E + D(1 − τ))
```

`β_D` se déduit du spread de crédit déjà calculé. Une prime de crédit ne
rémunère qu'en partie le risque systématique — le reste couvre la perte
attendue en cas de défaut et l'illiquidité — et en retenir la moitié est
l'approximation usuelle (Cooper & Davydenko, 2007), plafonnée à 0,4 : au-delà,
la créance se comporte comme une action et la séparation dette / capitaux
propres perd son sens.

### Ce que la correction ne supprime pas

Un couplage résiduel subsiste : la dé-leviérisation de Hamada prend `D/E` en
**valeur de marché** des capitaux propres, comme le veut la pratique — la
valeur comptable est déformée par les rachats d'actions, au point de devenir
négative. Un cours plus élevé abaisse donc `D/E`, relève `β_U` et `r_U`, et
abaisse la valeur actualisée.

Ce couplage est d'un ordre de grandeur inférieur à celui du modèle d'origine
(un cours triplé déplace `V_U` de moins de 10 %, contre 100 % auparavant) et il
joue dans le sens **stabilisant** : il renforce le signal au lieu de l'annuler.
Un test le vérifie (`test_valuation_is_independent_of_market_price`).

---

## 3. Paramètres exogènes et sensibilité

Trois paramètres ne sont pas estimés sur les données du titre :

| Paramètre | Valeur | Justification |
|---|---|---|
| Taux sans risque | 4,5 % | valeur de `TaurusConfig.risk_free_rate_annual` |
| Prime de risque actions | 5,0 % | moyenne longue période, marché américain |
| Croissance perpétuelle | 2,5 % | ordre de grandeur de la croissance nominale de long terme |

La valorisation par perpétuité y est **très sensible**. Plutôt que de livrer un
chiffre unique faussement précis, le pilier affiche une grille de neuf
scénarios : `r_U` à ± 1 point, `g` entre 1,5 % et 3,5 %. L'écart entre les
coins de la grille dépasse couramment un facteur deux — c'est l'ordre de
grandeur de l'incertitude réelle du modèle, et il doit être lu avant toute
décision.

Le plafond `g ≤ r_U − 2 points` empêche la perpétuité de diverger.

---

## 4. Choix de mise en œuvre

### Alpha : erreur-type robuste à l'hétéroscédasticité

L'erreur-type de l'intercept suit la correction HC1 de White. Les rendements
boursiers sont hétéroscédastiques : l'erreur-type MCO classique sous-estime
l'incertitude en période volatile et ferait passer pour significatif un alpha
qui ne l'est pas.

HC1 n'est pas systématiquement plus grande que l'erreur-type classique — elle
peut être plus faible lorsque les observations volatiles ont un faible levier.
Sa vertu est d'être **consistante** sous hétéroscédasticité. Le test vérifie
donc l'exactitude de la formule, pas une inégalité.

### Alpha : degrés de liberté

La valeur critique utilise `df = n_obs − K`, les degrés de liberté résiduels de
la régression. L'algorithme de production employait auparavant
`min(df_résiduel, ν = 5)`, ce qui portait le seuil à 2,57 et étouffait des
signaux légitimes ; son audit quantitatif a corrigé ce point, repris ici. Des
rendements à queues épaisses ne changent pas la loi de référence de la
statistique *t*.

### Alpha : allonger la fenêtre n'est pas la solution

Un alpha non significatif appelle naturellement la question « faudrait-il plus
de mois ? ». L'arithmétique est sans appel : le *t* croît comme la racine du
nombre d'observations, donc passer d'un *t* de 1,17 au seuil de 2,00 exige de
multiplier la fenêtre par (2,00 / 1,17)² ≈ 2,9 — soit environ 175 mois, près
de quinze ans.

Ce n'est pas le bon remède. Sur quinze ans, l'alpha d'une société n'est pas
constant : modèle d'affaires, position concurrentielle et direction changent.
On estimerait avec précision une moyenne qui ne décrit plus l'entreprise
d'aujourd'hui. La fenêtre de 60 mois de l'algorithme est ce compromis-là, et
le dashboard le conserve.

**Échantillonner plus finement ne sert à rien non plus.** Passer au rendement
quotidien multiplierait par vingt le nombre d'observations sans améliorer la
précision de l'alpha : l'erreur-type d'une moyenne de rendements dépend de la
DURÉE CALENDAIRE observée, pas du nombre de points à l'intérieur (Merton,
1980). Les données quotidiennes améliorent l'estimation de la volatilité et du
bêta, jamais celle du rendement espéré.

Reste une voie légitime : réduire la variance résiduelle en expliquant mieux
les rendements. Faire passer le R² de 49 % à 65 % relèverait le *t* de 1,17 à
1,41 — une amélioration réelle, insuffisante à elle seule.

La conclusion utile n'est donc pas « il manque des données » mais « l'effet est
faible au regard du bruit ». Le moteur en tient compte sans rien masquer : le
score du pilier vaut *t* / seuil, soit 0,58 dans cet exemple. Le pilier compte
pour une fraction de son poids au lieu d'être compté comme acquis ou rejeté,
et sa formulation distingue explicitement trois cas — significatif, penchant
sans être démontré, indiscernable du bruit.

### Alpha : pas d'imputation des mois manquants

Un rendement manquant remplacé par 0 % est presque parfaitement expliqué par
le facteur de marché : il réduit artificiellement la variance résiduelle et
gonfle le *t* de l'alpha. Les mois manquants sont écartés, jamais comblés.

### Momentum : fenêtres strictement comparables

Le momentum du titre, `P[fin] / P[début] − 1`, couvre **onze** rendements
mensuels pour douze relevés de cours — la convention de Jegadeesh & Titman,
reprise de l'algorithme. Le momentum du marché doit couvrir exactement ces
mois-là.

Deux écarts ont été corrigés par rapport à une transposition naïve :

1. **Décalage d'un mois.** Indexer le marché sur les douze relevés de cours lui
   donnait douze rendements contre onze au titre.
2. **Retard de publication.** Les facteurs de Kenneth French paraissent avec un
   à deux mois de décalage. Lorsque le marché s'arrête plus tôt, les deux
   mesures sont recalculées sur les seuls mois communs.

### Momentum : plancher de volatilité

Une volatilité annualisée inférieure à 0,5 % n'est pas crédible pour une
action : elle trahit un cours figé. Sans plancher, diviser le momentum par une
volatilité de 10⁻¹⁷ produit un score de l'ordre de 10¹⁶ qui saturerait le
verdict à lui seul. Sous ce plancher, la comparaison au marché bascule sur les
rendements bruts des deux côtés — comparer un ratio rendement/risque du titre
à un rendement brut du marché n'aurait aucun sens dimensionnel.

### Fondamentaux : reconstruction des flux sur douze mois

SEC EDGAR mélange dans un même tableau les valeurs trimestrielles (10-Q) et
annuelles (10-K). Le modèle raisonne en flux **annuels**. Prendre « la valeur
la plus récente » donnerait tantôt un trimestre, tantôt un exercice — un
facteur quatre selon le calendrier de publication.

Le moteur additionne donc quatre trimestres consécutifs et disjoints, en
écartant les périodes cumulées qui les chevauchent ; à défaut, il retient le
dernier exercice ; en dernier recours, il annualise la dernière période
cumulée.

### Fondamentaux : fraîcheur avant priorité

Les entreprises changent de concept XBRL au fil des ans et continuent parfois
d'exposer l'ancien, figé sur sa dernière valeur. Coca-Cola a cessé d'alimenter
`LongTermDebt` en 2024 au profit de `LongTermDebtAndCapitalLeaseObligations` ;
NextEra a abandonné `Revenues` en 2013. Suivre l'ordre de priorité des concepts
renverrait dans les deux cas un chiffre périmé de plusieurs années.

Le moteur retient donc le concept le plus **frais**, la priorité ne tranchant
qu'entre concepts d'arrêtés proches (moins d'un trimestre d'écart).

### Fondamentaux : taux d'imposition

La théorie Modigliani-Miller raisonne sur le taux **marginal**. Le taux effectif
ne s'y substitue que lorsqu'il reste plausible — entre 10 % et 40 %. Un exercice
à 0 % (crédits d'impôt des énergéticiens renouvelables) ou à 60 %
(redressement) reflète un accident comptable, pas la fiscalité structurelle de
la dette ; le taux statutaire reprend alors la main.

### Fondamentaux : capitaux propres négatifs

Les sociétés qui rachètent massivement leurs actions affichent des capitaux
propres comptables négatifs. Diviser la dette par ces capitaux propres
produirait un levier de l'ordre de la dette en valeur absolue, donc des coûts
d'agence délirants et un signal de survalorisation mécanique. La
capitalisation boursière sert alors d'assiette, et le ratio est plafonné à 10.

### Capitalisation boursière

Elle est reconstruite comme `actions en circulation × dernier cours`, plutôt
que reprise d'un fournisseur : deux chiffres frais valent mieux qu'une
capitalisation publiée qui peut dater de plusieurs semaines.

---

## 5. Titres étrangers

### Facteurs régionaux

Kenneth French publie cinq jeux de facteurs : Amérique du Nord, Europe, Japon,
Asie-Pacifique hors Japon, marchés émergents. Le moteur retient celui de la
région du **siège**, pas de la place de cotation : un certificat de dépôt
européen coté à New York reste exposé au risque européen, et les facteurs
européens de Kenneth French, libellés en dollars comme lui, en sont bien la
référence.

La région se déduit de trois signaux, du plus fiable au moins fiable : le
suffixe de place du ticker, le pays déclaré à la SEC, la devise de
publication. Le dashboard affiche lequel a tranché — une région devinée sur la
seule devise se trompe sur les groupes étrangers tenant leurs comptes en
dollars, Shell en étant l'exemple.

Le jeu américain historique remonte à 1963, contre 1990 pour les jeux
internationaux : la fenêtre de régression de 60 mois reste pleine dans tous
les cas.

### Deux conversions, pour deux raisons distinctes

Les facteurs internationaux sont libellés en dollars. Un titre coté en euros
ou en yens est donc **converti en dollars avant la régression**, faute de quoi
son alpha absorberait la variation de sa devise.

L'écran Modigliani-Miller, lui, rapproche des fondamentaux d'une
capitalisation : les deux doivent être dans une **même devise**, et c'est
celle des comptes qui est retenue. ASML publie en euros et cote en dollars ;
le rapprochement direct donnerait une divergence d'environ 16 %, qui n'est que
la parité EUR/USD. La capitalisation reste affichée dans la devise de
cotation, celle du cours qu'elle accompagne.

Les taux viennent de la Banque centrale européenne (30 devises). Une paire non
couverte — le dollar de Taïwan, par exemple — neutralise le pilier plutôt que
de supposer la parité : comparer une capitalisation en dollars à des comptes
en TWD représenterait un facteur trente, silencieusement.

### Capitalisation d'un certificat de dépôt

Le calcul naturel, actions en circulation × dernier cours, est faux pour un
ADR. Un certificat Toyota représente dix actions ordinaires, or SEC EDGAR
publie le nombre d'**ordinaires**. Le produit surestime la capitalisation d'un
facteur dix : Toyota ressortirait à 2 500 milliards de dollars au lieu de 250,
et l'écran le déclarerait massivement sur-évalué sans que rien ne le signale.

Le rapport ADR / action ordinaire n'est publié nulle part de façon
exploitable. La capitalisation est donc demandée à un fournisseur qui connaît
le titre coté ; la reconstitution n'intervient qu'à défaut, accompagnée d'un
avertissement. Le nombre d'actions servant à ramener la juste valeur à un prix
est lui-même déduit de la capitalisation et du cours, tous deux relatifs au
même titre.

### Deux taxonomies comptables

Les déposants américains publient en US-GAAP ; les émetteurs privés étrangers
publient le plus souvent en IFRS, avec des noms de concepts entièrement
différents — `ProfitLossFromOperatingActivities` au lieu de
`OperatingIncomeLoss`, `Borrowings` au lieu de `LongTermDebtNoncurrent`. Ne
connaître que l'US-GAAP privait le dashboard de SAP, TSMC, Shell, Unilever et
de la plupart des grandes capitalisations européennes et asiatiques cotées à
New York. Le moteur retient la taxonomie la mieux renseignée pour chaque
déposant.

### Sous-unités de cotation

Londres cote en pence et non en livres, Tel-Aviv en agorot. Le suffixe de
devise du fournisseur le signale (« GBp »), et l'ignorer diviserait la
capitalisation par cent.

---

## 6. Reconstitution du rendement total

Lorsque la source de cours ne réintègre pas les dividendes, le moteur les
reprend dans le fichier `companyfacts` de SEC EDGAR déjà téléchargé pour les
fondamentaux :

```
rendement total du mois t = (P_t + D_t) / P_{t-1} − 1
```

Correction mesurée sur le t-stat de l'alpha :

| Titre | Rendement reconstitué | *t* avant | *t* après | Écart |
|---|---|---|---|---|
| Altria | 8,1 % | −0,64 | **+0,30** | +0,94 |
| Verizon | 6,2 % | −1,59 | −0,85 | +0,74 |
| AT&T | 6,2 % | −1,15 | −0,53 | +0,62 |
| Coca-Cola | 3,0 % | −0,24 | +0,13 | +0,36 |
| Merck | 3,2 % | +0,22 | +0,54 | +0,33 |
| Apple | 0,5 % | +0,73 | +0,80 | +0,07 |

Altria change de signe : son alpha mesuré était négatif par pure omission des
dividendes.

### Le quatrième trimestre

Une société publie trois trimestres dans ses 10-Q puis l'exercice entier dans
son 10-K : le quatrième versement n'a donc pas de période de 90 jours propre.
La couverture plafonnait à 15 trimestres sur 20, soit un quart du rendement
perdu. Le résidu « annuel − somme des trois trimestres » le restitue, sous
réserve qu'il soit positif et du même ordre que les trois autres. La couverture
passe à 20 sur 20, et la correction de Verizon de +0,56 à +0,74.

### Trois garde-fous

**Divisions d'actions.** Les cours sont ajustés des divisions, les dividendes
par action déclarés à la SEC ne le sont pas : une division dans la fenêtre
décale le rapport D/P d'un facteur entier sur sa partie ancienne. Le rendement
de chaque période est donc ramené dans une bande de 0,4 à 2,5 fois sa médiane.
Un artefact de division en sort, une hausse ordinaire du dividende non.

**Couverture.** En dessous de huit trimestres dans la fenêtre, la
reconstitution est abandonnée et l'avertissement conservé — c'est le cas
d'ExxonMobil, qui ne publie que deux périodes trimestrielles, et de
Caterpillar.

**Devise.** Pour un certificat de dépôt, les dividendes sont libellés dans la
devise des comptes et le rapport au titre coté est inconnu : la reconstitution
est refusée plutôt que devinée.

### Ce que la reconstitution ne fait pas

EDGAR date un dividende par la fin de la période comptable où il est déclaré,
pas par sa date de détachement ; le décalage peut atteindre un trimestre. Sur
la moyenne de soixante mois que mesure l'alpha, c'est du second ordre, mais
cette série ne permet pas de dater un flux au mois près.

Le cours affiché et la capitalisation restent sur la base des cours : seuls la
régression et le momentum travaillent sur l'indice de rendement total, un
indice n'étant pas un prix de marché.

---

## 7. La zone d'achat

Les seuils du verdict portent sur un score sans dimension, or c'est un cours
que l'on regarde. Le dashboard traduit donc le seuil en prix : **à quel cours
ce titre basculerait-il en « sous-évaluée » ?**

Un seul pilier dépend du cours du jour. L'alpha mesure soixante mois de
performance passée, le momentum douze mois arrêtés il y a un mois : un prix
hypothétique aujourd'hui ne réécrit pas cette histoire. C'est donc la juste
valeur Modigliani-Miller, seule à confronter l'entreprise à son cours, qui
porte la variation, les deux autres piliers conservant leur contribution.

La résolution est itérative, non analytique : la juste valeur n'est pas tout à
fait indépendante du cours, puisque la dé-leviérisation de Hamada prend D/E en
valeur de marché et que le modèle de Merton fait entrer la capitalisation dans
la valeur de firme. Huit itérations suffisent largement à converger.

### Quand aucun cours ne suffit

Le pilier Modigliani-Miller pèse 0,30 et son score est borné à ±2 : il ne peut
apporter que 0,60 au composite, pour un seuil de verdict à 0,50. Si les deux
autres piliers contribuent ensemble moins de −0,10, aucune décote, si profonde
soit-elle, ne fait basculer le verdict — le pilier sature avant.

Le dashboard l'annonce alors comme tel plutôt que d'afficher un prix inventé.
C'est une information en soi : elle dit que la valorisation n'est pas ce qui
retient le modèle.

---

## 8. Diagnostic des sources

Le moteur enchaîne des fournisseurs de repli, ce qui le rend robuste et opaque
à la fois : l'utilisateur voit « prix : Nasdaq Data » sans savoir pourquoi
Yahoo a été écarté. Or quota atteint, ticker inconnu, réseau coupé et clé
absente produisent le même repli et appellent des réponses opposées.

`GET /api/diagnostics` interroge chaque source et rapporte son code de retour,
sa latence, le nombre de tentatives et ce qu'il faut en conclure. Le module ne
diagnostique rien de lui-même : il rend visible ce qui, sinon, se perd dans les
journaux du serveur.

Trois états, pas deux. Une clé facultative non configurée n'est pas une panne,
et l'afficher en rouge enverrait chercher un problème inexistant.

Le message d'échec d'une analyse suit le même principe. « Aucune donnée pour
MC.PA, vérifiez le ticker » envoie corriger une saisie correcte : les places
locales ne sont couvertes que par Yahoo et Financial Modeling Prep, et quand
le premier est au quota sans que le second soit configuré, le ticker n'y est
pour rien. Trois causes, trois messages : place locale sans fournisseur,
ticker qu'aucune source ne connaît, fournisseurs injoignables.

### Deux refus déterministes, pris pour des quotas

Yahoo répondait HTTP 429 à chaque requête, ce qui désigne un quota de débit.
La mesure a montré autre chose : la même requête, au même instant, passe ou
échoue selon la seule chaîne de **User-Agent**. Celle par défaut de nombreux
scripts — « Macintosh; Intel Mac OS X 10_15_7 … Chrome/124.0.0.0 » — est
refusée systématiquement ; une autre passe. Le code 429 décrivait donc un
filtrage, pas une saturation.

La conséquence était lourde : Yahoo est le seul fournisseur couvrant les
places locales et le seul à remonter au-delà de quelques années. Tout basculait
sur Nasdaq Data, limité aux cotations américaines et sans dividendes — d'où
la reconstitution décrite en section 6, qui reste utile mais n'est plus la
seule issue.

Un second refus, de même nature, touchait le jeton Yahoo : `getcrumb` renvoie
du texte brut et répondait 406 « Not Acceptable » à une session réclamant du
JSON. Sans jeton, les points d'entrée `quoteSummary` et `quote` restent
inaccessibles, et avec eux la raison sociale, le secteur et la capitalisation
des titres hors périmètre SEC.

La leçon vaut au-delà de ces deux cas : un code d'erreur HTTP nomme une
catégorie, pas une cause. Le diagnostic rapporte le code, mais c'est la mesure
comparative — faire varier un seul paramètre à la fois — qui a tranché.

### Ce que la mesure a écarté

Face au 429, la tentation était d'insister. La mesure l'a écarté avant que la
cause réelle ne soit connue : six tentatives espacées sur soixante secondes —
0, 2, 4, 8, 16 puis 30 — échouaient toutes, sur trois titres. Le nombre de
tentatives est donc resté à deux, et c'est heureux : le refus tenait au
User-Agent, et aucune patience n'en serait venue à bout.

Reste un vrai quota, lui, au-delà d'un certain volume de requêtes. Il se
traite par le cache — douze heures par défaut — et, le cas échéant, par une
clé Financial Modeling Prep.

---

## 9. Limites connues

- **Sociétés déficitaires.** Une perpétuité de flux négatifs n'a pas de sens :
  le pilier Modigliani-Miller est neutralisé lorsque l'EBIT sur douze mois est
  négatif, et son poids redistribué.
- **Valeurs de croissance.** Un modèle sans prime de croissance les sous-valorise
  structurellement. Un verdict « sur-évaluée » sur une telle valeur dit que le
  marché anticipe mieux que la perpétuité, pas nécessairement qu'il a tort.
- **Sociétés financières.** Le cadre Modigliani-Miller convient mal aux banques
  et aux assureurs, dont la dette est un intrant d'exploitation et non un choix
  de structure de capital.
- **Couverture géographique.** SEC EDGAR référence les déposants américains et
  les émetteurs privés étrangers déposant un 20-F. Une société sans lien avec
  la SEC — LVMH, Nestlé, la cotation locale de Toyota — n'a de fondamentaux
  que si une clé Financial Modeling Prep est fournie ; à défaut, le pilier
  Modigliani-Miller est neutralisé et son poids reporté sur les deux autres.
- **Dividendes.** C'est le trou de données qui compte le plus. Un cours non
  réajusté mesure un rendement en capital, inférieur au rendement total du
  montant du dividende, et le t-stat de l'alpha s'en trouve décalé :

  | Titre | Rendement | Décalage du *t* | *t* mesuré → corrigé |
  |---|---|---|---|
  | Alphabet | 0,4 % | +0,04 | 1,17 → 1,21 |
  | Johnson & Johnson | 3,0 % | +0,35 | 0,14 → 0,50 |
  | Verizon | 6,3 % | +0,67 | −1,59 → −0,93 |
  | Altria | 7,5 % | +0,81 | −0,64 → +0,16 |

  Le biais va toujours dans le même sens et croît avec le rendement : il
  pénalise systématiquement les valeurs de rendement, et peut inverser le
  signe de leur alpha. `PriceHistory.total_return` n'est donc vrai que
  lorsqu'une source réintègre démontrablement les dividendes — la série
  `adjClose` de Financial Modeling Prep, la série `adjclose` de Yahoo. Dès
  qu'un fournisseur retombe sur le cours brut, ou n'en dit rien, le drapeau
  passe à faux.

  Le moteur ne se contente alors pas d'avertir : il **reconstitue** le
  rendement total depuis les dividendes par action des comptes SEC EDGAR, que
  la même requête `companyfacts` a déjà rapportés — donc sans appel réseau
  supplémentaire. Voir la section 7.
- **Un seul titre à la fois.** Le dashboard ne reconstitue pas le classement
  cross-sectionnel de la stratégie ; il ne dit pas si un titre est plus
  attrayant qu'un autre, seulement s'il s'écarte de sa juste valeur théorique.
