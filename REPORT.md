# La quantification déterministe crée un biais d'agrégation que le moyennage ne peut pas éliminer — la stochastique non

**Question de recherche :** quand plusieurs répliques honnêtes compressent la même activation avant de l'envoyer à un serveur, la quantification *déterministe* introduit-elle un biais d'agrégation qui persiste quel que soit le nombre de répliques moyennées, alors que la quantification *stochastique* non biaisée n'en introduit (quasiment) pas ?

**Réponse courte : oui, et c'est démontré numériquement, pas juste supposé.**

## Méthode (en bref)

- **Modèle :** petit CNN entraîné sur CIFAR-10 (86.6 % d'accuracy propre), coupé après le 2ᵉ bloc convolutif → activation `a` de forme 64×8×8.
- **Compresseurs :** quantification uniforme déterministe (arrondi au plus proche) vs stochastique non biaisée (arrondi aléatoire, `E[Q_s(a)] = a` par construction), sur exactement la même grille (mêmes bornes, même nombre de niveaux) pour 8, 6, 4, 3, 2 bits.
- **Répliques :** R = 1, 2, 4, 8, 16, 32, chacune avec son propre générateur aléatoire indépendant pour le cas stochastique.
- **Décomposition biais/variance :** pour chaque (bits, R, méthode), on calcule séparément le **biais²** et la **variance** de la reconstruction moyennée `ā_R`, sur la même échelle que la MSE, puis on vérifie numériquement l'identité manuel `MSE = biais² + variance`.
- Code source et CSV bruts : [github.com/Ksentissi/split-quant-experiment](https://github.com/Ksentissi/split-quant-experiment).

## Preuve principale : la décomposition biais² / variance

**Vérification de l'identité `MSE = biais² + variance` :** calculée sur 30 configurations (5 taux de bits × 6 valeurs de R), l'écart maximal entre `biais² + variance` et la MSE réellement mesurée est de **4.16 × 10⁻⁹** — c'est-à-dire nul aux erreurs d'arrondi flottant près. La décomposition n'est donc pas une approximation, c'est une identité vérifiée sur les données réelles.

**Résultat central (exemple à 2 bits — voir `results/plot6_bias_variance_decomposition_bits{2,3,4,6,8}.png` pour tous les taux de bits) :**

| R | Biais² déterministe | Variance déterministe | Biais² stochastique | Variance stochastique |
|---|---|---|---|---|
| 1  | 0.1001 | **0.0 (exactement)** | 0.0200 | 0.1793 |
| 2  | 0.1001 | **0.0 (exactement)** | 0.0100 | 0.0897 |
| 4  | 0.1001 | **0.0 (exactement)** | 0.0050 | 0.0448 |
| 8  | 0.1001 | **0.0 (exactement)** | 0.0025 | 0.0224 |
| 16 | 0.1001 | **0.0 (exactement)** | 0.0012 | 0.0112 |
| 32 | 0.1001 | **0.0 (exactement)** | 0.0006 | 0.0056 |

Deux faits, mesurés et non supposés :

1. **La variance entre répliques déterministes est exactement 0.000 × 10⁰ à tous les taux de bits testés** (vérifié en générant 32 appels indépendants de `Q_d(a)` et en mesurant leur variance empirique — voir logs `[grid] ... variance-across-replicas ~ 0.000e+00`). C'est logique : `Q_d` est une fonction déterministe de `a`, il n'y a littéralement aucun hasard à moyenner. Donc `ā_R = Q_d(a)` pour **tout** R, et son erreur totale (`MSE = biais² + 0`) est **strictement constante en R** : 0.1001 à R=1 et encore 0.1001 à R=32, au chiffre près.
2. **Pour la quantification stochastique, l'erreur est presque entièrement de la variance, pas du biais** — et cette variance diminue en 1/R (moyennage). Le petit biais² résiduel mesuré (0.02 à R=1, tendant vers 0 avec R) n'est pas un vrai biais : `E[Q_s(a)] = a` est une identité algébrique exacte (voir `compressors.py`, preuve dans les commentaires), donc le biais théorique est nul à *tout* R. Ce qu'on mesure ici est simplement le bruit d'estimation Monte-Carlo dû au fait qu'on n'utilise que 10 seeds pour estimer une espérance — et ce bruit résiduel diminue lui aussi avec R, ce qui **confirme** la convergence vers un biais nul plutôt que de la contredire.

**Conséquence directe :** parce que le déterministe est 100 % biais (non réductible) et le stochastique est ~100 % variance (réductible par moyennage), l'écart entre les deux méthodes ne peut que se creuser en faveur du stochastique quand R augmente — ce qui est exactement ce qu'on observe (graphique `plot6_bias_variance_decomposition_bits2.png` : la droite bleue du déterministe est parfaitement horizontale, les courbes stochastiques (biais², variance, MSE totale) chutent toutes en ligne droite sur l'échelle log-log).

## Conséquences mesurées sur l'erreur et l'accuracy

| Bits | R | MSE déterministe | MSE stochastique | Accuracy déterministe | Accuracy stochastique |
|---|---|---|---|---|---|
| 2 | 1  | 0.1001 | 0.1992 | 0.842 | 0.829 ± 0.008 |
| 2 | 32 | 0.1001 | **0.0062** | 0.842 | **0.856** ± 0.003 |
| 4 | 1  | 0.0040 | 0.0079 | 0.852 | 0.857 ± 0.003 |
| 4 | 32 | 0.0040 | **0.0002** | 0.852 | **0.857** ± 0.001 |
| 8 | 1  | 0.00001 | 0.00003 | 0.856 | 0.857 |
| 8 | 32 | 0.00001 | 0.00000 | 0.856 | 0.857 |

(table complète avec biais/variance : `results/summary_table.csv` ; décomposition brute : `results/bias_variance.csv`)

- Comme le biais déterministe ne bouge jamais, sa MSE reste bloquée à 0.1001 (à 2 bits) quel que soit R.
- La MSE stochastique, elle, part plus haut (bruit sans moyennage) mais s'effondre avec R puisqu'elle n'a (presque) que de la variance à éliminer : **16× meilleure que le déterministe à R=32** (2 bits).
- Cet avantage se traduit en accuracy réelle : +1.4 point à 2 bits, +0.5 point à 4 bits ; négligeable à 8-6 bits car la compression y est déjà assez fine pour que l'erreur (de l'une ou l'autre méthode) n'affecte presque pas les prédictions.

## Limites

- Une seule couche de coupure testée (8×8×64) sur un seul petit CNN — pas généralisé à d'autres architectures/profondeurs.
- Grille répliques × seeds calculée sur un sous-échantillon fixe de 1000 images (pas les 10 000), pour des raisons de coût de calcul.
- Le « biais² » stochastique rapporté est une estimation Monte-Carlo sur seulement 10 seeds (pas le vrai biais théorique, qui est exactement nul par construction) — voir discussion ci-dessus.

## Suite naturelle du projet

Le déterministe n'a pas de variance à exploiter : ses répliques sont identiques, donc un attaquant Byzantin ne peut pas se cacher derrière un « désaccord légitime » puisqu'il n'y en a aucun. Le stochastique, en revanche, introduit une divergence honnête entre répliques (mesurée dans `replica_grid_raw.csv`, colonne `divergence`) qui croît avec la compression — c'est exactement la marge de tolérance qu'un consensus robuste devra accepter entre répliques honnêtes, et donc le budget de furtivité potentiel d'un attaquant Byzantin. C'est l'objet de la prochaine étape (non traitée ici).
