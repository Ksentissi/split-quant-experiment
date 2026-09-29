# Quantification stochastique vs déterministe en inférence distribuée répliquée — Résultats

**Question de recherche :** quand plusieurs répliques honnêtes traitent la même entrée et compressent leur activation avant de l'envoyer à un serveur, moyenner des quantifications *stochastiques* indépendantes apporte-t-il un avantage réel sur une quantification *déterministe*, à débit de bits identique ?

## Méthode (en bref)

- **Modèle :** petit CNN entraîné sur CIFAR-10 (86.6 % d'accuracy propre), coupé après le 2ᵉ bloc convolutif → activation `a` de forme 64×8×8.
- **Compresseurs :** quantification uniforme déterministe (arrondi au plus proche) vs stochastique non biaisée (arrondi aléatoire), sur exactement la même grille (mêmes bornes, même nombre de niveaux) pour 8, 6, 4, 3, 2 bits.
- **Répliques :** R = 1, 2, 4, 8, 16, 32, chacune avec son propre générateur aléatoire indépendant pour le cas stochastique.
- **Statistique :** chaque point stochastique est répété sur 10 seeds indépendantes (moyenne ± écart-type).
- **Données :** accuracy « simple » (R=1) mesurée sur les 10 000 images de test ; la grille complète (bits × R × seeds) sur un sous-échantillon fixe de 1000 images (coût de calcul de la grille complète trop élevé sur 10 000 images × 10 seeds × 32 répliques).
- Code source et CSV bruts : [github.com/Ksentissi/split-quant-experiment](https://github.com/Ksentissi/split-quant-experiment).

## Résultat principal

| Bits | R | MSE déterministe | MSE stochastique | Accuracy déterministe | Accuracy stochastique | Divergence honnête entre répliques |
|---|---|---|---|---|---|---|
| 2 | 1  | 0.1001 | 0.1992 ± 0.0001 | 0.842 | 0.829 ± 0.008 | 0.00 |
| 2 | 2  | 0.1001 | 0.0996 ± 0.0001 | 0.842 | 0.845 ± 0.006 | 39.9 |
| 2 | 32 | 0.1001 | **0.0062** ± 0.0000 | 0.842 | **0.856** ± 0.003 | 39.8 |
| 4 | 1  | 0.0040 | 0.0079 ± 0.0000 | 0.852 | 0.857 ± 0.003 | 0.00 |
| 4 | 32 | 0.0040 | **0.0002** ± 0.0000 | 0.852 | **0.857** ± 0.001 | 7.9 |
| 8 | 1  | 0.00001 | 0.00003 | 0.856 | 0.857 | 0.00 |
| 8 | 32 | 0.00001 | 0.00000 | 0.856 | 0.857 | 0.47 |

(table complète : `results/summary_table.csv`)

**Graphiques clés** (dossier `results/`) :
- `plot5_variance_loglog.png` — la variance de la reconstruction stochastique décroît en **exactement 1/R** (pente parfaite en log-log, confirmée pour tous les taux de bits).
- `plot1_mse_vs_replicas_bits2.png` — à 2 bits, la MSE stochastique croise et passe **sous** la MSE déterministe (constante) dès R=2, puis devient 16× plus petite à R=32.
- `plot2_accuracy_vs_replicas_bits2.png` — même croisement pour l'accuracy.
- `plot4_accuracy_vs_bits.png` — sans moyennage (R=1), stochastique et déterministe sont quasi équivalents à haut débit (8-6 bits) et stochastique est nettement **pire** à 2 bits.

## Réponse à la question de recherche

**Oui, un avantage réel apparaît — mais seulement grâce au moyennage, jamais avec une seule réplique stochastique.**

1. **Sans moyennage (R=1), le stochastique n'aide jamais sur la reconstruction** : sa MSE est systématiquement 2× à 2× pire que le déterministe, à tous les taux de bits. C'est attendu : une seule quantification stochastique ajoute du bruit sans bénéficier encore du moyennage.
2. **Avec moyennage, l'avantage apparaît dès R=2** et grandit avec R, conformément à la théorie (variance ∝ 1/R, vérifiée empiriquement). Il est **d'autant plus grand que la compression est agressive** :
   - À 2 bits : le stochastique+moyennage passe de 2× pire (R=1) à **16× meilleur** (R=32) en MSE, et de -1.3 point à **+1.4 point** d'accuracy par rapport au déterministe.
   - À 4-3 bits : gain plus modeste mais net (+0.5 à +1 point d'accuracy à R=32).
   - À 8-6 bits : l'avantage est négligeable — la compression est déjà assez fine pour que l'erreur de quantification n'affecte quasiment pas l'accuracy, avec ou sans moyennage.
3. **Coût de cet avantage** : la divergence honnête entre répliques (le désaccord légitime entre copies stochastiques de la même activation) croît fortement quand le débit baisse (≈0.47 à 8 bits vs ≈40 à 2 bits, sur l'échelle de cette expérience). C'est exactement le signal qui définira le budget de furtivité disponible pour un attaquant Byzantin dans la prochaine étape : plus on comprime agressivement pour gagner l'avantage stochastique, plus il y a de « bruit légitime » dans lequel un attaquant pourrait se dissimuler.

**Observation secondaire (à creuser) :** à 4 et 3 bits, l'accuracy stochastique est déjà légèrement meilleure que le déterministe *dès R=1*, alors que sa MSE est pire — le réseau semble plus tolérant au bruit non structuré (dithering) qu'à l'erreur systématique du déterministe. Cet effet s'inverse à 2 bits (le bruit devient trop grand). Un seul modèle/couche de coupure a été testé, donc cette observation n'est pas encore généralisable.

## Limites

- Une seule couche de coupure testée (8×8×64) sur un seul petit CNN — pas généralisé à d'autres architectures/profondeurs.
- Grille répliques × seeds calculée sur un sous-échantillon fixe de 1000 images (pas les 10 000), pour des raisons de coût de calcul.
- Range de clipping calculée par image (min/max global), pas par canal — une quantification par canal réduirait sans doute l'erreur pour les deux méthodes.

## Suite naturelle du projet

La divergence honnête mesurée ici (colonne « Divergence honnête ») définit la marge de manœuvre qu'un consensus robuste devra tolérer entre répliques honnêtes. Cette marge de tolérance légitime est exactement ce qu'un attaquant Byzantin pourrait exploiter comme budget de furtivité — c'est l'objet de la prochaine étape (non traitée ici).
