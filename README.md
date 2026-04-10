# 🔒 Privacy-Preserving Data Publishing (PPDP) — Exemples Python

Ce dépôt contient des implémentations Python pédagogiques des principales techniques de **Publication de Données Préservant la Vie Privée** (*Privacy-Preserving Data Publishing*, PPDP). Il accompagne un rapport "État de l'Art" universitaire sur le sujet.

---

## 📚 Contexte et Définitions

Avant de publier ou partager des données, il est essentiel de distinguer trois types d'attributs :

| Type d'attribut | Définition | Exemples |
|---|---|---|
| **Identifiants Explicites** | Attributs qui identifient directement et de façon unique un individu. | Nom, Prénom, Numéro de Sécurité Sociale, Adresse e-mail |
| **Quasi-Identifiants (QI)** | Attributs qui, combinés entre eux, peuvent permettre de ré-identifier un individu par recoupement avec d'autres sources. | Âge, Code Postal, Sexe, Profession |
| **Données Sensibles** | Attributs dont la divulgation pourrait nuire à l'individu (discrimination, atteinte à la vie privée). | Maladie, Salaire, Opinions politiques, Religion |

### Objectif du PPDP

Le PPDP vise à **trouver un équilibre optimal** entre deux impératifs contradictoires :
- **Confidentialité** : protéger les individus contre la ré-identification et la divulgation de leurs données sensibles.
- **Utilité** : conserver suffisamment d'information dans les données publiées pour qu'elles restent exploitables à des fins d'analyse ou de recherche.

---

## 🗂️ Structure du Dépôt

```
.
├── README.md
├── 1_Generalisation_Suppression.py   # k-Anonymat par généralisation et suppression
├── 2_Micro_Agregation.py             # Micro-agrégation pour les attributs numériques
├── 3_Ajout_Bruit.py                  # Confidentialité différentielle (bruit de Laplace/Gaussien)
├── 4_Slicing.py                      # Découpage (Slicing) pour briser les liens directs
└── 5_Algorithme_Mondrian.py          # Algorithme de Mondrian pour le k-anonymat
```

---

## ⚙️ Techniques Implémentées

### 1. Généralisation & Suppression (`k`-Anonymat)
La **suppression** retire les identifiants explicites. La **généralisation** remplace des valeurs précises par des intervalles ou des valeurs moins spécifiques (ex: `34 ans` → `[30-40[`) jusqu'à ce que chaque enregistrement soit indiscernable d'au moins `k-1` autres.

### 2. Micro-Agrégation
Les enregistrements sont regroupés en petits clusters. Les valeurs numériques individuelles (ex: salaires) sont remplacées par la **moyenne du groupe**, préservant les propriétés statistiques globales tout en masquant les valeurs individuelles.

### 3. Ajout de Bruit (Confidentialité Différentielle)
Un **bruit aléatoire** (distribution de Laplace ou Gaussienne) est ajouté aux valeurs sensibles. Le paramètre `ε` (epsilon) contrôle le compromis entre confidentialité et précision. Cette technique offre des garanties mathématiques formelles.

### 4. Slicing (Découpage)
Le jeu de données est divisé en **partitions (buckets)**. Au sein de chaque partition, la colonne des attributs sensibles est **permutée aléatoirement**, brisant le lien direct entre un individu et sa valeur sensible, tout en préservant la distribution globale.

### 5. Algorithme de Mondrian
Algorithme **top-down glouton** de partitionnement spatial. Il divise récursivement l'espace des quasi-identifiants le long de la médiane de la dimension la plus étendue, jusqu'à ce que chaque partition respecte le seuil `k`. Les valeurs sont ensuite généralisées par leur intervalle `[min-max]`.

---

## 🚀 Prérequis et Installation

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install pandas numpy matplotlib
```

## ▶️ Exécution

```bash
python 1_Generalisation_Suppression.py
python 2_Micro_Agregation.py
python 3_Ajout_Bruit.py
python 4_Slicing.py
python 5_Algorithme_Mondrian.py
```

---

## 📖 Références

- Sweeney, L. (2002). *k-anonymity: A model for protecting privacy.* International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems.
- Dwork, C. (2006). *Differential Privacy.* ICALP.
- LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). *Mondrian multidimensional k-anonymity.* ICDE.
- Li, T., Li, N., Zhang, J., & Molloy, I. (2010). *Slicing: A new approach for privacy preserving data publishing.* TKDE.

---

*Dépôt créé dans le cadre d'un projet universitaire sur le PPDP.*
