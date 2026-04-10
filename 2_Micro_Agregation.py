# =============================================================================
# Technique PPDP #2 : Micro-Agrégation
# =============================================================================
#
# Objectif : Protéger les valeurs numériques individuelles tout en
#            préservant les propriétés statistiques globales du jeu de données.
#
# Concepts clés :
#   - Micro-agrégation : Les enregistrements sont regroupés en petits
#                        clusters de taille >= k. Les valeurs numériques
#                        (quasi-identifiants continus) de chaque individu
#                        dans un cluster sont remplacées par la MOYENNE
#                        du groupe.
#
#   - Utilité statistique : La moyenne globale des données est préservée
#                           car on remplace chaque valeur individuelle par
#                           la moyenne de son groupe (la somme totale reste
#                           donc quasi inchangée).
#
#   - Approche : Tri des enregistrements par la valeur du quasi-identifiant,
#                puis regroupement en blocs contigus de taille k.
#
# =============================================================================

import pandas as pd
import numpy as np

# ==============================================================================
# ÉTAPE 1 : Création du jeu de données original
# ==============================================================================

# Graine aléatoire pour la reproductibilité des résultats
np.random.seed(42)

donnees_brutes = {
    "ID_Employe":   [f"EMP_{i:03d}" for i in range(1, 13)],

    # Quasi-identifiant continu : âge exact de chaque employé
    "Age":          [24, 27, 32, 35, 36, 41, 44, 45, 48, 53, 57, 62],

    # Quasi-identifiant continu : salaire annuel brut en euros
    "Salaire_Annuel": [
        28_500, 31_200, 42_000, 39_800, 55_000,
        61_500, 48_000, 52_300, 67_000, 71_200,
        58_900, 83_000
    ],

    # Attribut sensible : niveau de formation
    "Formation":    ["Licence", "Master", "Licence", "BTS", "Master",
                     "Doctorat", "Master", "Licence", "Doctorat", "Master",
                     "BTS", "Doctorat"]
}

df_original = pd.DataFrame(donnees_brutes)

print("=" * 70)
print("  JEU DE DONNÉES ORIGINAL (Avant micro-agrégation)")
print("=" * 70)
print(df_original.to_string(index=False))
print()


# ==============================================================================
# ÉTAPE 2 : Algorithme de micro-agrégation
# ==============================================================================

def micro_agreger_colonne(df, colonne, k=3):
    """
    Applique la micro-agrégation à une colonne numérique donnée.

    Algorithme :
      1. Trier les enregistrements selon la valeur de la colonne cible.
      2. Découper en groupes contigus de taille k.
      3. Remplacer chaque valeur individuelle par la MOYENNE de son groupe.
      4. Les valeurs sont arrondies à l'entier le plus proche.

    Arguments:
        df (pd.DataFrame) : Le DataFrame source.
        colonne (str)     : Nom de la colonne numérique à agréger.
        k (int)           : Taille minimale de chaque groupe (défaut = 3).

    Retourne:
        pd.Series : La colonne avec les valeurs agrégées.
    """
    # Tri par la valeur de la colonne (ordre croissant)
    df_trie = df.sort_values(by=colonne).copy()
    valeurs_agregees = df_trie[colonne].copy()
    n = len(df_trie)

    # Découpage en groupes de taille k
    for debut in range(0, n, k):
        fin = min(debut + k, n)
        indices_groupe = df_trie.index[debut:fin]

        # Calcul de la moyenne du groupe
        moyenne_groupe = df_trie.loc[indices_groupe, colonne].mean()

        # Remplacement des valeurs individuelles par la moyenne du groupe
        valeurs_agregees.loc[indices_groupe] = round(moyenne_groupe)

    # Réalignement sur l'ordre original du DataFrame d'entrée
    return valeurs_agregees.reindex(df.index)


def appliquer_micro_agregation(df, colonnes_a_agreger, k=3):
    """
    Applique la micro-agrégation sur plusieurs colonnes numériques.

    Arguments:
        df (pd.DataFrame)          : Le DataFrame source.
        colonnes_a_agreger (list)  : Liste des colonnes numériques à agréger.
        k (int)                    : Taille des groupes (défaut = 3).

    Retourne:
        pd.DataFrame : Le DataFrame avec les colonnes agrégées.
    """
    df_agregee = df.copy()

    for colonne in colonnes_a_agreger:
        print(f"  → Micro-agrégation sur '{colonne}' avec k = {k}")
        df_agregee[colonne] = micro_agreger_colonne(df_agregee, colonne, k=k)

    return df_agregee


# ==============================================================================
# ÉTAPE 3 : Application et affichage des résultats
# ==============================================================================

TAILLE_K = 3  # Taille des groupes pour la micro-agrégation
COLONNES_QI = ["Age", "Salaire_Annuel"]  # Quasi-identifiants à agréger

print(f">>> Lancement de la micro-agrégation (k = {TAILLE_K})...")
print()

df_agregee = appliquer_micro_agregation(df_original, COLONNES_QI, k=TAILLE_K)

print()
print("=" * 70)
print("  JEU DE DONNÉES APRÈS MICRO-AGRÉGATION")
print("=" * 70)
print(df_agregee.to_string(index=False))
print()

# ==============================================================================
# ÉTAPE 4 : Comparaison des statistiques pour mesurer l'utilité préservée
# ==============================================================================

print("=" * 70)
print("  COMPARAISON DES STATISTIQUES (Utilité des données)")
print("=" * 70)

for col in COLONNES_QI:
    moy_orig  = df_original[col].mean()
    moy_agr   = df_agregee[col].mean()
    std_orig  = df_original[col].std()
    std_agr   = df_agregee[col].std()

    print(f"\n  Colonne : '{col}'")
    print(f"  {'Métrique':<25} {'Original':>15} {'Agrégé':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Moyenne':.<25} {moy_orig:>15.2f} {moy_agr:>15.2f}")
    print(f"  {'Écart-type':.<25} {std_orig:>15.2f} {std_agr:>15.2f}")

print()
print("  ✅ La moyenne globale est préservée après micro-agrégation.")
print("     Les valeurs individuelles exactes ne sont plus accessibles.")
print("=" * 70)
