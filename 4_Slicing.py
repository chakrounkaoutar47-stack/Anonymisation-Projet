# =============================================================================
# Technique PPDP #4 : Slicing (Découpage)
# =============================================================================
#
# Objectif : Briser le lien direct entre un individu et son attribut sensible
#            en permutant les valeurs sensibles au sein de partitions (buckets),
#            tout en préservant la distribution statistique globale.
#
# Concepts clés :
#   - Slicing : Divise le jeu de données en colonnes (quasi-identifiants
#               et attributs sensibles) ET en lignes (partitions / buckets).
#
#   - Permutation intra-partition : Au sein de chaque bucket, la colonne
#               des attributs sensibles est mélangée aléatoirement.
#               → Un observateur ne peut plus associer un individu précis
#                 à sa valeur sensible (car plusieurs individus partagent
#                 les mêmes quasi-identifiants dans le bucket).
#
#   - Préservation de l'utilité : La distribution globale des maladies
#               est maintenue car on ne supprime ni ne modifie les valeurs,
#               on les redistribue seulement au sein des partitions.
#
#   - Avantage sur le k-anonymat : Le slicing gère mieux les attributs
#               de haute dimensionnalité car il ne nécessite pas de
#               généralisation des quasi-identifiants.
#
# =============================================================================

import pandas as pd
import numpy as np

# ==============================================================================
# ÉTAPE 1 : Création du jeu de données original
# ==============================================================================

np.random.seed(7)

donnees_brutes = {
    # Quasi-identifiants (attributs permettant potentiellement la ré-identification)
    "Age":    [22, 24, 25, 28, 30, 31, 35, 36, 40, 42,
               44, 45, 50, 52, 55, 58, 60, 62, 65, 68],

    "Sexe":   ["F", "H", "F", "H", "F", "H", "F", "H", "F", "H",
               "F", "H", "F", "H", "F", "H", "F", "H", "F", "H"],

    "Region": ["Nord", "Sud", "Nord", "Est", "Ouest", "Sud", "Nord", "Est",
               "Ouest", "Nord", "Sud", "Est", "Ouest", "Nord", "Sud", "Est",
               "Ouest", "Nord", "Sud", "Est"],

    # Attribut sensible : la maladie diagnostiquée
    "Maladie": ["Grippe", "Diabète", "Asthme", "Hypertension", "Grippe",
                "Cancer", "Diabète", "Grippe", "Asthme", "Diabète",
                "Hypertension", "Grippe", "Cancer", "Asthme", "Diabète",
                "Hypertension", "Grippe", "Cancer", "Asthme", "Diabète"]
}

df_original = pd.DataFrame(donnees_brutes)

print("=" * 70)
print("  JEU DE DONNÉES ORIGINAL (Avant slicing)")
print("=" * 70)
print(df_original.to_string(index=True))
print()
print(f"  Distribution originale de 'Maladie' :")
print(df_original["Maladie"].value_counts().to_string())
print()


# ==============================================================================
# ÉTAPE 2 : Découpage en partitions (buckets)
# ==============================================================================

def creer_partitions(df, nb_buckets):
    """
    Divise le DataFrame en partitions de taille approximativement égale.
    Le DataFrame est d'abord trié par Age pour que les individus similaires
    se retrouvent dans la même partition.

    Arguments:
        df (pd.DataFrame) : Le DataFrame à partitionner.
        nb_buckets (int)  : Nombre de partitions à créer.

    Retourne:
        list of pd.DataFrame : Liste des DataFrames partitionnés.
    """
    # Tri par Age pour regrouper des individus aux quasi-identifiants proches
    df_trie = df.sort_values("Age").reset_index(drop=True)
    taille_bucket = len(df_trie) // nb_buckets

    partitions = []
    for i in range(nb_buckets):
        debut = i * taille_bucket
        # La dernière partition prend tous les enregistrements restants
        fin = (i + 1) * taille_bucket if i < nb_buckets - 1 else len(df_trie)
        partitions.append(df_trie.iloc[debut:fin].copy())

    return partitions


# ==============================================================================
# ÉTAPE 3 : Permutation de l'attribut sensible au sein de chaque partition
# ==============================================================================

def permuter_attribut_sensible(partition, colonne_sensible):
    """
    Mélange aléatoirement les valeurs de l'attribut sensible UNIQUEMENT
    au sein d'une partition. Les quasi-identifiants restent inchangés.

    Cette opération brise le lien direct entre un individu et sa valeur
    sensible, tout en préservant la distribution de l'attribut dans
    la partition.

    Arguments:
        partition        (pd.DataFrame) : Une partition du jeu de données.
        colonne_sensible (str)          : Nom de la colonne à permuter.

    Retourne:
        pd.DataFrame : La partition avec la colonne sensible permutée.
    """
    partition_permutee = partition.copy()
    # np.random.permutation mélange les valeurs sans remise
    valeurs_melangees = np.random.permutation(partition[colonne_sensible].values)
    partition_permutee[colonne_sensible] = valeurs_melangees
    return partition_permutee


def appliquer_slicing(df, colonne_sensible, nb_buckets=4):
    """
    Applique l'algorithme de slicing complet :
      1. Partition du jeu de données en buckets.
      2. Permutation de l'attribut sensible dans chaque bucket.
      3. Reconstruction du jeu de données anonymisé.

    Arguments:
        df               (pd.DataFrame) : Le DataFrame original.
        colonne_sensible (str)          : Attribut sensible à protéger.
        nb_buckets       (int)          : Nombre de partitions (défaut = 4).

    Retourne:
        pd.DataFrame : Le DataFrame anonymisé par slicing.
    """
    print(f"  [1/3] Partitionnement en {nb_buckets} buckets (tri par 'Age')...")
    partitions = creer_partitions(df, nb_buckets)

    print(f"  [2/3] Permutation de '{colonne_sensible}' dans chaque bucket...")
    partitions_permutees = []
    for idx, partition in enumerate(partitions):
        ages_min = partition["Age"].min()
        ages_max = partition["Age"].max()
        print(f"         Bucket {idx + 1} : {len(partition)} individus "
              f"(Age : {ages_min}–{ages_max})")
        partitions_permutees.append(
            permuter_attribut_sensible(partition, colonne_sensible)
        )

    print(f"  [3/3] Reconstruction du jeu de données anonymisé...")
    df_anonymise = pd.concat(partitions_permutees).reset_index(drop=True)
    return df_anonymise


# ==============================================================================
# ÉTAPE 4 : Visualisation du contenu de chaque bucket
# ==============================================================================

def afficher_buckets(df, colonne_sensible, nb_buckets):
    """
    Affiche le contenu de chaque partition avant permutation
    pour illustrer comment les individus sont regroupés.
    """
    df_trie = df.sort_values("Age").reset_index(drop=True)
    taille_bucket = len(df_trie) // nb_buckets

    print("  Contenu des buckets (avant permutation) :")
    print()
    for i in range(nb_buckets):
        debut = i * taille_bucket
        fin = (i + 1) * taille_bucket if i < nb_buckets - 1 else len(df_trie)
        bucket = df_trie.iloc[debut:fin]
        print(f"  ┌─ Bucket {i + 1} (indices {debut}–{fin - 1}) ─────────────────")
        print(bucket[["Age", "Sexe", "Region", colonne_sensible]].to_string(index=True))
        print()


# ==============================================================================
# EXÉCUTION PRINCIPALE
# ==============================================================================

NB_BUCKETS        = 4
COLONNE_SENSIBLE  = "Maladie"

print(">>> Visualisation des buckets avant permutation :")
print()
afficher_buckets(df_original, COLONNE_SENSIBLE, NB_BUCKETS)

print(">>> Lancement du slicing...")
print()
df_sliced = appliquer_slicing(df_original, COLONNE_SENSIBLE, nb_buckets=NB_BUCKETS)

print()
print("=" * 70)
print("  JEU DE DONNÉES APRÈS SLICING (Attribut sensible permuté)")
print("=" * 70)
print(df_sliced.to_string(index=True))
print()

# ==============================================================================
# ÉTAPE 5 : Vérification de la préservation de la distribution
# ==============================================================================

print("=" * 70)
print("  VÉRIFICATION : Préservation de la distribution de 'Maladie'")
print("=" * 70)

dist_orig   = df_original["Maladie"].value_counts().sort_index()
dist_sliced = df_sliced["Maladie"].value_counts().sort_index()

comparaison = pd.DataFrame({
    "Distribution Originale":  dist_orig,
    "Distribution Après Slicing": dist_sliced
}).fillna(0).astype(int)

print()
print(comparaison.to_string())
print()
print("  ✅ La distribution globale de 'Maladie' est identique avant et après.")
print("     En revanche, le lien individu↔maladie a été brisé au sein de")
print(f"     chaque bucket ({NB_BUCKETS} partitions de ~{len(df_original)//NB_BUCKETS} individus).")
print("=" * 70)
