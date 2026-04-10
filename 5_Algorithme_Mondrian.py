# =============================================================================
# Technique PPDP #5 : Algorithme de Mondrian pour le k-Anonymat
# =============================================================================
#
# Objectif : Implémenter une version pédagogique simplifiée de l'algorithme
#            de Mondrian pour atteindre le k-anonymat sur des quasi-
#            identifiants continus.
#
# Concepts clés :
#   - Mondrian (LeFevre et al., 2006) : Algorithme de partitionnement spatial
#               multidimensionnel "top-down glouton". Il divise récursivement
#               l'espace des quasi-identifiants pour créer des régions dont
#               chacune contient au moins k individus.
#
#   - Partitionnement récursif : À chaque étape, on choisit la dimension
#               (colonne QI) ayant la PLUS GRANDE PLAGE de valeurs (max - min).
#               On coupe en deux au niveau de la MÉDIANE de cette dimension.
#               La récursion s'arrête si une coupe produirait une sous-partition
#               avec moins de k enregistrements.
#
#   - Généralisation par intervalle : Tous les enregistrements d'une partition
#               finale reçoivent la valeur généralisée "[min-max]" pour chaque
#               quasi-identifiant, rendant les individus indiscernables.
#
#   - Paramètre k : Contrôle la taille minimale des partitions finales.
#               Un k élevé → meilleure confidentialité, moins d'utilité.
#               Un k faible → moins de protection, données plus précises.
#
# =============================================================================

import pandas as pd
import numpy as np

# ==============================================================================
# ÉTAPE 1 : Création du jeu de données original
# ==============================================================================

np.random.seed(99)

n = 20  # Nombre d'individus dans le jeu de données

donnees_brutes = {
    # Quasi-identifiant 1 : âge (variable continue)
    "Age": [25, 27, 28, 31, 33, 35, 36, 38, 40, 42,
            44, 46, 48, 50, 52, 55, 58, 61, 65, 70],

    # Quasi-identifiant 2 : salaire annuel brut en milliers d'euros
    "Salaire_kEUR": [22, 35, 28, 41, 55, 38, 60, 45, 72, 48,
                     80, 53, 67, 90, 58, 75, 95, 62, 110, 85],

    # Attribut sensible (non utilisé pour le partitionnement)
    "Maladie": ["Grippe", "Asthme", "Diabète", "Hypertension", "Grippe",
                "Asthme", "Cancer", "Diabète", "Hypertension", "Grippe",
                "Asthme", "Diabète", "Cancer", "Hypertension", "Grippe",
                "Asthme", "Diabète", "Cancer", "Hypertension", "Grippe"]
}

df_original = pd.DataFrame(donnees_brutes)

print("=" * 65)
print("  JEU DE DONNÉES ORIGINAL (Avant Mondrian)")
print("=" * 65)
print(df_original.to_string(index=True))
print()


# ==============================================================================
# ÉTAPE 2 : Algorithme de Mondrian — Partitionnement récursif
# ==============================================================================

def choisir_dimension_de_coupe(df, colonnes_qi):
    """
    Choisit la dimension (colonne QI) dont la plage de valeurs est maximale.
    C'est le critère "glouton" de Mondrian : couper là où les données
    sont les plus dispersées.

    Arguments:
        df          (pd.DataFrame) : La partition courante.
        colonnes_qi (list)         : Liste des quasi-identifiants numériques.

    Retourne:
        str : Le nom de la colonne à couper.
    """
    plages = {col: df[col].max() - df[col].min() for col in colonnes_qi}
    # Retourne la colonne avec la plage maximale
    return max(plages, key=plages.get)


def partitionner_mondrian(df, colonnes_qi, k):
    """
    Partitionne récursivement le DataFrame selon l'algorithme de Mondrian.

    Stratégie :
      1. Choisir la dimension avec la plus grande plage.
      2. Calculer la médiane de cette dimension.
      3. Diviser en deux sous-partitions : valeurs ≤ médiane et > médiane.
      4. Vérifier que chaque sous-partition a au moins k enregistrements.
         - Si oui  → récursion sur chaque sous-partition.
         - Si non  → arrêt, la partition courante devient une feuille.
      5. Retourner la liste de toutes les partitions finales (feuilles).

    Arguments:
        df          (pd.DataFrame) : La partition de données courante.
        colonnes_qi (list)         : Noms des colonnes quasi-identifiants.
        k           (int)          : Taille minimale de chaque partition.

    Retourne:
        list of pd.DataFrame : Liste des partitions finales.
    """
    # --- Cas de base : la partition ne peut pas être divisée ---
    # Une partition est indivisible si elle a moins de 2k enregistrements
    # (toute coupe produirait une moitié avec < k enregistrements)
    if len(df) < 2 * k:
        return [df]

    # --- Choix de la dimension de coupe ---
    dim_coupe = choisir_dimension_de_coupe(df, colonnes_qi)
    mediane   = df[dim_coupe].median()

    # --- Division en deux sous-partitions autour de la médiane ---
    partition_gauche = df[df[dim_coupe] <= mediane]
    partition_droite = df[df[dim_coupe] >  mediane]

    # --- Vérification que les deux sous-partitions respectent le seuil k ---
    if len(partition_gauche) < k or len(partition_droite) < k:
        # La coupe violerait k → on ne coupe pas et on retourne la partition entière
        return [df]

    # --- Récursion sur chaque sous-partition ---
    return (partitionner_mondrian(partition_gauche, colonnes_qi, k) +
            partitionner_mondrian(partition_droite, colonnes_qi, k))


# ==============================================================================
# ÉTAPE 3 : Généralisation des partitions finales
# ==============================================================================

def generaliser_partition(partition, colonnes_qi):
    """
    Remplace les valeurs précises de chaque QI dans une partition finale
    par l'intervalle "[min-max]" correspondant à cette partition.

    Cela rend tous les individus de la partition indiscernables
    sur la base de leurs quasi-identifiants.

    Arguments:
        partition   (pd.DataFrame) : Une partition finale (feuille de Mondrian).
        colonnes_qi (list)         : Noms des colonnes quasi-identifiants.

    Retourne:
        pd.DataFrame : La partition avec les QI généralisés.
    """
    partition_generalisee = partition.copy()
    for col in colonnes_qi:
        val_min = partition[col].min()
        val_max = partition[col].max()
        if val_min == val_max:
            # Valeur unique dans la partition → pas de généralisation nécessaire
            valeur_generalisee = str(val_min)
        else:
            valeur_generalisee = f"{val_min}-{val_max}"
        partition_generalisee[col] = valeur_generalisee
    return partition_generalisee


def appliquer_mondrian(df, colonnes_qi, k):
    """
    Applique l'algorithme de Mondrian complet sur un DataFrame.

    Arguments:
        df          (pd.DataFrame) : Le DataFrame original.
        colonnes_qi (list)         : Quasi-identifiants numériques.
        k           (int)          : Paramètre de k-anonymat.

    Retourne:
        pd.DataFrame : Le DataFrame anonymisé (k-anonyme selon Mondrian).
    """
    print(f"  [1/3] Partitionnement récursif de Mondrian (k = {k})...")
    partitions = partitionner_mondrian(df, colonnes_qi, k)
    print(f"         → {len(partitions)} partition(s) finale(s) créée(s).")

    print(f"  [2/3] Généralisation des quasi-identifiants dans chaque partition...")
    partitions_generalisees = [
        generaliser_partition(p, colonnes_qi) for p in partitions
    ]

    print(f"  [3/3] Reconstruction du jeu de données anonymisé...")
    df_anonymise = pd.concat(partitions_generalisees).sort_index()
    return df_anonymise, partitions


# ==============================================================================
# ÉTAPE 4 : Affichage détaillé des partitions
# ==============================================================================

def afficher_partitions(partitions, colonnes_qi):
    """
    Affiche chaque partition finale avec ses statistiques clés.
    """
    print(f"\n  Détail des {len(partitions)} partition(s) finale(s) :")
    print()
    for idx, partition in enumerate(partitions):
        print(f"  ┌─ Partition {idx + 1} ({len(partition)} individu(s)) ────────────────")
        for col in colonnes_qi:
            print(f"  │  {col}: [{partition[col].min()}, {partition[col].max()}]  "
                  f"(médiane = {partition[col].median():.1f})")
        print(f"  │  Maladies présentes : "
              f"{', '.join(partition['Maladie'].unique())}")
        print()


# ==============================================================================
# EXÉCUTION PRINCIPALE
# ==============================================================================

K             = 3                       # Seuil de k-anonymat
COLONNES_QI   = ["Age", "Salaire_kEUR"] # Quasi-identifiants numériques

print(f">>> Lancement de l'algorithme de Mondrian (k = {K})...")
print()

df_anonymise, partitions = appliquer_mondrian(df_original, COLONNES_QI, k=K)

afficher_partitions(partitions, COLONNES_QI)

print("=" * 65)
print("  JEU DE DONNÉES ANONYMISÉ PAR MONDRIAN")
print("=" * 65)
print(df_anonymise.to_string(index=True))
print()

# ==============================================================================
# ÉTAPE 5 : Vérification du k-anonymat atteint
# ==============================================================================

print("=" * 65)
print("  VÉRIFICATION DU k-ANONYMAT")
print("=" * 65)

groupes = df_anonymise.groupby(COLONNES_QI).size()
k_atteint = groupes.min()

print(f"\n  Groupes d'équivalence formés (par QI généralisés) :")
print()
print(groupes.reset_index(name="Nb_Individus").to_string(index=False))
print()
print(f"  ✅ Valeur de k atteinte : k = {k_atteint}")
print(f"     → Chaque individu est indiscernable d'au moins {k_atteint - 1} autre(s).")
print(f"     → L'algorithme de Mondrian garantit k ≥ {K} par construction.")
print("=" * 65)
