# =============================================================================
# Technique PPDP #1 : Généralisation et Suppression pour le k-Anonymat
# =============================================================================
#
# Objectif : Transformer un jeu de données brut en un jeu de données k-anonyme.
#
# Concepts clés :
#   - Suppression      : Retirer complètement les identifiants explicites
#                        (ex: Nom) qui identifient directement un individu.
#   - Généralisation   : Remplacer une valeur précise par une valeur moins
#                        spécifique appartenant à un domaine plus large
#                        (ex: Age 34 → "30-40", Code Postal 75013 → "750**").
#   - k-Anonymat       : Propriété garantissant que chaque enregistrement est
#                        indiscernable d'au moins k-1 autres enregistrements
#                        sur la base des quasi-identifiants.
#
# =============================================================================

import pandas as pd

# ==============================================================================
# ÉTAPE 1 : Création du jeu de données original (données brutes)
# ==============================================================================

donnees_brutes = {
    # Identifiant explicite : identifie directement l'individu
    "Nom":          ["Alice Martin", "Bob Dupont", "Clara Simon",
                     "David Leroy",  "Emma Petit",  "Félix Morin",
                     "Grace Blanc",  "Hugo Noël",   "Iris Faure",
                     "Jules Garnier"],

    # Quasi-identifiants : peuvent ré-identifier par combinaison
    "Age":          [23, 45, 36, 52, 29, 41, 31, 67, 25, 58],
    "Code_Postal":  ["75013", "69002", "13001", "75015", "69008",
                     "13005", "75020", "69003", "13008", "75007"],

    # Attribut sensible : ne doit pas être lié directement à un individu
    "Maladie":      ["Diabète", "Hypertension", "Asthme", "Diabète",
                     "Grippe", "Asthme", "Hypertension", "Diabète",
                     "Grippe", "Cancer"]
}

df_original = pd.DataFrame(donnees_brutes)

print("=" * 65)
print("  JEU DE DONNÉES ORIGINAL (Avant anonymisation)")
print("=" * 65)
print(df_original.to_string(index=True))
print()


# ==============================================================================
# ÉTAPE 2 : Suppression des identifiants explicites
# ==============================================================================

def supprimer_identifiants(df, colonnes_a_supprimer):
    """
    Supprime les colonnes correspondant aux identifiants explicites.

    Arguments:
        df (pd.DataFrame)          : Le DataFrame source.
        colonnes_a_supprimer (list): Liste des noms de colonnes à supprimer.

    Retourne:
        pd.DataFrame : DataFrame sans les identifiants explicites.
    """
    return df.drop(columns=colonnes_a_supprimer)


# ==============================================================================
# ÉTAPE 3 : Généralisation des quasi-identifiants
# ==============================================================================

def generaliser_age(age, taille_intervalle=10):
    """
    Généralise un âge précis en un intervalle de classe.

    Exemple : 34 → "[30-40["
    """
    borne_basse = (age // taille_intervalle) * taille_intervalle
    borne_haute = borne_basse + taille_intervalle
    return f"[{borne_basse}-{borne_haute}["


def masquer_code_postal(code_postal, nb_chiffres_masques=2):
    """
    Masque les derniers chiffres d'un code postal par des astérisques.

    Exemple : "75013" → "750**" (masquage des 2 derniers chiffres)
    """
    # Conserve les premiers chiffres et remplace les derniers par '*'
    nb_a_conserver = len(code_postal) - nb_chiffres_masques
    return code_postal[:nb_a_conserver] + "*" * nb_chiffres_masques


# ==============================================================================
# ÉTAPE 4 : Application complète du pipeline d'anonymisation
# ==============================================================================

def anonymiser_k_anonymat(df):
    """
    Applique la suppression et la généralisation pour produire
    un jeu de données respectant le k-anonymat.

    Arguments:
        df (pd.DataFrame): Le DataFrame original non anonymisé.

    Retourne:
        pd.DataFrame: Le DataFrame anonymisé.
    """
    # Copie profonde pour ne pas modifier le DataFrame original
    df_anonymise = df.copy()

    # --- Suppression de l'identifiant explicite ---
    print("  [1/2] Suppression de l'identifiant explicite : 'Nom'")
    df_anonymise = supprimer_identifiants(df_anonymise, colonnes_a_supprimer=["Nom"])

    # --- Généralisation du quasi-identifiant 'Age' ---
    print("  [2/2] Généralisation des quasi-identifiants :")
    print("        → 'Age' : valeur exacte → intervalle de 10 ans")
    df_anonymise["Age"] = df_anonymise["Age"].apply(generaliser_age)

    # --- Généralisation du quasi-identifiant 'Code_Postal' ---
    print("        → 'Code_Postal' : masquage des 2 derniers chiffres")
    df_anonymise["Code_Postal"] = df_anonymise["Code_Postal"].apply(
        masquer_code_postal, nb_chiffres_masques=2
    )

    return df_anonymise


# ==============================================================================
# ÉTAPE 5 : Vérification du k-anonymat
# ==============================================================================

def verifier_k_anonymat(df, quasi_identifiants):
    """
    Vérifie la valeur de k pour un jeu de données anonymisé.
    Retourne la taille minimale des groupes formés par les quasi-identifiants,
    ce qui correspond à la valeur de k atteinte.

    Arguments:
        df (pd.DataFrame)         : Le DataFrame anonymisé.
        quasi_identifiants (list) : Les colonnes formant les quasi-identifiants.

    Retourne:
        int : La valeur de k (taille du plus petit groupe d'équivalence).
    """
    # Compte combien de fois chaque combinaison de QI apparaît
    groupes = df.groupby(quasi_identifiants).size()
    k_atteint = groupes.min()
    return k_atteint, groupes


# ==============================================================================
# EXÉCUTION PRINCIPALE
# ==============================================================================

print(">>> Lancement du pipeline d'anonymisation...")
print()

df_anonymise = anonymiser_k_anonymat(df_original)

print()
print("=" * 65)
print("  JEU DE DONNÉES ANONYMISÉ (Après suppression + généralisation)")
print("=" * 65)
print(df_anonymise.to_string(index=True))
print()

# --- Vérification de la propriété de k-anonymat ---
quasi_ids = ["Age", "Code_Postal"]
k_atteint, groupes_equivalence = verifier_k_anonymat(df_anonymise, quasi_ids)

print("=" * 65)
print("  ANALYSE DU k-ANONYMAT")
print("=" * 65)
print(f"\n  Quasi-identifiants utilisés : {quasi_ids}")
print(f"\n  Distribution des groupes d'équivalence :")
print()
print(groupes_equivalence.reset_index(name="Nb_Individus").to_string(index=False))
print()
print(f"  ✅ Valeur de k atteinte : k = {k_atteint}")
print(f"     → Chaque individu est indiscernable d'au moins {k_atteint - 1} autre(s).")
print("=" * 65)
