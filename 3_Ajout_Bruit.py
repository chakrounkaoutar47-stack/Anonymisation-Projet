# =============================================================================
# Technique PPDP #3 : Ajout de Bruit — Confidentialité Différentielle
# =============================================================================
#
# Objectif : Démontrer comment l'ajout de bruit aléatoire permet de protéger
#            les valeurs individuelles tout en conservant la distribution
#            statistique globale du jeu de données.
#
# Concepts clés :
#   - Confidentialité Différentielle (ε-Differential Privacy) :
#       Garantie mathématique formelle. Un algorithme M est ε-différentiel
#       si, pour deux jeux de données D et D' ne différant que d'un seul
#       enregistrement :
#           P[M(D) ∈ S] ≤ e^ε * P[M(D') ∈ S]
#       Un ε faible → forte protection, faible utilité.
#       Un ε grand  → faible protection, forte utilité.
#
#   - Mécanisme de Laplace :
#       Bruit calibré sur la SENSIBILITÉ GLOBALE Δf de la requête et ε.
#       Bruit ~ Laplace(0, Δf/ε)
#       Adapté aux données numériques continues.
#
#   - Mécanisme Gaussien :
#       Bruit ~ N(0, σ²), où σ est calibré sur Δf et ε.
#       Offre une confidentialité (ε, δ) au lieu de ε-DP stricte.
#       Utile quand une légère relaxation de la garantie est acceptable.
#
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Moteur sans interface graphique (pour serveurs/CI)
import matplotlib.pyplot as plt

# ==============================================================================
# ÉTAPE 1 : Création du jeu de données original
# ==============================================================================

np.random.seed(0)

n_individus = 200  # Nombre d'individus dans le jeu de données

donnees_brutes = {
    "ID_Patient":    [f"PAT_{i:04d}" for i in range(1, n_individus + 1)],

    # Attribut sensible numérique 1 : glycémie à jeun (mg/dL)
    "Glycemie_mgdL": np.random.normal(loc=100, scale=15, size=n_individus).round(1),

    # Attribut sensible numérique 2 : pression artérielle systolique (mmHg)
    "Pression_mmHg": np.random.normal(loc=120, scale=12, size=n_individus).round(1),

    # Attribut sensible numérique 3 : âge exact (années)
    "Age":           np.random.randint(20, 80, size=n_individus).astype(float),
}

df_original = pd.DataFrame(donnees_brutes)

print("=" * 70)
print("  JEU DE DONNÉES ORIGINAL — 5 premiers enregistrements (sur 200)")
print("=" * 70)
print(df_original.head().to_string(index=False))
print(f"\n  ... ({n_individus} enregistrements au total)")
print()


# ==============================================================================
# ÉTAPE 2 : Mécanisme de Laplace
# ==============================================================================

def ajouter_bruit_laplace(serie, sensibilite, epsilon):
    """
    Applique le mécanisme de Laplace pour la confidentialité différentielle.

    Le bruit est tiré d'une distribution de Laplace(0, b) où b = sensibilité/ε.
    Un epsilon faible → grand bruit → forte protection de la vie privée.
    Un epsilon grand  → petit bruit → faible protection (mais données utiles).

    Arguments:
        serie       (pd.Series) : La colonne de valeurs à protéger.
        sensibilite (float)     : Sensibilité globale Δf de la requête.
                                  Pour une valeur individuelle : max - min.
        epsilon     (float)     : Paramètre de confidentialité ε (> 0).

    Retourne:
        pd.Series : La série avec le bruit de Laplace ajouté.
    """
    # Paramètre d'échelle de la distribution de Laplace
    echelle_b = sensibilite / epsilon
    bruit = np.random.laplace(loc=0.0, scale=echelle_b, size=len(serie))
    return (serie + bruit).round(1)


# ==============================================================================
# ÉTAPE 3 : Mécanisme Gaussien
# ==============================================================================

def ajouter_bruit_gaussien(serie, sensibilite, epsilon, delta=1e-5):
    """
    Applique le mécanisme Gaussien pour la confidentialité (ε, δ)-différentielle.

    Le paramètre δ représente la probabilité d'un dépassement de la garantie ε.
    σ = (sensibilite / epsilon) * sqrt(2 * ln(1.25 / delta))

    Arguments:
        serie       (pd.Series) : La colonne de valeurs à protéger.
        sensibilite (float)     : Sensibilité globale Δf de la requête.
        epsilon     (float)     : Paramètre de confidentialité ε (> 0).
        delta       (float)     : Probabilité d'échec de la garantie (défaut 1e-5).

    Retourne:
        pd.Series : La série avec le bruit Gaussien ajouté.
    """
    # Calcul de l'écart-type σ selon la formule standard du mécanisme Gaussien
    sigma = (sensibilite / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    bruit = np.random.normal(loc=0.0, scale=sigma, size=len(serie))
    return (serie + bruit).round(1)


# ==============================================================================
# ÉTAPE 4 : Application sur les données avec différentes valeurs d'epsilon
# ==============================================================================

# Paramètres de confidentialité
EPSILON_FORT  = 0.1   # ε faible  → forte protection, bruit important
EPSILON_MOYEN = 1.0   # ε moyen   → compromis équilibré
EPSILON_FAIBLE = 10.0 # ε élevé   → faible protection, bruit réduit

# Sensibilité estimée pour chaque colonne (plage des valeurs possibles)
sensibilites = {
    "Glycemie_mgdL": 150.0,  # Plage réaliste : 50 à 200 mg/dL
    "Pression_mmHg": 100.0,  # Plage réaliste : 60 à 160 mmHg
    "Age":           60.0    # Plage dans ce jeu de données : 20 à 80 ans
}

colonnes_sensibles = ["Glycemie_mgdL", "Pression_mmHg", "Age"]

print(">>> Application du mécanisme de Laplace (ε = 1.0)...")
df_laplace = df_original.copy()
for col in colonnes_sensibles:
    df_laplace[col] = ajouter_bruit_laplace(
        df_original[col], sensibilites[col], epsilon=EPSILON_MOYEN
    )

print(">>> Application du mécanisme Gaussien (ε = 1.0, δ = 1e-5)...")
df_gaussien = df_original.copy()
for col in colonnes_sensibles:
    df_gaussien[col] = ajouter_bruit_gaussien(
        df_original[col], sensibilites[col], epsilon=EPSILON_MOYEN
    )


# ==============================================================================
# ÉTAPE 5 : Affichage des résultats
# ==============================================================================

print()
print("=" * 70)
print("  DONNÉES APRÈS BRUIT DE LAPLACE (ε = 1.0) — 5 premiers enregistrements")
print("=" * 70)
print(df_laplace.head().to_string(index=False))
print()

print("=" * 70)
print("  DONNÉES APRÈS BRUIT GAUSSIEN (ε = 1.0) — 5 premiers enregistrements")
print("=" * 70)
print(df_gaussien.head().to_string(index=False))
print()


# ==============================================================================
# ÉTAPE 6 : Comparaison statistique et visualisation
# ==============================================================================

print("=" * 70)
print("  COMPARAISON DES STATISTIQUES (colonne : Glycemie_mgdL)")
print("=" * 70)
print(f"\n  {'Métrique':<25} {'Original':>12} {'Laplace':>12} {'Gaussien':>12}")
print(f"  {'-'*62}")

col = "Glycemie_mgdL"
for nom, func in [("Moyenne", "mean"), ("Écart-type", "std"),
                   ("Médiane", "median")]:
    v_orig = getattr(df_original[col], func)()
    v_lap  = getattr(df_laplace[col], func)()
    v_gau  = getattr(df_gaussien[col], func)()
    print(f"  {nom:.<25} {v_orig:>12.2f} {v_lap:>12.2f} {v_gau:>12.2f}")

print()

# --- Génération d'un graphique comparatif des distributions ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(
    "Confidentialité Différentielle — Comparaison des distributions\n"
    "(Colonne : Glycémie en mg/dL)",
    fontsize=13, fontweight="bold"
)

params_graphique = [
    (df_original["Glycemie_mgdL"],  "steelblue",  "Données Originales\n(sans bruit)"),
    (df_laplace["Glycemie_mgdL"],   "darkorange", f"Mécanisme de Laplace\n(ε = {EPSILON_MOYEN})"),
    (df_gaussien["Glycemie_mgdL"],  "forestgreen",f"Mécanisme Gaussien\n(ε = {EPSILON_MOYEN}, δ = 1e-5)"),
]

for ax, (donnees, couleur, titre) in zip(axes, params_graphique):
    ax.hist(donnees, bins=30, color=couleur, alpha=0.8, edgecolor="white")
    ax.axvline(donnees.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Moyenne = {donnees.mean():.1f}")
    ax.set_title(titre, fontsize=11)
    ax.set_xlabel("Glycémie (mg/dL)")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=9)

plt.tight_layout()
chemin_graphique = "3_comparaison_distributions.png"
plt.savefig(chemin_graphique, dpi=120, bbox_inches="tight")
print(f"  📊 Graphique sauvegardé : '{chemin_graphique}'")
print()

# --- Démonstration de l'effet de epsilon sur le bruit ---
print("=" * 70)
print("  EFFET DE ε SUR L'AMPLITUDE DU BRUIT (Laplace, col: Glycemie_mgdL)")
print("=" * 70)
print(f"\n  {'Valeur de ε':<20} {'Échelle b=Δ/ε':>15} {'Écart-type du bruit':>22}")
print(f"  {'-'*58}")
for eps in [0.1, 0.5, 1.0, 2.0, 10.0]:
    b = sensibilites["Glycemie_mgdL"] / eps
    std_bruit = np.random.laplace(0, b, 10_000).std()
    print(f"  ε = {eps:<16.1f} {b:>15.2f} {std_bruit:>22.2f}")

print()
print("  ✅ Conclusion : un ε plus petit → bruit plus grand → meilleure protection.")
print("     La distribution globale reste cependant similaire à l'original.")
print("=" * 70)
