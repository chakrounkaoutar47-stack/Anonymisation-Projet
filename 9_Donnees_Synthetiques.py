"""
9_Donnees_Synthetiques.py
==========================
Synthetic data generation using a Gaussian Copula approach.

Principle: The copula separates the joint dependency structure
(correlations) from the marginal distributions.
Steps:
  1. Fit marginal distributions for each numerical column
  2. Transform to uniform via the CDF, then to Normal via the inverse CDF
  3. Fit a multivariate Normal to capture correlations
  4. Sample from the multivariate Normal
  5. Back-transform to the original marginal distributions

For categorical columns: frequency-based sampling is used.

This is a lightweight implementation (numpy/scipy only, no SDV/ctgan)
designed to be educational and reproducible without heavy dependencies.

Key limitation (illustrated in the evaluation section):
  Sub-populations with low cardinality (e.g. suspended sellers) are
  underrepresented or smoothed in the synthetic dataset.

Author : Kaoutar Chakroun — Master Data & IA, INPT / MarjaneMall
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import stats

np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# 1. Real dataset (simulating MarjaneMall OrderBook)
# ──────────────────────────────────────────────────────────────
N_REAL = 150
cities = ["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir"]
city_probs = [0.40, 0.25, 0.15, 0.12, 0.08]
statuts = ["Livrée", "Annulée", "En cours", "Retournée"]
statut_probs = [0.62, 0.14, 0.19, 0.05]

real_data = pd.DataFrame({
    "montant_mad":   np.random.exponential(350, N_REAL).round(2) + 50,
    "nb_articles":   np.random.randint(1, 8, N_REAL).astype(float),
    "delai_jours":   np.random.gamma(shape=2, scale=2, size=N_REAL).round(1) + 1,
    "ville":         np.random.choice(cities, N_REAL, p=city_probs),
    "statut":        np.random.choice(statuts, N_REAL, p=statut_probs),
})

NUM_COLS = ["montant_mad", "nb_articles", "delai_jours"]
CAT_COLS = ["ville", "statut"]

# ──────────────────────────────────────────────────────────────
# 2. Gaussian Copula synthesis
# ──────────────────────────────────────────────────────────────
class GaussianCopulaSynth:
    """Minimal Gaussian copula for mixed numerical/categorical data."""

    def __init__(self):
        self.marginals = {}   # {col: fitted continuous KDE or discrete freq}
        self.cat_freqs = {}   # {col: {value: prob}}
        self.corr_matrix = None
        self.num_cols = []
        self.cat_cols = []

    def fit(self, df: pd.DataFrame, num_cols: list, cat_cols: list):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        # Fit marginals for numerical columns (Kernel Density Estimation)
        for col in num_cols:
            self.marginals[col] = stats.gaussian_kde(df[col].values)

        # Fit frequency distributions for categorical columns
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True)
            self.cat_freqs[col] = freq.to_dict()

        # Transform numerical columns to standard normal via empirical CDF
        normal_data = pd.DataFrame(index=df.index)
        for col in num_cols:
            # Empirical CDF → uniform → normal
            ecdf_vals = stats.rankdata(df[col].values) / (len(df) + 1)
            ecdf_vals = np.clip(ecdf_vals, 1e-6, 1 - 1e-6)
            normal_data[col] = stats.norm.ppf(ecdf_vals)

        # Fit correlation matrix on the normal-transformed numerical data
        self.corr_matrix = normal_data.corr().values

    def sample(self, n: int) -> pd.DataFrame:
        """Generate n synthetic records."""
        k = len(self.num_cols)
        # Sample from correlated multivariate normal
        mean = np.zeros(k)
        mvn_samples = np.random.multivariate_normal(mean, self.corr_matrix, size=n)

        # Convert back: normal → uniform → original marginal (inverse CDF)
        syn_df = pd.DataFrame()
        for i, col in enumerate(self.num_cols):
            u = stats.norm.cdf(mvn_samples[:, i])
            u = np.clip(u, 1e-4, 1 - 1e-4)
            # Inverse transform via KDE-based quantile (Monte Carlo approximation)
            kde = self.marginals[col]
            kde_samples = kde.resample(10_000).flatten()
            quantiles = np.quantile(kde_samples, u)
            # Ensure non-negative for count/monetary columns
            syn_df[col] = np.maximum(quantiles, 0).round(2)

        # Sample categorical columns independently (frequency matching)
        for col in self.cat_cols:
            values = list(self.cat_freqs[col].keys())
            probs  = list(self.cat_freqs[col].values())
            syn_df[col] = np.random.choice(values, size=n, p=probs)

        # Enforce integer type for nb_articles
        if "nb_articles" in syn_df.columns:
            syn_df["nb_articles"] = syn_df["nb_articles"].round().astype(int).clip(1, 10)

        return syn_df

# ──────────────────────────────────────────────────────────────
# 3. Fit and generate
# ──────────────────────────────────────────────────────────────
synth = GaussianCopulaSynth()
synth.fit(real_data, NUM_COLS, CAT_COLS)
N_SYN = 150
syn_data = synth.sample(N_SYN)

print("=" * 60)
print("REAL DATA — descriptive statistics (numerical):")
print(real_data[NUM_COLS].describe().round(2).to_string())
print("\nSYNTHETIC DATA — descriptive statistics (numerical):")
print(syn_data[NUM_COLS].describe().round(2).to_string())

print("\nREAL — city distribution:")
print(real_data["ville"].value_counts(normalize=True).round(3).to_string())
print("\nSYNTHETIC — city distribution:")
print(syn_data["ville"].value_counts(normalize=True).round(3).to_string())

# ──────────────────────────────────────────────────────────────
# 4. Evaluation: rare sub-population smoothing
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUB-POPULATION ANALYSIS (rare category 'Retournée'):")
real_ret   = real_data[real_data["statut"] == "Retournée"]
syn_ret    = syn_data[syn_data["statut"]   == "Retournée"]
print(f"  Real count:      {len(real_ret)} ({len(real_ret)/len(real_data)*100:.1f}%)")
print(f"  Synthetic count: {len(syn_ret)} ({len(syn_ret)/len(syn_data)*100:.1f}%)")
if len(real_ret) > 0 and len(syn_ret) > 0:
    print(f"  Real mean montant_mad for 'Retournée':      {real_ret['montant_mad'].mean():.1f}")
    print(f"  Synthetic mean montant_mad for 'Retournée': {syn_ret['montant_mad'].mean():.1f}")
print("  → Rare sub-populations may be smoothed/under-represented")
print("    in synthetic data — a known limitation of copula-based methods.")

# ──────────────────────────────────────────────────────────────
# 5. Re-identification risk: membership inference check
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RE-IDENTIFICATION RISK — Nearest-neighbor distance check:")
from scipy.spatial.distance import cdist

# Normalize numerical columns
real_norm = (real_data[NUM_COLS] - real_data[NUM_COLS].mean()) / real_data[NUM_COLS].std()
syn_norm  = (syn_data[NUM_COLS]  - real_data[NUM_COLS].mean()) / real_data[NUM_COLS].std()

# Distance between each synthetic record and its nearest real neighbor
dists = cdist(syn_norm.values, real_norm.values, metric="euclidean")
min_dists = dists.min(axis=1)
print(f"  Mean nearest-neighbor distance (synthetic → real): {min_dists.mean():.3f}")
print(f"  % synthetic records with distance < 0.5 (risk zone): "
      f"{(min_dists < 0.5).mean()*100:.1f}%")
print("  (A low mean distance or high % in risk zone indicates membership leakage risk)")

# ──────────────────────────────────────────────────────────────
# 6. Visualization
# ──────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, col in enumerate(NUM_COLS):
    axes[i].hist(real_data[col], bins=25, alpha=0.6, label="Real", color="#2196F3", density=True)
    axes[i].hist(syn_data[col],  bins=25, alpha=0.6, label="Synthetic", color="#FF9800", density=True)
    axes[i].set_title(col)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Gaussian Copula Synthesis — Real vs Synthetic Distributions",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("output/9_synthetic_distributions.png", dpi=150, bbox_inches="tight")

syn_data.to_csv("output/synthetic_orders.csv", index=False)
print("\n✅ Synthetic dataset saved to output/synthetic_orders.csv")
print("✅ Distribution comparison saved to output/9_synthetic_distributions.png")
print("\nNOTE: Synthetic data does not correspond to any real individual.")
print("      However, membership inference attacks may still be possible.")
print("      For open data sharing, combine with Differential Privacy (script 7).")