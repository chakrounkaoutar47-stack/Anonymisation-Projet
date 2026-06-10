"""
8_K_Anonymat_L_Diversite.py
============================
Formal k-anonymity verification and l-diversity extension.

k-anonymity: Every combination of quasi-identifier values appears in at
             least k records → individualization is impossible.

l-diversity: Every k-anonymous equivalence class contains at least l
             distinct values for each sensitive attribute → protects
             against homogeneity attacks that k-anonymity misses.

This script:
  1. Checks whether a dataset satisfies k-anonymity for a given k
  2. Identifies violating groups and reports minimum k achieved
  3. Applies simple suppression-based generalization to reach target k
  4. Checks l-diversity on the sensitive attribute "statut_commande"

Author : Kaoutar Chakroun — Master Data & IA, INPT / MarjaneMall
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. Simulated dataset with quasi-identifiers and a sensitive attribute
# ──────────────────────────────────────────────────────────────
np.random.seed(7)
n = 60
data = pd.DataFrame({
    "client_id":       range(1, n + 1),
    "age":             np.random.choice([22, 25, 28, 30, 35, 40, 45, 50], size=n),
    "ville":           np.random.choice(["Casablanca", "Rabat", "Agadir",
                                          "Tanger", "Fès"], size=n,
                                        p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "domaine_email":   np.random.choice(["gmail.com", "yahoo.fr",
                                          "outlook.com", "hotmail.com"], size=n,
                                        p=[0.50, 0.20, 0.20, 0.10]),
    # Sensitive attribute — not a quasi-identifier
    "statut_commande": np.random.choice(
        ["Livrée", "Annulée", "En cours", "Retournée"], size=n,
        p=[0.60, 0.15, 0.20, 0.05]),
})

QUASI_IDS  = ["age", "ville", "domaine_email"]
SENSITIVE  = "statut_commande"

# ──────────────────────────────────────────────────────────────
# 2. k-anonymity check
# ──────────────────────────────────────────────────────────────
def check_k_anonymity(df: pd.DataFrame, quasi_ids: list) -> pd.DataFrame:
    """Return group sizes for each quasi-identifier combination."""
    return df.groupby(quasi_ids).size().reset_index(name="group_size")

def min_k(df: pd.DataFrame, quasi_ids: list) -> int:
    groups = check_k_anonymity(df, quasi_ids)
    return int(groups["group_size"].min())

def is_k_anonymous(df: pd.DataFrame, quasi_ids: list, k: int) -> bool:
    return min_k(df, quasi_ids) >= k

print("=" * 60)
print("STEP 1 — k-anonymity check on original dataset")
print(f"  Dataset size: {len(data)} records")
k_min = min_k(data, QUASI_IDS)
print(f"  Minimum group size (current k): {k_min}")
print(f"  Satisfies k=2? {is_k_anonymous(data, QUASI_IDS, 2)}")
print(f"  Satisfies k=3? {is_k_anonymous(data, QUASI_IDS, 3)}")

groups = check_k_anonymity(data, QUASI_IDS)
violating_2 = groups[groups["group_size"] < 2]
print(f"\n  Groups violating k=2: {len(violating_2)}")
if len(violating_2) > 0:
    print(violating_2.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 3. Achieve k=3 via generalization (age → age range, rare city → "Autre")
# ──────────────────────────────────────────────────────────────
def generalize_age(age: int) -> str:
    if age < 30:   return "20-29"
    elif age < 40: return "30-39"
    else:          return "40+"

def generalize_city(city: str, min_freq: int, df: pd.DataFrame) -> str:
    """Suppress cities with fewer than min_freq occurrences → 'Autre'."""
    freq = df["ville"].value_counts()
    return city if freq.get(city, 0) >= min_freq else "Autre"

TARGET_K = 3
data_anon = data.copy()
data_anon["age_range"] = data_anon["age"].apply(generalize_age)
data_anon["ville_gen"] = data_anon["ville"].apply(
    lambda c: generalize_city(c, min_freq=8, df=data))

QUASI_IDS_GEN = ["age_range", "ville_gen", "domaine_email"]

print("\n" + "=" * 60)
print("STEP 2 — After generalization (age → range, rare cities → 'Autre')")
k_min_gen = min_k(data_anon, QUASI_IDS_GEN)
print(f"  Minimum group size after generalization: {k_min_gen}")
print(f"  Satisfies k={TARGET_K}? {is_k_anonymous(data_anon, QUASI_IDS_GEN, TARGET_K)}")

groups_gen = check_k_anonymity(data_anon, QUASI_IDS_GEN)
print(f"\n  Top 10 equivalence class sizes:")
print(groups_gen.sort_values("group_size", ascending=False)
      .head(10).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 4. l-diversity check on sensitive attribute within each equivalence class
# ──────────────────────────────────────────────────────────────
def check_l_diversity(df: pd.DataFrame, quasi_ids: list, sensitive: str) -> pd.DataFrame:
    """Return the number of distinct sensitive values per equivalence class."""
    return (df.groupby(quasi_ids)[sensitive]
              .nunique()
              .reset_index(name="distinct_sensitive_values"))

def min_l(df: pd.DataFrame, quasi_ids: list, sensitive: str) -> int:
    return int(check_l_diversity(df, quasi_ids, sensitive)
               ["distinct_sensitive_values"].min())

print("\n" + "=" * 60)
print(f"STEP 3 — l-diversity on sensitive attribute '{SENSITIVE}'")
l_min = min_l(data_anon, QUASI_IDS_GEN, SENSITIVE)
print(f"  Minimum distinct values per equivalence class: {l_min}")
print(f"  Satisfies l=2? {l_min >= 2}")

l_div = check_l_diversity(data_anon, QUASI_IDS_GEN, SENSITIVE)
violations_l2 = l_div[l_div["distinct_sensitive_values"] < 2]
print(f"  Classes violating l=2 (homogeneity attack risk): {len(violations_l2)}")
if len(violations_l2) > 0:
    print(violations_l2.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 5. Privacy risk report
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRIVACY RISK SUMMARY")
print(f"  Original dataset  → k_min = {k_min}  (k-anonymity NOT guaranteed)")
print(f"  After generalization → k_min = {k_min_gen} (k={TARGET_K} {'✅ achieved' if k_min_gen >= TARGET_K else '❌ not achieved'})")
print(f"  l-diversity       → l_min = {l_min} (l=2 {'✅ satisfied' if l_min >= 2 else '❌ not satisfied'})")
print("\nRemaining risks:")
print("  - k-anonymity does not protect against inference attacks")
print("  - l-diversity does not guarantee t-closeness")
print("  - Linkage with external datasets (e.g., social media) can")
print("    still allow re-identification even when k≥3")
print("  → For strong formal guarantees, use Differential Privacy (see script 7)")

import os
os.makedirs("output", exist_ok=True)
data_anon[QUASI_IDS_GEN + [SENSITIVE]].to_csv(
    "output/k_anon_l_diverse.csv", index=False)
print("\n✅ Anonymized dataset saved to output/k_anon_l_diverse.csv")