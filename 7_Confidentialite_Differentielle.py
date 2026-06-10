"""
7_Confidentialite_Differentielle.py
=====================================
Differential Privacy (DP) using the Laplace mechanism.

Mathematical guarantee: A randomized mechanism M satisfies ε-differential
privacy if for any two datasets D and D' differing in one record,
and for any output set S:
    P[M(D) ∈ S] ≤ exp(ε) · P[M(D') ∈ S]

Lower ε  → stronger privacy, more noise, less utility.
Higher ε → weaker privacy, less noise, more utility.

This script applies DP to aggregate statistics on the MarjaneMall
order dataset (order counts per city, average order value per city),
which are the typical BI exports that benefit from DP protection.

Author : Kaoutar Chakroun — Master Data & IA, INPT / MarjaneMall
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────
# 1. Simulated OrderBook dataset
# ──────────────────────────────────────────────────────────────
np.random.seed(42)
n = 200
cities = ["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir"]
city_weights = [0.40, 0.25, 0.15, 0.12, 0.08]

orders = pd.DataFrame({
    "order_id":    range(1, n + 1),
    "ville":       np.random.choice(cities, size=n, p=city_weights),
    "montant_mad": np.random.exponential(scale=350, size=n).round(2) + 50,
    "nb_articles": np.random.randint(1, 8, size=n),
})

# ──────────────────────────────────────────────────────────────
# 2. Laplace mechanism
# ──────────────────────────────────────────────────────────────
def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """
    Add Laplace noise calibrated to sensitivity / epsilon.
    sensitivity = max possible change in query result from adding/removing one record.
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale)
    return true_value + noise

def dp_count(series: pd.Series, epsilon: float) -> float:
    """DP count query — sensitivity = 1 (one record changes count by at most 1)."""
    true_count = float(len(series))
    return max(0.0, laplace_mechanism(true_count, sensitivity=1.0, epsilon=epsilon))

def dp_mean(series: pd.Series, epsilon: float, value_range: float) -> float:
    """
    DP mean query via sum/count decomposition.
    sensitivity of sum = value_range (one record can change sum by at most range).
    """
    true_sum   = float(series.sum())
    true_count = float(len(series))
    # Split epsilon equally between sum and count queries (composition theorem)
    eps_half = epsilon / 2.0
    dp_sum_val   = laplace_mechanism(true_sum,   sensitivity=value_range, epsilon=eps_half)
    dp_count_val = max(1.0, laplace_mechanism(true_count, sensitivity=1.0, epsilon=eps_half))
    return dp_sum_val / dp_count_val

# ──────────────────────────────────────────────────────────────
# 3. Compare results across ε values
# ──────────────────────────────────────────────────────────────
ORDER_VALUE_RANGE = 2000.0  # MAD — plausible max order value in dataset

epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]

print("=" * 70)
print("TRUE aggregate statistics (no privacy):")
true_stats = orders.groupby("ville").agg(
    nb_commandes=("order_id", "count"),
    montant_moyen=("montant_mad", "mean"),
).round(2)
print(true_stats.to_string())

results_rows = []
for eps in epsilons:
    for city in cities:
        subset = orders[orders["ville"] == city]["montant_mad"]
        dp_cnt = dp_count(subset, epsilon=eps)
        dp_avg = dp_mean(subset, epsilon=eps, value_range=ORDER_VALUE_RANGE)
        results_rows.append({
            "epsilon": eps,
            "ville": city,
            "dp_nb_commandes": round(dp_cnt, 1),
            "dp_montant_moyen": round(dp_avg, 2),
        })

results_df = pd.DataFrame(results_rows)

print("\n" + "=" * 70)
print("DP aggregate statistics by epsilon:")
for eps in epsilons:
    print(f"\n  ε = {eps}:")
    subset = results_df[results_df["epsilon"] == eps][
        ["ville", "dp_nb_commandes", "dp_montant_moyen"]
    ]
    print(subset.to_string(index=False))

# ──────────────────────────────────────────────────────────────
# 4. Visualization: noise vs epsilon trade-off
# ──────────────────────────────────────────────────────────────
import os
os.makedirs("output", exist_ok=True)

true_counts = true_stats["nb_commandes"].reindex(cities).values.astype(float)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: count errors vs epsilon
errors_count = []
for eps in epsilons:
    dp_counts_eps = []
    for city in cities:
        subset = orders[orders["ville"] == city]["montant_mad"]
        dp_counts_eps.append(dp_count(subset, epsilon=eps))
    mae = np.mean(np.abs(np.array(dp_counts_eps) - true_counts))
    errors_count.append(mae)

axes[0].plot(epsilons, errors_count, "bo-", linewidth=2, markersize=8)
axes[0].set_xlabel("Privacy budget ε", fontsize=12)
axes[0].set_ylabel("Mean Absolute Error (order count)", fontsize=11)
axes[0].set_title("DP Count Query: Utility vs Privacy Budget", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="ε=1 (common default)")
axes[0].legend()

# Right: one epsilon, show DP vs true per city
eps_demo = 1.0
dp_counts_demo = []
for city in cities:
    subset = orders[orders["ville"] == city]["montant_mad"]
    dp_counts_demo.append(dp_count(subset, epsilon=eps_demo))

x = np.arange(len(cities))
width = 0.35
axes[1].bar(x - width/2, true_counts, width, label="True count", color="#2196F3", alpha=0.8)
axes[1].bar(x + width/2, dp_counts_demo, width, label=f"DP count (ε={eps_demo})",
            color="#FF9800", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(cities, rotation=15)
axes[1].set_ylabel("Number of orders")
axes[1].set_title("True vs DP Count per City", fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Differential Privacy — Laplace Mechanism on MarjaneMall Orders",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output/7_dp_comparison.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved to output/7_dp_comparison.png")

# ──────────────────────────────────────────────────────────────
# 5. Save DP-protected export (epsilon=1.0)
# ──────────────────────────────────────────────────────────────
export_rows = []
for city in cities:
    subset = orders[orders["ville"] == city]["montant_mad"]
    export_rows.append({
        "ville": city,
        "dp_nb_commandes (ε=1.0)": round(dp_count(subset, epsilon=1.0), 1),
        "dp_montant_moyen (ε=1.0)": round(
            dp_mean(subset, epsilon=1.0, value_range=ORDER_VALUE_RANGE), 2),
    })

pd.DataFrame(export_rows).to_csv("output/dp_aggregate_export.csv", index=False)
print("✅ DP export saved to output/dp_aggregate_export.csv")
print("\nInterpretation:")
print("  ε=0.1 → very strong privacy, high noise, low utility (research/open data)")
print("  ε=1.0 → standard balance (recommended default for most BI exports)")
print("  ε=5.0 → weak privacy, low noise, good utility (internal analytics only)")