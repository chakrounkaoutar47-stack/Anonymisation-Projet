"""
6_Pseudonymisation.py
=====================
Pseudonymization of PII attributes using two strategies:
  - SHA-256 hashing (deterministic, reversible with the hash)
  - Sequential token replacement (Shop_00001, Client_00001, ...)

Key concept: Pseudonymized data remain PERSONAL DATA under the GDPR
as long as the correspondence table exists. This script explicitly
stores the correspondence table to illustrate this legal risk.

Author : Kaoutar Chakroun — Master Data & IA, INPT / MarjaneMall
"""

import hashlib
import pandas as pd
import json
import os

# ──────────────────────────────────────────────────────────────
# 1. Sample dataset (simulating ClientBook + SellerBook)
# ──────────────────────────────────────────────────────────────
clients = pd.DataFrame({
    "client_id":   [1, 2, 3, 4, 5],
    "nom":         ["Alami", "Benali", "Tazi", "Idrissi", "Bennani"],
    "prenom":      ["Ahmed", "Salma", "Karim", "Omar", "Sara"],
    "email":       ["ahmed.alami@gmail.com", "salma.b@yahoo.fr",
                    "karim.t@outlook.com", "omar.idrissi@live.com",
                    "sara.b@gmail.com"],
    "telephone":   ["0611223344", "0799887766", "0600000000",
                    "0655443322", "0712345678"],
    "ville":       ["Casablanca", "Rabat", "Casablanca", "Tanger", "Agadir"],
})

sellers = pd.DataFrame({
    "seller_id": [1, 2, 3, 4, 5],
    "shop_name": ["Fashion Store", "Tech Market", "Deco Home",
                  "Sport Plus", "Bio Nature"],
    "email":     ["contact@fashion.com", "sales@tech.ma",
                  "info@deco.fr", "sport@plus.com", "bio@nature.ma"],
})

# ──────────────────────────────────────────────────────────────
# 2. SHA-256 hashing pseudonymization
# ──────────────────────────────────────────────────────────────
SALT = "marjanemall_secret_salt_2026"   # In production: store securely (env var / vault)

def sha256_pseudonymize(value: str, salt: str = SALT) -> str:
    """Return a truncated SHA-256 hex digest of salt+value."""
    payload = (salt + str(value)).encode("utf-8")
    return "HASH_" + hashlib.sha256(payload).hexdigest()[:16].upper()

# Pseudonymize email on clients
clients_pseudo = clients.copy()
clients_pseudo["email_pseudo"] = clients_pseudo["email"].apply(sha256_pseudonymize)
clients_pseudo["telephone_pseudo"] = clients_pseudo["telephone"].apply(sha256_pseudonymize)

# ──────────────────────────────────────────────────────────────
# 3. Sequential token pseudonymization (like fn_PseudoShop in T-SQL)
# ──────────────────────────────────────────────────────────────
correspondence_table = {}  # {original_value: pseudo_token}
counter = {"client": 0, "shop": 0}

def sequential_pseudo(value: str, prefix: str) -> str:
    """Replace value with a sequential token; record in correspondence table."""
    if value not in correspondence_table:
        counter[prefix] += 1
        token = f"{prefix.capitalize()}_{counter[prefix]:05d}"
        correspondence_table[value] = token
    return correspondence_table[value]

clients_pseudo["nom_pseudo"]    = clients_pseudo["nom"].apply(
    lambda x: sequential_pseudo(x, "client"))
clients_pseudo["prenom_pseudo"] = clients_pseudo["prenom"].apply(
    lambda x: sequential_pseudo(x, "client"))

sellers_pseudo = sellers.copy()
sellers_pseudo["shop_pseudo"] = sellers_pseudo["shop_name"].apply(
    lambda x: sequential_pseudo(x, "shop"))
sellers_pseudo["email_pseudo"] = sellers_pseudo["email"].apply(sha256_pseudonymize)

# ──────────────────────────────────────────────────────────────
# 4. Build TRUSTED_ZONE view (drop original PII columns)
# ──────────────────────────────────────────────────────────────
clients_trusted = clients_pseudo[[
    "client_id", "nom_pseudo", "prenom_pseudo",
    "email_pseudo", "telephone_pseudo", "ville"
]].rename(columns={
    "nom_pseudo": "nom",
    "prenom_pseudo": "prenom",
    "email_pseudo": "email",
    "telephone_pseudo": "telephone",
})

sellers_trusted = sellers_pseudo[[
    "seller_id", "shop_pseudo", "email_pseudo"
]].rename(columns={
    "shop_pseudo": "shop_name",
    "email_pseudo": "email",
})

# ──────────────────────────────────────────────────────────────
# 5. Results
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("RAW ClientBook (before pseudonymization):")
print(clients[["client_id", "nom", "prenom", "email", "telephone"]].to_string(index=False))

print("\nTRUSTED ClientBook (after pseudonymization):")
print(clients_trusted.to_string(index=False))

print("\nRAW SellerBook (before pseudonymization):")
print(sellers.to_string(index=False))

print("\nTRUSTED SellerBook (after pseudonymization):")
print(sellers_trusted.to_string(index=False))

print("\n⚠  CORRESPONDENCE TABLE (illustrates GDPR re-identification risk):")
for original, token in correspondence_table.items():
    print(f"   {token:20s} ← {original}")

# Save correspondence table (in production: encrypted & access-restricted)
os.makedirs("output", exist_ok=True)
with open("output/correspondence_table.json", "w", encoding="utf-8") as f:
    json.dump(correspondence_table, f, indent=2, ensure_ascii=False)

clients_trusted.to_csv("output/clients_trusted.csv", index=False)
sellers_trusted.to_csv("output/sellers_trusted.csv", index=False)

print("\n✅ Results saved to output/")
print("\nNOTE: As long as 'output/correspondence_table.json' exists,")
print("      these data remain PERSONAL DATA under GDPR Article 4(5).")
print("      True anonymization requires destroying this table.")