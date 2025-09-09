import numpy as np
import pandas as pd
from datetime import datetime
import arrival_wait_pairs as awp  # your existing module

# ----------------------------
# User settings
# ----------------------------
def lamE(t):  # E2(t)
    return 2.0 + np.sin(0.5 * t)
lamE_max = 3.0

def lamH1(t):  # H2(t) branch 1
    return 2.0 + 0.6*np.sin(0.5 * t)
def lamH2(t):  # H2(t) branch 2
    return 0.8 + 0.4*np.cos(0.3 * t)
lamH1_max, lamH2_max, p = 2.6, 1.2, 0.6

mu, c      = 1, 8        # service rate, capacity
t_end      = 16.0          # simulation horizon
n_reps     = 100     # number of independent runs
seed_base  = 7             # base seed (replication r uses seed_base + r)

# ----------------------------
# Run simulations
# ----------------------------
rows_E2 = []
rows_H2 = []

for i in range(1, n_reps + 1):
    seed = seed_base + i
    
    # E2(t)
    pairs_E2 = awp.generate_pairs_E2(t_end, mu, c, lamE, lamE_max, seed=seed)
    for (t, v) in pairs_E2:
        rows_E2.append((i, t, v))
    
    # H2(t)
    pairs_H2 = awp.generate_pairs_H2(t_end, mu, c, p, lamH1, lamH1_max, lamH2, lamH2_max, seed=seed)
    for (t, v) in pairs_H2:
        rows_H2.append((i, t, v))

# ----------------------------
# Convert to DataFrames
# ----------------------------
df_E2 = pd.DataFrame(rows_E2, columns=["i", "t", "V(t)"])
df_H2 = pd.DataFrame(rows_H2, columns=["i", "t", "V(t)"])

# ----------------------------
# Save to CSV
# ----------------------------
ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
path_E2 = f"E2_pairs.csv"
path_H2 = f"H2_pairs.csv"
df_E2.to_csv(path_E2, index=False)
df_H2.to_csv(path_H2, index=False)

print(f"E2(t) pairs saved to {path_E2}")
print(f"H2(t) pairs saved to {path_H2}")

# Peek at first few rows
print("\nE2(t) sample:\n", df_E2.head())
print("\nH2(t) sample:\n", df_H2.head())