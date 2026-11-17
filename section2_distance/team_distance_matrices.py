"""
A5 Section 2: Team diversity audit (Jaccard and Gower distance).
Group 6, MGT 8040 Data Mining, Fall 2026.
Generated with the course-approved Prompt 1 (ChatGPT) and verified against the A5 report.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

members = ["Isis", "Nana", "Tejas"]

traits = [
    "Data outside class",
    "Stats course",
    "Merged datasets",
    "GitHub for analysis",
    "SQL outside class",
    "Writes code",
    "Non-technical presentation",
    "Collected own data",
    "Double-checks AI",
    "A2/A3 model used logged data"
]

tooling = pd.DataFrame([
    [1,0,0,0,0,1,0,1,1,1],
    [1,1,1,0,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,0]
], index=members, columns=traits)

# 1. Jaccard distance
jaccard = pd.DataFrame(
    squareform(pdist(tooling.to_numpy().astype(bool), metric="jaccard")),
    index=members, columns=members
)

print("Jaccard Distance Matrix:")
print(jaccard.round(4))

# 2. Validation checks
identical = [
    (a, b) for a, b in combinations(members, 2)
    if np.array_equal(tooling.loc[a], tooling.loc[b])
]

if identical:
    print("FLAG: Identical vectors:", identical)
else:
    print("No identical vectors.")

if np.allclose(jaccard, 0):
    print("FLAG: Jaccard matrix is all zeros.")
else:
    print("Jaccard matrix is not all zeros.")

# 3. Missing and unique traits
counts = tooling.sum()

print("\nTraits held by nobody:")
print(counts[counts == 0].index.tolist() or "None")

print("\nTraits held by only one member:")
for trait in counts[counts == 1].index:
    holders = tooling.index[tooling[trait] == 1].tolist()
    print(trait, ":", holders)

# 4. Model and tracking profiles
profile = pd.DataFrame({
    "Tracking volume": [40, 40, 3],
    "Raw max |Delta|": [7.4004, 13.43, 1.888],
    "SD(Y)": [7.8260, 12.64, 1.099],
    "Industry domain": [
        "Health/wellness",
        "Campus services",
        "Career prep"
    ]
}, index=members)

profile["Delta_std"] = (
    profile["Raw max |Delta|"] / profile["SD(Y)"]
)

print("\nMember Profiles:")
print(profile.round(4).to_string())

# 5. Range-normalized numeric distance
def numeric_distance(values):
    values = np.asarray(values, dtype=float)
    observed_range = np.ptp(values)

    if observed_range == 0:
        return np.zeros((len(values), len(values)))

    return (
        np.abs(values[:, None] - values[None, :])
        / observed_range
    )

# 6. Gower contributions, equally weighted
domain = profile["Industry domain"].to_numpy()

contributions = {
    "Volume": numeric_distance(
        profile["Tracking volume"]
    ) / 3,

    "Delta_std": numeric_distance(
        profile["Delta_std"]
    ) / 3,

    "Domain": (
        domain[:, None] != domain[None, :]
    ).astype(float) / 3
}

gower = pd.DataFrame(
    sum(contributions.values()),
    index=members,
    columns=members
)

print("\nGower Distance Matrix:")
print(gower.round(4))

# 7. Per-pair attribute contributions
rows = []

for i, j in combinations(range(len(members)), 2):
    rows.append({
        "Pair": f"{members[i]}-{members[j]}",
        **{
            name: matrix[i, j]
            for name, matrix in contributions.items()
        },
        "Total": gower.iloc[i, j]
    })

contribution_table = pd.DataFrame(rows)

print("\nGower Attribute Contributions:")
print(contribution_table.round(4).to_string(index=False))
