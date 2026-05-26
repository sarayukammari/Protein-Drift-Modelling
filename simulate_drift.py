import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from landscape import TimeVaryingLandscape

np.random.seed(42)

T = 200
NOISE_STD = 0.01
DRIFT_STRENGTH = 0.002
SIMILARITY_THRESHOLD = 0.90

df = pd.read_csv("embeddings.csv")

labels = df["id"].values
embedding_cols = [c for c in df.columns if c.startswith("e")]
embeddings = df[embedding_cols].values.astype(float)

wt_index = list(labels).index("WT")
wt_embedding = embeddings[wt_index]

all_similarity_rows = []
all_projection_rows_2d = []
all_projection_rows_3d = []
ihl_rows = []

pca_3d = PCA(n_components=3)
pca_3d.fit(embeddings)

plt.figure(figsize=(10, 6))

for i, label in enumerate(labels):
    if label == "WT":
        continue

    current = embeddings[i].copy()

    # mutation-specific drift direction
    direction = current - wt_embedding
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # time-varying landscape per mutant (mixed deterministic + stochastic)
    landscape = TimeVaryingLandscape(
        dim=current.shape[0],
        base_strength=DRIFT_STRENGTH,
        noise_std=NOISE_STD,
        seed=42 + i
    )

    similarities = []
    identity_half_life = None

    for t in range(T):
        landscape.step()
        current = landscape.apply(current, direction)

        sim = cosine_similarity(
            wt_embedding.reshape(1, -1),
            current.reshape(1, -1)
        )[0, 0]

        similarities.append(sim)

        if identity_half_life is None and sim < SIMILARITY_THRESHOLD:
            identity_half_life = t

        point_3d = pca_3d.transform(current.reshape(1, -1))[0]

        all_similarity_rows.append({
            "mutant": label,
            "time": t,
            "similarity_to_wt": sim
        })

        all_projection_rows_2d.append({
            "mutant": label,
            "time": t,
            "pc1": point_3d[0],
            "pc2": point_3d[1]
        })

        all_projection_rows_3d.append({
            "mutant": label,
            "time": t,
            "pc1": point_3d[0],
            "pc2": point_3d[1],
            "pc3": point_3d[2]
        })

    if identity_half_life is None:
        identity_half_life = T

    ihl_rows.append({
        "mutant": label,
        "identity_half_life": identity_half_life
    })

    plt.plot(range(T), similarities, label=label)

plt.axhline(
    y=SIMILARITY_THRESHOLD,
    linestyle="--",
    color="black",
    label="IHL threshold"
)

plt.xlabel("Time step")
plt.ylabel("Cosine similarity to WT")
plt.title("Protein Identity Drift")
plt.legend()
plt.tight_layout()
plt.savefig("identity_decay.png", dpi=200)
plt.close()

pd.DataFrame(all_similarity_rows).to_csv("drift_similarity.csv", index=False)
pd.DataFrame(all_projection_rows_2d).to_csv("embedding_trajectory.csv", index=False)
pd.DataFrame(all_projection_rows_3d).to_csv("embedding_trajectory_3d.csv", index=False)
pd.DataFrame(ihl_rows).sort_values("identity_half_life").to_csv("ihl.csv", index=False)

print("Saved:")
print("  identity_decay.png")
print("  drift_similarity.csv")
print("  embedding_trajectory.csv")
print("  embedding_trajectory_3d.csv")
print("  ihl.csv")