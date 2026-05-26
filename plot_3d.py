import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

df = pd.read_csv("embedding_trajectory_3d.csv")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for mutant in df["mutant"].unique():
    sub = df[df["mutant"] == mutant].sort_values("time")

    ax.plot(
        sub["pc1"],
        sub["pc2"],
        sub["pc3"],
        label=mutant,
        linewidth=2
    )

    # start point
    ax.scatter(
        sub.iloc[0]["pc1"],
        sub.iloc[0]["pc2"],
        sub.iloc[0]["pc3"],
        s=60,
        marker="o"
    )

    # end point
    ax.scatter(
        sub.iloc[-1]["pc1"],
        sub.iloc[-1]["pc2"],
        sub.iloc[-1]["pc3"],
        s=80,
        marker="^"
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D Protein Identity Drift Trajectory")
ax.legend()
plt.tight_layout()
plt.savefig("embedding_trajectory_3d.png", dpi=200)
plt.show()