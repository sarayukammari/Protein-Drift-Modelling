import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------
# STEP 1: Load 3D trajectory CSV
# -------------------------------
df = pd.read_csv("embedding_trajectory_3d.csv")

mutants = df["mutant"].unique().tolist()
T = int(df["time"].max()) + 1

# Pre-split for speed
by_mutant = {
    m: df[df["mutant"] == m].sort_values("time")
    for m in mutants
}

# -------------------------------
# STEP 2: Set up plot
# -------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

lines = {}
points = {}

for m in mutants:
    line, = ax.plot([], [], [], linewidth=2, label=m)
    point, = ax.plot([], [], [], "o")
    lines[m] = line
    points[m] = point

# Set bounds based on data
ax.set_xlim(df["pc1"].min(), df["pc1"].max())
ax.set_ylim(df["pc2"].min(), df["pc2"].max())
ax.set_zlim(df["pc3"].min(), df["pc3"].max())

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Protein Identity Drift (3D)")
ax.legend()

# -------------------------------
# STEP 3: Animation update
# -------------------------------
def update(frame):
    artists = []
    for m in mutants:
        sub = by_mutant[m]
        sub = sub[sub["time"] <= frame]
        x = sub["pc1"].values
        y = sub["pc2"].values
        z = sub["pc3"].values

        if len(x) == 0:
            continue

        lines[m].set_data(x, y)
        lines[m].set_3d_properties(z)

        points[m].set_data([x[-1]], [y[-1]])
        points[m].set_3d_properties([z[-1]])

        artists.extend([lines[m], points[m]])

    return artists

ani = FuncAnimation(fig, update, frames=T, interval=60)

ani.save("drift_animation.gif", writer="pillow")

print("Saved drift_animation.gif")

plt.show()