import numpy as np


class DynamicLandscape:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.t = 0

        # basin centers
        self.native = np.array([0.0, 0.0])
        self.misfold = np.array([2.0, 1.2])
        self.agg = np.array([-2.0, 1.5])

    def step(self):
        self.t += 1

        # slow movement of basins
        tt = self.t * 0.01
        self.native = np.array([0.3*np.sin(tt), 0.2*np.cos(tt)])
        self.misfold = np.array([2 + 0.3*np.cos(tt), 1.2 + 0.2*np.sin(tt)])
        self.agg = np.array([-2 + 0.2*np.sin(tt), 1.5 + 0.2*np.cos(tt)])

    def energy(self, X, Y):
        def well(cx, cy, depth, width):
            return -depth * np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*width**2))

        native = well(*self.native, 6, 0.9)
        misfold = well(*self.misfold, 4, 0.7)
        agg = well(*self.agg, 3, 0.6)

        # ridge barrier
        ridge = 2.5 * np.exp(-((X+0.5)**2)/0.8)

        # rugged surface
        ripple = 0.8 * np.sin(1.5*X + 0.2*self.t) * np.cos(1.2*Y)

        # tilt (aging / stress)
        tilt = 0.15*X - 0.1*Y

        return native + misfold + agg + ridge + ripple + tilt

    def grad(self, x, y):
        eps = 1e-3
        dx = (self.energy(x+eps, y) - self.energy(x-eps, y)) / (2*eps)
        dy = (self.energy(x, y+eps) - self.energy(x, y-eps)) / (2*eps)
        return np.array([dx, dy])


def simulate(initial_positions, steps=300, eta=0.05, noise=0.05):
    landscape = DynamicLandscape()
    rng = np.random.default_rng(42)

    traj = {k: [v.copy()] for k, v in initial_positions.items()}

    for _ in range(steps):
        landscape.step()

        for k in traj:
            cur = traj[k][-1]

            grad = landscape.grad(cur[0], cur[1])

            new = cur - eta * grad + rng.normal(0, noise, 2)
            traj[k].append(new)

    return {k: np.array(v) for k, v in traj.items()}


def load_trajectories_from_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if not {"mutant", "time", "pc1", "pc2"}.issubset(df.columns):
        raise ValueError("CSV must contain mutant, time, pc1, pc2 columns")

    trajectories = {}
    for mutant in df["mutant"].unique():
        sub = df[df["mutant"] == mutant].sort_values("time")
        trajectories[mutant] = sub[["pc1", "pc2"]].values

    return trajectories


def animate(traj, output="drift.gif", frames=300, grid_size=60, interval=200, trail_len=60, elev=35, azim=-45, spin=0.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    landscape = DynamicLandscape()

    all_points = np.vstack(list(traj.values()))
    xs = np.linspace(all_points[:,0].min()-6, all_points[:,0].max()+6, grid_size)
    ys = np.linspace(all_points[:,1].min()-6, all_points[:,1].max()+6, grid_size)
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111, projection="3d")

    lines = {}
    points = {}

    for k in traj:
        line, = ax.plot([], [], [], lw=3)
        point, = ax.plot([], [], [], 'o', markersize=10)
        lines[k] = line
        points[k] = point

    def update(frame):
        ax.clear()

        landscape.step()
        Z = landscape.energy(X, Y)

        ax.plot_surface(X, Y, Z, alpha=0.8)

        # Camera control (set a fixed angle; optional slow spin via spin)
        ax.view_init(elev=elev, azim=azim + spin * frame)

        for k, arr in traj.items():
            idx = frame % len(arr)

            start = max(0, idx - trail_len)
            trail = arr[start:idx+1]

            z = np.array([
                landscape.energy(p[0], p[1]) for p in trail
            ])

            ax.plot(trail[:,0], trail[:,1], z, lw=6)

            cur = trail[-1]
            ax.scatter(cur[0], cur[1],
                       landscape.energy(cur[0], cur[1]),
                       s=80, color='red')

        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_zlim(-8, 6)

        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    ani.save(output, writer="pillow")

    print("Saved:", output)


# ------------------ RUN ------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animate time-varying landscape")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--out", default="drift.gif")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--grid", type=int, default=60)
    parser.add_argument("--interval", type=int, default=200)
    parser.add_argument("--trail", type=int, default=60)
    parser.add_argument("--elev", type=float, default=35)
    parser.add_argument("--azim", type=float, default=-45)
    parser.add_argument("--spin", type=float, default=0, help="Degrees per frame to rotate azimuth (0 = no rotation)")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--noise", type=float, default=0.08)

    args = parser.parse_args()

    trajectories = None
    if args.csv:
        try:
            trajectories = load_trajectories_from_csv(args.csv)
        except Exception:
            trajectories = None

    if trajectories is None:
        initial = {
            "WT": np.array([5.0, 5.0]),
            "Mut1": np.array([-5.0, 5.0]),
            "Mut2": np.array([5.0, -5.0]),
        }
        trajectories = simulate(initial, steps=args.steps, eta=args.eta, noise=args.noise)

    animate(
        trajectories,
        output=args.out,
        frames=args.frames,
        grid_size=args.grid,
        interval=args.interval,
        trail_len=args.trail,
        elev=args.elev,
        azim=args.azim,
        spin=args.spin,
    )  
