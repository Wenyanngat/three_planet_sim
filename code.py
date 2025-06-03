import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import argparse

# Gravitational constant (arbitrary units)
G = 1.0


def generate_system(mode="random", seed=None):
    """Generate initial masses, positions and velocities."""
    rng = np.random.default_rng(seed)
    num_bodies = 3
    masses = rng.uniform(0.5, 5.0, size=num_bodies)
    if mode == "close":
        positions = rng.uniform(-0.3, 0.3, size=(num_bodies, 3))
        velocities = rng.uniform(-0.3, 0.3, size=(num_bodies, 3))
    else:
        positions = rng.uniform(-1.5, 1.5, size=(num_bodies, 3))
        velocities = rng.uniform(-1.0, 1.0, size=(num_bodies, 3))
    return masses, positions, velocities


def compute_accelerations(masses, positions):
    """Compute gravitational accelerations for each body."""
    num_bodies = len(positions)
    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        acc = np.zeros(3)
        for j in range(num_bodies):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            dist = np.linalg.norm(diff)
            acc += G * masses[j] * diff / (dist ** 3 + 1e-6)
        accelerations[i] = acc
    return accelerations


def rk4_step(masses, positions, velocities, dt):
    """Perform a single Runge-Kutta 4th order integration step."""
    a1 = compute_accelerations(masses, positions)
    k1v = a1 * dt
    k1x = velocities * dt

    a2 = compute_accelerations(masses, positions + 0.5 * k1x)
    k2v = a2 * dt
    k2x = (velocities + 0.5 * k1v) * dt

    a3 = compute_accelerations(masses, positions + 0.5 * k2x)
    k3v = a3 * dt
    k3x = (velocities + 0.5 * k2v) * dt

    a4 = compute_accelerations(masses, positions + k3x)
    k4v = a4 * dt
    k4x = (velocities + k3v) * dt

    new_velocities = velocities + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    new_positions = positions + (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
    return new_positions, new_velocities


parser = argparse.ArgumentParser(description="3D three-body simulation")
parser.add_argument("--mode", choices=["random", "close"], default="random",
                    help="Initial configuration mode")
args = parser.parse_args()

masses, positions, velocities = generate_system(args.mode)

# Base time step (smaller => more accurate but slower)
dt_base = 0.01
speed_multiplier = 5.0

colors = ["red", "green", "blue"]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=colors, s=masses * 20)
lines = [ax.plot([], [], [], c=c, lw=1)[0] for c in colors]

plt.subplots_adjust(bottom=0.2)

speed_ax = plt.axes([0.25, 0.05, 0.5, 0.03])
speed_slider = Slider(speed_ax, "Speed", valmin=0.1, valmax=10.0,
                      valinit=speed_multiplier, valstep=0.1)


def update_speed(val):
    global speed_multiplier
    speed_multiplier = speed_slider.val


speed_slider.on_changed(update_speed)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_title("Three-body simulation (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

histories = [np.empty((0, 3)) for _ in range(3)]


def update(frame):
    global positions, velocities
    dt = dt_base * speed_multiplier
    positions, velocities = rk4_step(masses, positions, velocities, dt)
    for i in range(3):
        histories[i] = np.vstack([histories[i], positions[i]])
        lines[i].set_data(histories[i][:, 0], histories[i][:, 1])
        lines[i].set_3d_properties(histories[i][:, 2])
    scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return [scat] + lines


ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.show()
