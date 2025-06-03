import sys
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Gravitational constant from https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation
G = 6.67430e-11  # m^3 kg^-1 s^-2


class ThreeBodySimulation:
    """Run a simple three body simulation using RK4 integration."""

    def __init__(self):
        # Masses for Sun, Earth and Mars (kg)
        self.masses = np.array([
            1.9885e30,
            5.972e24,
            6.39e23
        ])
        self.positions = np.array([
            [0.0, 0.0, 0.0],
            [1.496e11, 0.0, 0.0],
            [2.279e11, 0.0, 0.0]
        ])
        self.velocities = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 29780.0, 0.0],
            [0.0, 24070.0, 0.0]
        ])
    def compute_accelerations(self, positions):
        num = len(positions)
        a = np.zeros_like(positions)
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)
                a[i] += G * self.masses[j] * diff / (dist ** 3 + 1e-9)
        return a

    def rk4_step(self, dt):
        a1 = self.compute_accelerations(self.positions)
        k1v = a1 * dt
        k1x = self.velocities * dt

        a2 = self.compute_accelerations(self.positions + 0.5 * k1x)
        k2v = a2 * dt
        k2x = (self.velocities + 0.5 * k1v) * dt

        a3 = self.compute_accelerations(self.positions + 0.5 * k2x)
        k3v = a3 * dt
        k3x = (self.velocities + 0.5 * k2v) * dt

        a4 = self.compute_accelerations(self.positions + k3x)
        k4v = a4 * dt
        k4x = (self.velocities + k3v) * dt

        self.velocities += (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        self.positions += (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Three Body Simulation")

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(self)
        control_frame.pack(fill=tk.X)
        self.start_btn = tk.Button(control_frame, text="Start", command=self.toggle)
        self.start_btn.pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=10.0)
        tk.Label(control_frame, text="Speed").pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(control_frame, from_=1, to=100,
                                     orient=tk.HORIZONTAL, variable=self.speed_var)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.sim = ThreeBodySimulation()
        self.history = [np.empty((0, 3)) for _ in range(3)]
        self.running = False
        self.after_id = None
        self.redraw()
    def toggle(self):
        self.running = not self.running
        self.start_btn.configure(text="Pause" if self.running else "Start")
        if self.running:
            self.run_step()
        elif self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def run_step(self):
        dt = self.speed_var.get() * 3600.0
        self.sim.rk4_step(dt)
        for i in range(3):
            self.history[i] = np.vstack([self.history[i], self.sim.positions[i]])
        self.redraw()
        if self.running:
            self.after_id = self.after(50, self.run_step)

    def redraw(self):
        self.ax.clear()
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_zlabel("z (m)")
        self.ax.set_title("Three Body Simulation")
        for i in range(3):
            self.ax.plot(self.history[i][:, 0], self.history[i][:, 1], self.history[i][:, 2])
        self.ax.scatter(
            self.sim.positions[:, 0],
            self.sim.positions[:, 1],
            self.sim.positions[:, 2],
            c=["yellow", "blue", "red"],
            s=[50, 10, 10],
        )
        self.canvas.draw()


if __name__ == "__main__":
    app = Application()
    app.mainloop()
