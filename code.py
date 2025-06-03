import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# Гравитационная постоянная (можно изменить под свои единицы измерения)
G = 1.0

# Массив масс трёх тел
masses = np.array([1.0, 1.0, 1.0])

# Начальные положения (x, y) и скорости (vx, vy) трёх тел
positions = np.array([
    [-1.0,  0.0],  # тело 1
    [ 1.0,  0.0],  # тело 2
    [ 0.0,  0.5]   # тело 3
])
velocities = np.array([
    [ 0.0,  0.3],  # скорость тела 1
    [ 0.0, -0.3],  # скорость тела 2
    [ 0.0,  0.0]   # скорость тела 3
])

# Базовый шаг по времени (чем меньше, тем точнее, но медленнее расчёты)
dt_base = 0.01

# Умножитель для регулирования скорости симуляции:
# >1 — симуляция идёт быстрее, <1 — медленнее
speed_multiplier = 5.0

def compute_accelerations(positions):
    """
    Вычисляет ускорения каждой массы под действием гравитации остальных тел.
    Формула: a_i = G * sum_{j≠i} m_j * (r_j - r_i) / |r_j - r_i|^3
    """
    num_bodies = len(positions)
    accelerations = np.zeros_like(positions)
    for i in range(num_bodies):
        acc = np.zeros(2)
        for j in range(num_bodies):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            dist = np.linalg.norm(diff)
            # Добавляем небольшую защиту от деления на ноль
            acc += G * masses[j] * diff / (dist**3 + 1e-6)
        accelerations[i] = acc
    return accelerations

def rk4_step(positions, velocities, dt):
    """
    Один шаг интегрирования методом Рунге–Кутты 4-го порядка.
    На вход: текущие позиции, скорости и шаг dt.
    Возвращает: новые позиции и скорости.
    """
    # k1
    a1 = compute_accelerations(positions)
    k1v = a1 * dt
    k1x = velocities * dt

    # k2
    a2 = compute_accelerations(positions + 0.5 * k1x)
    k2v = a2 * dt
    k2x = (velocities + 0.5 * k1v) * dt

    # k3
    a3 = compute_accelerations(positions + 0.5 * k2x)
    k3v = a3 * dt
    k3x = (velocities + 0.5 * k2v) * dt

    # k4
    a4 = compute_accelerations(positions + k3x)
    k4v = a4 * dt
    k4x = (velocities + k3v) * dt

    new_velocities = velocities + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
    new_positions  = positions  + (k1x + 2*k2x + 2*k3x + k4x) / 6.0
    return new_positions, new_velocities

# Настройка графика
fig, ax = plt.subplots(figsize=(6, 6))
colors = ['red', 'green', 'blue']
scat = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=50)

# Оставляем место под слайдер скорости внизу графика
plt.subplots_adjust(bottom=0.2)

# Создаём ползунок для управления скоростью
speed_ax = plt.axes([0.25, 0.05, 0.5, 0.03])
speed_slider = Slider(speed_ax, 'Скорость', valmin=0.1, valmax=10.0,
                      valinit=speed_multiplier, valstep=0.1)

def update_speed(val):
    global speed_multiplier
    speed_multiplier = speed_slider.val

speed_slider.on_changed(update_speed)

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Трёхтелая симуляция")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    """
    Функция обновления для анимации:
    выполняет один шаг интегрирования и перерисовывает положения точек.
    """
    global positions, velocities
    dt = dt_base * speed_multiplier
    positions, velocities = rk4_step(positions, velocities, dt)
    scat.set_offsets(positions)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=600, interval=50, blit=True)
plt.show()
