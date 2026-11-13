import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt



def read_data():
    path = Path(__file__).resolve().parent / "input" / "task4_1.txt"
    tokens = path.read_text(encoding="utf-8").split()
    a = float(tokens[0])
    b = float(tokens[1])
    h = float(tokens[2])
    y0 = float(tokens[3])
    z0 = float(tokens[4])
    return a, b, h, y0, z0


def make_grid(a, b, h):
    n = round((b - a) / h)
    grid = []
    for i in range(n + 1):
        grid.append(a + i * h)
    return grid


def f(x, y, z):
    return -z * math.tan(x) - y * (math.cos(x) ** 2)


def exact(x, a, y0, z0):
    t = math.sin(x) - math.sin(a)
    return y0 * math.cos(t) + (z0 / math.cos(a)) * math.sin(t)


def rk4_step(x, y, z, h):
    k1 = h * z
    l1 = h * f(x, y, z)

    k2 = h * (z + l1 / 2.0)
    l2 = h * f(x + h / 2.0, y + k1 / 2.0, z + l1 / 2.0)

    k3 = h * (z + l2 / 2.0)
    l3 = h * f(x + h / 2.0, y + k2 / 2.0, z + l2 / 2.0)

    k4 = h * (z + l3)
    l4 = h * f(x + h, y + k3, z + l3)

    next_y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    next_z = z + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    return next_y, next_z


def euler_method(x, y0, z0):
    h = x[1] - x[0]
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        next_y = y[i] + h * z[i]
        next_z = z[i] + h * f(x[i], y[i], z[i])
        y.append(next_y)
        z.append(next_z)
    return y, z


def rk4_method(x, y0, z0):
    h = x[1] - x[0]
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        next_y, next_z = rk4_step(x[i], y[i], z[i], h)
        y.append(next_y)
        z.append(next_z)
    return y, z


def adams_method(x, y0, z0):
    h = x[1] - x[0]
    y = [y0]
    z = [z0]

    for i in range(3):
        next_y, next_z = rk4_step(x[i], y[i], z[i], h)
        y.append(next_y)
        z.append(next_z)

    for i in range(3, len(x) - 1):
        next_y = y[i] + h * (55.0 * z[i] - 59.0 * z[i - 1] + 37.0 * z[i - 2] - 9.0 * z[i - 3]) / 24.0

        f0 = f(x[i], y[i], z[i])
        f1 = f(x[i - 1], y[i - 1], z[i - 1])
        f2 = f(x[i - 2], y[i - 2], z[i - 2])
        f3 = f(x[i - 3], y[i - 3], z[i - 3])
        next_z = z[i] + h * (55.0 * f0 - 59.0 * f1 + 37.0 * f2 - 9.0 * f3) / 24.0

        y.append(next_y)
        z.append(next_z)

    return y, z


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    a, b, h, y0, z0 = read_data()
    x = make_grid(a, b, h)
    fine_x = make_grid(a, b, h / 2.0)

    print("Задание 4.1. Вариант 10")
    print("Уравнение: y″ + tg(x)·y′ + cos²(x)·y = 0")
    print(f"Отрезок: [{a:g}; {b:g}], шаги: {h:g} и {h / 2.0:g}")
    print(f"Начальные условия: y({a:g}) = {y0:g}, y′({a:g}) = {z0:g}")
    print("Точное решение: y₀·cos(sin(x) − sin(a)) + y′₀·sin(sin(x) − sin(a))/cos(a)")

    methods = [
        ("Эйлер", 1, "tab:blue", euler_method),
        ("Рунге–Кутта", 4, "tab:orange", rk4_method),
        ("Адамс", 4, "tab:green", adams_method),
    ]

    figure, axis = plt.subplots(figsize=(10, 6))

    for name, order, color, solver in methods:
        y, _ = solver(x, y0, z0)
        fine_y, _ = solver(fine_x, y0, z0)

        print(f"\nМетод: {name}, порядок: {order}")
        print(f"{'Узел':>8} {'Решение, h':>15} {'Решение, h/2':>16} {'Точное':>15} {'Ошибка, h':>14} {'Ошибка, h/2':>15} {'Рунге, h/2':>15}")

        errors = []
        estimates = []
        for i in range(len(x)):
            t = x[i]
            value = exact(t, a, y0, z0)
            error = abs(y[i] - value)
            fine_error = abs(fine_y[2 * i] - value)
            estimate = abs(fine_y[2 * i] - y[i]) / (2 ** order - 1)

            errors.append(error)
            estimates.append(estimate)
            print(f"{t:8.3f} {y[i]:15.10f} {fine_y[2 * i]:16.10f} {value:15.10f} {error:14.6e} {fine_error:15.6e} {estimate:15.6e}")

        fine_errors = []
        for j in range(len(fine_x)):
            fine_errors.append(abs(fine_y[j] - exact(fine_x[j], a, y0, z0)))

        print(f"Максимальная абсолютная ошибка на сетке h: {max(errors):.8e}")
        print(f"Максимальная абсолютная ошибка на всей сетке h/2: {max(fine_errors):.8e}")
        print(f"Максимальная оценка Рунге–Ромберга на общих узлах для h/2: {max(estimates):.8e}")

        axis.plot(x, y, "o", color=color, label=f"{name}, шаг {h:g}")
        axis.plot(fine_x, fine_y, color=color, label=f"{name}, шаг {h / 2.0:g}")

    exact_x = []
    exact_y = []
    for i in range(501):
        t = a + (b - a) * i / 500.0
        exact_x.append(t)
        exact_y.append(exact(t, a, y0, z0))

    axis.plot(exact_x, exact_y, color="red", label="Точное решение")
    axis.set(title="Задача Коши — вариант 10", xlabel="Аргумент x", ylabel="Решение y")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(__file__).resolve().parent / "plot_task4_1.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
