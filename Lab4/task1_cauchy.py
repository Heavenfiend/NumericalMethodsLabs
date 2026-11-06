import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_data():
    path = Path(__file__).resolve().parent / "input" / "task4_1.txt"
    a, b, h, y0, z0 = map(float, path.read_text(encoding="utf-8").split())
    if not all(math.isfinite(t) for t in (a, b, h, y0, z0)):
        raise ValueError("Входные данные должны быть конечными числами.")
    if a >= b or h <= 0 or not -math.pi / 2 < a < b < math.pi / 2:
        raise ValueError("Требуются положительный шаг и отрезок внутри (−π/2; π/2).")
    return a, b, h, y0, z0


def make_grid(a, b, h):
    n = round((b - a) / h)
    if n < 4 or abs(n * h - (b - a)) > 1e-10:
        raise ValueError("Шаг должен делить отрезок хотя бы на четыре части.")
    return [a + i * h for i in range(n + 1)]


def f(x, y, z):
    return -z * math.tan(x) - y * math.cos(x) ** 2


def exact(x, a, y0, z0):
    t = math.sin(x) - math.sin(a)
    return y0 * math.cos(t) + z0 / math.cos(a) * math.sin(t)


def rk4_step(x, y, z, h):
    k1 = h * z
    l1 = h * f(x, y, z)
    k2 = h * (z + l1 / 2)
    l2 = h * f(x + h / 2, y + k1 / 2, z + l1 / 2)
    k3 = h * (z + l2 / 2)
    l3 = h * f(x + h / 2, y + k2 / 2, z + l2 / 2)
    k4 = h * (z + l3)
    l4 = h * f(x + h, y + k3, z + l3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6, z + (l1 + 2 * l2 + 2 * l3 + l4) / 6


def solve(x, y0, z0, method):
    h = x[1] - x[0]
    y, z = [y0], [z0]
    for i in range(len(x) - 1):
        if method == "Эйлер":
            next_y = y[i] + h * z[i]
            next_z = z[i] + h * f(x[i], y[i], z[i])
        elif method == "Адамс" and i >= 3:
            next_y = y[i] + h * (55 * z[i] - 59 * z[i - 1] + 37 * z[i - 2] - 9 * z[i - 3]) / 24
            values = [f(x[i - j], y[i - j], z[i - j]) for j in range(4)]
            next_z = z[i] + h * (55 * values[0] - 59 * values[1] + 37 * values[2] - 9 * values[3]) / 24
        else:
            next_y, next_z = rk4_step(x[i], y[i], z[i], h)
        y.append(next_y)
        z.append(next_z)
    return y, z


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a, b, h, y0, z0 = read_data()
    x = make_grid(a, b, h)
    fine_x = make_grid(a, b, h / 2)
    print("Задание 4.1. Вариант 10")
    print("Уравнение: y″ + tg(x)·y′ + cos²(x)·y = 0")
    print(f"Отрезок: [{a:g}; {b:g}], шаги: {h:g} и {h / 2:g}")
    print(f"Начальные условия: y({a:g}) = {y0:g}, y′({a:g}) = {z0:g}")
    print("Точное решение: y₀·cos(sin(x) − sin(a)) + y′₀·sin(sin(x) − sin(a))/cos(a)")
    figure, axis = plt.subplots(figsize=(10, 6))
    for name, order, color in (("Эйлер", 1, "tab:blue"), ("Рунге–Кутта", 4, "tab:orange"), ("Адамс", 4, "tab:green")):
        y, _ = solve(x, y0, z0, name)
        fine_y, _ = solve(fine_x, y0, z0, name)
        print(f"\nМетод: {name}, порядок: {order}")
        print(f"{'Узел':>8} {'Решение, h':>15} {'Решение, h/2':>16} {'Точное':>15} {'Ошибка, h':>14} {'Ошибка, h/2':>15} {'Рунге, h/2':>15}")
        errors, estimates = [], []
        for i, t in enumerate(x):
            value = exact(t, a, y0, z0)
            error = abs(y[i] - value)
            fine_error = abs(fine_y[2 * i] - value)
            estimate = abs(fine_y[2 * i] - y[i]) / (2 ** order - 1)
            errors.append(error)
            estimates.append(estimate)
            print(f"{t:8.3f} {y[i]:15.10f} {fine_y[2 * i]:16.10f} {value:15.10f} {error:14.6e} {fine_error:15.6e} {estimate:15.6e}")
        fine_errors = [abs(value - exact(t, a, y0, z0)) for t, value in zip(fine_x, fine_y)]
        print(f"Максимальная абсолютная ошибка на сетке h: {max(errors):.8e}")
        print(f"Максимальная абсолютная ошибка на всей сетке h/2: {max(fine_errors):.8e}")
        print(f"Максимальная оценка Рунге–Ромберга на общих узлах для h/2: {max(estimates):.8e}")
        axis.plot(x, y, "o", color=color, label=f"{name}, шаг {h:g}")
        axis.plot(fine_x, fine_y, color=color, label=f"{name}, шаг {h / 2:g}")
    exact_x = [a + (b - a) * i / 500 for i in range(501)]
    axis.plot(exact_x, [exact(t, a, y0, z0) for t in exact_x], color="red", label="Точное решение")
    axis.set(title="Задача Коши — вариант 10", xlabel="Аргумент x", ylabel="Решение y")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(__file__).resolve().parent / "plot_task4_1.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
