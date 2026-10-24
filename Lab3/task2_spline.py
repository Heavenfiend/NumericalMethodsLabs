import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    point = float(lines[0])
    x, y = [list(map(float, line.split())) for line in lines[1:]]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Необходимы одинаковые количества узлов и значений.")
    if any(not math.isfinite(t) for t in x + y + [point]):
        raise ValueError("Все данные должны быть конечными.")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Узлы должны строго возрастать.")
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка находится за пределами таблицы.")
    return point, x, y


def thomas(lower, diagonal, upper, rhs):
    n = len(diagonal)
    p, q = [0.0] * n, [0.0] * n
    for i in range(n):
        denominator = diagonal[i] + (lower[i] * p[i - 1] if i else 0.0)
        if denominator == 0.0:
            raise ValueError("Нулевой знаменатель в методе прогонки.")
        p[i] = -upper[i] / denominator
        q[i] = (rhs[i] - (lower[i] * q[i - 1] if i else 0.0)) / denominator
    result = [0.0] * n
    result[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        result[i] = p[i] * result[i + 1] + q[i]
    return result


def build_spline(x, y):
    n = len(x)
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    lower, diagonal, upper, rhs = ([0.0] * n for _ in range(4))
    diagonal[0] = diagonal[-1] = 1.0
    for i in range(1, n - 1):
        lower[i], diagonal[i], upper[i] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        rhs[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = thomas(lower, diagonal, upper, rhs)
    coefficients = []
    for i in range(n - 1):
        b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h[i])
        coefficients.append((y[i], b, c[i], d))
    return coefficients


def evaluate_piece(coefficients, offset, order=0):
    a, b, c, d = coefficients
    if order == 1:
        return b + 2 * c * offset + 3 * d * offset ** 2
    if order == 2:
        return 2 * c + 6 * d * offset
    return a + offset * (b + offset * (c + offset * d))


def find_interval(x, point):
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка находится за пределами сплайна.")
    for i in range(len(x) - 1):
        if point < x[i + 1]:
            return i
    return len(x) - 2


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    point, x, y = read_data(Path(__file__).resolve().parent / "input" / "task3_2.txt")
    coefficients = build_spline(x, y)
    print("Задание 3.2. Вариант 10. Естественный кубический сплайн")
    print(f"{'Номер':>7} {'Узел':>12} {'Значение':>16}")
    for i in range(len(x)):
        print(f"{i:7d} {x[i]:12.6f} {y[i]:16.8f}")
    print("Локальная форма: Сᵢ(x) = aᵢ + bᵢ·t + cᵢ·t² + dᵢ·t³, t = x − xᵢ")
    print(f"{'Участок':>8} {'Левая граница':>14} {'Правая граница':>15}" + "".join(f"{name:>17}" for name in ("aᵢ", "bᵢ", "cᵢ", "dᵢ")))
    for i, values in enumerate(coefficients):
        print(f"{i + 1:8d} {x[i]:14.6f} {x[i + 1]:15.6f}" + "".join(f"{v:17.10f}" for v in values))
        a, b, c, d = values
        print(f"  На [{x[i]:g}; {x[i + 1]:g}]: {a:.10g} {b:+.10g}·t {c:+.10g}·t² {d:+.10g}·t³; t = x − ({x[i]:g})")
    index = find_interval(x, point)
    value = evaluate_piece(coefficients[index], point - x[index])
    print(f"Точка: {point:g}; номер участка (с единицы): {index + 1}")
    print(f"Значение сплайна: {value:.12f}")
    print("Проверка прохождения через узлы:")
    print(f"{'Узел':>12} {'Табличное значение':>20} {'Сплайн':>20} {'Невязка':>16}")
    for t, exact in zip(x, y):
        i = find_interval(x, t)
        actual = evaluate_piece(coefficients[i], t - x[i])
        print(f"{t:12.6f} {exact:20.12f} {actual:20.12f} {actual - exact:16.6e}")
    print("Скачки значения и производных во внутренних узлах:")
    print(f"{'Узел':>12} {'Значение':>16} {'Первая произв.':>18} {'Вторая произв.':>18}")
    for i in range(1, len(x) - 1):
        jumps = [evaluate_piece(coefficients[i - 1], x[i] - x[i - 1], k)
                 - evaluate_piece(coefficients[i], 0.0, k) for k in range(3)]
        print(f"{x[i]:12.6f}" + "".join(f"{v:18.6e}" for v in jumps))
    print(f"Вторая производная слева: {evaluate_piece(coefficients[0], 0, 2):.12e}")
    print(f"Вторая производная справа: {evaluate_piece(coefficients[-1], x[-1] - x[-2], 2):.12e}")
    figure, axis = plt.subplots(figsize=(9, 5))
    for i, values in enumerate(coefficients):
        grid = np.linspace(x[i], x[i + 1], 150)
        axis.plot(grid, [evaluate_piece(values, t - x[i]) for t in grid], label=f"Участок {i + 1}")
    axis.scatter(x, y, color="black", label="Исходные узлы", zorder=4)
    axis.scatter([point], [value], color="red", marker="*", s=150, label="Заданная точка на сплайне", zorder=5)
    axis.set(title="Естественный кубический сплайн", xlabel="Аргумент", ylabel="Значение сплайна")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output = Path(__file__).resolve().parent / "plot_task3_2.png"
    figure.savefig(output, dpi=180)
    print(f"График сохранён: {output}")
    plt.show()


if __name__ == "__main__":
    main()
