import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    point = float(lines[1])
    grids = [list(map(float, line.split())) for line in lines[2:]]
    if len(grids) != count or any(len(x) < 2 for x in grids):
        raise ValueError("Неверное количество наборов или узлов.")
    for x in grids:
        if any(not math.isfinite(t) for t in x) or len(set(x)) != len(x):
            raise ValueError("Узлы должны быть конечными и различными.")
        if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
            raise ValueError("Узлы должны возрастать.")
        if not x[0] <= point <= x[-1]:
            raise ValueError("Точка должна находиться внутри отрезка.")
    return point, grids


def lagrange(x, y, point):
    result = 0.0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if i != j:
                term *= (point - x[j]) / (x[i] - x[j])
        result += term
    return result


def divided_differences(x, y):
    n = len(x)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for order in range(1, n):
        for i in range(n - order):
            table[i][order] = (
                (table[i + 1][order - 1] - table[i][order - 1])
                / (x[i + order] - x[i])
            )
    return table


def newton(x, coefficients, point):
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = coefficients[i] + (point - x[i]) * result
    return result


def multiply_linear(coefficients, root):
    result = [0.0] * (len(coefficients) + 1)
    for i, value in enumerate(coefficients):
        result[i] -= root * value
        result[i + 1] += value
    return result


def expanded_lagrange(x, y):
    result = [0.0] * len(x)
    for i in range(len(x)):
        basis = [1.0]
        denominator = 1.0
        for j in range(len(x)):
            if i != j:
                basis = multiply_linear(basis, x[j])
                denominator *= x[i] - x[j]
        for k, value in enumerate(basis):
            result[k] += y[i] * value / denominator
    return result


def polynomial_text(coefficients):
    return " ".join(
        f"{value:+.12g}" + (f"·x^{i}" if i > 1 else "·x" if i else "")
        for i, value in enumerate(coefficients)
    ).lstrip("+")


def print_polynomials(x, y, table):
    terms = []
    for i in range(len(x)):
        denominator = 1.0
        factors = []
        for j in range(len(x)):
            if i != j:
                denominator *= x[i] - x[j]
                factors.append(f"(x - ({x[j]:g}))")
        terms.append(f"({y[i] / denominator:.12g})·" + "·".join(factors))
    print("Многочлен Лагранжа:")
    print("Л(x) = " + " + ".join(terms))
    print("После раскрытия скобок: " + polynomial_text(expanded_lagrange(x, y)))
    terms = [f"({table[0][0]:.12g})"]
    for i in range(1, len(x)):
        terms.append(f"({table[0][i]:.12g})·" + "·".join(
            f"(x - ({x[j]:g}))" for j in range(i)
        ))
    print("Многочлен Ньютона:")
    print("Н(x) = " + " + ".join(terms))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    point, grids = read_data(Path(__file__).resolve().parent / "input" / "task3_1.txt")
    figure, axes = plt.subplots(1, len(grids), figsize=(14, 5), squeeze=False)
    for number, x in enumerate(grids, 1):
        y = [math.atan(t) for t in x]
        table = divided_differences(x, y)
        print(f"\nЗадание 3.1. Вариант 10. Набор {number}. Функция: арктангенс x")
        print(f"{'Номер':>7} {'Узел':>12} {'Значение функции':>20}")
        for i in range(len(x)):
            print(f"{i:7d} {x[i]:12.6f} {y[i]:20.12f}")
        print("Таблица разделённых разностей (столбец 0 — значения функции):")
        print(f"{'Узел':>12}" + "".join(f"{'Порядок ' + str(k):>18}" for k in range(len(x))))
        for i in range(len(x)):
            print(f"{x[i]:12.6f}" + "".join(f"{table[i][k]:18.10f}" for k in range(len(x) - i)))
        print_polynomials(x, y, table)
        left = lagrange(x, y, point)
        right = newton(x, table[0], point)
        exact = math.atan(point)
        print(f"Заданная точка: {point:g}")
        print(f"Значение Лагранжа: {left:.12f}")
        print(f"Значение Ньютона:  {right:.12f}")
        print(f"Точное значение:   {exact:.12f}")
        print(f"Абсолютная погрешность Лагранжа: {abs(left - exact):.12e}")
        print(f"Абсолютная погрешность Ньютона:  {abs(right - exact):.12e}")
        print(f"Разность Лагранж − Ньютон: {left - right:.12e}")
        grid = np.linspace(x[0], x[-1], 500)
        axis = axes[0][number - 1]
        axis.plot(grid, [math.atan(t) for t in grid], label="Исходная функция", color="black")
        axis.plot(grid, [lagrange(x, y, t) for t in grid], label="Многочлен Лагранжа", linewidth=3, alpha=0.6)
        axis.plot(grid, [newton(x, table[0], t) for t in grid], "--", label="Многочлен Ньютона")
        axis.scatter(x, y, color="black", label="Узлы", zorder=4)
        axis.scatter([point], [exact], marker="*", s=150, label="Точное значение в заданной точке", zorder=5)
        axis.scatter([point], [left], marker="s", s=45, label="Интерполяция в заданной точке", zorder=5)
        axis.axvline(point, color="gray", linestyle=":")
        axis.set(title=f"Набор {number}: интерполяция арктангенса", xlabel="Аргумент", ylabel="Значение функции")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output = Path(__file__).resolve().parent / "plot_task3_1.png"
    figure.savefig(output, dpi=180)
    print(f"\nГрафик сохранён: {output}")
    plt.show()


if __name__ == "__main__":
    main()
