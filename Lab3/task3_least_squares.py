import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    degrees = list(map(int, lines[0].split()))
    x, y = [list(map(float, line.split())) for line in lines[1:]]
    if len(x) != len(y) or not x or not degrees:
        raise ValueError("Неверные размеры входных данных.")
    if any(not math.isfinite(t) for t in x + y):
        raise ValueError("Все значения должны быть конечными.")
    if any(degree < 0 or degree >= len(set(x)) for degree in degrees):
        raise ValueError("Для выбранной степени недостаточно различных узлов.")
    return degrees, x, y


def gaussian_elimination(matrix, rhs):
    n = len(rhs)
    a = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, rhs)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(a[i][k]))
        if a[pivot][k] == 0.0:
            raise ValueError("Система вырождена.")
        a[k], a[pivot] = a[pivot], a[k]
        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k + 1, n + 1):
                a[i][j] -= factor * a[k][j]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = a[i][n]
        for j in range(i + 1, n):
            total -= a[i][j] * result[j]
        result[i] = total / a[i][i]
    return result


def normal_system(x, y, degree):
    n = degree + 1
    matrix = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n
    for t, value in zip(x, y):
        powers = [1.0]
        for _ in range(2 * degree):
            powers.append(powers[-1] * t)
        for i in range(n):
            rhs[i] += value * powers[i]
            for j in range(n):
                matrix[i][j] += powers[i + j]
    return matrix, rhs


def polynomial(coefficients, point):
    result = 0.0
    for value in reversed(coefficients):
        result = result * point + value
    return result


def polynomial_text(coefficients):
    return " ".join(
        f"{value:+.12g}" + (f"·x^{i}" if i > 1 else "·x" if i else "")
        for i, value in enumerate(coefficients)
    ).lstrip("+")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    degrees, x, y = read_data(Path(__file__).resolve().parent / "input" / "task3_3.txt")
    print("Задание 3.3. Вариант 10. Метод наименьших квадратов")
    print(f"{'Номер':>7} {'Узел':>12} {'Значение':>16}")
    for i, (t, value) in enumerate(zip(x, y)):
        print(f"{i:7d} {t:12.6f} {value:16.8f}")
    results = []
    figure, axis = plt.subplots(figsize=(9, 5))
    grid = np.linspace(min(x), max(x), 500)
    styles = ["-", "--", "-."]
    for index, degree in enumerate(degrees):
        matrix, rhs = normal_system(x, y, degree)
        coefficients = gaussian_elimination(matrix, rhs)
        print(f"\nСтепень {degree}. Нормальная система (последний столбец — правая часть):")
        for row, value in zip(matrix, rhs):
            print(" ".join(f"{t:14.6f}" for t in row) + f" | {value:14.6f}")
        print("Коэффициенты при степенях от нулевой до старшей:")
        for i, value in enumerate(coefficients):
            print(f"Коэффициент {i}: {value:.12f}")
        print(f"П{degree}(x) = {polynomial_text(coefficients)}")
        errors = [polynomial(coefficients, t) - value for t, value in zip(x, y)]
        sse = sum(error * error for error in errors)
        mse = sse / len(x)
        print(f"Сумма квадратов ошибок: {sse:.12f}")
        print(f"Средний квадрат ошибки (MSE): {mse:.12f}")
        residual = max(abs(sum(matrix[i][j] * coefficients[j] for j in range(degree + 1)) - rhs[i])
                       for i in range(degree + 1))
        print(f"Максимальная невязка нормальной системы: {residual:.6e}")
        print(f"{'Узел':>12} {'Табличное значение':>20} {'Приближение':>18} {'Отклонение':>18}")
        for t, value, error in zip(x, y, errors):
            print(f"{t:12.6f} {value:20.10f} {value + error:18.10f} {error:18.10f}")
        results.append((degree, sse, mse))
        axis.plot(grid, [polynomial(coefficients, t) for t in grid], styles[index % len(styles)],
                  label=f"Многочлен степени {degree}", linewidth=2)
    print(f"\n{'Степень':>9} {'Сумма квадратов':>20} {'Средний квадрат':>20}")
    for degree, sse, mse in results:
        print(f"{degree:9d} {sse:20.12f} {mse:20.12f}")
    best = min(results, key=lambda row: row[2])
    print(f"По исходным точкам лучшее приближение даёт многочлен степени {best[0]}.")
    print("Сравнение относится только к ошибке на исходных данных.")
    axis.scatter(x, y, color="black", label="Исходные данные", zorder=5)
    axis.set(title="Приближение методом наименьших квадратов", xlabel="Аргумент", ylabel="Значение")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output = Path(__file__).resolve().parent / "plot_task3_3.png"
    figure.savefig(output, dpi=180)
    print(f"График сохранён: {output}")
    plt.show()


if __name__ == "__main__":
    main()
