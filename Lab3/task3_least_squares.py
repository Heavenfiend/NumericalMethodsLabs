import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = []
    for line in raw_lines:
        if line.strip():
            lines.append(line.strip())

    degrees = [int(val) for val in lines[0].split()]
    x = [float(val) for val in lines[1].split()]
    y = [float(val) for val in lines[2].split()]

    if len(x) != len(y) or len(x) == 0 or len(degrees) == 0:
        raise ValueError("Неверные размеры входных данных.")

    return degrees, x, y

# решаем систему уравнений методом Гаусса
def gaussian_elimination(matrix, rhs):
    n = len(rhs)
    a = np.array(matrix, dtype=float)
    b = np.array(rhs, dtype=float)

    for k in range(n):
        pivot = k
        max_val = abs(a[k, k])
        for i in range(k + 1, n):
            if abs(a[i, k]) > max_val:
                max_val = abs(a[i, k])
                pivot = i

        if max_val < 1e-15:
            raise ValueError("Система вырождена.")

        if pivot != k:
            a[[k, pivot]] = a[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]

        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k] = 0.0
            for j in range(k + 1, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]

    result = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total -= a[i, j] * result[j]
        result[i] = total / a[i, i]

    return list(result)

# составляем нормальную систему для метода наименьших квадратов
def normal_system(x, y, degree):
    n = degree + 1
    matrix = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    for i in range(n):
        for j in range(n):
            sum_x = 0.0
            for k in range(len(x)):
                sum_x += x[k] ** (i + j)
            matrix[i, j] = sum_x

        sum_yx = 0.0
        for k in range(len(x)):
            sum_yx += y[k] * (x[k] ** i)
        rhs[i] = sum_yx

    return matrix.tolist(), rhs.tolist()

# находим значение многочлена в заданной точке
def polynomial(coefficients, point):
    result = 0.0
    for i in range(len(coefficients)):
        result += coefficients[i] * (point ** i)
    return result


def polynomial_text(coefficients):
    parts = []
    for i in range(len(coefficients)):
        val = coefficients[i]
        if i == 0:
            parts.append(f"{val:+.12g}")
        elif i == 1:
            parts.append(f"{val:+.12g}·x")
        else:
            parts.append(f"{val:+.12g}·x^{i}")
    return " ".join(parts).lstrip("+")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(__file__).resolve().parent / "input" / "task3_3.txt"
    degrees, x, y = read_data(input_path)

    print("Задание 3.3. Вариант 10. Метод наименьших квадратов")
    print(f"{'Номер':>7} {'Узел':>12} {'Значение':>16}")
    for i in range(len(x)):
        print(f"{i:7d} {x[i]:12.6f} {y[i]:16.8f}")

    results = []
    figure, axis = plt.subplots(figsize=(9, 5))
    grid = np.linspace(min(x), max(x), 500)
    styles = ["-", "--", "-."]

    for index in range(len(degrees)):
        degree = degrees[index]
        matrix, rhs = normal_system(x, y, degree)
        coefficients = gaussian_elimination(matrix, rhs)

        print(f"\nСтепень {degree}. Нормальная система (последний столбец — правая часть):")
        for i in range(len(matrix)):
            row_str = " ".join(f"{val:14.6f}" for val in matrix[i])
            print(f"{row_str} | {rhs[i]:14.6f}")

        print("Коэффициенты при степенях от нулевой до старшей:")
        for i in range(len(coefficients)):
            print(f"Коэффициент {i}: {coefficients[i]:.12f}")
        print(f"П{degree}(x) = {polynomial_text(coefficients)}")

        errors = []
        sse = 0.0
        for k in range(len(x)):
            approx = polynomial(coefficients, x[k])
            err = approx - y[k]
            errors.append(err)
            sse += err * err

        mse = sse / len(x)
        print(f"Сумма квадратов ошибок: {sse:.12f}")
        print(f"Средний квадрат ошибки (MSE): {mse:.12f}")
        # проверка правильности
        max_residual = 0.0
        for i in range(degree + 1):
            row_sum = 0.0
            for j in range(degree + 1):
                row_sum += matrix[i][j] * coefficients[j]
            res = abs(row_sum - rhs[i])
            if res > max_residual:
                max_residual = res

        print(f"Максимальная невязка нормальной системы: {max_residual:.6e}")
        print(f"{'Узел':>12} {'Табличное значение':>20} {'Приближение':>18} {'Отклонение':>18}")
        for k in range(len(x)):
            approx = y[k] + errors[k]
            print(f"{x[k]:12.6f} {y[k]:20.10f} {approx:18.10f} {errors[k]:18.10f}")

        results.append((degree, sse, mse))

        y_grid = [polynomial(coefficients, t) for t in grid]
        style = styles[index % len(styles)]
        axis.plot(grid, y_grid, style, label=f"Многочлен степени {degree}", linewidth=2)

    print(f"\n{'Степень':>9} {'Сумма квадратов':>20} {'Средний квадрат':>20}")
    for degree, sse, mse in results:
        print(f"{degree:9d} {sse:20.12f} {mse:20.12f}")

    best_degree = results[0][0]
    best_mse = results[0][2]
    for degree, sse, mse in results:
        if mse < best_mse:
            best_mse = mse
            best_degree = degree

    print(f"По исходным точкам лучшее приближение даёт многочлен степени {best_degree}.")
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

