import sys
from pathlib import Path

import numpy as np


INPUT_FILE = Path(__file__).resolve().parent / "input" / "thomas.txt"


def read_system(path):
    tokens = path.read_text(encoding="utf-8").split()
    n = int(tokens[0])
    expected = 4 * n - 1
    if len(tokens) != expected:
        raise ValueError(f"Ожидалось {expected} чисел, получено {len(tokens)}")
    values = [float(token) for token in tokens[1:]]
    position = 0
    lower = np.array(values[position:position + n - 1], dtype=float)
    position += n - 1
    main_diagonal = np.array(values[position:position + n], dtype=float)
    position += n
    upper = np.array(values[position:position + n - 1], dtype=float)
    position += n - 1
    right_side = np.array(values[position:position + n], dtype=float)
    return lower, main_diagonal, upper, right_side


def thomas_method(lower, main_diagonal, upper, right_side, tolerance=1e-14):
    n = len(main_diagonal)
    p = np.zeros(n, dtype=float)
    q = np.zeros(n, dtype=float)
    denominator = main_diagonal[0]
    if abs(denominator) <= tolerance:
        raise ValueError("Нулевой знаменатель на первом шаге")
    if n > 1:
        p[0] = -upper[0] / denominator
    q[0] = right_side[0] / denominator

    for index in range(1, n):
        denominator = main_diagonal[index] + lower[index - 1] * p[index - 1]
        if abs(denominator) <= tolerance:
            raise ValueError(f"Нулевой знаменатель на шаге {index + 1}")
        if index < n - 1:
            p[index] = -upper[index] / denominator
        q[index] = (right_side[index] - lower[index - 1] * q[index - 1]) / denominator

    solution = np.zeros(n, dtype=float)
    solution[-1] = q[-1]
    for index in range(n - 2, -1, -1):
        solution[index] = p[index] * solution[index + 1] + q[index]
    return solution, p, q


def residual_norm(lower, main_diagonal, upper, right_side, solution):
    maximum = 0.0
    for index in range(len(main_diagonal)):
        value = main_diagonal[index] * solution[index]
        if index > 0:
            value += lower[index - 1] * solution[index - 1]
        if index < len(main_diagonal) - 1:
            value += upper[index] * solution[index + 1]
        maximum = max(maximum, abs(value - right_side[index]))
    return maximum


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    lower, main_diagonal, upper, right_side = read_system(INPUT_FILE)
    solution, p, q = thomas_method(lower, main_diagonal, upper, right_side)

    print("1.2. Метод прогонки")
    print("Нижняя побочная диагональ:\n", lower)
    print("Главная диагональ:\n", main_diagonal)
    print("Верхняя побочная диагональ:\n", upper)
    print("Правая часть:\n", right_side)
    print("Прогоночные коэффициенты:")
    for index in range(len(p)):
        print(f"i = {index + 1}: P_i = {p[index]:.12g}, Q_i = {q[index]:.12g}")
    print("Итоговое решение:\n", solution)
    print(f"Норма невязки: {residual_norm(lower, main_diagonal, upper, right_side, solution):.12e}")


if __name__ == "__main__":
    main()
