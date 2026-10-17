import sys
from pathlib import Path
from time import perf_counter

import numpy as np


INPUT_FILE = Path(__file__).resolve().parent / "input" / "iterations.txt"


def read_system(path):
    tokens = path.read_text(encoding="utf-8").split()
    n = int(tokens[0])
    expected = 1 + n * n + n + 2
    if len(tokens) != expected:
        raise ValueError(f"Ожидалось {expected} чисел, получено {len(tokens)}")
    values = [float(token) for token in tokens[1:]]
    matrix = np.array(values[:n * n], dtype=float).reshape(n, n)
    right_side = np.array(values[n * n:n * n + n], dtype=float)
    epsilon = values[-2]
    max_iterations = int(values[-1])
    return matrix, right_side, epsilon, max_iterations


def vector_norm_inf(vector):
    return max(abs(value) for value in vector)


def matrix_norm_inf(matrix):
    return max(sum(abs(value) for value in row) for row in matrix) # для проверки сходимости


def build_iteration_form(matrix, right_side):
    n = len(right_side)
    B = np.zeros((n, n), dtype=float)
    c = np.zeros(n, dtype=float)
    for i in range(n):
        if matrix[i, i] == 0.0:
            raise ValueError("Нулевой диагональный элемент")
        c[i] = right_side[i] / matrix[i, i]
        for j in range(n):
            if i != j:
                B[i, j] = -matrix[i, j] / matrix[i, i]
    return B, c


def simple_iterations(B, c, epsilon, max_iterations):
    n = len(c)
    previous = np.zeros(n, dtype=float)
    started = perf_counter() # запоминаем время

    for iteration in range(1, max_iterations + 1):
        current = B @ previous + c
        difference = vector_norm_inf(current - previous)
        if difference <= epsilon:
            return current, iteration, difference, perf_counter() - started
        previous = current
    raise RuntimeError("Метод простых итераций не сошёлся")


def gauss_seidel(B, c, epsilon, max_iterations):
    n = len(c)
    previous = np.zeros(n, dtype=float)
    started = perf_counter()

    for iteration in range(1, max_iterations + 1):
        current = previous.copy()
        for i in range(n):
            current[i] = c[i]
            for j in range(i):
                current[i] += B[i, j] * current[j]
            for j in range(i + 1, n):
                current[i] += B[i, j] * previous[j]
        difference = vector_norm_inf(current - previous)
        if difference <= epsilon:
            return current, iteration, difference, perf_counter() - started
        previous = current
    raise RuntimeError("Метод Зейделя не сошёлся")


def reference_solution(matrix, right_side):
    return np.linalg.solve(matrix, right_side)


def main():
    if hasattr(sys.stdout, "reconfigure"): # решение вывода русского текста в консоль
        sys.stdout.reconfigure(encoding="utf-8")
    matrix, right_side, epsilon, max_iterations = read_system(INPUT_FILE)
    B, c = build_iteration_form(matrix, right_side)
    norm_B = matrix_norm_inf(B)

    print("1.3. Метод простых итераций и метод Зейделя")
    print(f"Заданная точность ε: {epsilon:.12g}")
    print(f"Матрица B:\n{B}")
    print(f"Вектор c:\n{c}")
    print(f"Норма ||B||∞: {norm_B:.6f}")
    if norm_B >= 1.0:
        print("Предупреждение: достаточное условие сходимости ||B|| < 1 не выполнено")

    reference = reference_solution(matrix, right_side)
    simple_solution, simple_count, simple_difference, simple_time = simple_iterations(
        B, c, epsilon, max_iterations
    )
    seidel_solution, seidel_count, seidel_difference, seidel_time = gauss_seidel(
        B, c, epsilon, max_iterations
    )

    print("Эталонное решение:\n", reference)
    print("Решение методом простых итераций:\n", simple_solution)
    print(f"Количество итераций: {simple_count}")
    print(f"Норма разности соседних приближений: {simple_difference:.12e}")
    print(f"Норма невязки: {vector_norm_inf(matrix @ simple_solution - right_side):.12e}")
    print(f"Время: {simple_time:.9f} с")
    print("Решение методом Зейделя:\n", seidel_solution)
    print(f"Количество итераций: {seidel_count}")
    print(f"Норма разности соседних приближений: {seidel_difference:.12e}")
    print(f"Норма невязки: {vector_norm_inf(matrix @ seidel_solution - right_side):.12e}")
    print(f"Время: {seidel_time:.9f} с")
    if simple_count < seidel_count:
        faster = "метод простых итераций"
    elif seidel_count < simple_count:
        faster = "метод Зейделя"
    else:
        faster = "оба метода"
    print(f"Быстрее сходится: {faster}")


if __name__ == "__main__":
    main()
