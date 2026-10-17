import sys
from pathlib import Path

import numpy as np


INPUT_FILE = Path(__file__).resolve().parent / "input" / "lu.txt"


def read_system(path):
    tokens = path.read_text(encoding="utf-8").split()
    n = int(tokens[0])
    expected = 1 + n * n + n
    if len(tokens) != expected:
        raise ValueError(f"Ожидалось {expected} чисел, получено {len(tokens)}")
    values = [float(token) for token in tokens[1:]]
    matrix = np.array(values[:n * n], dtype=float).reshape(n, n)
    right_side = np.array(values[n * n:], dtype=float)
    return matrix, right_side


def matrix_norm_inf(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def vector_norm_inf(vector):
    return max(abs(value) for value in vector)


def lu_decomposition(matrix, tolerance=1e-14):
    n = matrix.shape[0]
    lower = np.eye(n, dtype=float)
    upper = matrix.astype(float)
    permutation = np.eye(n, dtype=float)
    swaps = 0

    for column in range(n):
        pivot_row = column
        for row in range(column + 1, n):
            if abs(upper[row, column]) > abs(upper[pivot_row, column]):
                pivot_row = row
        if abs(upper[pivot_row, column]) <= tolerance:
            raise ValueError("Матрица вырождена")
        if pivot_row != column:
            upper[[column, pivot_row], :] = upper[[pivot_row, column], :]
            permutation[[column, pivot_row], :] = permutation[[pivot_row, column], :]
            if column > 0:
                lower[[column, pivot_row], :column] = lower[[pivot_row, column], :column]
            swaps += 1
        for row in range(column + 1, n):
            factor = upper[row, column] / upper[column, column]
            lower[row, column] = factor
            upper[row, column:] -= factor * upper[column, column:]
            upper[row, column] = 0.0

    return lower, upper, permutation, swaps


def forward_substitution(lower, right_side):
    n = len(right_side)
    solution = np.zeros(n, dtype=float)
    for row in range(n):
        value = right_side[row]
        for column in range(row):
            value -= lower[row, column] * solution[column]
        solution[row] = value / lower[row, row]
    return solution


def backward_substitution(upper, right_side):
    n = len(right_side)
    solution = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        value = right_side[row]
        for column in range(row + 1, n):
            value -= upper[row, column] * solution[column]
        solution[row] = value / upper[row, row]
    return solution


def solve_lu(lower, upper, permutation, right_side):
    intermediate = forward_substitution(lower, permutation @ right_side)
    return backward_substitution(upper, intermediate)


def inverse_from_lu(lower, upper, permutation):
    n = lower.shape[0]
    inverse = np.zeros((n, n), dtype=float)
    for column in range(n):
        basis = np.zeros(n, dtype=float)
        basis[column] = 1.0
        inverse[:, column] = solve_lu(lower, upper, permutation, basis)
    return inverse


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    matrix, right_side = read_system(INPUT_FILE)
    lower, upper, permutation, swaps = lu_decomposition(matrix)
    solution = solve_lu(lower, upper, permutation, right_side)
    inverse = inverse_from_lu(lower, upper, permutation)
    determinant = (-1.0 if swaps % 2 else 1.0) * np.prod(np.diag(upper))
    lower_upper = lower @ upper
    permuted_matrix = permutation @ matrix
    identity_check = matrix @ inverse

    print("1.1. LU-разложение с выбором опорного элемента")
    print("Матрица A:\n", matrix)
    print("Вектор b:\n", right_side)
    print("Матрица перестановок P:\n", permutation)
    print("Матрица L:\n", lower)
    print("Матрица U:\n", upper)
    print("Произведение L · U:\n", lower_upper)
    print("Произведение P · A:\n", permuted_matrix)
    print(f"Проверка L · U = P · A, норма разности: {matrix_norm_inf(lower_upper - permuted_matrix):.12e}")
    print("Решение системы A x = b:\n", solution)
    print(f"Норма невязки ||A x - b||∞: {vector_norm_inf(matrix @ solution - right_side):.12e}")
    print("Обратная матрица A⁻¹:\n", inverse)
    print(f"Определитель det(A): {determinant:.12g}")
    print("Проверка A · A⁻¹:\n", identity_check)
    print(f"Норма отклонения от E: {matrix_norm_inf(identity_check - np.eye(matrix.shape[0])):.12e}")


if __name__ == "__main__":
    main()
