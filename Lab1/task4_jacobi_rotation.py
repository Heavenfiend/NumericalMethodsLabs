import math
import sys
from pathlib import Path

import numpy as np


INPUT_FILE = Path(__file__).resolve().parent / "input" / "jacobi_rotation.txt"


def read_task(path):
    tokens = path.read_text(encoding="utf-8").split()
    n = int(tokens[0])
    expected = 1 + n * n + 2
    if len(tokens) != expected:
        raise ValueError(f"Ожидалось {expected} чисел, получено {len(tokens)}")
    values = [float(token) for token in tokens[1:]]
    matrix = np.array(values[:n * n], dtype=float).reshape(n, n)
    epsilon = values[-2]
    max_iterations = int(values[-1])
    return matrix, epsilon, max_iterations


def matrix_norm_inf(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def largest_off_diagonal(matrix):
    n = matrix.shape[0]
    if n < 2:
        return 0, 0, 0.0
    selected_row = 0
    selected_column = 1
    maximum = abs(matrix[0, 1])
    for row in range(n):
        for column in range(row + 1, n):
            if abs(matrix[row, column]) > maximum:
                maximum = abs(matrix[row, column])
                selected_row = row
                selected_column = column
    return selected_row, selected_column, maximum


def jacobi_rotation(matrix, epsilon, max_iterations):
    transformed = matrix.astype(float, copy=True)
    n = matrix.shape[0]
    if matrix_norm_inf(transformed - transformed.T) > 1e-12:
        raise ValueError("Матрица должна быть симметричной")
    eigenvectors = np.eye(n, dtype=float)

    for iteration in range(max_iterations + 1):
        row, column, maximum = largest_off_diagonal(transformed)
        if maximum <= epsilon:
            eigenvalues = np.diag(transformed).copy()
            order = np.argsort(eigenvalues)
            return eigenvalues[order], eigenvectors[:, order], iteration
        if iteration == max_iterations:
            break
        angle = 0.5 * math.atan2(
            2.0 * transformed[row, column],
            transformed[row, row] - transformed[column, column],
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.eye(n, dtype=float)
        rotation[row, row] = cosine
        rotation[column, column] = cosine
        rotation[row, column] = -sine
        rotation[column, row] = sine
        transformed = rotation.T @ transformed @ rotation
        transformed = 0.5 * (transformed + transformed.T) # принудительно возвращаем идеальную симметричность
        eigenvectors = eigenvectors @ rotation
    raise RuntimeError("Метод вращений Якоби не сошёлся")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    matrix, epsilon, max_iterations = read_task(INPUT_FILE)
    eigenvalues, eigenvectors, iterations = jacobi_rotation(
        matrix, epsilon, max_iterations
    )
    diagonal = np.diag(eigenvalues)
    left = matrix @ eigenvectors
    right = eigenvectors @ diagonal

    print("1.4. Метод вращений Якоби")
    print(f"Заданная точность ε: {epsilon:.12g}")
    print(f"Количество вращений: {iterations}")
    print("Найденные собственные значения:\n", eigenvalues)
    for index in range(eigenvectors.shape[1]):
        print(f"Собственный вектор v_{index + 1}:\n", eigenvectors[:, index])
    print("Матрица собственных векторов V:\n", eigenvectors)
    print("Левая часть A · V:\n", left)
    print("Правая часть V · Λ:\n", right)
    print(f"Проверка A · V = V · Λ, норма разности: {matrix_norm_inf(left - right):.12e}")


if __name__ == "__main__":
    main()
