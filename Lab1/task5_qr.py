import cmath
import math
import sys
from pathlib import Path

import numpy as np


INPUT_FILE = Path(__file__).resolve().parent / "input" / "qr.txt"


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


def squared_length(vector):
    return float(sum(abs(value) ** 2 for value in vector))


def qr_decomposition(matrix, tolerance=1e-15):
    rows, columns = matrix.shape
    value_type = complex if np.iscomplexobj(matrix) else float
    upper = matrix.astype(value_type, copy=True)
    orthogonal = np.eye(rows, dtype=value_type)

    for column in range(min(rows, columns)):
        vector = upper[column:, column].copy()
        length = math.sqrt(squared_length(vector))
        if length <= tolerance:
            continue
        first = vector[0]
        phase = first / abs(first) if abs(first) > tolerance else 1.0
        vector[0] += phase * length
        divisor = squared_length(vector) # для v^T * v
        if divisor <= tolerance:
            continue
        upper[column:, column:] -= (2.0 / divisor) * np.outer( # HU = (E - 2 * (v @ v^T) v^T @ v) * U т.е. накапливаем
            vector, np.conjugate(vector) @ upper[column:, column:]
        )
        orthogonal[:, column:] -= (2.0 / divisor) * np.outer( # накапливаем Q матрицу Q <- QH
            orthogonal[:, column:] @ vector, np.conjugate(vector)
        )
    return orthogonal, upper


def hessenberg_reduction(matrix, tolerance=1e-15):
    hessenberg = matrix.astype(complex, copy=True)
    n = matrix.shape[0]
    for column in range(n - 2):
        vector = hessenberg[column + 1:, column].copy()
        length = math.sqrt(squared_length(vector)) # как в QR
        if length <= tolerance:
            continue
        first = vector[0]
        phase = first / abs(first) if abs(first) > tolerance else 1.0
        vector[0] += phase * length
        divisor = squared_length(vector)
        if divisor <= tolerance:
            continue
        hessenberg[column + 1:, column:] -= (2.0 / divisor) * np.outer( # A <- HA
            vector, np.conjugate(vector) @ hessenberg[column + 1:, column:]
        )
        hessenberg[:, column + 1:] -= (2.0 / divisor) * np.outer( # HA <- HAH
            hessenberg[:, column + 1:] @ vector, np.conjugate(vector)
        )
        hessenberg[column + 2:, column] = 0.0
    return hessenberg


def eigenvalues_two_by_two(block): # квадратное ур блока 2х2
    trace = block[0, 0] + block[1, 1]
    determinant = block[0, 0] * block[1, 1] - block[0, 1] * block[1, 0]
    root = cmath.sqrt(trace * trace - 4.0 * determinant)
    return (trace + root) / 2.0, (trace - root) / 2.0


def small_subdiagonal(matrix, row, epsilon): # элемент под диаг около 0 ? ?
    scale = 1.0 + abs(matrix[row - 1, row - 1]) + abs(matrix[row, row])
    return abs(matrix[row, row - 1]) <= epsilon * scale


def qr_eigenvalues(matrix, epsilon, max_iterations):
    hessenberg = hessenberg_reduction(matrix)
    active = matrix.shape[0] # размер не обработанной части
    eigenvalues = []
    iterations = 0

    while active > 0:
        if active == 1:
            eigenvalues.append(hessenberg[0, 0])
            active -= 1
        elif small_subdiagonal(hessenberg, active - 1, epsilon):
            hessenberg[active - 1, active - 2] = 0.0
            eigenvalues.append(hessenberg[active - 1, active - 1])
            active -= 1
        elif active == 2 or small_subdiagonal(hessenberg, active - 2, epsilon): # либо ост 2х2 либо блок 2х2 отделился от осталльной матицы
            if active > 2:
                hessenberg[active - 2, active - 3] = 0.0
            eigenvalues.extend(
                eigenvalues_two_by_two(hessenberg[active - 2:active, active - 2:active])
            )
            active -= 2
        else:
            if iterations >= max_iterations:
                raise RuntimeError("QR-алгоритм не сошёлся")
            pair = eigenvalues_two_by_two(
                hessenberg[active - 2:active, active - 2:active]
            )
            bottom_right = hessenberg[active - 1, active - 1]
            shift = pair[0] if abs(pair[0] - bottom_right) < abs(pair[1] - bottom_right) else pair[1]
            identity = np.eye(active, dtype=complex)
            orthogonal, upper = qr_decomposition( # A - мю * E = QR
                hessenberg[:active, :active] - shift * identity
            )
            hessenberg[:active, :active] = upper @ orthogonal + shift * identity # A = RQ + мю * E
            for row in range(active):
                for column in range(row - 1):
                    hessenberg[row, column] = 0.0 # убираем остатки округления под поддиагональю
            iterations += 1

    result = np.array(eigenvalues, dtype=complex)
    order = np.lexsort((result.imag, result.real))
    return result[order], iterations


def format_eigenvalue(value):
    if abs(value.imag) < 1e-10:
        return f"{value.real:.12g}"
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{value.real:.12g} {sign} {abs(value.imag):.12g}i"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    matrix, epsilon, max_iterations = read_task(INPUT_FILE)
    orthogonal, upper = qr_decomposition(matrix)
    eigenvalues, iterations = qr_eigenvalues(matrix, epsilon, max_iterations)
    product = orthogonal @ upper

    print("1.5. QR-разложение и QR-алгоритм")
    print("Исходная матрица A:\n", matrix)
    print("Матрица Q:\n", orthogonal)
    print("Матрица R:\n", upper)
    print("Произведение Q · R:\n", product)
    print(f"Норма разности ||Q · R - A||∞: {matrix_norm_inf(product - matrix):.12e}")
    print(f"Количество QR-итераций: {iterations}")
    print("Найденные собственные значения:")
    for index, value in enumerate(eigenvalues, start=1):
        print(f"λ_{index} = {format_eigenvalue(value)}")
    complex_values = [value for value in eigenvalues if abs(value.imag) >= 1e-10]
    if len(complex_values) >= 2:
        print(
            "Комплексно-сопряжённая пара: "
            + ", ".join(format_eigenvalue(value) for value in complex_values)
        )


if __name__ == "__main__":
    main()
