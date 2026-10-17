import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_config(path):
    tokens = path.read_text(encoding="utf-8").split()
    epsilon = float(tokens[0])
    max_iterations = int(tokens[1]) if len(tokens) > 1 else 1000
    return epsilon, max_iterations


def vector_norm_inf(vec):
    return float(max(abs(float(vec[0])), abs(float(vec[1]))))


def system_f(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        x1 - np.cos(x2) - 1.0,
        x2 - np.sin(x1) - 1.0
    ], dtype=float)


def phi_vector(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        1.0 + np.cos(x2),
        1.0 + np.sin(x1)
    ], dtype=float)


def phi_jacobian_norm_inf(x):
    x1, x2 = float(x[0]), float(x[1])
    row1 = abs(-np.sin(x2))
    row2 = abs(np.cos(x1))
    return float(max(row1, row2))


def solve_2x2_cramer(matrix, rhs):
    a11, a12 = float(matrix[0, 0]), float(matrix[0, 1])
    a21, a22 = float(matrix[1, 0]), float(matrix[1, 1])
    b1, b2 = float(rhs[0]), float(rhs[1])
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        raise ZeroDivisionError("Матрица Якоби вырождена")
    dx1 = (b1 * a22 - b2 * a12) / det
    dx2 = (a11 * b2 - a21 * b1) / det
    return np.array([dx1, dx2], dtype=float)


def jacobian_f(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        [1.0, float(np.sin(x2))],
        [float(-np.cos(x1)), 1.0]
    ], dtype=float)


def simple_iteration_system(x0, epsilon, max_iterations):
    history = []
    previous = x0.copy()
    for iteration in range(1, max_iterations + 1):
        current = phi_vector(previous)
        diff = vector_norm_inf(current - previous)
        res = vector_norm_inf(system_f(current))
        history.append((iteration, current.copy(), diff, res))
        if diff <= epsilon:
            return current, history
        previous = current
    raise RuntimeError("Метод простой итерации для системы не сошёлся")


def newton_system(x0, epsilon, max_iterations):
    history = []
    previous = x0.copy()
    for iteration in range(1, max_iterations + 1):
        jac = jacobian_f(previous)
        rhs = -system_f(previous)
        delta = solve_2x2_cramer(jac, rhs)
        current = previous + delta
        diff = vector_norm_inf(delta)
        res = vector_norm_inf(system_f(current))
        history.append((iteration, current.copy(), diff, res))
        if diff <= epsilon:
            return current, history
        previous = current
    raise RuntimeError("Метод Ньютона для системы не сошёлся")


def print_system_history(title, history):
    print(title)
    print(f"{'k':>3} | {'x1_k':>15} | {'x2_k':>15} | {'||X_k - X_{k-1}||∞':>18} | {'||F(X_k)||∞':>15}")
    print("-" * 78)
    for k, vec, diff, res in history:
        print(f"{k:3d} | {vec[0]:15.12f} | {vec[1]:15.12f} | {diff:18.6e} | {res:15.6e}")


def plot_system_results(sol, x0, hist_si, hist_nm, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x2_grid = np.linspace(0.0, 2.5, 400)
    x1_curve1 = 1.0 + np.cos(x2_grid)
    x1_grid = np.linspace(0.0, 2.5, 400)
    x2_curve2 = 1.0 + np.sin(x1_grid)
    ax1.plot(x1_curve1, x2_grid, label="x1 - cos(x2) = 1", color="blue")
    ax1.plot(x1_grid, x2_curve2, label="x2 - sin(x1) = 1", color="green")
    ax1.scatter([x0[0]], [x0[1]], color="orange", zorder=5, label=f"X0 = ({x0[0]}, {x0[1]})")
    ax1.scatter([sol[0]], [sol[1]], color="red", zorder=5, label=f"X* ≈ ({sol[0]:.4f}, {sol[1]:.4f})")
    ax1.set_title("Графическое отделение корней системы (2.2)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_xlim(0.0, 2.5)
    ax1.set_ylim(0.0, 2.5)
    ax1.grid(True, linestyle=":")
    ax1.legend()

    k_si = [item[0] for item in hist_si]
    err_si = [item[2] for item in hist_si]
    k_nm = [item[0] for item in hist_nm]
    err_nm = [item[2] for item in hist_nm]
    ax2.semilogy(k_si, err_si, marker="o", label="Простая итерация")
    ax2.semilogy(k_nm, err_nm, marker="s", label="Метод Ньютона")
    ax2.set_title("Сходимость методов")
    ax2.set_xlabel("Номер итерации k")
    ax2.set_ylabel("||X_k - X_{k-1}||∞ (лог. шкала)")
    ax2.grid(True, which="both", linestyle=":")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    input_path = Path(__file__).resolve().parent / "input" / "task2_2.txt"
    epsilon, max_iterations = read_config(input_path)

    x0 = np.array([0.85, 1.75], dtype=float)
    q_init = phi_jacobian_norm_inf(x0)

    print("2.2. Решение системы нелинейных уравнений (вариант 10, a = 1):")
    print("  x1 - cos(x2) = 1")
    print("  x2 - sin(x1) = 1")
    print(f"Заданная точность ε: {epsilon:g}")
    print(f"Начальное приближение X0: [{x0[0]}, {x0[1]}]")
    print(f"Норма матрицы Якоби отображения ||Phi'(X0)||∞: {q_init:.6f}\n")

    sol_si, hist_si = simple_iteration_system(x0, epsilon, max_iterations)
    print_system_history("Таблица итераций (Метод простой итерации):", hist_si)
    print(f"Решение (простая итерация): x1 = {sol_si[0]:.12f}, x2 = {sol_si[1]:.12f}")
    print(f"Количество итераций: {len(hist_si)}")
    print(f"Норма невязки ||F(X)||∞: {vector_norm_inf(system_f(sol_si)):.6e}\n")

    sol_nm, hist_nm = newton_system(x0, epsilon, max_iterations)
    print_system_history("Таблица итераций (Метод Ньютона):", hist_nm)
    print(f"Решение (метод Ньютона):    x1 = {sol_nm[0]:.12f}, x2 = {sol_nm[1]:.12f}")
    print(f"Количество итераций: {len(hist_nm)}")
    print(f"Норма невязки ||F(X)||∞: {vector_norm_inf(system_f(sol_nm)):.6e}\n")

    print(f"Норма разности решений: {vector_norm_inf(sol_si - sol_nm):.6e}")
    plot_system_results(sol_nm, x0, hist_si, hist_nm, Path(__file__).resolve().parent / "plot_task2_2.png")


if __name__ == "__main__":
    main()
