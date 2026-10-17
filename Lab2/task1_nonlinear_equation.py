import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_config(path):
    tokens = path.read_text(encoding="utf-8").split()
    epsilon = float(tokens[0])
    max_iterations = int(tokens[1]) if len(tokens) > 1 else 1000
    return epsilon, max_iterations


def f(x):
    return float(np.sin(x) - 2.0 * x * x + 0.5)


def df(x):
    return float(np.cos(x) - 4.0 * x)


def ddf(x):
    return float(-np.sin(x) - 4.0)


def phi(x):
    return float(np.sqrt((np.sin(x) + 0.5) / 2.0))


def dphi(x):
    return float(np.cos(x) / (2.0 * np.sqrt(2.0 * (np.sin(x) + 0.5))))


def simple_iteration(x0, epsilon, max_iterations, q):
    history = []
    previous = x0
    scale = q / (1.0 - q) if q < 0.5 else 1.0
    for iteration in range(1, max_iterations + 1):
        current = phi(previous)
        diff = abs(current - previous)
        error_est = scale * diff
        history.append((iteration, current, diff, error_est, abs(f(current))))
        if error_est <= epsilon and diff <= epsilon:
            return current, history
        previous = current
    raise RuntimeError("Метод простой итерации не сошёлся за заданное число шагов")


def newton_method(x0, epsilon, max_iterations):
    history = []
    previous = x0
    for iteration in range(1, max_iterations + 1):
        derivative = df(previous)
        if abs(derivative) < 1e-15:
            raise ZeroDivisionError("Производная близка к нулю")
        current = previous - f(previous) / derivative
        diff = abs(current - previous)
        history.append((iteration, current, diff, abs(f(current))))
        if diff <= epsilon:
            return current, history
        previous = current
    raise RuntimeError("Метод Ньютона не сошёлся за заданное число шагов")


def print_history_simple(history):
    print("Таблица итераций (Метод простой итерации):")
    print(f"{'k':>3} | {'x_k':>16} | {'|x_k - x_{k-1}|':>16} | {'Оценка погр.':>16} | {'|f(x_k)|':>16}")
    print("-" * 79)
    for k, x_val, diff, err, res in history:
        print(f"{k:3d} | {x_val:16.12f} | {diff:16.6e} | {err:16.6e} | {res:16.6e}")


def print_history_newton(history):
    print("Таблица итераций (Метод Ньютона):")
    print(f"{'k':>3} | {'x_k':>16} | {'|x_k - x_{k-1}|':>16} | {'|f(x_k)|':>16}")
    print("-" * 60)
    for k, x_val, diff, res in history:
        print(f"{k:3d} | {x_val:16.12f} | {diff:16.6e} | {res:16.6e}")


def plot_results(root, x0, hist_si, hist_nm, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x_vals = np.linspace(0.0, 1.5, 400)
    y_vals = np.sin(x_vals) - 2.0 * x_vals ** 2 + 0.5
    ax1.plot(x_vals, y_vals, label="f(x) = sin(x) - 2x² + 0.5", color="blue")
    ax1.axhline(0, color="black", linewidth=1, linestyle="--")
    ax1.scatter([x0], [f(x0)], color="orange", zorder=5, label=f"x0 = {x0}")
    ax1.scatter([root], [0.0], color="red", zorder=5, label=f"Корень x* ≈ {root:.6f}")
    ax1.set_title("Графическое отделение корня (2.1)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
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
    ax2.set_ylabel("|x_k - x_{k-1}| (лог. шкала)")
    ax2.grid(True, which="both", linestyle=":")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    input_path = Path(__file__).resolve().parent / "input" / "task2_1.txt"
    epsilon, max_iterations = read_config(input_path)

    a, b = 0.5, 1.0
    x0 = 0.8

    grid = np.linspace(a, b, 500)
    q = max(abs(dphi(val)) for val in grid)
    fourier_check = f(x0) * ddf(x0)

    print("2.1. Решение нелинейного уравнения: sin(x) - 2*x^2 + 0.5 = 0")
    print(f"Заданная точность ε: {epsilon:g}")
    print(f"Отрезок локализации положительного корня: [{a}; {b}]")
    print(f"f({a}) = {f(a):.6f}, f({b}) = {f(b):.6f}")
    print(f"Начальное приближение x0: {x0}")
    print(f"Коэффициент сжатия q = max|phi'(x)| на [{a}; {b}]: {q:.6f} (< 1)")
    print(f"Проверка условия Фурье f(x0)*f''(x0): {fourier_check:.6f} (> 0)\n")

    root_si, hist_si = simple_iteration(x0, epsilon, max_iterations, q)
    print_history_simple(hist_si)
    print(f"Корень (простая итерация): {root_si:.12f}")
    print(f"Количество итераций: {len(hist_si)}")
    print(f"Невязка |f(x)|: {abs(f(root_si)):.6e}\n")

    root_nm, hist_nm = newton_method(x0, epsilon, max_iterations)
    print_history_newton(hist_nm)
    print(f"Корень (метод Ньютона):    {root_nm:.12f}")
    print(f"Количество итераций: {len(hist_nm)}")
    print(f"Невязка |f(x)|: {abs(f(root_nm)):.6e}\n")

    print(f"Разность решений |x_SI - x_Newton|: {abs(root_si - root_nm):.6e}")
    plot_results(root_nm, x0, hist_si, hist_nm, Path(__file__).resolve().parent / "plot_task2_1.png")


if __name__ == "__main__":
    main()
