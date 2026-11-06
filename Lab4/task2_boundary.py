import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_data():
    path = Path(__file__).resolve().parent / "input" / "task4_2.txt"
    values = list(map(float, path.read_text(encoding="utf-8").split()))
    a, b, h, left, k, right, guess0, guess1, epsilon, limit = values
    if not all(math.isfinite(t) for t in values):
        raise ValueError("Входные данные должны быть конечными числами.")
    if not 0 < a < b or h <= 0 or epsilon <= 0 or limit < 1 or not limit.is_integer():
        raise ValueError("Требуются 0 < a < b, положительные шаг, точность и целый предел итераций.")
    if guess0 == guess1:
        raise ValueError("Начальные приближения для стрельбы должны различаться.")
    return a, b, h, left, k, right, guess0, guess1, epsilon, int(limit)


def make_grid(a, b, h):
    n = round((b - a) / h)
    if n < 2 or abs(n * h - (b - a)) > 1e-10:
        raise ValueError("Шаг должен делить отрезок хотя бы на две части.")
    return [a + i * h for i in range(n + 1)]


def p(x):
    return -(2 * x + 1) / x


def q(x):
    return (x + 1) / x


def f(x, y, z):
    return -p(x) * z - q(x) * y


def exact_coefficients(a, b, left, k, right):
    first = a * a + 2 * a
    second = (1 - k) * b * b + 2 * b
    denominator = second - (1 - k) * first
    if denominator == 0:
        raise ValueError("Краевые условия не определяют единственное решение.")
    c1 = (right / math.exp(b) - (1 - k) * left / math.exp(a)) / denominator
    c2 = left / math.exp(a) - first * c1
    return c1, c2


def exact(x, c1, c2):
    return (c1 * x * x + c2) * math.exp(x)


def rk4(x, y0, z0):
    h = x[1] - x[0]
    y, z = [y0], [z0]
    for i in range(len(x) - 1):
        t, u, v = x[i], y[i], z[i]
        k1, l1 = h * v, h * f(t, u, v)
        k2 = h * (v + l1 / 2)
        l2 = h * f(t + h / 2, u + k1 / 2, v + l1 / 2)
        k3 = h * (v + l2 / 2)
        l3 = h * f(t + h / 2, u + k2 / 2, v + l2 / 2)
        k4 = h * (v + l3)
        l4 = h * f(t + h, u + k3, v + l3)
        y.append(u + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        z.append(v + (l1 + 2 * l2 + 2 * l3 + l4) / 6)
    return y, z


def shooting(x, left, k, right, guess0, guess1, epsilon, limit):
    y, z = rk4(x, guess0, left)
    residual0 = z[-1] - k * y[-1] - right
    if abs(residual0) <= epsilon:
        return y, z, guess0, residual0
    y, z = rk4(x, guess1, left)
    residual1 = z[-1] - k * y[-1] - right
    if abs(residual1) <= epsilon:
        return y, z, guess1, residual1
    for _ in range(limit):
        denominator = residual1 - residual0
        if denominator == 0:
            raise ValueError("Метод секущих: совпали невязки двух выстрелов.")
        next_guess = guess1 - residual1 * (guess1 - guess0) / denominator
        guess0, guess1 = guess1, next_guess
        residual0 = residual1
        y, z = rk4(x, guess1, left)
        residual1 = z[-1] - k * y[-1] - right
        if abs(residual1) <= epsilon:
            return y, z, guess1, residual1
    raise ValueError("Метод стрельбы не сошёлся за заданное число итераций.")


def thomas(lower, diagonal, upper, rhs):
    n = len(rhs)
    c, d = [0.0] * n, [0.0] * n
    for i in range(n):
        denominator = diagonal[i] - (lower[i] * c[i - 1] if i else 0)
        if denominator == 0:
            raise ValueError("Нулевой знаменатель в методе прогонки.")
        c[i] = upper[i] / denominator
        d[i] = (rhs[i] - (lower[i] * d[i - 1] if i else 0)) / denominator
    y = [0.0] * n
    y[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        y[i] = d[i] - c[i] * y[i + 1]
    return y


def finite_differences(x, left, k, right):
    h = x[1] - x[0]
    n = len(x)
    lower, diagonal, upper, rhs = ([0.0] * n for _ in range(4))
    for i in range(1, n - 1):
        lower[i] = 1 - h * p(x[i]) / 2
        diagonal[i] = -2 + h * h * q(x[i])
        upper[i] = 1 + h * p(x[i]) / 2
    diagonal[0] = -2 + h * h * q(x[0])
    upper[0] = 2
    rhs[0] = (2 * h - h * h * p(x[0])) * left
    lower[-1] = 2
    diagonal[-1] = -2 + 2 * h * k + h * h * (p(x[-1]) * k + q(x[-1]))
    rhs[-1] = -(2 * h + h * h * p(x[-1])) * right
    return thomas(lower, diagonal, upper, rhs)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a, b, h, left, k, right, guess0, guess1, epsilon, limit = read_data()
    c1, c2 = exact_coefficients(a, b, left, k, right)
    x = make_grid(a, b, h)
    fine_x = make_grid(a, b, h / 2)
    print("Задание 4.2. Вариант 10")
    print("Уравнение: x·y″ − (2x + 1)·y′ + (x + 1)·y = 0")
    print(f"Отрезок: [{a:g}; {b:g}], шаги: {h:g} и {h / 2:g}")
    print(f"Краевые условия: y′({a:g}) = {left:.12f}; y′({b:g}) − {k:g}·y({b:g}) = {right:g}")
    print(f"Точное решение: y = ({c1:.12g}·x² + ({c2:.12g}))·exp(x)")
    coarse_y, _, initial, residual = shooting(x, left, k, right, guess0, guess1, epsilon, limit)
    fine_y, _, fine_initial, fine_residual = shooting(fine_x, left, k, right, guess0, guess1, epsilon, limit)
    print(f"Стрельба, шаг h: найденное y(a) = {initial:.12f}; невязка справа = {residual:.6e}")
    print(f"Стрельба, шаг h/2: найденное y(a) = {fine_initial:.12f}; невязка справа = {fine_residual:.6e}")
    print("В разностном методе краевые производные аппроксимируются со вторым порядком через фиктивные узлы.")
    results = [
        ("Метод стрельбы", 4, coarse_y, fine_y),
        ("Конечно-разностный метод", 2, finite_differences(x, left, k, right), finite_differences(fine_x, left, k, right)),
    ]
    figure, axis = plt.subplots(figsize=(10, 6))
    for name, order, y, fine_y in results:
        print(f"\n{name}, порядок: {order}")
        print(f"{'Узел':>8} {'Решение, h':>15} {'Решение, h/2':>16} {'Точное':>15} {'Ошибка, h':>14} {'Ошибка, h/2':>15} {'Рунге, h/2':>15}")
        errors, estimates = [], []
        for i, t in enumerate(x):
            value = exact(t, c1, c2)
            error = abs(y[i] - value)
            fine_error = abs(fine_y[2 * i] - value)
            estimate = abs(fine_y[2 * i] - y[i]) / (2 ** order - 1)
            errors.append(error)
            estimates.append(estimate)
            print(f"{t:8.3f} {y[i]:15.10f} {fine_y[2 * i]:16.10f} {value:15.10f} {error:14.6e} {fine_error:15.6e} {estimate:15.6e}")
        fine_errors = [abs(value - exact(t, c1, c2)) for t, value in zip(fine_x, fine_y)]
        print(f"Максимальная абсолютная ошибка на сетке h: {max(errors):.8e}")
        print(f"Максимальная абсолютная ошибка на всей сетке h/2: {max(fine_errors):.8e}")
        print(f"Максимальная оценка Рунге–Ромберга на общих узлах для h/2: {max(estimates):.8e}")
        axis.plot(x, y, "o", label=f"{name}, шаг {h:g}")
        axis.plot(fine_x, fine_y, ".-", label=f"{name}, шаг {h / 2:g}")
    exact_x = [a + (b - a) * i / 500 for i in range(501)]
    axis.plot(exact_x, [exact(t, c1, c2) for t in exact_x], color="black", label="Точное решение")
    axis.set(title="Краевая задача — вариант 10", xlabel="Аргумент x", ylabel="Решение y")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(__file__).resolve().parent / "plot_task4_2.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
