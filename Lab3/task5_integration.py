import math
import sys
from pathlib import Path


def read_data(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    a, b = map(float, lines[0].split())
    h1, h2 = map(float, lines[1].split())
    parameter = float(lines[2])
    if not all(math.isfinite(t) for t in (a, b, h1, h2, parameter)):
        raise ValueError("Все параметры должны быть конечными.")
    if not a < b or not 0 < h2 < h1:
        raise ValueError("Требуются возрастающие границы и шаги: 0 < h₂ < h₁.")
    if parameter <= 0:
        raise ValueError("Параметр в знаменателе должен быть положительным.")
    ratio = h1 / h2
    if abs(ratio - round(ratio)) > 1e-10:
        raise ValueError("Отношение шагов должно быть целым для вложенных сеток.")
    return a, b, h1, h2, parameter


def f(x, parameter):
    return x * x / (x * x + parameter)


def antiderivative(x, parameter):
    scale = math.sqrt(parameter)
    return x - scale * math.atan(x / scale)


def interval_count(a, b, h):
    count = round((b - a) / h)
    if count < 1 or abs(count * h - (b - a)) > 1e-10 * max(1.0, abs(b - a)):
        raise ValueError("Длина отрезка должна быть кратна шагу.")
    return count


def rectangles(function, a, b, h, offset):
    count = interval_count(a, b, h)
    total = 0.0
    for i in range(count):
        total += function(a + (i + offset) * h)
    return h * total


def trapezoids(function, a, b, h):
    count = interval_count(a, b, h)
    total = (function(a) + function(b)) / 2
    for i in range(1, count):
        total += function(a + i * h)
    return h * total


def simpson(function, a, b, h):
    count = interval_count(a, b, h)
    if count % 2:
        raise ValueError("Для метода Симпсона требуется чётное число отрезков.")
    total = function(a) + function(b)
    for i in range(1, count):
        total += (4 if i % 2 else 2) * function(a + i * h)
    return h * total / 3


def runge_romberg(coarse, fine, ratio, order):
    correction = (fine - coarse) / (ratio ** order - 1)
    return fine + correction, abs(correction)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a, b, h1, h2, parameter = read_data(Path(__file__).resolve().parent / "input" / "task3_5.txt")
    methods = [
        ("Левые прямоугольники", 1, lambda g, a, b, h: rectangles(g, a, b, h, 0.0)),
        ("Правые прямоугольники", 1, lambda g, a, b, h: rectangles(g, a, b, h, 1.0)),
        ("Средние прямоугольники", 2, lambda g, a, b, h: rectangles(g, a, b, h, 0.5)),
        ("Трапеции", 2, trapezoids),
        ("Симпсон", 4, simpson),
    ]
    exact = antiderivative(b, parameter) - antiderivative(a, parameter)
    print("Задание 3.5. Вариант 10. Численное интегрирование")
    print(f"Подынтегральная функция: x² / (x² + {parameter:g})")
    print(f"Границы: [{a:g}; {b:g}]; шаг h₁ = {h1:g}; шаг h₂ = {h2:g}")
    print(f"Число отрезков: {interval_count(a, b, h1)} и {interval_count(a, b, h2)}")
    scale = math.sqrt(parameter)
    print(f"Первообразная: x − {scale:g}·арктангенс(x / {scale:g})")
    print(f"Точное значение интеграла: {exact:.15f}")
    print(f"\n{'Метод':<24} {'Шаг h₁':>16} {'Шаг h₂':>16} {'Порядок':>8} {'Уточнение':>16} {'Оценка для h₂':>16}")
    results = []
    for name, order, method in methods:
        function = lambda t: f(t, parameter)
        coarse = method(function, a, b, h1)
        fine = method(function, a, b, h2)
        refined, estimate = runge_romberg(coarse, fine, h1 / h2, order)
        results.append((name, abs(coarse - exact), abs(fine - exact), abs(refined - exact)))
        print(f"{name:<24} {coarse:16.12f} {fine:16.12f} {order:8d} {refined:16.12f} {estimate:16.6e}")
    print(f"\n{'Метод':<24} {'Ошибка для h₁':>18} {'Ошибка для h₂':>18} {'После уточнения':>18}")
    for name, coarse_error, fine_error, refined_error in results:
        print(f"{name:<24} {coarse_error:18.10e} {fine_error:18.10e} {refined_error:18.10e}")
    best_fine = min(results, key=lambda row: row[2])
    best_refined = min(results, key=lambda row: row[3])
    print(f"Наиболее точный метод с шагом h₂: {best_fine[0]}.")
    print(f"Наиболее точный метод после уточнения: {best_refined[0]}.")
    print("Оценка Рунге–Ромберга относится к исходному результату с шагом h₂.")
    print("Это асимптотическая оценка, а не гарантированная верхняя граница ошибки.")


if __name__ == "__main__":
    main()
