import math
import sys
from pathlib import Path


def read_data(path):
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    a, b = [float(val) for val in lines[0].split()]
    h1, h2 = [float(val) for val in lines[1].split()]
    parameter = float(lines[2])

    return a, b, h1, h2, parameter

# функция, интеграл которой нужно найти
def f(x, parameter):
    return (x * x) / (x * x + parameter)

# первообразная для вычисления точного значения интеграла
def antiderivative(x, parameter):
    scale = math.sqrt(parameter)
    return x - scale * math.atan(x / scale)

# считаем, на сколько частей разбивается отрезок
def interval_count(a, b, h):
    return round((b - a) / h)

# метод левых прямоугольников берём значения в начале отрезков
def left_rectangles(a, b, h, parameter):
    n = interval_count(a, b, h)
    total = 0.0
    for i in range(n):
        x = a + i * h
        total += f(x, parameter)
    return h * total

# метод правых прямоугольников берём значения в конце отрезков
def right_rectangles(a, b, h, parameter):
    n = interval_count(a, b, h)
    total = 0.0
    for i in range(1, n + 1):
        x = a + i * h
        total += f(x, parameter)
    return h * total

# метод средних прямоугольников берём середину каждого отрезка
def midpoint_rectangles(a, b, h, parameter):
    n = interval_count(a, b, h)
    total = 0.0
    for i in range(n):
        x = a + (i + 0.5) * h
        total += f(x, parameter)
    return h * total

# метод трапеций учитываем значения на концах отрезков
def trapezoids(a, b, h, parameter):
    n = interval_count(a, b, h)
    total = (f(a, parameter) + f(b, parameter)) / 2.0
    for i in range(1, n):
        x = a + i * h
        total += f(x, parameter)
    return h * total

# метод симпсона
def simpson(a, b, h, parameter):
    n = interval_count(a, b, h)
    total = f(a, parameter) + f(b, parameter)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 1:
            total += 4.0 * f(x, parameter)
        else:
            total += 2.0 * f(x, parameter)
    return h * total / 3.0

# метод Рунге  Ромберга уточняем интеграл R = I2-I1/r^n - 1
def runge_romberg(coarse, fine, ratio, order):
    correction = (fine - coarse) / (ratio ** order - 1.0)
    refined = fine + correction
    return refined, abs(correction)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(__file__).resolve().parent / "input" / "task3_5.txt"
    a, b, h1, h2, parameter = read_data(input_path)

    methods = [
        ("Левые прямоугольники", 1, left_rectangles),
        ("Правые прямоугольники", 1, right_rectangles),
        ("Средние прямоугольники", 2, midpoint_rectangles),
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
    ratio = h1 / h2

    for name, order, method_func in methods:
        coarse = method_func(a, b, h1, parameter)
        fine = method_func(a, b, h2, parameter)
        refined, estimate = runge_romberg(coarse, fine, ratio, order)

        err_coarse = abs(coarse - exact)
        err_fine = abs(fine - exact)
        err_refined = abs(refined - exact)
        results.append((name, err_coarse, err_fine, err_refined))

        print(f"{name:<24} {coarse:16.12f} {fine:16.12f} {order:8d} {refined:16.12f} {estimate:16.6e}")

    print(f"\n{'Метод':<24} {'Ошибка для h₁':>18} {'Ошибка для h₂':>18} {'После уточнения':>18}")
    for name, coarse_error, fine_error, refined_error in results:
        print(f"{name:<24} {coarse_error:18.10e} {fine_error:18.10e} {refined_error:18.10e}")

    best_fine_name = results[0][0]
    min_fine_err = results[0][2]
    best_refined_name = results[0][0]
    min_refined_err = results[0][3]

    for name, coarse_error, fine_error, refined_error in results:
        if fine_error < min_fine_err:
            min_fine_err = fine_error
            best_fine_name = name
        if refined_error < min_refined_err:
            min_refined_err = refined_error
            best_refined_name = name

    print(f"Наиболее точный метод с шагом h₂: {best_fine_name}.")
    print(f"Наиболее точный метод после уточнения: {best_refined_name}.")
    print("Оценка Рунге–Ромберга относится к исходному результату с шагом h₂.")
    print("Это асимптотическая оценка, а не гарантированная верхняя граница ошибки.")


if __name__ == "__main__":
    main()

