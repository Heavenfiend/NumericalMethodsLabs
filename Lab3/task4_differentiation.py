import math
import sys
from pathlib import Path


def read_data(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    point = float(lines[0])
    x, y = [list(map(float, line.split())) for line in lines[1:]]
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Необходимы хотя бы три узла и соответствующие значения.")
    if any(not math.isfinite(t) for t in x + y + [point]):
        raise ValueError("Все данные должны быть конечными.")
    if any(x[i] >= x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("Узлы должны строго возрастать.")
    if not x[0] <= point <= x[-1]:
        raise ValueError("Точка находится за пределами таблицы.")
    return point, x, y


def choose_nodes(x, point):
    nearest = min(range(len(x)), key=lambda i: abs(x[i] - point))
    start = max(0, min(nearest - 1, len(x) - 3))
    return start


def derivatives(x, y, point):
    first_slope = (y[1] - y[0]) / (x[1] - x[0])
    second_slope = (y[2] - y[1]) / (x[2] - x[1])
    second_difference = (second_slope - first_slope) / (x[2] - x[0])
    first = first_slope + second_difference * (2 * point - x[0] - x[1])
    second = 2 * second_difference
    return first, second, first_slope, second_slope, second_difference


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    point, x, y = read_data(Path(__file__).resolve().parent / "input" / "task3_4.txt")
    print("Задание 3.4. Вариант 10. Численное дифференцирование")
    print(f"{'Номер':>7} {'Узел':>12} {'Значение':>16}")
    for i, (t, value) in enumerate(zip(x, y)):
        print(f"{i:7d} {t:12.6f} {value:16.8f}")
    print(f"Точка вычисления: {point:g}")
    start = choose_nodes(x, point)
    selected_x, selected_y = x[start:start + 3], y[start:start + 3]
    print("Использованы три соседних узла; номера отсчитываются от нуля:")
    for i in range(start, start + 3):
        print(f"Узел {i}: аргумент = {x[i]:g}, значение = {y[i]:.8f}")
    first, second, left, right, difference = derivatives(selected_x, selected_y, point)
    print(f"Наклон на левом отрезке:  {left:.12f}")
    print(f"Наклон на правом отрезке: {right:.12f}")
    print(f"Разделённая разность второго порядка: {difference:.12f}")
    print(f"Первая производная: {first:.12f}")
    print(f"Вторая производная: {second:.12f}")
    print("Производные вычислены по локальному интерполяционному многочлену второй степени.")


if __name__ == "__main__":
    main()
