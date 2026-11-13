import sys
from pathlib import Path


def read_data(path):
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    point = float(lines[0])
    x = [float(val) for val in lines[1].split()]
    y = [float(val) for val in lines[2].split()]

    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Необходимы хотя бы три узла и соответствующие значения.")

    return point, x, y

# выбираем три соседних узла для вычисления производных
def choose_nodes(x, point):
    n = len(x)
    start = 0
    for i in range(n - 1):
        if x[i] <= point <= x[i + 1]:
            start = i
            break

    if start > n - 3:
        start = n - 3

    return start

# находим разделённые разности и по ним вычисляем обе производные
def derivatives(x, y, point):
    x0, x1, x2 = x[0], x[1], x[2]
    y0, y1, y2 = y[0], y[1], y[2]

    first_slope = (y1 - y0) / (x1 - x0)
    second_slope = (y2 - y1) / (x2 - x1)
    second_difference = (second_slope - first_slope) / (x2 - x0)

    first = first_slope + second_difference * (2.0 * point - x0 - x1)
    second = 2.0 * second_difference

    return first, second, first_slope, second_slope, second_difference


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = Path(__file__).resolve().parent / "input" / "task3_4.txt"
    point, x, y = read_data(input_path)

    print("Задание 3.4. Вариант 10. Численное дифференцирование")
    print(f"{'Номер':>7} {'Узел':>12} {'Значение':>16}")
    for i in range(len(x)):
        print(f"{i:7d} {x[i]:12.6f} {y[i]:16.8f}")
    print(f"Точка вычисления: {point:g}")

    start = choose_nodes(x, point)
    selected_x = x[start : start + 3]
    selected_y = y[start : start + 3]

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

