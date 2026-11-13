import java.io.File
import java.io.PrintStream
import java.util.Locale
import kotlin.math.abs

private data class Derivatives(
    val first: Double,
    val second: Double,
    val leftSlope: Double,
    val rightSlope: Double,
    val secondDifference: Double
)

private fun numbers(line: String) = line.trim().split(Regex("\\s+")).map(String::toDouble)

private fun chooseNodes(x: List<Double>, point: Double): Int {
    val nearest = x.indices.minByOrNull { abs(x[it] - point) }!!
    return (nearest - 1).coerceIn(0, x.size - 3)
}

private fun derivatives(x: List<Double>, y: List<Double>, point: Double): Derivatives {
    val (x0, x1, x2) = x
    val (y0, y1, y2) = y
    val left = (y1 - y0) / (x1 - x0)
    val right = (y2 - y1) / (x2 - x1)
    val difference = (right - left) / (x2 - x0)
    val first = left + difference * (2 * point - x0 - x1)
    return Derivatives(first, 2 * difference, left, right, difference)
}

fun main() {
    System.setOut(PrintStream(System.out, true, Charsets.UTF_8))
    val input = File("input/task3_4.txt").takeIf { it.isFile } ?: File("Lab3/input/task3_4.txt")
    val lines = input.readLines()
    val point = lines[0].trim().toDouble()
    val x = numbers(lines[1])
    val y = numbers(lines[2])
    require(x.size == y.size && x.size >= 3) { "Нужны хотя бы три узла и соответствующие значения." }
    require((x + y + point).all { it.isFinite() }) { "Все данные должны быть конечными." }
    require(x.zipWithNext().all { (a, b) -> a < b }) { "Узлы должны строго возрастать." }
    require(point in x.first()..x.last()) { "Точка находится за пределами таблицы." }

    println("Задание 3.4. Вариант 10. Численное дифференцирование")
    System.out.printf(Locale.ROOT, "%7s %12s %16s%n", "Номер", "Узел", "Значение")
    for (i in x.indices) {
        System.out.printf(Locale.ROOT, "%7d %12.6f %16.8f%n", i, x[i], y[i])
    }

    println("Точка вычисления: $point")
    val start = chooseNodes(x, point)
    val indices = start until start + 3
    println("Использованы три соседних узла; номера отсчитываются от нуля:")
    for (i in indices) {
        System.out.printf(Locale.ROOT, "Узел %d: аргумент = %s, значение = %.8f%n", i, x[i], y[i])
    }

    val result = derivatives(x.slice(indices), y.slice(indices), point)
    System.out.printf(Locale.ROOT, "Наклон на левом отрезке:  %.12f%n", result.leftSlope)
    System.out.printf(Locale.ROOT, "Наклон на правом отрезке: %.12f%n", result.rightSlope)
    System.out.printf(Locale.ROOT, "Разделённая разность второго порядка: %.12f%n", result.secondDifference)
    System.out.printf(Locale.ROOT, "Первая производная: %.12f%n", result.first)
    System.out.printf(Locale.ROOT, "Вторая производная: %.12f%n", result.second)
    println("Производные вычислены по локальному интерполяционному многочлену второй степени.")
}
