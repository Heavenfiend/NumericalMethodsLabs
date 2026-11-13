import java.io.File
import java.io.PrintStream
import java.util.Locale
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.sqrt

private data class Method(val name: String, val order: Int, val integrate: (Double) -> Double)
private data class Errors(val name: String, val coarse: Double, val fine: Double, val refined: Double)

private fun intervalCount(a: Double, b: Double, h: Double): Int {
    val count = round((b - a) / h)
    require(count in 1.0..Int.MAX_VALUE.toDouble() && abs(count * h - (b - a)) <= 1e-10 * maxOf(1.0, abs(b - a))) {
        "Длина отрезка должна быть кратна шагу."
    }
    return count.toInt()
}

private fun rectangles(f: (Double) -> Double, a: Double, b: Double, h: Double, offset: Double): Double {
    val count = intervalCount(a, b, h)
    var total = 0.0
    for (i in 0 until count) {
        total += f(a + (i + offset) * h)
    }
    return h * total
}

private fun trapezoids(f: (Double) -> Double, a: Double, b: Double, h: Double): Double {
    val count = intervalCount(a, b, h)
    var total = (f(a) + f(b)) / 2
    for (i in 1 until count) {
        total += f(a + i * h)
    }
    return h * total
}

private fun simpson(f: (Double) -> Double, a: Double, b: Double, h: Double): Double {
    val count = intervalCount(a, b, h)
    require(count % 2 == 0) { "Для метода Симпсона требуется чётное число отрезков." }
    var total = f(a) + f(b)
    for (i in 1 until count) {
        val weight = if (i % 2 == 0) 2 else 4
        total += weight * f(a + i * h)
    }
    return h * total / 3
}

private fun rungeRomberg(coarse: Double, fine: Double, ratio: Double, order: Int): Pair<Double, Double> {
    val correction = (fine - coarse) / (ratio.pow(order) - 1)
    return Pair(fine + correction, abs(correction))
}

fun main() {
    System.setOut(PrintStream(System.out, true, Charsets.UTF_8))
    val input = File("input/task3_5.txt").takeIf { it.isFile } ?: File("Lab3/input/task3_5.txt")
    val values = input.readText().trim().split(Regex("\\s+")).map(String::toDouble)
    require(values.size == 5 && values.all { it.isFinite() }) { "Нужно пять конечных чисел." }
    val (a, b, h1, h2, parameter) = values
    require(a < b && h2 > 0 && h2 < h1) { "Требуются возрастающие границы и шаги: 0 < h₂ < h₁." }
    require(parameter > 0) { "Параметр в знаменателе должен быть положительным." }
    val ratio = h1 / h2
    require(abs(ratio - round(ratio)) <= 1e-10) { "Отношение шагов должно быть целым для вложенных сеток." }

    val f: (Double) -> Double = { x -> x * x / (x * x + parameter) }
    val scale = sqrt(parameter)
    fun antiderivative(x: Double) = x - scale * atan(x / scale)
    val exact = antiderivative(b) - antiderivative(a)
    val methods = listOf(
        Method("Левые прямоугольники", 1) { h -> rectangles(f, a, b, h, 0.0) },
        Method("Правые прямоугольники", 1) { h -> rectangles(f, a, b, h, 1.0) },
        Method("Средние прямоугольники", 2) { h -> rectangles(f, a, b, h, 0.5) },
        Method("Трапеции", 2) { h -> trapezoids(f, a, b, h) },
        Method("Симпсон", 4) { h -> simpson(f, a, b, h) }
    )

    println("Задание 3.5. Вариант 10. Численное интегрирование")
    println("Подынтегральная функция: x² / (x² + $parameter)")
    println("Границы: [$a; $b]; шаг h₁ = $h1; шаг h₂ = $h2")
    println("Число отрезков: ${intervalCount(a, b, h1)} и ${intervalCount(a, b, h2)}")
    println("Первообразная: x − $scale·арктангенс(x / $scale)")
    System.out.printf(Locale.ROOT, "Точное значение интеграла: %.15f%n", exact)
    System.out.printf(Locale.ROOT, "%n%-24s %16s %16s %8s %16s %16s%n",
        "Метод", "Шаг h₁", "Шаг h₂", "Порядок", "Уточнение", "Оценка для h₂")

    val errors = mutableListOf<Errors>()
    for (method in methods) {
        val coarse = method.integrate(h1)
        val fine = method.integrate(h2)
        val (refined, estimate) = rungeRomberg(coarse, fine, ratio, method.order)
        errors.add(Errors(method.name, abs(coarse - exact), abs(fine - exact), abs(refined - exact)))
        System.out.printf(Locale.ROOT, "%-24s %16.12f %16.12f %8d %16.12f %16.6e%n",
            method.name, coarse, fine, method.order, refined, estimate)
    }

    System.out.printf(Locale.ROOT, "%n%-24s %18s %18s %18s%n", "Метод", "Ошибка для h₁", "Ошибка для h₂", "После уточнения")
    for (error in errors) {
        System.out.printf(Locale.ROOT, "%-24s %18.10e %18.10e %18.10e%n", error.name, error.coarse, error.fine, error.refined)
    }
    val bestFine = errors.minByOrNull { it.fine }!!
    val bestRefined = errors.minByOrNull { it.refined }!!
    println("Наиболее точный метод с шагом h₂: ${bestFine.name}.")
    println("Наиболее точный метод после уточнения: ${bestRefined.name}.")
    println("Оценка Рунге–Ромберга относится к исходному результату с шагом h₂.")
    println("Это асимптотическая оценка, а не гарантированная верхняя граница ошибки.")
}
