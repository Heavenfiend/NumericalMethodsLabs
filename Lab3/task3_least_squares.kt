import java.awt.BasicStroke
import java.awt.Color
import java.awt.Font
import java.awt.GraphicsEnvironment
import java.awt.RenderingHints
import java.awt.geom.Path2D
import java.awt.image.BufferedImage
import java.io.File
import java.io.PrintStream
import java.util.Locale
import javax.imageio.ImageIO
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.SwingUtilities
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.roundToInt

private data class Fit(val degree: Int, val coefficients: DoubleArray, val sse: Double, val mse: Double)

private fun numbers(line: String) = line.trim().split(Regex("\\s+")).map(String::toDouble)

private fun normalSystem(x: List<Double>, y: List<Double>, degree: Int): Pair<Array<DoubleArray>, DoubleArray> {
    val size = degree + 1
    val matrix = Array(size) { DoubleArray(size) }
    val rhs = DoubleArray(size)
    for (k in x.indices) {
        val powers = DoubleArray(2 * degree + 1)
        powers[0] = 1.0
        for (i in 1 until powers.size) powers[i] = powers[i - 1] * x[k]
        for (i in 0 until size) {
            rhs[i] += y[k] * powers[i]
            for (j in 0 until size) matrix[i][j] += powers[i + j]
        }
    }
    return Pair(matrix, rhs)
}

private fun gaussianElimination(matrix: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
    val n = rhs.size
    val a = Array(n) { i -> matrix[i] + rhs[i] }
    for (k in 0 until n) {
        val pivot = (k until n).maxByOrNull { abs(a[it][k]) }!!
        require(a[pivot][k] != 0.0) { "Система вырождена." }
        val row = a[k]
        a[k] = a[pivot]
        a[pivot] = row
        for (i in k + 1 until n) {
            val factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for (j in k + 1..n) a[i][j] -= factor * a[k][j]
        }
    }
    val result = DoubleArray(n)
    for (i in n - 1 downTo 0) {
        var total = a[i][n]
        for (j in i + 1 until n) total -= a[i][j] * result[j]
        result[i] = total / a[i][i]
    }
    return result
}

private fun polynomial(coefficients: DoubleArray, x: Double): Double {
    var result = 0.0
    for (i in coefficients.indices.reversed()) result = result * x + coefficients[i]
    return result
}

private fun polynomialText(coefficients: DoubleArray): String = coefficients.mapIndexed { i, value ->
    val power = when (i) {
        0 -> ""
        1 -> "·x"
        else -> "·x^$i"
    }
    String.format(Locale.ROOT, "%+.12g", value) + power
}.joinToString(" ").removePrefix("+")

private fun fit(x: List<Double>, y: List<Double>, degree: Int): Fit {
    val (matrix, rhs) = normalSystem(x, y, degree)
    val coefficients = gaussianElimination(matrix, rhs)
    println("\nСтепень $degree. Нормальная система (последний столбец — правая часть):")
    for (i in rhs.indices) {
        val row = matrix[i].joinToString(" ") { String.format(Locale.ROOT, "%14.6f", it) }
        System.out.printf(Locale.ROOT, "%s | %14.6f%n", row, rhs[i])
    }
    println("Коэффициенты при степенях от нулевой до старшей:")
    for (i in coefficients.indices) System.out.printf(Locale.ROOT, "Коэффициент %d: %.12f%n", i, coefficients[i])
    println("П$degree(x) = ${polynomialText(coefficients)}")
    val errors = x.indices.map { polynomial(coefficients, x[it]) - y[it] }
    val sse = errors.sumOf { it * it }
    val mse = sse / x.size
    val residual = rhs.indices.maxOf { i -> abs(rhs.indices.sumOf { j -> matrix[i][j] * coefficients[j] } - rhs[i]) }
    System.out.printf(Locale.ROOT, "Сумма квадратов ошибок: %.12f%n", sse)
    System.out.printf(Locale.ROOT, "Средний квадрат ошибки (MSE): %.12f%n", mse)
    System.out.printf(Locale.ROOT, "Максимальная невязка нормальной системы: %.6e%n", residual)
    System.out.printf(Locale.ROOT, "%12s %20s %18s %18s%n", "Узел", "Табличное значение", "Приближение", "Отклонение")
    for (i in x.indices) {
        System.out.printf(Locale.ROOT, "%12.6f %20.10f %18.10f %18.10f%n", x[i], y[i], y[i] + errors[i], errors[i])
    }
    return Fit(degree, coefficients, sse, mse)
}

private fun ticks(min: Double, max: Double): List<Double> {
    val target = (max - min) / 6
    val scale = 10.0.pow(floor(log10(target)))
    val step = listOf(1.0, 2.0, 2.5, 5.0, 10.0).first { it * scale >= target } * scale
    val first = ceil(min / step) * step
    val count = floor((max - first) / step).toInt() + 1
    return List(count) { first + it * step }
}

private fun drawPlot(x: List<Double>, y: List<Double>, fits: List<Fit>, output: File) {
    val width = 1080
    val height = 600
    val left = 100
    val right = width - 28
    val top = 58
    val bottom = height - 66
    val xmin = x.min()
    val xmax = x.max()
    val grid = List(500) { xmin + (xmax - xmin) * it / 499 }
    val curves = fits.map { fit -> grid.map { polynomial(fit.coefficients, it) } }
    val values = y + curves.flatten()
    val ymin = values.min()
    val ymax = values.max()
    val xMargin = (xmax - xmin).takeIf { it > 0 }?.times(0.05) ?: 1.0
    val yMargin = (ymax - ymin).takeIf { it > 0 }?.times(0.05) ?: 1.0
    fun px(value: Double) = left + (right - left) * (value - xmin + xMargin) / (xmax - xmin + 2 * xMargin)
    fun py(value: Double) = bottom - (bottom - top) * (value - ymin + yMargin) / (ymax - ymin + 2 * yMargin)

    val colors = listOf(Color(0x1F77B4), Color(0xFF7F0E), Color(0x2CA02C))
    val strokes = listOf(
        BasicStroke(3f),
        BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 10f, floatArrayOf(10f, 5f), 0f),
        BasicStroke(3f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 10f, floatArrayOf(11f, 4f, 2f, 4f), 0f)
    )
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g = image.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    g.color = Color.WHITE
    g.fillRect(0, 0, width, height)
    g.font = Font(Font.SANS_SERIF, Font.PLAIN, 17)
    fun centered(text: String, x: Int, y: Int) = g.drawString(text, x - g.fontMetrics.stringWidth(text) / 2, y)
    fun tickText(value: Double) = String.format(Locale.ROOT, "%.6f", value).trimEnd('0').trimEnd('.')
    for (tick in ticks(xmin - xMargin, xmax + xMargin)) {
        val position = px(tick).roundToInt()
        g.color = Color(0xDDDDDD)
        g.drawLine(position, top, position, bottom)
        g.color = Color.BLACK
        centered(tickText(tick), position, bottom + 24)
    }
    for (tick in ticks(ymin - yMargin, ymax + yMargin)) {
        val position = py(tick).roundToInt()
        g.color = Color(0xDDDDDD)
        g.drawLine(left, position, right, position)
        g.color = Color.BLACK
        val label = tickText(tick)
        g.drawString(label, left - 12 - g.fontMetrics.stringWidth(label), position + 6)
    }
    g.color = Color.BLACK
    g.drawRect(left, top, right - left, bottom - top)
    centered("Аргумент", (left + right) / 2, height - 18)
    val transform = g.transform
    g.rotate(-Math.PI / 2)
    centered("Значение", -(top + bottom) / 2, 26)
    g.transform = transform
    g.font = Font(Font.SANS_SERIF, Font.PLAIN, 23)
    centered("Приближение методом наименьших квадратов", (left + right) / 2, 37)

    for (i in curves.indices) {
        val path = Path2D.Double()
        path.moveTo(px(grid.first()), py(curves[i].first()))
        for (j in 1 until grid.size) path.lineTo(px(grid[j]), py(curves[i][j]))
        g.color = colors[i % colors.size]
        g.stroke = strokes[i % strokes.size]
        g.draw(path)
    }
    g.color = Color.BLACK
    for (i in x.indices) g.fillOval(px(x[i]).roundToInt() - 5, py(y[i]).roundToInt() - 5, 10, 10)

    val labels = fits.map { "Многочлен степени ${it.degree}" } + "Исходные данные"
    g.font = Font(Font.SANS_SERIF, Font.PLAIN, 17)
    val legendWidth = labels.maxOf { g.fontMetrics.stringWidth(it) } + 62
    val legendHeight = labels.size * 24 + 12
    val legendX = left + 12
    val legendY = top + 12
    g.color = Color(255, 255, 255, 235)
    g.fillRoundRect(legendX, legendY, legendWidth, legendHeight, 8, 8)
    g.color = Color(0xCCCCCC)
    g.stroke = BasicStroke(1f)
    g.drawRoundRect(legendX, legendY, legendWidth, legendHeight, 8, 8)
    for (i in labels.indices) {
        val row = legendY + 18 + i * 24
        if (i < fits.size) {
            g.color = colors[i % colors.size]
            g.stroke = strokes[i % strokes.size]
            g.drawLine(legendX + 10, row, legendX + 42, row)
        } else {
            g.color = Color.BLACK
            g.fillOval(legendX + 21, row - 5, 10, 10)
        }
        g.color = Color.BLACK
        g.drawString(labels[i], legendX + 52, row + 6)
    }
    g.dispose()

    ImageIO.write(image, "png", output)
    println("График сохранён: ${output.absolutePath}")
    if (!GraphicsEnvironment.isHeadless()) {
        SwingUtilities.invokeLater {
            val window = JFrame("Метод наименьших квадратов")
            window.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
            window.contentPane.add(JLabel(ImageIcon(image)))
            window.pack()
            window.setLocationRelativeTo(null)
            window.isVisible = true
        }
    }
}

fun main() {
    System.setOut(PrintStream(System.out, true, Charsets.UTF_8))
    val input = File("input/task3_3.txt").takeIf { it.isFile } ?: File("Lab3/input/task3_3.txt")
    val lines = input.readLines()
    val degrees = lines[0].trim().split(Regex("\\s+")).map(String::toInt)
    val x = numbers(lines[1])
    val y = numbers(lines[2])
    require(x.size == y.size && x.isNotEmpty()) { "Неверные размеры входных данных." }
    require((x + y).all { it.isFinite() }) { "Все значения должны быть конечными." }
    val distinct = x.toSet().size
    require(degrees.all { it in 0 until distinct }) { "Для выбранной степени недостаточно различных узлов." }

    println("Задание 3.3. Вариант 10. Метод наименьших квадратов")
    System.out.printf(Locale.ROOT, "%7s %12s %16s%n", "Номер", "Узел", "Значение")
    for (i in x.indices) System.out.printf(Locale.ROOT, "%7d %12.6f %16.8f%n", i, x[i], y[i])
    val fits = degrees.map { fit(x, y, it) }
    System.out.printf(Locale.ROOT, "%n%9s %20s %20s%n", "Степень", "Сумма квадратов", "Средний квадрат")
    for (fit in fits) System.out.printf(Locale.ROOT, "%9d %20.12f %20.12f%n", fit.degree, fit.sse, fit.mse)
    val best = fits.minByOrNull { it.mse }!!
    println("По исходным точкам лучшее приближение даёт многочлен степени ${best.degree}.")
    println("Сравнение относится только к ошибке на исходных данных.")
    val output = input.absoluteFile.parentFile.parentFile.resolve("plot_task3_3_kotlin.png")
    drawPlot(x, y, fits, output)
}
