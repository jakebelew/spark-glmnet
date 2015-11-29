package org.apache.spark.mllib

import scala.math.abs

object TestUtil {

  def equalWithinTolerance(actual: Array[Array[Double]], expected: Array[Array[Double]], tolerance: Double, testName: String): Unit = {
    if (actual.length != expected.length)
      sys.error(s"$testName: The actual number of rows ${actual.length} do not match the expected number of rows ${expected.length}")
    actual.zip(expected).zipWithIndex.foreach {
      case ((a, e), row) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Array[Double], expected: Array[Double], tolerance: Double, testName: String): Unit = {
    if (actual.length != expected.length)
      sys.error(s"$testName: The actual number of columns ${actual.length} do not match the expected number of columns ${expected.length}")
    actual.zip(expected).zipWithIndex.foreach {
      case ((a, e), column) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Double, expected: Double, tolerance: Double, testName: String): Unit =
    if (abs(expected - actual) > tolerance)
      sys.error(s"$testName: The difference between the expected [$expected] and actual [$actual] value is not within the tolerance of [$tolerance]")

  def equal(actual: Array[String], expected: Array[String], testName: String): Unit =
    actual.zip(expected).foreach {
      case (a, e) => if (a != e) sys.error(s"The actual [$a] is not equal to the expected [$e] value")
    }
}