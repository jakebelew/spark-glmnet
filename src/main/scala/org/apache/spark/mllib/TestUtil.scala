package org.apache.spark.mllib

import scala.math.abs

object TestUtil {

  def equalWithinTolerance(actual: Array[Array[Double]], expected: Array[Array[Double]], tolerance: Double, testName: String): Unit = {
    actual.zip(expected).foreach {
      case (a, e) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Array[Double], expected: Array[Double], tolerance: Double, testName: String): Unit = {
    actual.zip(expected).foreach {
      case (a, e) => equalWithinTolerance(a, e, tolerance, testName)
    }
  }

  def equalWithinTolerance(actual: Double, expected: Double, tolerance: Double, testName: String): Unit = {
    if (abs(expected - actual) > tolerance)
      throw new Exception(s"$testName: The difference between the expected [$expected] and actual [$actual] value is not within the tolerance of [$tolerance]")
  }

  def equal(actual: Array[String], expected: Array[String], testName: String): Unit = {
    actual.zip(expected).foreach {
      case (a, e) =>
        if (a != e)
          throw new Exception(s"The actual [$a] is not equal to the expected [$e] value")
    }
  }
}