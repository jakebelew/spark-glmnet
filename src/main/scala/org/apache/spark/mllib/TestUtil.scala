package org.apache.spark.mllib

import scala.math.abs

object TestUtil {

  def equalWithinTolerance(actual: Array[Array[Double]], expected: Array[Array[Double]], tolerance: Double): Unit = {
    actual.zip(expected).foreach {
      case (a, e) => equalWithinTolerance(a, e, tolerance)
    }
  }

  def equalWithinTolerance(actual: Array[Double], expected: Array[Double], tolerance: Double): Unit = {
    actual.zip(expected).foreach {
      case (a, e) =>
        if (abs(e - a) > tolerance)
          throw new Exception(s"The difference between the expected [$e] and actual [$a] value is not within the tolerance of [$tolerance]")
    }
  }

  def equal(actual: Array[String], expected: Array[String]): Unit = {
    actual.zip(expected).foreach {
      case (a, e) =>
        if (a != e)
          throw new Exception(s"The actual [$a] is not equal to the expected [$e] value")
    }
  }
}