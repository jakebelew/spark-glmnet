/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package nonsubmit.utils

import scala.util.Random

/**
 * :: DeveloperApi ::
 * Generate sample data used for Linear Data. This class generates
 * uniformly random values for every feature and adds Gaussian noise with mean `eps` to the
 * response variable `Y`.
 */
object LinearDataGenerator {

  def generateLinearInputAsMatrix(
    nexamples: Int,
    nfeatures: Int,
    eps: Double = 0.1,
    intercept: Double = 0.0): Array[Double] = {
    toMatrix(generateLinearInput(nexamples, nfeatures, eps, intercept))
  }

  def toMatrix(x: Array[Array[Double]]): Array[Double] = {
    val numRows = x.size
    val numCols = x(0).size
    var row = 0
    var col = 0
    val matrix = Array.ofDim[Double](numRows * numCols)
    while (row < numRows) {
      while (col < numCols) {
        matrix(col * numRows + row) = x(row)(col)
        col += 1
      }
      col = 0
      row += 1
    }
    matrix
  }

  /**
   * Generate a sequence containing sample data for Linear Regression models - including Ridge, Lasso,
   * and uregularized variants.
   *
   * For compatibility, the generated data
   * will have zero mean and variance of (1.0/3.0) since the original output range is
   * [-1, 1] with uniform distribution, and the variance of uniform distribution
   * is (b - a)^2^ / 12 which will be (1.0/3.0)
   *
   * @param nexamples Number of examples that will be contained in the RDD.
   * @param nfeatures Number of features to generate for each example.
   * @param eps Epsilon factor by which examples are scaled.
   * @param intercept Data intercept
   *
   * @return Seq of (label: Double, features: Array[Double]) containing sample data.
   */
  def generateLinearInput(
    nexamples: Int,
    nfeatures: Int,
    eps: Double = 0.1,
    intercept: Double = 0.0): Array[Array[Double]] = {

    val random = new Random(42)
    // Random values distributed uniformly in [-0.5, 0.5]
    val weights = Array.fill(nfeatures)(random.nextDouble() - 0.5)
    //println(s"weights: ${weights.mkString(",")}")
    val nPoints = nexamples
    val seed = 42
    val data = generateLinearInput(intercept, weights, nPoints, seed, eps)
    //println(s"generating data of $nexamples rows, $nfeatures columns and ${data.size * data(0).size * 8 / 1048576} MiB")
    data
  }

  private def generateLinearInput(
    intercept: Double,
    weights: Array[Double],
    nPoints: Int,
    seed: Int,
    eps: Double): Array[Array[Double]] = {
    generateLinearInput(intercept, weights,
      Array.fill[Double](weights.length)(0.0),
      Array.fill[Double](weights.length)(1.0 / 3.0),
      nPoints, seed, eps)
  }

  private def generateLinearInput(
    intercept: Double,
    weights: Array[Double],
    xMean: Array[Double],
    xVariance: Array[Double],
    nPoints: Int,
    seed: Int,
    eps: Double): Array[Array[Double]] = {

    val rnd = new Random(seed)
    val x = Array.fill[Array[Double]](nPoints)(
      Array.fill[Double](weights.length)(rnd.nextDouble()))

    x.foreach { v =>
      var i = 0
      val len = v.length
      while (i < len) {
        v(i) = (v(i) - 0.5) * math.sqrt(12.0 * xVariance(i)) + xMean(i)
        i += 1
      }
    }
    x
  }
}
