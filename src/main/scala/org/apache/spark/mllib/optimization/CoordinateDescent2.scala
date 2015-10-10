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

package org.apache.spark.mllib.optimization

import scala.annotation.tailrec
import scala.collection.mutable.MutableList
import scala.math.{ abs, exp }
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.{ DenseVector, Matrices, Matrix, DenseMatrix, Vector, Vectors }
import org.apache.spark.mllib.linalg.mlmatrix.RowPartionedTransformer
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{ *, Vector => BV, DenseMatrix => BDM }
import nonsubmit.utils.Timer

/**
 * Class used to solve an optimization problem using Coordinate Descent.
 */
//Version Two of CD that will provide a BLAS Level 3 computation of the XCorrelation
//computing XCorrelation originally took 97% of CD time for large number of rows and 45% of CD time for a large number of columns
private[spark] class CoordinateDescent2 extends CDOptimizer // with CoordinateDescentParams
  with Logging {

  def optimize(data: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    CoordinateDescent2.runCD(
      data,
      initialWeights,
      xy,
      elasticNetParam,
      lambdaShrink,
      numLambdas,
      maxIter,
      tol,
      numFeatures, numRows)
  }

  //TODO - Temporary to allow testing multiple versions of CoordinateDescent with minimum code duplication - remove to Object method only later
  def computeXY(data: RDD[(DenseVector, DenseMatrix)], numFeatures: Int, numRows: Long): Array[Double] = {
    CoordinateDescent2.computeXY(data, numFeatures, numRows)
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run coordinate descent.
 */
@DeveloperApi
object CoordinateDescent2 extends Logging {

  def runCD(data: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numLambdas: Int, maxIter: Int, tol: Double, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    logInfo(s"Performing coordinate descent with: [elasticNetParam: $alpha, lamShrnk: $lamShrnk, numLambdas: $numLambdas, maxIter: $maxIter, tol: $tol]")

    val lambdas = computeLambdas(xy, alpha, lamShrnk, numLambdas, numLambdas, numRows): Array[Double]
    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, maxIter, tol, numFeatures, numRows)
  }

  private def optimize(data: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], lambdas: Array[Double], alpha: Double, lamShrnk: Double, maxIter: Int, tol: Double, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    var totalNumNewBeta = 0
    val results = new MutableList[(Double, Vector)]

    val indexStart = xy.zipWithIndex.filter(xyi => abs(xyi._1) > (lambdas(0) * alpha)).map(_._2)
    totalNumNewBeta += indexStart.length
    logNewBeta(indexStart.length, totalNumNewBeta)
    val xx = CDSparseMatrix2(numFeatures, indexStart)
    populateXXMatrix(data, indexStart, xx, numRows)

    loop(initialWeights, 0)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldBeta: Vector, n: Int): Unit = {
      if (n < lambdas.length) {
        logDebug(s"Lambda number: ${n + 1}")
        val newLambda = lambdas(n)
        val (newBeta, numNewBeta) = cdIter(data, oldBeta, newLambda, alpha, xy, xx, tol, maxIter, numRows)
        totalNumNewBeta += numNewBeta
        logNewBeta(numNewBeta, totalNumNewBeta)
        results += Pair(newLambda, newBeta.copy)
        loop(newBeta, n + 1)
      }
    }
    //println(s"CD ET: ${sw.elapsedTime / 1000} seconds")
    data.unpersist()
    logDebug(s"totalNumNewBeta $totalNumNewBeta")
    results.toList
  }

  private def logNewBeta(numNewBeta: Int, totalNumNewBeta: Int) = {
    if (numNewBeta > 0) {
      logDebug(s"numNewBeta: $numNewBeta,  totalNumNewBeta: $totalNumNewBeta")
    }
  }

  private def computeXY(data: RDD[(DenseVector, DenseMatrix)], numFeatures: Int, numRows: Long): Array[Double] = {
    val xy = data.treeAggregate(new InitLambda2(numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy

    //xy.toArray.map(_ / numRows)
    (xy.toBreeze / numRows.toDouble).toArray
  }

  private def computeLambdas(xy: Array[Double], alpha: Double, lamShrnk: Double, lambdaRange: Int, numLambdas: Int, numRows: Long): Array[Double] = {
    //logDebug(s"alpha: $alpha, lamShrnk: $lamShrnk, maxIter: $lambdaRange, numRows: $numRows")

    val maxXY = xy.map(abs).max(Ordering.Double)
    val lambdaInit = maxXY / alpha

    val lambdaMult = exp(scala.math.log(lamShrnk) / lambdaRange)

    val lambdas = new MutableList[Double]

    loop(lambdaInit, numLambdas)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldLambda: Double, n: Int): Unit = {
      if (n > 0) {
        val newLambda = oldLambda * lambdaMult
        lambdas += newLambda
        loop(newLambda, n - 1)
      }
    }
    logDebug(s"lambdas: ${lambdas.mkString(",")}")
    lambdas.toArray
  }

  private def populateXXMatrix(data: RDD[(DenseVector, DenseMatrix)], newIndexes: Array[Int], xx: CDSparseMatrix2, numRows: Long): Unit = {
    Timer("xCorrelation").start
    val correlatedX = xCorrelation(data, newIndexes, xx.numFeatures, numRows)
    Timer("xCorrelation").end
    Timer("xx.update").start
    xx.update(newIndexes, correlatedX)
    Timer("xx.update").end
  }

  private def xCorrelation(data: RDD[(DenseVector, DenseMatrix)], newColIndexes: Array[Int], numFeatures: Int, numRows: Long): Matrix = {
    val numNewBeta = newColIndexes.size

    val xx = data.treeAggregate(new XCorrelation2(newColIndexes, numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xx

    Matrices.fromBreeze(xx.toBreeze :/= (numRows.toDouble))
  }

  private def S(z: Double, gamma: Double): Double = if (gamma >= abs(z)) 0.0 else (z / abs(z)) * (abs(z) - gamma)

  private def cdIter(data: RDD[(DenseVector, DenseMatrix)], oldBeta: Vector, newLambda: Double, alpha: Double, xy: Array[Double], xx: CDSparseMatrix2, tol: Double, maxIter: Int, numRows: Long): (Vector, Int) = {
    var numNewBeta = 0
    val ridgePenaltyShrinkage = 1 + newLambda * (1 - alpha)
    val gamma = newLambda * alpha

    @tailrec
    def loop(beta: Vector, deltaBeta: Double, firstPass: Boolean, n: Int): Vector = {
      if (deltaBeta <= tol || n == 0) {
        beta
      } else {
        val betaStart = beta.copy
        Timer("coordinateWiseUpdate").start
        coordinateWiseUpdate(beta.toBreeze)
        Timer("coordinateWiseUpdate").end

        if (firstPass) {
          val newIndexes = xx.newIndices(beta.toBreeze)
          if (!newIndexes.isEmpty) {
            numNewBeta += newIndexes.size
            populateXXMatrix(data, newIndexes.toArray, xx, numRows)
          }
        }
        val sumDiff = (beta.toArray zip betaStart.toArray) map (b => abs(b._1 - b._2)) sum
        val sumBeta = beta.toArray.map(abs).sum
        val deltaBeta = sumDiff / sumBeta
        loop(beta, deltaBeta, false, n - 1)
      }
    }

    def coordinateWiseUpdate(beta: BV[Double]) = {
      for (j <- 0 until xx.numFeatures) {
        val xyj = xy(j) - xx.dot(j, beta)
        val uncBeta = xyj + beta(j)
        beta(j) = S(uncBeta, gamma) / ridgePenaltyShrinkage
      }
    }

    (loop(oldBeta, 100.0, true, maxIter), numNewBeta)
  }
}

private class InitLambda2(numFeatures: Int) extends Serializable {

  var xy: DenseVector = _

  def compute(row: (DenseVector, DenseMatrix)): this.type = {
    assert(xy == null, "More than one matrix per partition")
    xy = gemv(row._2.transpose, row._1)
    this
  }

  def combine(other: InitLambda2): this.type = {
    assert(xy != null, "Partition does not contain a matrix")
    assert(other.xy != null, "Other Partition does not contain a matrix")
    xy.toBreeze :+= other.xy.toBreeze
    this
  }

  private def gemv(
    A: DenseMatrix,
    x: DenseVector): DenseVector = {
    val y = new DenseVector(new Array[Double](A.numRows))
    BLAS.gemv(1.0, A, x, 1.0, y)
    y
  }
}

//private class XCorrelation2(newColIndexes: Array[Int], numFeatures: Int) extends Serializable {
class XCorrelation2(newColIndexes: Array[Int], numFeatures: Int) extends Serializable {

  var xx: DenseMatrix = _

  def compute(row: (DenseVector, DenseMatrix)): this.type = {
    assert(xx == null, "More than one matrix per partition")
    xx = computeJxK(row._2, newColIndexes)
    this
  }

  def combine(other: XCorrelation2): this.type = {
    assert(xx != null, "Partition does not contain a matrix")
    assert(other.xx != null, "Other Partition does not contain a matrix")
    xx.toBreeze :+= other.xx.toBreeze
    this
  }

  //TODO - Test which way is the fastest in the entire CD process (computeJxK or computeKxJ)
  //private
  def computeJxK(m: DenseMatrix, newColIndexes: Array[Int]): DenseMatrix = {
    val xk = sliceMatrixByColumns(m, newColIndexes)
    gemm(m.transpose, xk)
  }

  //private
  def computeKxJ(m: DenseMatrix, newColIndexes: Array[Int]): DenseMatrix = {
    val xk = sliceMatrixByColumns(m, newColIndexes)
    gemm(xk.transpose, m)
  }

  //TODO - does breeze have this functionality?
  private def sliceMatrixByColumns(m: DenseMatrix, sliceIndices: Array[Int]): DenseMatrix = {
    //val startTime = System.currentTimeMillis()
    val nIndices = sliceIndices.length
    val nRows = m.numRows
    val slice = Array.ofDim[Double](nRows * nIndices)
    var i = 0
    while (i < nIndices) {
      Array.copy(m.values, sliceIndices(i) * nRows, slice, i * nRows, nRows)
      i += 1
    }
    val sm = new DenseMatrix(nRows, sliceIndices.length, slice)
    //TODO - add Timer instead
    //println(s"sliceMatrixByColumns time: ${(System.currentTimeMillis() - startTime) / 1000} seconds")
    sm
  }

  private def gemm(
    A: DenseMatrix,
    B: DenseMatrix): DenseMatrix = {
    val C = DenseMatrix.zeros(A.numRows, B.numCols)
    BLAS.gemm(1.0, A, B, 1.0, C)
    C
  }
}