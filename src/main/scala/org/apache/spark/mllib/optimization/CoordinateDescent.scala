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

import java.lang.Math.abs
import scala.annotation.tailrec
import scala.collection.mutable.MutableList
import scala.math.exp
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{ Vector => BV }
import nonsubmit.utils.Timer
//import org.apache.spark.ml.regression.LinearRegressionWithCDParams
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.param.shared._
//import org.apache.spark.ml.regression.tempSharedParams

/**
 * Params for coordinate descent.
 */
private[spark] trait CoordinateDescentParams {
  var elasticNetParam: Double = 0.01
  var numLambdas: Int = 100
  var lambdaShrink: Double = 0.001
  var maxIter: Int = 100
  var tol: Double = 1E-3
  var logSaveAll: Boolean = false

  /**
   * Set the elasticNetParam. Default 0.01.
   */
  def setElasticNetParam(elasticNetParam: Double): this.type = {
    this.elasticNetParam = elasticNetParam
    this
  }

    /**
   * Set the number of lambdas for CD. Default 100.
   */
  def setNumLambdas(numLambdas: Int): this.type = {
    this.numLambdas = numLambdas
    this
  }

  /**
   * Set the lambda shrinkage parameter. Default 0.001.
   */
  def setLambdaShrink(lambdaShrink: Double): this.type = {
    this.lambdaShrink = lambdaShrink
    this
  }

  /**
   * Set the number of iterations for CD. Default 100.
   */
  def setMaxIter(maxIter: Int): this.type = {
    this.maxIter = maxIter
    this
  }

  /**
   * Set the tol. Default 0.01.
   */
  def setTol(tol: Double): this.type = {
    this.tol = tol
    this
  }

  /**
   * Set logSaveAll. Default false.
   */
  def setLogSaveAll(logSaveAll: Boolean): this.type = {
    this.logSaveAll = logSaveAll
    this
  }
}

/**
 * Class used to solve an optimization problem using Coordinate Descent.
 */
//TODO - Faster version in the works
private[spark] class CoordinateDescent extends CDOptimizer //with CoordinateDescentParams
  with Logging {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    CoordinateDescent.runCD(
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

//  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector = {
//    logInfo(s"CoordinateDescent_params logSaveAll: $logSaveAll")
//    CoordinateDescent.runCD(
//      data,
//      initialWeights,
//      xy,
//      elasticNetParam,
//      lambdaShrink,
//      maxIter,
//      tol,
//      lambdaIndex,
//      numFeatures, numRows)
//  }

  //TODO - Temporary to allow testing multiple versions of CoordinateDescent with minimum code duplication - remove to Object method only later
  def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double] = {
    CoordinateDescent.computeXY(data, numFeatures, numRows)
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run coordinate descent.
 */
@DeveloperApi
object CoordinateDescent extends Logging {

  def runCD(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numLambdas: Int, maxIter: Int, tol: Double, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    logInfo(s"Performing coordinate descent with: [elasticNetParam: $alpha, lamShrnk: $lamShrnk, numLambdas: $numLambdas, maxIter: $maxIter, tol: $tol]")

    val lambdas = computeLambdas(xy, alpha, lamShrnk, numLambdas, numLambdas, numRows): Array[Double]
    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, maxIter, tol, numFeatures, numRows)
  }

//  def runCD(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, maxIter: Int, tol: Double, lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector = {
//
//    logInfo(s"CoordinateDescent_params elasticNetParam: $alpha")
//    logInfo(s"CoordinateDescent_params lamShrnk: $lamShrnk")
//    logInfo(s"CoordinateDescent_params maxIter: $maxIter")
//    logInfo(s"CoordinateDescent_params tol: $tol")
//
//    val lambdas = computeLambdas(xy, alpha, lamShrnk, maxIter, lambdaIndex + 1, numRows): Array[Double]
//    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, maxIter, tol, numFeatures, numRows).last._2
//  }

  private def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdas: Array[Double], alpha: Double, lamShrnk: Double, maxIter: Int, tol: Double, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    //data.persist(StorageLevel.MEMORY_AND_DISK)
    //logRDD("data before persist", data)
    var totalNumNewBeta = 0
    val results = new MutableList[(Double, Vector)]

    val indexStart = xy.zipWithIndex.filter(xyi => abs(xyi._1) > (lambdas(0) * alpha)).map(_._2)
    totalNumNewBeta += indexStart.length
    logNewBeta(indexStart.length, totalNumNewBeta)
    val xx = CDSparseMatrix(numFeatures, indexStart)
    populateXXMatrix(data, indexStart, xx, numFeatures, numRows)

    loop(initialWeights, 0)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldBeta: Vector, n: Int): Unit = {
      if (n < lambdas.length) {
        logDebug(s"Lambda number: ${n + 1}")
        val newLambda = lambdas(n)
        val (newBeta, numNewBeta) = cdIter(data, oldBeta, newLambda, alpha, xy, xx, tol, maxIter, numFeatures, numRows)
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

  private def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double] = {
    val xy = data.treeAggregate(new InitLambda(numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy

    xy.map(_ / numRows)
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

  private def populateXXMatrix(data: RDD[(Double, Vector)], newIndexes: Array[Int], xx: CDSparseMatrix, numFeatures: Int, numRows: Long): Unit = {
    Timer("xCorrelation").start
    val correlatedX = xCorrelation(data, newIndexes, numFeatures, numRows)
    Timer("xCorrelation").end
    Timer("xx.update").start
    xx.update(newIndexes, correlatedX)
    Timer("xx.update").end
  }

  private def xCorrelation(data: RDD[(Double, Vector)], newColIndexes: Array[Int], numFeatures: Int, numRows: Long): Array[Array[Double]] = {
    val numNewBeta = newColIndexes.size

    val xx = data.treeAggregate(new XCorrelation(newColIndexes, numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xx

    xx.map { _.map(_ / numRows) }
  }

  private def S(z: Double, gamma: Double): Double = if (gamma >= abs(z)) 0.0 else (z / abs(z)) * (abs(z) - gamma)

  private def cdIter(data: RDD[(Double, Vector)], oldBeta: Vector, newLambda: Double, alpha: Double, xy: Array[Double], xx: CDSparseMatrix, tol: Double, maxIter: Int, numFeatures: Int, numRows: Long): (Vector, Int) = {
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
            populateXXMatrix(data, newIndexes.toArray, xx, numFeatures, numRows)
          }
        }
        val sumDiff = (beta.toArray zip betaStart.toArray) map (b => abs(b._1 - b._2)) sum
        val sumBeta = beta.toArray.map(abs).sum
        val deltaBeta = sumDiff / sumBeta
        loop(beta, deltaBeta, false, n - 1)
      }
    }

    def coordinateWiseUpdate(beta: BV[Double]) = {
      for (j <- 0 until numFeatures) {
        val xyj = xy(j) - xx.dot(j, beta)
        val uncBeta = xyj + beta(j)
        beta(j) = S(uncBeta, gamma) / ridgePenaltyShrinkage
      }
    }

    (loop(oldBeta, 100.0, true, maxIter), numNewBeta)
  }
}

private class InitLambda(numFeatures: Int) extends Serializable {

  lazy val xy: Array[Double] = Array.ofDim[Double](numFeatures)

  def compute(row: (Double, Vector)): this.type = {
    val loadedXY = xy
    val y = row._1
    val x = row._2.toArray
    var j = 0
    while (j < numFeatures) {
      loadedXY(j) += x(j) * y
      j += 1
    }
    this
  }

  def combine(other: InitLambda): this.type = {
    val thisXX = xy
    val otherXX = other.xy
    var j = 0
    while (j < numFeatures) {
      thisXX(j) += otherXX(j)
      j += 1
    }
    this
  }
}

//private class XCorrelation(newColIndexes: Array[Int], numFeatures: Int) extends Serializable {
class XCorrelation(newColIndexes: Array[Int], numFeatures: Int) extends Serializable {

  private val numNewBeta = newColIndexes.size

  lazy val xx: Array[Array[Double]] = Array.ofDim[Double](numNewBeta, numFeatures)

  def compute(row: (Double, Vector)): this.type = {
    val loadedXX = xx
    val x = row._2.toArray
    var k = 0
    var j = 0
    while (k < numNewBeta) {
      while (j < numFeatures) {
        loadedXX(k)(j) += x(j) * x(newColIndexes(k))
        j += 1
      }
      j = 0
      k += 1
    }
    this
  }

  def combine(other: XCorrelation): this.type = {
    val thisXX = xx
    val otherXX = other.xx
    var k = 0
    var j = 0
    while (k < numNewBeta) {
      while (j < numFeatures) {
        thisXX(k)(j) = thisXX(k)(j) + otherXX(k)(j)
        j += 1
      }
      j = 0
      k += 1
    }
    this
  }
}