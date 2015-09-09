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
import scala.Array.canBuildFrom
import scala.annotation.tailrec
import scala.collection.mutable.MutableList
import scala.math.exp
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{ Vector => BV }
import nonsubmit.utils.StopWatch
import nonsubmit.utils.Log.logRDD
import nonsubmit.utils.Timer

/**
 * Class used to solve an optimization problem using Coordinate Descent.
 */
//TODO - Faster version in the works
class CoordinateDescent private[spark]
  extends Logging {

  private var alpha: Double = 1.0
  private var lamShrnk: Double = 0.001
  private var numIterations: Int = 100

  /**
   * Set the alpha. Default 1.0.
   */
  def setAlpha(step: Double): this.type = {
    this.alpha = step
    this
  }

  /**
   * Set the lambda shrinkage parameter. Default 0.001.
   */
  def setLamShrnk(regParam: Double): this.type = {
    this.lamShrnk = regParam
    this
  }

  /**
   * Set the number of iterations for CD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    CoordinateDescent.runCD2(
      data,
      initialWeights,
      xy,
      alpha,
      lamShrnk,
      numIterations,
      numFeatures, numRows)
  }

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector = {
    CoordinateDescent.runCD2(
      data,
      initialWeights,
      xy,
      alpha,
      lamShrnk,
      numIterations,
      lambdaIndex,
      numFeatures, numRows)
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run coordinate descent.
 */
@DeveloperApi
object CoordinateDescent extends Logging {

  def runCD2(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    val lambdas = computeLambdas(xy, alpha, lamShrnk, numIterations, numIterations, numRows): Array[Double]
    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, numIterations, numFeatures, numRows)
  }

  def runCD2(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector = {
    val lambdas = computeLambdas(xy, alpha, lamShrnk, numIterations, lambdaIndex + 1, numRows): Array[Double]
    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, numIterations, numFeatures, numRows).last._2
  }

  private def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdas: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
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
        val (newBeta, numNewBeta) = cdIter(data, oldBeta, newLambda, alpha, xy, xx, numFeatures, numRows)
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

  def logNewBeta(numNewBeta: Int, totalNumNewBeta: Int) = {
    if (numNewBeta > 0) {
      logDebug(s"numNewBeta: $numNewBeta,  totalNumNewBeta: $totalNumNewBeta")
    }
  }

  def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double] = {
    val xy = data.treeAggregate(new InitLambda(numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy

    xy.map(_ / numRows)
  }

  def computeLambdas(xy: Array[Double], alpha: Double, lamShrnk: Double, lambdaRange: Int, numLambdas: Int, numRows: Long): Array[Double] = {
    logDebug(s"alpha: $alpha, lamShrnk: $lamShrnk, numIterations: $lambdaRange, numRows: $numRows")

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

  def populateXXMatrix(data: RDD[(Double, Vector)], newIndexes: Array[Int], xx: CDSparseMatrix, numFeatures: Int, numRows: Long): Unit = {
    Timer("xCorrelation").start
    val correlatedX = xCorrelation(data, newIndexes, numFeatures, numRows)
    Timer("xCorrelation").end
    Timer("xx.update").start
    xx.update(newIndexes, correlatedX)
    Timer("xx.update").end
  }

  def xCorrelation(data: RDD[(Double, Vector)], newColIndexes: Array[Int], numFeatures: Int, numRows: Long): Array[Array[Double]] = {
    val numNewBeta = newColIndexes.size

    val xx = data.treeAggregate(new XCorrelation(newColIndexes, numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xx

    xx.map { _.map(_ / numRows) }
  }

  def S(z: Double, gamma: Double): Double = if (gamma >= abs(z)) 0.0 else (z / abs(z)) * (abs(z) - gamma)

  def cdIter(data: RDD[(Double, Vector)], oldBeta: Vector, newLambda: Double, alpha: Double, xy: Array[Double], xx: CDSparseMatrix, numFeatures: Int, numRows: Long): (Vector, Int) = {
    var numNewBeta = 0
    //val eps = 0.01
    val eps = 0.001
    val numCDIter = 100
    val ridgePenaltyShrinkage = 1 + newLambda * (1 - alpha)
    val gamma = newLambda * alpha

    @tailrec
    def loop(beta: Vector, deltaBeta: Double, firstPass: Boolean, n: Int): Vector = {
      if (deltaBeta <= eps || n == 0) {
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

    (loop(oldBeta, 100.0, true, numCDIter), numNewBeta)
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

private class XCorrelation(newColIndexes: Array[Int], numFeatures: Int) extends Serializable {

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