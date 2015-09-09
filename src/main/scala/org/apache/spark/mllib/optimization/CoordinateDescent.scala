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
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
//class CoordinateDescent private[mllib]
class CoordinateDescent private[spark]
  //extends Optimizer with Logging {
  extends Logging {

  private var alpha: Double = 1.0
  private var lamShrnk: Double = 0.001
  private var numIterations: Int = 100
  //private var numIterations: Int = 3

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

  //  def optimizeSingleModel(data: RDD[(Double, Vector)], initialWeights: Vector, numFeatures: Int, numRows: Long): Vector = {
  //    Vectors.dense(Array.ofDim[Double](numFeatures))
  //  }

  //  def optimize1(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)] = ???
  //
  //  def optimize2(data: RDD[(Double, Vector)], initialWeights: Vector, lambda: Double, numFeatures: Int, numRows: Long): Vector = ???
  //
  //  private def optimize3(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdas: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)] = ???

  //  /**
  //   * :: DeveloperApi ::
  //   * Runs coordinate descent on the given training data.
  //   * @param data training data
  //   * @param initialWeights initial weights
  //   * @return solution vector
  //   */
  //  @DeveloperApi
  //  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
  //    CoordinateDescent.runCD(
  //      data,
  //      alpha,
  //      lamShrnk,
  //      numIterations,
  //      initialWeights,
  //      numFeatures, numRows)
  //  }

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
  /**
   * Run stochastic coordinate descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data - Input data for SGD. RDD of the set of data examples, each of
   *               the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                   one single data example)
   * @param updater - Updater function to actually perform a gradient step in a given direction.
   * @param stepSize - initial step size for the first step
   * @param numIterations - number of iterations that SGD should be run.
   * @param regParam - regularization parameter
   * @param miniBatchFraction - fraction of the input data set that should be used for
   *                            one iteration of SGD. Default value 1.0.
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */

  def runCD2(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    //println(s"data1: ${data.collect.mkString("\n")}")
    val lambdas = computeLambdas(xy, alpha, lamShrnk, numIterations, numIterations, numRows): Array[Double]
    println(s"lambdas: ${lambdas.mkString(";")}")
    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, numIterations, numFeatures, numRows)
  }

  def runCD2(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector = {
    //println(s"alpha: $alpha")
    //println(s"data2: ${data.collect.mkString("\n")}")
    //println(s"data2: ${data}")
    val lambdas = computeLambdas(xy, alpha, lamShrnk, numIterations, lambdaIndex + 1, numRows): Array[Double]

    //println(s"lambdas: ${lambdas.mkString(";")}, lambda: $lambda")
    println(s"lambdas: ${lambdas.mkString(";")})")
    //val lambdaIndex = indexOfClosestLambda(lambdas, lambda)
    // handle the case where there is no match and lambdaIndex is -1

    //val subLambdas = lambdas.take(lambdaIndex + 1)
    //println(s"lambdaIndex: $lambdaIndex, subLambdas.length: ${subLambdas.length}")

    optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, numIterations, numFeatures, numRows).last._2
  }

  private def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdas: Array[Double], alpha: Double, lamShrnk: Double, numIterations: Int, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
    //data.persist(StorageLevel.MEMORY_AND_DISK)
    //logRDD("data before persist", data)
    println(s"alpha: $alpha")
    println(s"lambdas: ${lambdas.mkString(",")}")

    val results = new MutableList[(Double, Vector)]

    //val lambdaMult = exp(scala.math.log(lamShrnk) / numIterations)

    //val sw = new StopWatch()
    //Timer("initLambda").start
    //val (xy, lambdaInit) = initLambda(data, alpha, sw, numFeatures, numRows)
    //Timer("initLambda").end

    val indexStart = xy.zipWithIndex.filter(xyi => abs(xyi._1) > (lambdas(0) * alpha)).map(_._2)

    val xx = CDSparseMatrix(numFeatures, indexStart)
    populateXXMatrix(data, indexStart, xx, numFeatures, numRows)

    //    val results = for {
    //      lambda <- lambdas
    //      newBeta = cdIter(data, oldBeta, newLambda, alpha, xy, xx, numFeatures, numRows)
    //    } yield lambda

    println("Y")
    loop(initialWeights, 0)
    println("Z")
    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldBeta: Vector, n: Int): Unit = {
      if (n < lambdas.length) {
        println(s"Lambda number: ${n + 1}")
        val newLambda = lambdas(n)
        val newBeta = cdIter(data, oldBeta, newLambda, alpha, xy, xx, numFeatures, numRows)
        results += Pair(newLambda, newBeta.copy)
        loop(newBeta, n + 1)
      }
    }
    //println(s"CD ET: ${sw.elapsedTime / 1000} seconds")
    data.unpersist()
    println(s"totalNumNewBeta: $totalNumNewBeta")
    results.toList
  }

  //  /*Function to calculate starting lambda value*/
  //  def initLambda(data: RDD[(Double, Vector)], alpha: Double, sw: StopWatch, numFeatures: Int, numRows: Long): (Array[Double], Double) = {
  //    sw.restart
  //
  //    val xy = data.treeAggregate(new InitLambda(numFeatures))(
  //      (aggregate, row) => aggregate.compute(row),
  //      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy
  //
  //    //logRDD("data after persist", data)
  //    //unpersist.unpersist()
  //    //logRDD("data after persist and labelsAndFeatures unpersist", data)
  //
  //    //def maxBy[B](f: (A) ⇒ B): A
  //    //Finds the first element which yields the largest value measured by function f
  //    //val maxXY = xy.maxBy(abs) / numRows
  //    val maxXY = xy.map(abs).max(Ordering.Double) / numRows
  //    val lambdaInit = maxXY / alpha
  //
  //    (xy.map(_ / numRows), lambdaInit)
  //  }

  //
  //    loop(oldBeta, 100.0, true, numCDIter)
  //  }
  //
  //    ???
  //  }

//  def main(args: Array[String]) {
//
//    val list = Array(10.0, 9.0, 8.0, 7.0, 6.0, 5.0)
//    //lambdas: 0.018850669313042794;0.0018850669313042797;1.88506693130428E-4, lambda: 0.014540099339240853
//
//    val list2 = Array(0.018850669313042794, 0.0018850669313042797, 1.88506693130428E-4)
//    assert(indexOfClosestLambda(list2, 0.014540099339240853) == 0)
//
//    assert(indexOfClosestLambda(list, 7.51) == 2)
//    assert(indexOfClosestLambda(list, 7.50) == 2)
//    assert(indexOfClosestLambda(list, 7.49) == 3)
//    assert(indexOfClosestLambda(list, 8.0) == 2)
//    assert(indexOfClosestLambda(list, 6.25) == 4)
//  }
//
//  // def indexOf(elem: A): Int
//  //Finds index of first occurrence of some value in this sequence.   
//  //TODO - revisit this implementation. F3 into scala impl of indexOf() and also look at scala cookbook
//  def indexOfClosestLambda(lambdas: Array[Double], lambda: Double): Int = {
//    @tailrec
//    def loop(n: Int, prevDiff: Double): Int = {
//      val currDiff = lambdas(n) - lambda
//      println(s"n: $n, prevDiff: $prevDiff < currDiff: $currDiff")
//      if (n == 0) { if (lambda < lambdas(0)) 0 else -1 }
//      //else if (abs(prevDiff) == abs(currDiff)) n 
//      else if (abs(prevDiff) < abs(currDiff)) n + 1
//      else loop(n - 1, currDiff)
//    }
//
//    val lastIndex = lambdas.length - 1
//    loop(lastIndex - 1, lambdas(lastIndex) - lambda)
//  }

  def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double] = {
    //val (xy, lambdaInit) = initLambda(data, alpha, sw, numFeatures, numRows)

    //def initLambda(data: RDD[(Double, Vector)], alpha: Double, sw: StopWatch, numFeatures: Int, numRows: Long): (Array[Double], Double) = {
    //sw.restart

    val xy = data.treeAggregate(new InitLambda(numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy

    //logRDD("data after persist", data)
    //unpersist.unpersist()
    //logRDD("data after persist and labelsAndFeatures unpersist", data)

    //    val maxXY = xy.map(abs).max(Ordering.Double) / numRows
    //    val lambdaInit = maxXY / alpha

    //(xy.map(_ / numRows), lambdaInit)
    xy.map(_ / numRows)
  }

  //def ?? : Nothing = throw new NotImplementedError

  def computeLambdas(xy: Array[Double], alpha: Double, lamShrnk: Double, lambdaRange: Int, numLambdas: Int, numRows: Long): Array[Double] = {
    println(s"computeLambdas() xy: ${xy.mkString(",")}, alpha: $alpha, lamShrnk: $lamShrnk, numIterations: $lambdaRange, numRows: $numRows")

    val maxXY = xy.map(abs).max(Ordering.Double)
    val lambdaInit = maxXY / alpha

    val lambdaMult = exp(scala.math.log(lamShrnk) / lambdaRange)

    val lambdas = new MutableList[Double]

    loop(lambdaInit, numLambdas)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldLambda: Double, n: Int): Unit = {
      if (n > 0) {
        //println(s"lamda number: ${101 - n}")
        val newLambda = oldLambda * lambdaMult
        lambdas += newLambda
        loop(newLambda, n - 1)
      }
    }
    lambdas.toArray
  }

  //  def runCD(
  //    data: RDD[(Double, Vector)],
  //    alpha: Double,
  //    lamShrnk: Double,
  //    numIterations: Int,
  //    initialWeights: Vector, numFeatures: Int, numRows: Long): List[(Double, Vector)] = {
  //
  //    data.persist(StorageLevel.MEMORY_AND_DISK)
  //    //logRDD("data before persist", data)
  //    println(s"alpha: $alpha")
  //
  //    val results = new MutableList[(Double, Vector)]
  //
  //    val lambdaMult = exp(scala.math.log(lamShrnk) / numIterations)
  //
  //    //val sw = new StopWatch()
  //    Timer("initLambda").start
  //    val (xy, lambdaInit) = initLambda(data, alpha, sw, numFeatures, numRows)
  //    Timer("initLambda").end
  //
  //    val indexStart = xy.zipWithIndex.filter(xyi => abs(xyi._1) > (lambdaInit * lambdaMult * alpha)).map(_._2)
  //
  //    val xx = CDSparseMatrix(numFeatures, indexStart)
  //    populateXXMatrix(data, indexStart, xx, numFeatures, numRows)
  //
  //    loop(initialWeights, lambdaInit, numIterations)
  //
  //    /*loop to decrement lambda and perform iteration for betas*/
  //    @tailrec
  //    def loop(oldBeta: Vector, oldLambda: Double, n: Int): Unit = {
  //      if (n > 0) {
  //        println(s"lamda number: ${101 - n}")
  //        val newLambda = oldLambda * lambdaMult
  //        val newBeta = cdIter(data, oldBeta, newLambda, alpha, xy, xx, numFeatures, numRows)
  //        results += Pair(newLambda, newBeta.copy)
  //        loop(newBeta, newLambda, n - 1)
  //      }
  //    }
  //    println(s"CD ET: ${sw.elapsedTime / 1000} seconds")
  //    data.unpersist()
  //    println(s"totalNumNewBeta: $totalNumNewBeta")
  //    results.toList
  //  }
  //
  def populateXXMatrix(data: RDD[(Double, Vector)], newIndexes: Array[Int], xx: CDSparseMatrix, numFeatures: Int, numRows: Long): Unit = {
    Timer("xCorrelation").start
    val correlatedX = xCorrelation(data, newIndexes, numFeatures, numRows)
    Timer("xCorrelation").end
    Timer("xx.update").start
    xx.update(newIndexes, correlatedX)
    Timer("xx.update").end
  }

  var totalNumNewBeta = 0

  def xCorrelation(data: RDD[(Double, Vector)], newColIndexes: Array[Int], numFeatures: Int, numRows: Long): Array[Array[Double]] = {
    val numNewBeta = newColIndexes.size
    totalNumNewBeta += numNewBeta
    println(s"numNewBeta: $numNewBeta,  totalNumNewBeta: $totalNumNewBeta")

    val xx = data.treeAggregate(new XCorrelation(newColIndexes, numFeatures))(
      (aggregate, row) => aggregate.compute(row),
      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xx

    xx.map { _.map(_ / numRows) }
  }
  //
  //  /*Function to calculate starting lambda value*/
  //  def initLambda(data: RDD[(Double, Vector)], alpha: Double, sw: StopWatch, numFeatures: Int, numRows: Long): (Array[Double], Double) = {
  //    sw.restart
  //
  //    val xy = data.treeAggregate(new InitLambda(numFeatures))(
  //      (aggregate, row) => aggregate.compute(row),
  //      (aggregate1, aggregate2) => aggregate1.combine(aggregate2)).xy
  //
  //    //logRDD("data after persist", data)
  //    //unpersist.unpersist()
  //    //logRDD("data after persist and labelsAndFeatures unpersist", data)
  //
  //    //def maxBy[B](f: (A) ⇒ B): A
  //    //Finds the first element which yields the largest value measured by function f
  //    //val maxXY = xy.maxBy(abs) / numRows
  //    val maxXY = xy.map(abs).max(Ordering.Double) / numRows
  //    val lambdaInit = maxXY / alpha
  //
  //    (xy.map(_ / numRows), lambdaInit)
  //  }

  def S(z: Double, gamma: Double): Double = if (gamma >= abs(z)) 0.0 else (z / abs(z)) * (abs(z) - gamma)

  def cdIter(data: RDD[(Double, Vector)], oldBeta: Vector, newLambda: Double, alpha: Double, xy: Array[Double], xx: CDSparseMatrix, numFeatures: Int, numRows: Long): Vector = {
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

    loop(oldBeta, 100.0, true, numCDIter)
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

//http://rgg.zone/2014/12/07/the-cost-of-laziness/
//http://docs.scala-lang.org/sips/pending/improved-lazy-val-initialization.html
//http://docs.scala-lang.org/sips/sip-list.html
//http://stackoverflow.com/questions/3041253/whats-the-hidden-cost-of-scalas-lazy-val

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