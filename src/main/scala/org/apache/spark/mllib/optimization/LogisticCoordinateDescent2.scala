package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.Logging
import org.apache.spark.ml.classification.Stats3
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import scala.collection.mutable.MutableList
import scala.math.{ abs, exp, sqrt }
import scala.annotation.tailrec
import TempTestUtil.verifyResults

private[spark] class LogisticCoordinateDescent2 extends CoordinateDescentParams
  with Logging {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    println("\nExecuting LogisticCoordinateDescent2\n")
    LogisticCoordinateDescent2.runCD(
      data,
      initialWeights,
      xy,
      elasticNetParam,
      lambdaShrink,
      numLambdas,
      maxIter,
      tol,
      stats,
      numRows)
  }

  //TODO - Temporary to allow testing multiple versions of CoordinateDescent with minimum code duplication - remove to Object method only later
  def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double] = {
    //CoordinateDescent.computeXY(data, numFeatures, numRows)
    Array.ofDim[Double](100)
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run coordinate descent.
 */
@DeveloperApi
object LogisticCoordinateDescent2 extends Logging {

  def runCD(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numLambdas: Int, maxIter: Int, tol: Double, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    logInfo(s"Performing coordinate descent with: [elasticNetParam: $alpha, lamShrnk: $lamShrnk, numLambdas: $numLambdas, maxIter: $maxIter, tol: $tol]")

    val (labelsSeq, xNormalizedSeq) = data.toArray.unzip
    val labels = labelsSeq.toArray
    val xNormalized = xNormalizedSeq.map(_.toArray).toArray
    val lamMult = 0.93

    val (lambdas, initialBeta0) = computeLambdasAndInitialBeta0(labels, xNormalized, alpha, lamMult, numLambdas, stats, numRows)
    // optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, maxIter, tol, numFeatures, numRows)
    val lambdasAndBetas = optimize(labels, xNormalized, lambdas, initialBeta0, alpha, stats, numRows)

    //TODO - Return the column order and put that into the model as part of the history. Or better yet, 
    // columnOrder should be calculated in the example code from the List of models containing the beta history using a util class
    val columnOrder = determineColumnOrder(lambdasAndBetas.unzip._2)
    lambdasAndBetas
  }

  //private def computeLambdas(xy: Array[Double], alpha: Double, lamShrnk: Double, lambdaRange: Int, numLambdas: Int, numRows: Long): Array[Double] = {
  //private def computeLambdasAndInitialBeta0(lamdaInit: Double, lambdaMult: Double, numLambdas: Int, stats: Stats3, numRows: Long): Array[Double] = {
  private def computeLambdasAndInitialBeta0(labels: Array[Double], xNormalized: Array[Array[Double]], alpha: Double, lambdaMult: Double, numLambdas: Int, stats: Stats3, numRows: Long): (Array[Double], Double) = {
    //logDebug(s"alpha: $alpha, lamShrnk: $lamShrnk, maxIter: $lambdaRange, numRows: $numRows")

    //    val maxXY = xy.map(abs).max(Ordering.Double)
    //    val lambdaInit = maxXY / alpha
    //
    //    val lambdaMult = exp(scala.math.log(lamShrnk) / lambdaRange)
    //    
    //---------------------------------------------------------------------------------------------------

    val nrow = numRows.toInt
    val ncol = stats.numFeatures
    val meanLabel = stats.yMean

    //calculate starting points for betas
    val (sumWxr, sumWxx, sumWr, sumW) = new ComputeInitialWeights(meanLabel).aggregate(labels, xNormalized, nrow, ncol)

    val avgWxr = for (i <- 0 until ncol) yield sumWxr(i) / nrow
    val avgWxx = for (i <- 0 until ncol) yield sumWxx(i) / nrow

    var maxWxr = 0.0
    for (i <- 0 until ncol) {
      val value = abs(avgWxr(i))
      maxWxr = if (value > maxWxr) value else maxWxr
    }
    //calculate starting value for lambda
    val lamdaInit = maxWxr / alpha

    //this value of lambda corresponds to beta = list of 0's
    val beta0 = sumWr / sumW

    // val lambdaMult = 0.93 //100 steps gives reduction by factor of 1000 in lambda (recommended by authors)

    //TODO - The following Array.iterate method can be used in the other CoordinateDescent objects to replace 13 lines of code with 1 line
    val lambdas = Array.iterate[Double](lamdaInit * lambdaMult, numLambdas)(_ * lambdaMult)
    (lambdas, beta0)
  }

  private class ComputeInitialWeights(p: Double) {
    val w = p * (1.0 - p)

    def aggregate(labels: Array[Double], xNormalized: Array[Array[Double]], nrow: Int, ncol: Int): (Array[Double], Array[Double], Double, Double) = {
      var sumWxr = Array.ofDim[Double](ncol)
      var sumWxx = Array.ofDim[Double](ncol)
      var sumWr = 0.0
      var sumW = 0.0

      for (iRow <- 0 until nrow) {
        //residual for logistic
        val r = (labels(iRow) - p) / w
        val x = xNormalized(iRow)
        val wr = w * r
        sumWxr = (for (i <- 0 until ncol) yield (sumWxr(i) + wr * x(i))).toArray
        sumWxx = (for (i <- 0 until ncol) yield (sumWxx(i) + w * x(i) * x(i))).toArray
        sumWr = sumWr + wr
        sumW = sumW + w
      }
      (sumWxr, sumWxx, sumWr, sumW)
    }
  }

  private def optimize(labels: Array[Double], xNormalized: Array[Array[Double]], lambdas: Array[Double], initialBeta0: Double, alpha: Double, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    //initial value of lambda corresponds to beta = list of 0's
    var beta = Array.ofDim[Double](stats.numFeatures)
    val beta0 = initialBeta0

    val betaMat = MutableList.empty[Array[Double]]
    //TODO - Do not return the initial beta value of all zero's, only return a list of 100
    betaMat += beta.clone

    //TODO - beta0 does not change value, so no need to collect it in a list    
    val beta0List = MutableList.empty[Double]
    beta0List += beta0

    val nzList = MutableList.empty[Int]

    loop(beta, 0)

    /*loop to decrement lambda and perform iteration for betas*/
    @tailrec
    def loop(oldBeta: Array[Double], n: Int): Unit = {
      if (n < lambdas.length) {
        val newLambda = lambdas(n)
        val newBeta = outerLoop(n + 1, labels, xNormalized, oldBeta, beta0, newLambda, alpha, stats.numFeatures, numRows)
        betaMat += newBeta.clone
        beta0List += beta0
        loop(newBeta, n + 1)
      }
    }

    verifyResults(stats, stats.yMean, stats.yStd, betaMat, beta0List)

    //TODO - Do not combine beta0 with the other betas, return it as a separate member of the tuples -> List[(Double, Double, Vector)]
    val fullBetas = beta0List.zip(betaMat).map { case (b0, beta) => Vectors.dense(b0 +: beta) }
    lambdas.zip(fullBetas).toList
  }

  private def outerLoop(iStep: Int, labels: Array[Double], xNormalized: Array[Array[Double]], oldBeta: Array[Double], beta0: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): Array[Double] = {
    //Use incremental change in betas to control inner iteration
    //set middle loop values for betas = to outer values
    // values are used for calculating weights and probabilities
    //inner values are used for calculating penalized regression updates
    //take pass through data to calculate averages over data require for iteration
    //initilize accumulators

    def loop(iterIRLS: Int, betaIRLS: Array[Double], distIRLS: Double): Array[Double] = {
      if (distIRLS <= 0.01) betaIRLS
      else {
        val (newBetaIRLS, newDistIRLS) = middleLoop(iStep, iterIRLS, labels, xNormalized, betaIRLS, beta0, lambda, alpha, numColumns, numRows)
        loop(0, newBetaIRLS, newDistIRLS)
      }
    }

    loop(0, oldBeta, 100.0)
  }

  private def middleLoop(iStep: Int, iterIRLS: Int, labels: Array[Double], xNormalized: Array[Array[Double]], betaIRLS: Array[Double], beta0IRLS: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double) = {
    @tailrec
    def loop(iterInner: Int, distInner: Double, oldBeta0Inner: Double, mutableBetaInner: Array[Double]): (Int, Array[Double]) = {
      if (iterInner >= 100 || distInner <= 0.01) (iterInner, mutableBetaInner)
      else {
        val (newDistInner, newBeta0Inner) = innerLoop(labels, xNormalized, mutableBetaInner, oldBeta0Inner, beta0IRLS, betaIRLS, lambda, alpha, numColumns, numRows)
        loop(iterInner + 1, newDistInner, newBeta0Inner, mutableBetaInner)
      }
    }

    val (iterInner, betaInner) = loop(0, 100.0, beta0IRLS, betaIRLS.clone)

    println(iStep, iterIRLS, iterInner)

    //Check change in betaMiddle to see if IRLS is converged
    val a = (for (i <- 0 until numColumns) yield (abs(betaIRLS(i) - betaInner(i)))).sum
    val b = (for (i <- 0 until numColumns) yield abs(betaIRLS(i))).sum
    val distIRLS = a / (b + 0.0001)
    val dBeta = for (i <- 0 until numColumns) yield (betaInner(i) - betaIRLS(i))
    //val gradStep = 1.0
    //val newBetaIRLS = for (i <- 0 until numColumns) yield (betaIRLS(i) + gradStep * dBeta(i))
    val newBetaIRLS = for (i <- 0 until numColumns) yield (betaIRLS(i) + dBeta(i))
    (newBetaIRLS.toArray, distIRLS)
  }

  /** The betaInner input parameter will be mutated. */
  private def innerLoop(labels: Array[Double], xNormalized: Array[Array[Double]], betaInner: Array[Double], oldBeta0Inner: Double, beta0IRLS: Double, betaIRLS: Array[Double], lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Double, Double) = {
    val nrow = numRows.toInt
    val ncol = numColumns

    var beta0Inner = oldBeta0Inner

    //cycle through attributes and update one-at-a-time
    //record starting value for comparison
    val betaStart = betaInner.clone
    for (iCol <- 0 until ncol) {
      var sumWxrC = 0.0
      var sumWxxC = 0.0
      var sumWr = 0.0
      var sumW = 0.0

      for (iRow <- 0 until nrow) {
        val x = xNormalized(iRow).clone
        val y = labels(iRow)
        val pr = Pr(beta0IRLS, betaIRLS, x)
        val (p, w) = if (abs(pr) < 1e-5) (0.0, 1e-5)
        else if (abs(1.0 - pr) < 1e-5) (1.0, 1e-5)
        else (pr, pr * (1.0 - pr))
        val z = (y - p) / w + beta0IRLS + (for (i <- 0 until ncol) yield (x(i) * betaIRLS(i))).sum
        val r = z - beta0Inner - (for (i <- 0 until ncol) yield (x(i) * betaInner(i))).sum
        sumWxrC += w * x(iCol) * r
        sumWxxC += w * x(iCol) * x(iCol)
        sumWr += w * r
        sumW += w
        //println(s"sumWxrC: ${sumWxrC}, sumWxxC: ${sumWxxC}, sumWr: ${sumWr}, sumW: ${sumW}")              
      }
      val avgWxr = sumWxrC / nrow
      val avgWxx = sumWxxC / nrow

      beta0Inner = beta0Inner + sumWr / sumW
      //println(s"beta0Inner: ${beta0Inner}")
      val uncBeta = avgWxr + avgWxx * betaInner(iCol)
      betaInner(iCol) = S(uncBeta, lambda * alpha) / (avgWxx + lambda * (1.0 - alpha))
      //println(s"betaInner(iCol): ${betaInner(iCol)}")
    }
    val sumDiff = (for (n <- 0 until ncol) yield (abs(betaInner(n) - betaStart(n)))).sum
    val sumBeta = (for (n <- 0 until ncol) yield abs(betaInner(n))).sum
    val distInner = sumDiff / sumBeta

    (distInner, beta0Inner)
  }

  private def S(z: Double, gamma: Double): Double =
    if (gamma >= abs(z)) 0.0
    else if (z > 0.0) z - gamma
    else z + gamma

  private def Pr(b0: Double, b: Array[Double], x: Array[Double]): Double = {
    val n = x.length
    var sum = b0
    for (i <- 0 until n) {
      sum += b(i) * x(i)
      sum = if (sum < -100) -100 else sum
    }
    1.0 / (1.0 + exp(-sum))
  }

  private def determineColumnOrder(betas: List[Vector]): Array[Int] = {
    val nzList = betas
      .map(_.toArray.drop(1).zipWithIndex.filter(_._1 != 0.0).map(_._2))
      .flatMap(f => f)
      .distinct

    //make up names for columns of xNum
    val nameList = nzList.map(index => s"V$index")

    println(nameList)
    verifyResults(nameList)

    nzList.toArray
  }
}