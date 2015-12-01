package org.apache.spark.mllib.optimization

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.Logging
import org.apache.spark.ml.classification.Stats3
import org.apache.spark.mllib.{ FileUtil, TestUtil }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import scala.collection.mutable.MutableList
import scala.math.{ abs, exp, sqrt }
import TempTestUtil.verifyResults

private[spark] class LogisticCoordinateDescent3 extends CoordinateDescentParams
  with Logging {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    LogisticCoordinateDescent3.runCD(
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
object LogisticCoordinateDescent3 extends Logging {

  def runCD(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], alpha: Double, lamShrnk: Double, numLambdas: Int, maxIter: Int, tol: Double, stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    logInfo(s"Performing coordinate descent with: [elasticNetParam: $alpha, lamShrnk: $lamShrnk, numLambdas: $numLambdas, maxIter: $maxIter, tol: $tol]")

    println("LogisticCoordinateDescent3")

    // val lambdas = computeLambdas(xy, alpha, lamShrnk, numLambdas, numLambdas, numRows): Array[Double]
    // optimize(data, initialWeights, xy, lambdas, alpha, lamShrnk, maxIter, tol, numFeatures, numRows)
    runScala(data, stats, numRows)
    //Array.fill[(Double, Vector)](numLambdas)((0.0, Vectors.zeros(stats.numFeatures))).toList
  }

  //private def computeLambdas(xy: Array[Double], alpha: Double, lamShrnk: Double, lambdaRange: Int, numLambdas: Int, numRows: Long): Array[Double] = {
  private def computeLambdas(lamdaInit: Double, lambdaMult: Double, numLambdas: Int): Array[Double] = {
    //logDebug(s"alpha: $alpha, lamShrnk: $lamShrnk, maxIter: $lambdaRange, numRows: $numRows")

    //    val maxXY = xy.map(abs).max(Ordering.Double)
    //    val lambdaInit = maxXY / alpha
    //
    //    val lambdaMult = exp(scala.math.log(lamShrnk) / lambdaRange)
    //
    //    val lambdas = new MutableList[Double]
    //
    //    loop(lambdaInit, numLambdas)
    //
    //    /*loop to decrement lambda and perform iteration for betas*/
    //    @tailrec
    //    def loop(oldLambda: Double, n: Int): Unit = {
    //      if (n > 0) {
    //        val newLambda = oldLambda * lambdaMult
    //        lambdas += newLambda
    //        loop(newLambda, n - 1)
    //      }
    //    }
    //    logDebug(s"lambdas: ${lambdas.mkString(",")}")
    //    lambdas.toArray
    //TODO - The following Array.iterate method can be used in the other CoordinateDescent objects to replace 13 lines of code with 1 line
    Array.iterate[Double](lamdaInit * lambdaMult, numLambdas)(_ * lambdaMult)
  }

  private def runScala(data: RDD[(Double, Vector)], stats: Stats3, numRows: Long): List[(Double, Vector)] = {
    val (labelsSeq, xNormalizedSeq) = data.toArray.unzip
    val labels = labelsSeq.toArray
    val xNormalized = xNormalizedSeq.map(_.toArray).toArray

    //number of rows and columns in x matrix
    val nrow = numRows.toInt
    val ncol = stats.numFeatures

    val alpha = 1.0

    //Do Not Normalize labels but do calculate averages
    val meanLabel = stats.yMean
    val sdLabel = stats.yStd

    //initialize probabilities and weights
    var sumWxr = Array.ofDim[Double](ncol)
    var sumWxx = Array.ofDim[Double](ncol)
    var sumWr = 0.0
    var sumW = 0.0

    //calculate starting points for betas
    for (iRow <- 0 until nrow) {
      val p = meanLabel
      val w = p * (1.0 - p)
      //residual for logistic
      val r = (labels(iRow) - p) / w
      val x = xNormalized(iRow)
      sumWxr = (for (i <- 0 until ncol) yield (sumWxr(i) + w * x(i) * r)).toArray
      sumWxx = (for (i <- 0 until ncol) yield (sumWxx(i) + w * x(i) * x(i))).toArray
      sumWr = sumWr + w * r
      sumW = sumW + w
    }
    val avgWxr = for (i <- 0 until ncol) yield sumWxr(i) / nrow
    val avgWxx = for (i <- 0 until ncol) yield sumWxx(i) / nrow

    var maxWxr = 0.0
    for (i <- 0 until ncol) {
      val value = abs(avgWxr(i))
      maxWxr = if (value > maxWxr) value else maxWxr
    }
    //calculate starting value for lambda
    var lam = maxWxr / alpha

    //this value of lambda corresponds to beta = list of 0's
    //initialize a vector of coefficients beta
    var beta = Array.ofDim[Double](ncol)
    var beta0 = sumWr / sumW

    //initialize matrix of betas at each step
    val betaMat = MutableList.empty[Array[Double]]
    betaMat += beta.clone

    val beta0List = MutableList.empty[Double]
    beta0List += beta0

    //begin iteration
    val nSteps = 100
    val lamMult = 0.93 //100 steps gives reduction by factor of 1000 in lambda (recommended by authors)
    val nzList = MutableList.empty[Int]
    for (iStep <- 0 until nSteps) {
      // OUTER LOOP --------------------------------------------------------------------------------------------------------------      
      //decrease lambda
      lam = lam * lamMult

      val (newBeta, newBeta0, newSumWr, newSumW) = outerLoop(iStep, labels, xNormalized, beta, beta0, sumWr, sumW, lam, alpha, ncol, numRows)
      beta = newBeta
      beta0 = newBeta0
      sumWr = newSumWr
      sumW = newSumW

      betaMat += newBeta.clone
      //println("betaMat:")
      //betaMat.foreach(f => println(f.mkString(",")))
      beta0List += newBeta0
      //println(s"betaIRLS: ${beta0List.mkString(",")}")

      val nzBeta = for (index <- 0 until ncol if beta(index) != 0.0) yield index
      for (q <- nzBeta) {
        if (!nzList.contains(q)) {
          nzList += q
        }
      }
      // END OUTER LOOP --------------------------------------------------------------------------------------------------------------      
    }

    //make up names for columns of xNum
    val names = for (i <- 0 until ncol) yield "V" + i
    val nameList = for (i <- 0 until nzList.length) yield names(nzList(i))

    println(nameList)

    verifyResults(stats, meanLabel, sdLabel, betaMat, beta0List)
    verifyResults(nameList.toList)

    //TODO - Return actual lambdas once they are supplied as a list. Also return the column order and put that into the model as part of the history.
    val fullBetas = beta0List.zip(betaMat).map { case (b0, beta) => Vectors.dense(b0 +: beta) }
    val lambdas = Array.ofDim[Double](fullBetas.size)
    lambdas.zip(fullBetas).toList
  }

  private def outerLoop(iStep: Int, labels: Array[Double], xNormalized: Array[Array[Double]], oldBeta: Array[Double], oldBeta0: Double, oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double, Double, Double) = {
    val nrow = numRows.toInt
    val ncol = numColumns
    val lam = lambda

    var sumWr = oldSumWr
    var sumW = oldSumW

    //Use incremental change in betas to control inner iteration

    //set middle loop values for betas = to outer values
    // values are used for calculating weights and probabilities
    //inner values are used for calculating penalized regression updates

    //take pass through data to calculate averages over data require for iteration
    //initilize accumulators

    var betaIRLS = oldBeta.clone
    val beta0IRLS = oldBeta0
    var distIRLS = 100.0
    //Middle loop to calculate new betas with fixed IRLS weights and probabilities
    var iterIRLS = 0
    while (distIRLS > 0.01) {
      // MIDDLE LOOP --------------------------------------------------------------------------------------------------------------      
      iterIRLS += 1

      val (newBetaIRLS, newDistIRLS, newSumWr, newSumW) = middleLoop(iStep, iterIRLS, labels, xNormalized, betaIRLS, beta0IRLS, sumWr, sumW, lam, alpha, ncol, numRows)
      betaIRLS = newBetaIRLS
      distIRLS = newDistIRLS
      sumWr = newSumWr
      sumW = newSumW

      //println(s"iterIRLS: $iterIRLS")
      //      var iterInner = 0
      //
      //      val betaInner = betaIRLS.clone
      //      var beta0Inner = beta0IRLS
      //      var distInner = 100.0
      //      while (distInner > 0.01 && iterInner < 100) {
      //        iterInner += 1
      //        //println(s"iterInner: $iterInner")
      //        //if (iterInner > 100) break
      //
      //        //cycle through attributes and update one-at-a-time
      //        //record starting value for comparison
      //        val betaStart = betaInner.clone
      //        for (iCol <- 0 until ncol) {
      //          var sumWxrC = 0.0
      //          var sumWxxC = 0.0
      //          sumWr = 0.0
      //          sumW = 0.0
      //          //println(s"sumWr: ${sumWr}, sumW: ${sumW}")
      //
      //          for (iRow <- 0 until nrow) {
      //            val x = xNormalized(iRow).clone
      //            val y = labels(iRow)
      //            val pr = Pr(beta0IRLS, betaIRLS, x)
      //            val (p, w) = if (abs(pr) < 1e-5) (0.0, 1e-5)
      //            else if (abs(1.0 - pr) < 1e-5) (1.0, 1e-5)
      //            else (pr, pr * (1.0 - pr))
      //            val z = (y - p) / w + beta0IRLS + (for (i <- 0 until ncol) yield (x(i) * betaIRLS(i))).sum
      //            val r = z - beta0Inner - (for (i <- 0 until ncol) yield (x(i) * betaInner(i))).sum
      //            sumWxrC += w * x(iCol) * r
      //            sumWxxC += w * x(iCol) * x(iCol)
      //            sumWr += w * r
      //            sumW += w
      //            //println(s"sumWxrC: ${sumWxrC}, sumWxxC: ${sumWxxC}, sumWr: ${sumWr}, sumW: ${sumW}")              
      //          }
      //          val avgWxr = sumWxrC / nrow
      //          val avgWxx = sumWxxC / nrow
      //
      //          beta0Inner = beta0Inner + sumWr / sumW
      //          //println(s"beta0Inner: ${beta0Inner}")
      //          val uncBeta = avgWxr + avgWxx * betaInner(iCol)
      //          betaInner(iCol) = S(uncBeta, lam * alpha) / (avgWxx + lam * (1.0 - alpha))
      //          //println(s"betaInner(iCol): ${betaInner(iCol)}")
      //        }
      //        val sumDiff = (for (n <- 0 until ncol) yield (abs(betaInner(n) - betaStart(n)))).sum
      //        val sumBeta = (for (n <- 0 until ncol) yield abs(betaInner(n))).sum
      //        distInner = sumDiff / sumBeta
      //        //println(s"distInner: ${distInner}")
      //      }
      //
      //      println(iStep, iterIRLS, iterInner)
      //
      //      //if exit inner while loop, then set betaMiddle = betaMiddle and run through middle loop again.
      //
      //      //Check change in betaMiddle to see if IRLS is converged
      //      val a = (for (i <- 0 until ncol) yield (abs(betaIRLS(i) - betaInner(i)))).sum
      //      val b = (for (i <- 0 until ncol) yield abs(betaIRLS(i))).sum
      //      distIRLS = a / (b + 0.0001)
      //      //println(s"distIRLS: ${distIRLS}")
      //      val dBeta = for (i <- 0 until ncol) yield (betaInner(i) - betaIRLS(i))
      //      val gradStep = 1.0
      //      val temp = for (i <- 0 until ncol) yield (betaIRLS(i) + gradStep * dBeta(i))
      //      betaIRLS = temp.toArray.clone
      //      //println(s"betaIRLS: ${betaIRLS.mkString(",")}")
      // END MIDDLE LOOP --------------------------------------------------------------------------------------------------------------      
    }

    val beta = betaIRLS.clone
    //println(s"beta: ${beta.mkString(",")}")
    val beta0 = beta0IRLS
    //println(s"beta0: ${beta0}")
    //      betaMat += beta.clone
    //      //println(s"betaMat: ${betaMat}")
    //      println("betaMat:")
    //      betaMat.foreach(f => println(f.mkString(",")))
    //      beta0List += beta0
    //      println(s"betaIRLS: ${beta0List.mkString(",")}")              
    //
    //      val nzBeta = for (index <- 0 until ncol if beta(index) != 0.0) yield index
    //      for (q <- nzBeta) {
    //        if (!nzList.contains(q)) {
    //          nzList += q
    //        }
    //      }
    (beta, beta0, sumWr, sumW)
  }

  //private def outerLoop(iStep: Int, labels: Array[Double], xNormalized: Array[Array[Double]], oldBeta: Array[Double], oldBeta0: Double, oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double, Double, Double) = {
  //private def middleLoop(oldBetaIRLS: Array[Double], oldBeta0IRLS: Double) = {
  private def middleLoop(iStep: Int, iterIRLS: Int, labels: Array[Double], xNormalized: Array[Array[Double]], oldBetaIRLS: Array[Double], oldBeta0IRLS: Double, oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double, Double, Double) = {
    var betaIRLS = oldBetaIRLS
    val beta0IRLS = oldBeta0IRLS
    val nrow = numRows.toInt
    val ncol = numColumns
    val lam = lambda

    var sumWr = oldSumWr
    var sumW = oldSumW

    var iterInner = 0

    val betaInner = betaIRLS.clone
    var beta0Inner = beta0IRLS
    var distInner = 100.0
    while (distInner > 0.01 && iterInner < 100) {
      // INNER LOOP --------------------------------------------------------------------------------------------------------------      
      iterInner += 1

      val (newDistInner, newSumWr, newSumW, newBeta0Inner, newBetaInner) = innerLoop(labels, xNormalized, betaInner, beta0Inner, beta0IRLS, betaIRLS, sumWr, sumW, lambda, alpha, numColumns, numRows)
      //                                                                         innerLoop(labels, xNormalized, oldBetaInner, oldBeta0Inner, beta0IRLS, betaIRLS, oldSumWr, oldSumW, lambda, alpha, numColumns, numRows): (Double, Double, Double, Double, Array[Double]) = {

      //innerLoop(labels: Array[Double], xNormalized: Array[Array[Double]], oldBetaInner: Array[Double], oldBeta0Inner: Double, beta0IRLS: Double, betaIRLS: Array[Double], oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Double, Double, Double, Double, Array[Double]) = {
      //    (distInner, sumWr, sumW, beta0Inner, betaInner)
      //betaIRLS = newBetaIRLS
      distInner = newDistInner
      sumWr = newSumWr
      sumW = newSumW
      beta0Inner = newBeta0Inner
      //betaInner = newBetaInner
      //println(s"iterInner: $iterInner")
      //if (iterInner > 100) break

      //      //cycle through attributes and update one-at-a-time
      //      //record starting value for comparison
      //      val betaStart = betaInner.clone
      //      for (iCol <- 0 until ncol) {
      //        var sumWxrC = 0.0
      //        var sumWxxC = 0.0
      //        sumWr = 0.0
      //        sumW = 0.0
      //        //println(s"sumWr: ${sumWr}, sumW: ${sumW}")
      //
      //        for (iRow <- 0 until nrow) {
      //          val x = xNormalized(iRow).clone
      //          val y = labels(iRow)
      //          val pr = Pr(beta0IRLS, betaIRLS, x)
      //          val (p, w) = if (abs(pr) < 1e-5) (0.0, 1e-5)
      //          else if (abs(1.0 - pr) < 1e-5) (1.0, 1e-5)
      //          else (pr, pr * (1.0 - pr))
      //          val z = (y - p) / w + beta0IRLS + (for (i <- 0 until ncol) yield (x(i) * betaIRLS(i))).sum
      //          val r = z - beta0Inner - (for (i <- 0 until ncol) yield (x(i) * betaInner(i))).sum
      //          sumWxrC += w * x(iCol) * r
      //          sumWxxC += w * x(iCol) * x(iCol)
      //          sumWr += w * r
      //          sumW += w
      //          //println(s"sumWxrC: ${sumWxrC}, sumWxxC: ${sumWxxC}, sumWr: ${sumWr}, sumW: ${sumW}")              
      //        }
      //        val avgWxr = sumWxrC / nrow
      //        val avgWxx = sumWxxC / nrow
      //
      //        beta0Inner = beta0Inner + sumWr / sumW
      //        //println(s"beta0Inner: ${beta0Inner}")
      //        val uncBeta = avgWxr + avgWxx * betaInner(iCol)
      //        betaInner(iCol) = S(uncBeta, lam * alpha) / (avgWxx + lam * (1.0 - alpha))
      //        //println(s"betaInner(iCol): ${betaInner(iCol)}")
      //      }
      //      val sumDiff = (for (n <- 0 until ncol) yield (abs(betaInner(n) - betaStart(n)))).sum
      //      val sumBeta = (for (n <- 0 until ncol) yield abs(betaInner(n))).sum
      //      distInner = sumDiff / sumBeta
      //println(s"distInner: ${distInner}")
      // END INNER LOOP --------------------------------------------------------------------------------------------------------------      
    }

    println(iStep, iterIRLS, iterInner)

    //if exit inner while loop, then set betaMiddle = betaMiddle and run through middle loop again.

    //Check change in betaMiddle to see if IRLS is converged
    val a = (for (i <- 0 until ncol) yield (abs(betaIRLS(i) - betaInner(i)))).sum
    val b = (for (i <- 0 until ncol) yield abs(betaIRLS(i))).sum
    val distIRLS = a / (b + 0.0001)
    //println(s"distIRLS: ${distIRLS}")
    val dBeta = for (i <- 0 until ncol) yield (betaInner(i) - betaIRLS(i))
    val gradStep = 1.0
    val temp = for (i <- 0 until ncol) yield (betaIRLS(i) + gradStep * dBeta(i))
    betaIRLS = temp.toArray.clone
    //println(s"betaIRLS: ${betaIRLS.mkString(",")}")  

    (betaIRLS, distIRLS, sumWr, sumW)
  }

  //private def middleLoop(iStep: Int, iterIRLS: Int, labels: Array[Double], xNormalized: Array[Array[Double]], oldBetaIRLS: Array[Double], oldBeta0IRLS: Double, oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Array[Double], Double, Double, Double) = {
  private def innerLoop(labels: Array[Double], xNormalized: Array[Array[Double]], oldBetaInner: Array[Double], oldBeta0Inner: Double, beta0IRLS: Double, betaIRLS: Array[Double], oldSumWr: Double, oldSumW: Double, lambda: Double, alpha: Double, numColumns: Int, numRows: Long): (Double, Double, Double, Double, Array[Double]) = {
    val nrow = numRows.toInt
    val ncol = numColumns
    val lam = lambda

    var sumWr = oldSumWr
    var sumW = oldSumW
    var beta0Inner = oldBeta0Inner
    var betaInner = oldBetaInner

    //cycle through attributes and update one-at-a-time
    //record starting value for comparison
    val betaStart = betaInner.clone
    for (iCol <- 0 until ncol) {
      var sumWxrC = 0.0
      var sumWxxC = 0.0
      sumWr = 0.0
      sumW = 0.0
      //println(s"sumWr: ${sumWr}, sumW: ${sumW}")

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
      betaInner(iCol) = S(uncBeta, lam * alpha) / (avgWxx + lam * (1.0 - alpha))
      //println(s"betaInner(iCol): ${betaInner(iCol)}")
    }
    val sumDiff = (for (n <- 0 until ncol) yield (abs(betaInner(n) - betaStart(n)))).sum
    val sumBeta = (for (n <- 0 until ncol) yield abs(betaInner(n))).sum
    val distInner = sumDiff / sumBeta

    (distInner, sumWr, sumW, beta0Inner, betaInner)
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
}