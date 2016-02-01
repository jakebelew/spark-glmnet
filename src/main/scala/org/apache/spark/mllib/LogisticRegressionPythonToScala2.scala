package org.apache.spark.mllib

import scala.collection.mutable.MutableList
import scala.io.Source
import scala.math.{ abs, exp, sqrt }
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

// Temporary direct conversion from the Python implementation to Scala as preparation for conversion to Spark
// This version is modified to remove unnecessary repetitive calculations
object LogisticRegressionPythonToScala2 extends App {
  //__author__ = 'mike_bowles'

  def S(z: Double, gamma: Double): Double =
    if (gamma >= abs(z)) 0.0
    else if (z > 0.0) z - gamma
    else z + gamma

  def Pr(b0: Double, b: Array[Double], x: Array[Double]): Double = {
    val n = x.length
    var sum = b0
    for (i <- 0 until n) {
      sum += b(i) * x(i)
      sum = if (sum < -100) -100 else sum
    }
    1.0 / (1.0 + exp(-sum))
  }

  // does adjustments recommended by Friedman for num stability
  def adjPW(b0: Double, b: Array[Double], x: Array[Double]): (Double, Double) = {
    val pr = Pr(b0, b, x)
    if (abs(pr) < 1e-5) (0.0, 1e-5)
    else if (abs(1.0 - pr) < 1e-5) (1.0, 1e-5)
    else (pr, pr * (1.0 - pr))
  }

  def calcOuter(X: MutableList[Array[Double]], Y: MutableList[Double], beta0: Double, beta: Array[Double]) = {
    val nRow = X.length
    val nCol = X(0).length

    var wXX = DenseMatrix.zeros[Double](nCol, nCol)
    var wX = DenseVector.zeros[Double](nCol)
    var wXz = DenseVector.zeros[Double](nCol)
    var wZ = 0.0
    var wSum = 0.0

    for (iRow <- 0 until nRow) {
      val y = Y(iRow)
      val x = X(iRow)
      val (p, w) = adjPW(beta0, beta, x)
      val xNP = DenseVector(x)
      wXX += w * (xNP * xNP.t)
      wX += w * xNP
      // residual for logistic
      val z = (y - p) / w + beta0 + (for (i <- 0 until ncol) yield (x(i) * beta(i))).sum
      wXz += w * xNP * z
      wZ += w * z
      wSum += w
    }
    (wXX, wX, wXz, wZ, wSum)
  }

  //read data from uci data repository
  //"https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
  val data = FileUtil.readFile("data/sonar.all-data")

  val xList = MutableList.empty[Array[String]]

  for (line <- data) {
    val row = line.trim.split(",")
    xList += row
  }

  //separate labels from attributes, convert from attributes from string to numeric and convert "M" to 1 and "R" to 0
  val xNum = MutableList.empty[Array[Double]]
  val labels = MutableList.empty[Double]

  for (row <- xList) {
    val lastColChar = row.last
    val lastCol = if (lastColChar == "M") 1.0 else 0.0
    labels += lastCol
    val attrRow = row.take(row.length - 1).map(_.toDouble)
    xNum += attrRow
  }

  //number of rows and columns in x matrix
  val nrow = xNum.length
  val ncol = xNum(1).length

  val alpha = 0.8

  //calculate means and variances
  val xMeans = MutableList.empty[Double]
  val xSD = MutableList.empty[Double]
  for (i <- 0 until ncol) {
    val col = for (j <- 0 until nrow) yield xNum(j)(i)
    val mean = col.sum / nrow
    xMeans += mean
    val colDiff = for (j <- 0 until nrow) yield (xNum(j)(i) - mean)
    val sumSq = (for (i <- 0 until nrow) yield (colDiff(i) * colDiff(i))).sum
    val stdDev = sqrt(sumSq / (nrow - 1))
    xSD += stdDev
  }

  //use calculate mean and standard deviation to normalize xNum
  val xNormalized = MutableList.empty[Array[Double]]
  for (i <- 0 until nrow) {
    val rowNormalized = for (j <- 0 until ncol) yield (xNum(i)(j) - xMeans(j)) / xSD(j)
    xNormalized += rowNormalized.toArray
  }

  //Do Not Normalize labels but do calculate averages
  val meanLabel = labels.sum / nrow
  val sdLabel = sqrt((for (i <- 0 until nrow) yield ((labels(i) - meanLabel) * (labels(i) - meanLabel))).sum / (nrow - 1))

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
    //decrease lambda
    lam = lam * lamMult

    //Use incremental change in betas to control inner iteration

    //set middle loop values for betas = to outer values
    // values are used for calculating weights and probabilities
    //inner values are used for calculating penalized regression updates

    //take pass through data to calculate averages over data require for iteration
    //initilize accumulators

    var betaIRLS = beta.clone
    val beta0IRLS = beta0
    var distIRLS = 100.0

    //calculations for xx, wxx, etc
    val (wXX, wX, wXz, wZ, wSum) = calcOuter(xNormalized, labels, beta0IRLS, betaIRLS)

    //Middle loop to calculate new betas with fixed IRLS weights and probabilities
    var iterIRLS = 0
    while (distIRLS > 0.01) {
      iterIRLS += 1
      var iterInner = 0

      val betaInner = DenseVector(betaIRLS.clone)
      var beta0Inner = beta0IRLS
      var distInner = 100.0
      while (distInner > 0.01 && iterInner < 100) {
        iterInner += 1

        //cycle through attributes and update one-at-a-time
        //record starting value for comparison
        val betaStart = betaInner.toArray.clone
        for (iCol <- 0 until ncol) {
          val sumWxrC = wXz(iCol) - wX(iCol) * beta0Inner - (wXX(::, iCol) dot betaInner)
          val sumWxxC = wXX(iCol, iCol)
          val sumWrC = wZ - wSum * beta0Inner - (wX dot betaInner)
          val sumWC = wSum

          val avgWxr = sumWxrC / nrow
          val avgWxx = sumWxxC / nrow

          beta0Inner = beta0Inner + sumWrC / sumWC
          val uncBeta = avgWxr + avgWxx * betaInner(iCol)
          betaInner(iCol) = S(uncBeta, lam * alpha) / (avgWxx + lam * (1.0 - alpha))
        }
        val sumDiff = (for (n <- 0 until ncol) yield (abs(betaInner(n) - betaStart(n)))).sum
        val sumBeta = (for (n <- 0 until ncol) yield abs(betaInner(n))).sum
        distInner = sumDiff / sumBeta
      }

      println(iStep, iterIRLS, iterInner)

      //if exit inner while loop, then set betaMiddle = betaMiddle and run through middle loop again.

      //Check change in betaMiddle to see if IRLS is converged
      val a = (for (i <- 0 until ncol) yield (abs(betaIRLS(i) - betaInner(i)))).sum
      val b = (for (i <- 0 until ncol) yield abs(betaIRLS(i))).sum
      distIRLS = a / (b + 0.0001)
      val dBeta = for (i <- 0 until ncol) yield (betaInner(i) - betaIRLS(i))
      val gradStep = 1.0
      val temp = for (i <- 0 until ncol) yield (betaIRLS(i) + gradStep * dBeta(i))
      betaIRLS = temp.toArray.clone
    }

    beta = betaIRLS.clone
    beta0 = beta0IRLS
    betaMat += beta.clone
    beta0List += beta0

    val nzBeta = for (index <- 0 until ncol if beta(index) != 0.0) yield index
    for (q <- nzBeta) {
      if (!nzList.contains(q)) {
        nzList += q
      }
    }
  }

  //make up names for columns of xNum
  val names = for (i <- 0 until ncol) yield "V" + i
  val nameList = for (i <- 0 until nzList.length) yield names(nzList(i))

  println(nameList)

  verifyResults()

  def verifyResults() = {

    val tolerance = 1e-14
    val yTolerance = 1e-12

    val expectedXmeans = FileUtil.readFile("results/logistic-regression/xMeans.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(xMeans.toArray, expectedXmeans, tolerance, "xMeans")

    val expectedXSD = FileUtil.readFile("results/logistic-regression/xSDwithBesselsCorrection.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(xSD.toArray, expectedXSD, tolerance, "xSD")

    val expectedYmean = FileUtil.readFile("results/logistic-regression/yMean.txt")(0).toDouble
    TestUtil.equalWithinTolerance(meanLabel, expectedYmean, yTolerance, "yMean")

    val expectedYSD = FileUtil.readFile("results/logistic-regression/ySDwithBesselsCorrection.txt")(0).toDouble
    TestUtil.equalWithinTolerance(sdLabel, expectedYSD, yTolerance, "ySD")

    val expectedBetaMat = FileUtil.readFile("results/logistic-regression/betaMatWithBesselsCorrectionV2.txt")
      .map(_.split(",").map(_.toDouble)).toArray
    TestUtil.equalWithinTolerance(betaMat.toArray, expectedBetaMat, yTolerance, "betas")

    val expectedBeta0List = FileUtil.readFile("results/logistic-regression/beta0List.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(beta0List.toArray, expectedBeta0List, tolerance, "beta0s")

    val expectedNamelist = FileUtil.readFile("results/logistic-regression/namelistV2.txt")(0)
      .split(",").map(_.trim).toArray
    TestUtil.equal(nameList.toArray, expectedNamelist, "columnOrder")
  }
}

// Temporary test for this Scala version. Will use unit test similar to existing logistic regression test for Spark version.
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
