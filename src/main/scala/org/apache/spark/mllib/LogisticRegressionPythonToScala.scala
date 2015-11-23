package org.apache.spark.mllib

import scala.collection.mutable.MutableList
import scala.io.Source
import scala.math.{ abs, exp, sqrt }

object LogisticRegressionPythonToScala extends App {
  //__author__ = 'mike_bowles'
  //import urllib2
  //import sys
  //from math import sqrt, fabs, exp
  //import matplotlib.pyplot as plot

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

  //read data from uci data repository
  //target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
  //data = urllib2.urlopen(target_url)
  val data = FileUtil.readFile("data/sonar.all-data")

  //arrange data into list for labels and list of lists for attributes
  val xList = MutableList.empty[Array[String]]

  for (line <- data) {
    //split on comma
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

  val alpha = 1.0

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
    //Middle loop to calculate new betas with fixed IRLS weights and probabilities
    var iterIRLS = 0
    while (distIRLS > 0.01) {
      iterIRLS += 1
      var iterInner = 0.0

      val betaInner = betaIRLS.clone
      var beta0Inner = beta0IRLS
      var distInner = 100.0
      while (distInner > 0.01 && iterInner < 100) {
        iterInner += 1
        //if (iterInner > 100) break

        //cycle through attributes and update one-at-a-time
        //record starting value for comparison
        val betaStart = betaInner.clone
        for (iCol <- 0 until ncol) {
          var sumWxrC = 0.0
          var sumWxxC = 0.0
          sumWr = 0.0
          sumW = 0.0

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
          }
          val avgWxr = sumWxrC / nrow
          val avgWxx = sumWxxC / nrow

          beta0Inner = beta0Inner + sumWr / sumW
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
  //for i in range(ncol):
  //    //plot range of beta values for each attribute
  //    coefCurve = (betaMat(k](i] for k in range(nSteps)]
  //    xaxis = range(nSteps)
  //    plot.plot(xaxis, coefCurve)
  //
  //plot.xlabel("Steps Taken")
  //plot.ylabel("Coefficient Values")
  //plot.show()

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

    val expectedBetaMat = FileUtil.readFile("results/logistic-regression/betaMatWithBesselsCorrection.txt")
      .map(_.split(",").map(_.toDouble)).toArray
    TestUtil.equalWithinTolerance(betaMat.toArray, expectedBetaMat, tolerance, "betas")

    val expectedBeta0List = FileUtil.readFile("results/logistic-regression/beta0List.txt")(0)
      .split(",").map(_.toDouble).toArray
    TestUtil.equalWithinTolerance(beta0List.toArray, expectedBeta0List, tolerance, "beta0s")

    val expectedNamelist = FileUtil.readFile("results/logistic-regression/namelist.txt")(0)
      .split(",").map(_.trim).toArray
    TestUtil.equal(nameList.toArray, expectedNamelist, "columnOrder")
  }
}

object FileUtil {
  def readFile(filename: String): List[String] = {
    val bufferedSource = Source.fromFile(filename)
    val lines = bufferedSource.getLines.toList
    bufferedSource.close
    lines
  }
}
