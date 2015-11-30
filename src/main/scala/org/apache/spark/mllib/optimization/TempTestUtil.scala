package org.apache.spark.mllib.optimization

import org.apache.spark.ml.classification.Stats3
import org.apache.spark.mllib.{ FileUtil, TestUtil }
import scala.collection.mutable.MutableList

object TempTestUtil {

  def verifyResults(stats: Stats3, meanLabel: Double, sdLabel: Double, betaMat: MutableList[Array[Double]], beta0List: MutableList[Double]) = {
//    val tolerance = 1e-12
//
//    val expectedXmeans = FileUtil.readFile("results/logistic-regression/xMeans.txt")(0)
//      .split(",").map(_.toDouble).toArray
//    TestUtil.equalWithinTolerance(stats.featuresMean.toArray, expectedXmeans, tolerance, "xMeans")
//
//    val expectedXSD = FileUtil.readFile("results/logistic-regression/xSDwithBesselsCorrection.txt")(0)
//      .split(",").map(_.toDouble).toArray
//    TestUtil.equalWithinTolerance(stats.featuresStd.toArray, expectedXSD, tolerance, "xSD")
//
//    val expectedYmean = FileUtil.readFile("results/logistic-regression/yMean.txt")(0).toDouble
//    TestUtil.equalWithinTolerance(meanLabel, expectedYmean, tolerance, "yMean")
//
//    val expectedYSD = FileUtil.readFile("results/logistic-regression/ySDwithBesselsCorrection.txt")(0).toDouble
//    TestUtil.equalWithinTolerance(sdLabel, expectedYSD, tolerance, "ySD")
//
//    val expectedBetaMat = FileUtil.readFile("results/logistic-regression/betaMatWithBesselsCorrection.txt")
//      .map(_.split(",").map(_.toDouble)).toArray
//    TestUtil.equalWithinTolerance(betaMat.toArray, expectedBetaMat, tolerance, "betas")
//
//    val expectedBeta0List = FileUtil.readFile("results/logistic-regression/beta0List.txt")(0)
//      .split(",").map(_.toDouble).toArray
//    TestUtil.equalWithinTolerance(beta0List.toArray, expectedBeta0List, tolerance, "beta0s")
//
//    println("\nResults Verified\n")
  }

  def verifyResults(nameList: List[String]) = {
//    val expectedNamelist = FileUtil.readFile("results/logistic-regression/namelist.txt")(0)
//      .split(",").map(_.trim).toArray
//    TestUtil.equal(nameList.toArray, expectedNamelist, "columnOrder")
//
//    println("\nnameList Verified\n")
  }
}