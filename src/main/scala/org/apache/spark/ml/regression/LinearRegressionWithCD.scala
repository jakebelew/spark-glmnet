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

package org.apache.spark.ml.regression

import org.apache.spark.mllib.feature.{ StandardScaler, StandardScalerModel }
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable.MutableList
import org.apache.spark.mllib.optimization.CDOptimizer
import org.apache.spark.mllib.optimization.CoordinateDescent
import org.apache.spark.mllib.optimization.CoordinateDescent2

//Modifed from org.apache.spark.ml.regression.LinearRegression

/**
 * Params for linear regression.
 */
private[spark] trait LinearRegressionWithCDParams extends PredictorParams with HasLambdaIndex with HasLogSaveAll with HasElasticNetParam
  with HasNumLambdas with HasMaxIter with HasTol with HasLambdaShrink with HasFitIntercept with HasOptimizerVersion {

  def setLambdaIndex(value: Int): this.type = set(lambdaIndex, value)
  setDefault(lambdaIndex -> 99)

  //TODO - Temporary param to allow testing multiple versions of CoordinateDescent with minimum code duplication
  def setOptimizerVersion(value: Int): this.type = set(optimizerVersion, value)
  setDefault(optimizerVersion -> 1)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0.01 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.01.
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.01)

  /**
   * Set the number of Lambdas.
   * Default is 100.
   * @group setParam
   */
  def setNumLambdas(value: Int): this.type = set(numLambdas, value)
  setDefault(numLambdas -> 100)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-3)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-3.
   * @group setParam
   */
  def setLambdaShrink(value: Double): this.type = set(lambdaShrink, value)
  setDefault(lambdaShrink -> 1E-3)

  /**
   * Set whether to fit intercept.
   * Default is false.
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
   * Set whether to put intermediate values in a log file.
   * Default is false.
   * @group setParam
   */
  def setLogSaveAll(value: Boolean): this.type = set(logSaveAll, value)
  setDefault(logSaveAll -> false)
}

class LinearRegressionWithCD(override val uid: String)
  extends Regressor[Vector, LinearRegressionWithCD, LinearRegressionWithCDModel]
  with LinearRegressionWithCDParams with Logging {

  def this() = this(Identifiable.randomUID("linReg"))

  /**
   * Fits multiple models to the input data with multiple sets of parameters.
   * The default implementation uses a for loop on each parameter map.
   * Subclasses could override this to optimize multi-model training.
   *
   * @param dataset input dataset
   * @param paramMaps An array of parameter maps.
   *                  These values override any specified in this Estimator's embedded ParamMap.
   * @return fitted models, matching the input parameter maps
   */
  override def fit(dataset: DataFrame, paramMaps: Array[ParamMap]): Seq[LinearRegressionWithCDModel] = {
    fit(dataset, fitMultiModel, paramMaps)
  }

  override protected def train(dataset: DataFrame): LinearRegressionWithCDModel = {
    fit(dataset, fitSingleModel)(0)
  }

  private def newOptimizer =
    if ($(optimizerVersion) == 2) new CoordinateDescent2() else new CoordinateDescent()

  private val fitMultiModel = (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats, paramMaps: Array[ParamMap]) => {
    val boundaryIndices = new Range(0, paramMaps.length, $(maxIter))
    val models = new MutableList[LinearRegressionWithCDModel]

    boundaryIndices.foreach(index => {
      //copy(paramMap).fit(dataset)    
      val alphaS = paramMaps(index).get(elasticNetParam).getOrElse(1.0)
      val optimizer = newOptimizer
        //.setAlpha($(elasticNetParam))
        .setElasticNetParam(alphaS)
      //TODO set these parameters
      //  var elasticNetParam: Double = 0.01
      //  var lambdaShrink: Double = 0.001
      //  var maxIter: Int = 100
      //  var tol: Double = 1E-3
      //  var logSaveAll: Boolean = false

      val (lambdas, rawWeights) = optimizer.optimize(normalizedInstances, initialWeights, xy, stats.numFeatures, numRows).unzip
      //rawWeights.foreach { rw => logDebug(s"Raw Weights ${rw.toArray.mkString(",")}") }

      models ++= rawWeights.map(rw => createModel(rw.toArray, stats))
    })

    models
  }

  private val fitSingleModel = (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats, paramMaps: Array[ParamMap]) => {
    val optimizer = newOptimizer
      .setElasticNetParam($(elasticNetParam))

    logDebug(s"Best fit lambda index: ${$(lambdaIndex)}")
    val rawWeights = optimizer.optimize(normalizedInstances, initialWeights, xy, $(lambdaIndex), stats.numFeatures, numRows).toArray
    val model = createModel(rawWeights, stats)
    Seq(model)
  }

  // f: (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats, paramMaps: Array[ParamMap])
  private def fit(dataset: DataFrame, f: (RDD[(Double, Vector)], Vector, Array[Double], Long, Stats, Array[ParamMap]) => Seq[LinearRegressionWithCDModel], paramMaps: Array[ParamMap] = Array()): Seq[LinearRegressionWithCDModel] = {
    // paramMaps.map(fit(dataset, _))

    val (normalizedInstances, scalerModel) = normalizeDataSet(dataset)
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) normalizedInstances.persist(StorageLevel.MEMORY_AND_DISK)

    val stats = Stats(scalerModel)
    //logDebug(s"stats:\nnumFeatures: ${stats.numFeatures}\nyMean: ${stats.yMean}\nStd: ${stats.yStd}\nfeaturesMean: ${stats.featuresMean.mkString(",")}\nfeaturesStd: ${stats.featuresStd.mkString(",")}")

    // If the yStd is zero, then the intercept is yMean with zero weights;
    // as a result, training is not needed.
    if (stats.yStd == 0.0) {
      if (handlePersistence) normalizedInstances.unpersist()
      return createModelsWithInterceptAndWeightsOfZeros(dataset, stats.yMean, stats.numFeatures)
    }

    val numRows = normalizedInstances.count
    val initialWeights = Vectors.zeros(stats.numFeatures)

    val xy = newOptimizer.computeXY(normalizedInstances, stats.numFeatures, numRows)
    logDebug(s"xy: ${xy.mkString(",")}")

    val models = f(normalizedInstances, initialWeights, xy, numRows, stats, paramMaps)

    if (handlePersistence) normalizedInstances.unpersist()

    models
  }

  case class Stats(scalerModel: StandardScalerModel) {
    val numFeatures = scalerModel.mean.size - 1
    val yMean = scalerModel.mean.toArray(0)
    val yStd = scalerModel.std.toArray(0)
    lazy val featuresMean = scalerModel.mean.toArray.drop(1)
    lazy val featuresStd = scalerModel.std.toArray.drop(1)
  }

  private def normalizeDataSet(dataset: DataFrame): (RDD[(Double, Vector)], StandardScalerModel) = {
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => Vectors.dense(label +: features.toArray)
    }

    val scalerModel = new StandardScaler(withMean = true, withStd = true)
      .fit(instances)

    val normalizedInstances = scalerModel
      .transform(instances)
      .map(row => (row.toArray.take(1)(0), Vectors.dense(row.toArray.drop(1))))

    (normalizedInstances, scalerModel)
  }

  private def createModelsWithInterceptAndWeightsOfZeros(dataset: DataFrame, yMean: Double, numFeatures: Int): Seq[LinearRegressionWithCDModel] = {
    logWarning(s"The standard deviation of the label is zero, so the weights will be zeros " +
      s"and the intercept will be the mean of the label; as a result, training is not needed.")

    val weights = Vectors.sparse(numFeatures, Seq())
    val intercept = yMean

    val model = new LinearRegressionWithCDModel(uid, weights, intercept)
    //    val trainingSummary = new LinearRegressionTrainingSummary(
    //      model.transform(dataset),
    //      $(predictionCol),
    //      $(labelCol),
    //      $(featuresCol),
    //      Array(0D))
    //    Seq(copyValues(model.setSummary(trainingSummary)))
    Seq(copyValues(model))
  }

  private def createModel(rawWeights: Array[Double], stats: Stats): LinearRegressionWithCDModel = {
    /* The weights are trained in the scaled space; we're converting them back to the original space. */
    val weights = {
      var i = 0
      val len = rawWeights.length
      while (i < len) {
        rawWeights(i) *= { if (stats.featuresStd(i) != 0.0) stats.yStd / stats.featuresStd(i) else 0.0 }
        i += 1
      }
      Vectors.dense(rawWeights).compressed
    }
    //logDebug(s"Weights ${weights.toArray.mkString(",")}")

    /*
       The intercept in R's GLMNET is computed using closed form after the coefficients are
       converged. See the following discussion for detail.
       http://stats.stackexchange.com/questions/13617/how-is-the-intercept-computed-in-glmnet
     */
    //val intercept = if ($(fitIntercept)) yMean - dot(weights, Vectors.dense(featuresMean)) else 0.0
    val intercept = stats.yMean - dot(weights, Vectors.dense(stats.featuresMean))

    val model = copyValues(new LinearRegressionWithCDModel(uid, weights, intercept))
    //    val trainingSummary = new LinearRegressionTrainingSummary(
    //      model.transform(dataset),
    //      $(predictionCol),
    //      $(labelCol),
    //      $(featuresCol),
    //      objectiveHistory)
    //    model.setSummary(trainingSummary)
    model
  }

  override def copy(extra: ParamMap): LinearRegressionWithCD = defaultCopy(extra)
}

/**
 * :: Experimental ::
 * Model produced by [[LinearRegression]].
 */
@Experimental
class LinearRegressionWithCDModel private[ml] (
  override val uid: String,
  val weights: Vector,
  val intercept: Double)
  extends RegressionModel[Vector, LinearRegressionWithCDModel]
  with LinearRegressionParams {

  /** The order of weights - from largest to smallest. Returns the indexes of the weights in descending order of the absolute value. */
  def orderOfWeights(): Array[Int] =
    weights.toArray.map(math.abs).zipWithIndex.sortBy(-_._1).map(_._2).toArray

  override protected def predict(features: Vector): Double = {
    dot(features, weights) + intercept
  }

  override def copy(extra: ParamMap): LinearRegressionWithCDModel = {
    copyValues(new LinearRegressionWithCDModel(uid, weights, intercept), extra)
  }
}

object ModelLogger extends Logging {
  override def logInfo(msg: => String) = { super.logInfo(msg) }
}

