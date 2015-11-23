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

package org.apache.spark.ml.classification

import org.apache.spark.annotation.Experimental
import org.apache.spark.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.{ HasOptimizerVersion, HasLambdaIndex, HasLambdaShrink, HasNumLambdas, Regressor, RegressionModel }
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.feature.{ StandardScaler, StandardScalerModel }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.optimization.{ CoordinateDescentParams, LogisticCoordinateDescent1 }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer_Modified
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter
import scala.collection.mutable.MutableList

//Modifed from org.apache.spark.ml.regression.LinearRegression

/**
 * Params for linear regression.
 */
private[spark] trait LogisticRegressionWithCDParams extends PredictorParams with HasLambdaIndex with HasElasticNetParam
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
}

class LogisticRegressionWithCD1(override val uid: String)
  extends Regressor[Vector, LogisticRegressionWithCD1, LogisticRegressionWithCDModel]
  with LogisticRegressionWithCDParams with Logging {

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
  override def fit(dataset: DataFrame, paramMaps: Array[ParamMap]): Seq[LogisticRegressionWithCDModel] = {
    fit(dataset, fitMultiModel, paramMaps)
  }

  override protected def train(dataset: DataFrame): LogisticRegressionWithCDModel = {
    fit(dataset, fitSingleModel)(0)
  }

  private def newOptimizer = new LogisticCoordinateDescent1()

  private val fitMultiModel = (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats3, paramMaps: Array[ParamMap]) => {
    val boundaryIndices = new Range(0, paramMaps.length, $(numLambdas))
    val models = new MutableList[LogisticRegressionWithCDModel]

    boundaryIndices.foreach(index => {
      val optimizer = newOptimizer
      copyValues(optimizer)
      copyValues(optimizer, paramMaps(index))

      val (lambdas, rawWeights) = optimizer.optimize(normalizedInstances, initialWeights, xy, stats, numRows).unzip
      //rawWeights.foreach { rw => logDebug(s"Raw Weights ${rw.toArray.mkString(",")}") }

      models ++= rawWeights.map(rw => createModel(rw.toArray, stats))
    })

    models
  }

  private val fitSingleModel = (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats3, paramMaps: Array[ParamMap]) => {
    val optimizer = newOptimizer
    copyValues(optimizer)

    logDebug(s"Best fit lambda index: ${$(lambdaIndex)}")
    //val rawWeights = optimizer.optimize(normalizedInstances, initialWeights, xy, $(lambdaIndex), stats.numFeatures, numRows).toArray
    val rawWeights = optimizer.optimize(normalizedInstances, initialWeights, xy, stats, numRows)($(lambdaIndex))._2.toArray
    val model = createModel(rawWeights, stats)
    Seq(model)
  }

  // f: (normalizedInstances: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats, paramMaps: Array[ParamMap])
  private def fit(dataset: DataFrame, f: (RDD[(Double, Vector)], Vector, Array[Double], Long, Stats3, Array[ParamMap]) => Seq[LogisticRegressionWithCDModel], paramMaps: Array[ParamMap] = Array()): Seq[LogisticRegressionWithCDModel] = {
    // paramMaps.map(fit(dataset, _))

    val (normalizedInstances, stats) = normalizeDataSet(dataset)
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) normalizedInstances.persist(StorageLevel.MEMORY_AND_DISK)

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

  private def normalizeDataSet(dataset: DataFrame): (RDD[(Double, Vector)], Stats3) = {
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }

    // Calculate statistics but do not normalize labels
    val (summarizer, statCounter) = instances.treeAggregate(
      (new MultivariateOnlineSummarizer_Modified, new StatCounter))(
        seqOp = (c, v) => (c, v) match {
          case ((summarizer: MultivariateOnlineSummarizer_Modified, statCounter: StatCounter),
            (label: Double, features: Vector)) =>
            (summarizer.add(features), statCounter.merge(label))
        },
        combOp = (c1, c2) => (c1, c2) match {
          case ((summarizer1: MultivariateOnlineSummarizer_Modified, statCounter1: StatCounter),
            (summarizer2: MultivariateOnlineSummarizer_Modified, statCounter2: StatCounter)) =>
            (summarizer1.merge(summarizer2), statCounter1.merge(statCounter2))
        })

    val stats = Stats3(summarizer, statCounter)

    val scalerModel = new StandardScalerModel(stats.featuresStd, stats.featuresMean, true, true)

    val normalizedInstances = instances
      .map { case (label, features) => (label, scalerModel.transform(features)) }

    (normalizedInstances, stats)
  }

  protected[classification] def createModelsWithInterceptAndWeightsOfZeros(dataset: DataFrame, yMean: Double, numFeatures: Int): Seq[LogisticRegressionWithCDModel] = {
    logWarning(s"The standard deviation of the label is zero, so the weights will be zeros " +
      s"and the intercept will be the mean of the label; as a result, training is not needed.")

    val weights = Vectors.sparse(numFeatures, Seq())
    val intercept = yMean

    val model = new LogisticRegressionWithCDModel(uid, weights, intercept)
    //    val trainingSummary = new LogisticRegressionTrainingSummary(
    //      model.transform(dataset),
    //      $(predictionCol),
    //      $(labelCol),
    //      $(featuresCol),
    //      Array(0D))
    //    Seq(copyValues(model.setSummary(trainingSummary)))
    Seq(copyValues(model))
  }

  protected[classification] def createModel(rawWeights: Array[Double], stats: Stats3): LogisticRegressionWithCDModel = {
    /* The weights are trained in the scaled space; we're converting them back to the original space. */
    val weights = {
      //TODO - Logistic coordinate descent has added beta0 (the intercept) to the rawWeights. Is this correct? 
      var i = 0
      val len = rawWeights.length - 1
      while (i < len) {
        rawWeights(i + 1) *= { if (stats.featuresStd(i) != 0.0) stats.yStd / stats.featuresStd(i) else 0.0 }
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
    //TODO - Logistic coordinate descent has added beta0 (the intercept) to the rawWeights. Is this correct? 
    //val intercept = if ($(fitIntercept)) yMean - dot(weights, Vectors.dense(featuresMean)) else 0.0
    //val intercept = stats.yMean - dot(weights, stats.featuresMean)
    val intercept = rawWeights(0)
    val model = copyValues(new LogisticRegressionWithCDModel(uid, weights, intercept))
    //    val trainingSummary = new LogisticRegressionTrainingSummary(
    //      model.transform(dataset),
    //      $(predictionCol),
    //      $(labelCol),
    //      $(featuresCol),
    //      objectiveHistory)
    //    model.setSummary(trainingSummary)
    model
  }

  protected[classification] def copyValues(optimizer: CoordinateDescentParams, map: ParamMap) = {
    params.foreach { param =>
      if (map.contains(param)) {
        //logDebug(s"Copy ParamMap values: [param.name: ${param.name}, param.value: ${map(param)}, param.type: ${param.getClass().getName()}]")
        param.name match {
          case "elasticNetParam" => optimizer.setElasticNetParam(map(param).asInstanceOf[Double])
          case "lambdaShrink" => optimizer.setLambdaShrink(map(param).asInstanceOf[Double])
          case "numLambdas" => optimizer.setNumLambdas(map(param).asInstanceOf[Int])
          case "maxIter" => optimizer.setMaxIter(map(param).asInstanceOf[Int])
          case "tol" => optimizer.setTol(map(param).asInstanceOf[Double])
          case _ =>
        }
      }
    }
  }

  protected[classification] def copyValues(optimizer: CoordinateDescentParams) = {
    params.foreach { param =>
      //logDebug(s"Copy LR values: [param.name: ${param.name}, param.value: ${$(param)}, param.type: ${param.getClass().getName()}]")
      param.name match {
        case "elasticNetParam" => optimizer.setElasticNetParam($(elasticNetParam))
        case "lambdaShrink" => optimizer.setLambdaShrink($(lambdaShrink))
        case "numLambdas" => optimizer.setNumLambdas($(numLambdas))
        case "maxIter" => optimizer.setMaxIter($(maxIter))
        case "tol" => optimizer.setTol($(tol))
        case _ =>
      }
    }
  }

  override def copy(extra: ParamMap): LogisticRegressionWithCD1 = defaultCopy(extra)
}

case class Stats3(summarizer: MultivariateOnlineSummarizer_Modified, statCounter: StatCounter) {
  val numRows = statCounter.count
  val numFeatures = summarizer.mean.size
  val yMean = statCounter.mean
  val yStd = statCounter.sampleStdev
  val featuresMean = summarizer.mean
  val featuresStd = Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
}

/**
 * :: Experimental ::
 * Model produced by [[LogisticRegression]].
 */
@Experimental
class LogisticRegressionWithCDModel private[ml] (
  override val uid: String,
  val weights: Vector,
  val intercept: Double)
  extends RegressionModel[Vector, LogisticRegressionWithCDModel]
  with LogisticRegressionWithCDParams {

  /** The order of weights - from largest to smallest. Returns the indexes of the weights in descending order of the absolute value. */
  def orderOfWeights(): Array[Int] =
    weights.toArray.map(math.abs).zipWithIndex.sortBy(-_._1).map(_._2).toArray

  override protected def predict(features: Vector): Double = {
    dot(features, weights) + intercept
  }

  override def copy(extra: ParamMap): LogisticRegressionWithCDModel = {
    copyValues(new LogisticRegressionWithCDModel(uid, weights, intercept), extra)
  }
}

object ModelLogger extends Logging {
  override def logInfo(msg: => String) = { super.logInfo(msg) }
}

