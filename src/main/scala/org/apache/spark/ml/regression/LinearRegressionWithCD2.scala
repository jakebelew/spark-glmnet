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

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.{ ParamMap, Params, IntParam, ParamValidators }
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.feature.{ StandardScaler, StandardScalerModel }
import org.apache.spark.mllib.linalg.{ DenseMatrix, DenseVector, Vector, Vectors }
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.mlmatrix.RowPartionedTransformer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{ MatrixMultivariateOnlineSummarizer => Summarizer }
import org.apache.spark.mllib.optimization.{ CoordinateDescent2, CDOptimizer }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter
import scala.collection.mutable.MutableList
import breeze.linalg.{ *, DenseMatrix => BDM }

//Modifed from org.apache.spark.ml.regression.LinearRegression

/**
 * Params for linear regression.
 */
class LinearRegressionWithCD2(override val uid: String)
  extends LinearRegressionWithCD {

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
    fit2(dataset, fitMultiModel, paramMaps)
  }

  override protected def train(dataset: DataFrame): LinearRegressionWithCDModel = {
    fit2(dataset, fitSingleModel)(0)
  }

  private def newOptimizer = new CoordinateDescent2()
  //if ($(optimizerVersion) == 2) new CoordinateDescent2() else new CoordinateDescent()

  private val fitMultiModel = (normalizedInstances: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats2, paramMaps: Array[ParamMap]) => {
    val boundaryIndices = new Range(0, paramMaps.length, $(numLambdas))
    val models = new MutableList[LinearRegressionWithCDModel]

    boundaryIndices.foreach(index => {
      val optimizer = newOptimizer
      copyValues(optimizer)
      copyValues(optimizer, paramMaps(index))

      val (lambdas, rawWeights) = optimizer.optimize(normalizedInstances, initialWeights, xy, stats.numFeatures, numRows).unzip
      //rawWeights.foreach { rw => logDebug(s"Raw Weights ${rw.toArray.mkString(",")}") }

      models ++= rawWeights.map(rw => createModel(rw.toArray, stats))
    })

    models
  }

  private val fitSingleModel = (normalizedInstances: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats2, paramMaps: Array[ParamMap]) => {
    val optimizer = newOptimizer
    copyValues(optimizer)

    logDebug(s"Best fit lambda index: ${$(lambdaIndex)}")
    //val rawWeights = optimizer.optimize(normalizedInstances, initialWeights, xy, $(lambdaIndex), stats.numFeatures, numRows).toArray
    val rawWeights = optimizer.optimize(normalizedInstances, initialWeights, xy, stats.numFeatures, numRows)($(lambdaIndex))._2.toArray
    val model = createModel(rawWeights, stats)
    Seq(model)
  }

  // f: (normalizedInstances: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], numRows: Long, stats: Stats2, paramMaps: Array[ParamMap])
  private def fit2(dataset: DataFrame, f: (RDD[(DenseVector, DenseMatrix)], Vector, Array[Double], Long, Stats2, Array[ParamMap]) => Seq[LinearRegressionWithCDModel], paramMaps: Array[ParamMap] = Array()): Seq[LinearRegressionWithCDModel] = {
    val (normalizedInstances, stats, handlePersistence) = normalizeDataSet(dataset)

    // If the yStd is zero, then the intercept is yMean with zero weights;
    // as a result, training is not needed.
    if (stats.yStd == 0.0) {
      if (handlePersistence) normalizedInstances.unpersist()
      return createModelsWithInterceptAndWeightsOfZeros(dataset, stats.yMean, stats.numFeatures)
    }

    //TODO - remove after verifying count is correct
    logDebug(s"stats.numRows: ${stats.numRows}")
    val numRows = stats.numRows
    val initialWeights = Vectors.zeros(stats.numFeatures)

    val xy = newOptimizer.computeXY(normalizedInstances, stats.numFeatures, numRows)
    logDebug(s"xy: ${xy.mkString(",")}")

    val models = f(normalizedInstances, initialWeights, xy, numRows, stats, paramMaps)

    if (handlePersistence) normalizedInstances.unpersist()

    models
  }

  case class Stats2(summarizer: Summarizer, statCounter: StatCounter) {
    val numRows = statCounter.count
    val numFeatures = summarizer.mean.size
    val yMean = statCounter.mean
    val yStd = statCounter.sampleStdev
    lazy val featuresMean = summarizer.mean
    lazy val featuresStd = Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
  }

  private def normalizeDataSet(dataset: DataFrame): (RDD[(DenseVector, DenseMatrix)], Stats2, Boolean) = {
    val denseRDD = RowPartionedTransformer.labeledPointsToMatrix(dataset)

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    //TODO - Need to add logWarn that persisting dataset before here is probably not needed and will slow LR down -
    //and/or do we persist this anyway and unpersist the initial data?
    if (handlePersistence) denseRDD.persist(StorageLevel.MEMORY_AND_DISK)

    val (summarizer, statCounter) = denseRDD.treeAggregate(
      (new Summarizer, new StatCounter))(
        seqOp = (c, v) => (c, v) match {
          case ((summarizer: Summarizer, statCounter: StatCounter),
            (labels: DenseVector, features: DenseMatrix)) =>
            statCounter.merge(labels.toArray)
            summarizer.add(features)
            (summarizer, statCounter)
        },
        combOp = (c1, c2) => (c1, c2) match {
          case ((summarizer1: Summarizer, statCounter1: StatCounter),
            (summarizer2: Summarizer, statCounter2: StatCounter)) =>
            (summarizer1.merge(summarizer2), statCounter1.merge(statCounter2))
        })

    val stats = Stats2(summarizer, statCounter)

    denseRDD.foreach {
      case (labels: DenseVector, features: DenseMatrix) =>
        normalizePartition(labels, features, stats)
    }

    (denseRDD, stats, handlePersistence)
  }

  private def normalizePartition(labels: DenseVector, features: DenseMatrix, stats: Stats2) = {
    val bLabels = labels.toBreeze
    bLabels :-= stats.yMean
    bLabels :/= stats.yStd
    val bFeatures = new BDM(features.numRows, features.numCols, features.values)(*, ::)
    bFeatures :-= stats.featuresMean.toBreeze
    bFeatures :/= stats.featuresStd.toBreeze
  }

  //TODO - temp until all speed and func testing is complete and a final version selected
  private def createModel(rawWeights: Array[Double], stats: Stats2): LinearRegressionWithCDModel = {
    super.createModel(rawWeights, new Stats(
      new StandardScalerModel(Vectors.dense(stats.yStd +: (stats.featuresStd.toArray)),
        Vectors.dense(stats.yMean +: (stats.featuresMean.toArray)))))
  }

  override def copy(extra: ParamMap): LinearRegressionWithCD2 = defaultCopy(extra)
}