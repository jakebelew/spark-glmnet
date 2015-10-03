package nonsubmit.utils

import org.apache.spark.{ SparkContext, SparkConf, Logging }
import org.apache.spark.mllib.linalg.{ DenseMatrix, Vectors, DenseVector }
import org.apache.spark.mllib.linalg.mlmatrix.RowPartionedTransformer
import org.apache.spark.mllib.stat.{ MatrixMultivariateOnlineSummarizer => Summarizer }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ SQLContext, DataFrame }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter

//Transform Input Dataset to a normalized and persisted RDD[(Vector, Matrix)]
//
//The RDD[(Vector, Matrix)] contains a single row per partition. 
//Each row contains a tuple (y: RowPartitionedVector, x: RowPartitionedMatrix) where the Vector and Matrix contain the rows for that partition.
//
//LinearRegressionWithCDExample --------------------------
//generate an RDD[LabeledPoints] 
//convert to DataFrame
//
//LinearRegressionWithCD ---------------------------------
//convert to RDD[(Vector, Matrix)]
//persist()
//calculate statistics		ACTION - RDD[(Vector, Matrix)] is actually persisted 
//normalize data in place         
//
//(NO double reads or double memory this way)
//
//CoordinateDescent  ---------------------------------
//RDD[(Vector, Matrix)] - Much faster BLAS operations

//LinearRegressionWithCDExample
object DataTransformPrototype extends Logging {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("DataTransformPrototype").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val training = org.apache.spark.mllib.util.LinearDataGenerator.generateLinearRDD(sc, 2000, 2000, 0.1, 2, 6.2)

    new LR().fit(training.toDF())

    sc.stop()
  }
}

//LinearRegressionWithCD
class LR extends Logging { //with HasLabelCol with HasFeaturesCol {

  private def normalizeDataSet(dataset: DataFrame): (RDD[(DenseVector, DenseMatrix)], Stats) = {
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

    val stats = Stats(summarizer, statCounter)

    //TODO - normalize in place to avoid double memory or double reads, but map so that a partition can be recreated correctly? 
    //TODO - Test this on AWS by stopping an executor instance on a long running LR
    denseRDD.foreach {
      case (labels: DenseVector, features: DenseMatrix) =>
      //TODO - perform in-place normalization with Breeze
    }

    (denseRDD, stats)
  }

  case class Stats(summarizer: Summarizer, statCounter: StatCounter) {
    val numFeatures = summarizer.mean.size
    val yMean = statCounter.mean
    val yStd = statCounter.sampleStdev
    lazy val featuresMean = summarizer.mean
    lazy val featuresStd = Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
  }

  def fit(dataset: DataFrame) = {
    val (normalizedInstances, stats) = normalizeDataSet(dataset)
    println(s"stats:\nnumFeatures: ${stats.numFeatures}\nyMean: ${stats.yMean}\nStd: ${stats.yStd}\nfeaturesMean: ${stats.featuresMean}\nfeaturesStd: ${stats.featuresStd}")
    println(s"\nnormalizedInstances:\n${normalizedInstances.toDebugString}")
  }

  //  private def fit(dataset: DataFrame, f: (RDD[(Double, Vector)], Vector, Array[Double], Long, Stats, Array[ParamMap]) => Seq[LinearRegressionWithCDModel], paramMaps: Array[ParamMap] = Array()): Seq[LinearRegressionWithCDModel] = {
  //    // paramMaps.map(fit(dataset, _))
  //
  //    val (normalizedInstances, scalerModel) = normalizeDataSet(dataset)
  //    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
  //    if (handlePersistence) normalizedInstances.persist(StorageLevel.MEMORY_AND_DISK)
  //
  //    val stats = Stats(scalerModel)
  //
  //    // If the yStd is zero, then the intercept is yMean with zero weights;
  //    // as a result, training is not needed.
  //    if (stats.yStd == 0.0) {
  //      if (handlePersistence) normalizedInstances.unpersist()
  //      return createModelsWithInterceptAndWeightsOfZeros(dataset, stats.yMean, stats.numFeatures)
  //    }
  //
  //    val numRows = normalizedInstances.count
  //    val initialWeights = Vectors.zeros(stats.numFeatures)
  //
  //    val xy = newOptimizer.computeXY(normalizedInstances, stats.numFeatures, numRows)
  //    logDebug(s"xy: ${xy.mkString(",")}")
  //
  //    val models = f(normalizedInstances, initialWeights, xy, numRows, stats, paramMaps)
  //
  //    if (handlePersistence) normalizedInstances.unpersist()
  //
  //    models //TODO - .compressed
  //  }  
}
