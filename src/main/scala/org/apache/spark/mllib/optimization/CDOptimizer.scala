package org.apache.spark.mllib.optimization

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ DenseVector, DenseMatrix, Vector }

//TODO - Temporary trait to allow testing multiple versions of CoordinateDescent with minimum code duplication
trait CDOptimizer extends Serializable with CoordinateDescentParams {

  def optimize(data: RDD[(DenseVector, DenseMatrix)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)]

  def computeXY(data: RDD[(DenseVector, DenseMatrix)], numFeatures: Int, numRows: Long): Array[Double]
}