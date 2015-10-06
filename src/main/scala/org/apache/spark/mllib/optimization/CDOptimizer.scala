package org.apache.spark.mllib.optimization

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector

//TODO - Temporary trait to allow testing multiple versions of CoordinateDescent with minimum code duplication
trait CDOptimizer extends Serializable with CoordinateDescentParams {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], numFeatures: Int, numRows: Long): List[(Double, Vector)]

  //def optimize(data: RDD[(Double, Vector)], initialWeights: Vector, xy: Array[Double], lambdaIndex: Int, numFeatures: Int, numRows: Long): Vector

  def computeXY(data: RDD[(Double, Vector)], numFeatures: Int, numRows: Long): Array[Double]
}