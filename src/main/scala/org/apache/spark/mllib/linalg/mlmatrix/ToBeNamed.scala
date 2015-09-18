//package edu.berkeley.cs.amplab.mlmatrix
package org.apache.spark.mllib.linalg.mlmatrix

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.rdd.RDD
//import org.apache.spark.mllib.regression.LabeledPoint

// Adapted from https://github.com/amplab/ml-matrix/blob/master/src/main/scala/edu/berkeley/cs/amplab/mlmatrix/RowPartitionedMatrix.scala
//TODO - RowPartitionedMatrix and DistributedMatrix are not needed now except for possible reference and can be removed from the code base later
//TODO - Investigate usage as a org.apache.spark.ml.Transformer
/** Note: [[breeze.linalg.DenseMatrix]] by default uses column-major layout. */
/**
 * Transforms RDD's into a single row per partition, allowing higher level BLAS operations to operate on the RDD contents.
 *  For example, Doubles can be converted to DenseVectors, Array[Double]s or DenseVectors can be converted to DenseMatrix's.
 */
object ToBeNamed {

  //TODO - Implement me. Probably want to make a function that allows arbitrary conversions.
  //def labeledPointsToMatrix(rdd: RDD[LabeledPoint]): RDD[(DenseVector, DenseMatrix)] = ???

  def arrayToMatrix(matrixRDD: RDD[Array[Double]]): RDD[DenseMatrix] = {
    val rowsColsPerPartition = matrixRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        if (iter.hasNext) {
          val nCols = iter.next().size
          Iterator((part, 1 + iter.size, nCols))
        } else {
          Iterator((part, 0, 0))
        }
    }.collect().sortBy(x => (x._1, x._2, x._3)).map(x => (x._1, (x._2, x._3))).toMap

    val rBroadcast = matrixRDD.context.broadcast(rowsColsPerPartition)

    val data = matrixRDD.mapPartitionsWithIndex {
      case (part, iter) =>
        val (rows, cols) = rBroadcast.value(part)
        val matData = new Array[Double](rows * cols)
        var nRow = 0
        while (iter.hasNext) {
          val arr = iter.next()
          var idx = 0
          while (idx < arr.size) {
            matData(nRow + idx * rows) = arr(idx)
            idx = idx + 1
          }
          nRow += 1
        }
        Iterator(new DenseMatrix(rows, cols, matData.toArray))
    }
    data
  }
}
