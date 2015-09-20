package org.apache.spark.mllib.optimization

import java.lang.Math.abs
import scala.Array.canBuildFrom
import scala.collection.JavaConversions.seqAsJavaList
import scala.collection.mutable.MutableList
import scala.util.Sorting
import org.apache.spark.mllib.linalg.Matrix

/** SparseMatrix optimized for Coordinate Descent algorithm. */
protected case class CDSparseMatrix2(numFeatures: Int, startingActiveIndices: Array[Int]) {

  type Vector = breeze.linalg.Vector[Double]
  type SparseVector = breeze.linalg.SparseVector[Double]

  val xx = Array.ofDim[SparseVector](numFeatures)

  private var empty = true

  // newXX is JxK
  def update(newActiveIndices: Array[Int], newXX: Matrix) = {
    if (empty) initialUpdate(newActiveIndices, newXX)
    else regularUpdate(newActiveIndices, newXX)
  }

  private def initialUpdate(newActiveIndices: Array[Int], newXX: Matrix) = {
    empty = false
    val numIndices = newActiveIndices.size
    var j = 0
    while (j < numFeatures) {
      xx(j) = new SparseVector(newActiveIndices, getValues(numIndices, j, newXX), numFeatures)
      j += 1
    }
  }

  private def getValues(numIndices: Int, j: Int, newXX: Matrix): Array[Double] = {
    val values = Array.ofDim[Double](numIndices)
    val numK = newXX.numCols
    var k = 0
    while (k < numK) {
      values(k) = newXX(j,k)
      k += 1
    }
    values
  }

  private def regularUpdate(newActiveIndices: Array[Int], newXX: Matrix) = {
    val xxIndices = xx(0).index

    val (newXXIndices, xxTranslation, newXXTranslation) = computeTranslationIndices(xxIndices, newActiveIndices)

    val numIndices = newXXIndices.size
    var j = 0
    while (j < numFeatures) {
      xx(j) = new SparseVector(newXXIndices, combineValues(numIndices, j, xxTranslation, newXX, newXXTranslation), numFeatures)
      j += 1
    }
  }

  private def combineValues(numIndices: Int, j: Int, xxTranslation: Array[Int], newXX: Matrix, newXXTranslation: Array[Int]): Array[Double] = {
    val values = Array.ofDim[Double](numIndices)

    val xxjData = xx(j).data
    val numK = xxjData.size
    var k = 0
    while (k < numK) {
      values(xxTranslation(k)) = xxjData(k)
      k += 1
    }

    val numNewK = newXX.numCols
    k = 0
    while (k < numNewK) {
      values(newXXTranslation(k)) = newXX(j,k)
      k += 1
    }

    values
  }

  // This requires that xxIndices and newActiveIndices are already sorted. 
  // Another tuple value (the original indices) would need to be added to the computation to allow unsorted newActiveIndices.
  private def computeTranslationIndices(xxIndices: Array[Int], newActiveIndices: Array[Int]): (Array[Int], Array[Int], Array[Int]) = {

    val indicesWithSource = Array.concat(xxIndices.map((_, true)), newActiveIndices.map((_, false)))

    Sorting.quickSort(indicesWithSource)(Ordering.by[(Int, Boolean), Int](_._1))

    val newXXIndices = indicesWithSource.map(_._1)

    val indicesWithSourceAndTranslationIndices = indicesWithSource.zipWithIndex

    val xxTranslation = indicesWithSourceAndTranslationIndices.filter(_._1._2).map(_._2)

    val newXXTranslation = indicesWithSourceAndTranslationIndices.filter(!_._1._2).map(_._2)

    (newXXIndices, xxTranslation, newXXTranslation)
  }

  def dot(j: Int, b: Vector): Double = {
    xx(j) dot b
  }

  override def toString() = xx.map(_.toString).mkString("\n")

  private val inactiveIndices = new java.util.LinkedList((0 until numFeatures).toList)

  startingActiveIndices.foreach {
    inactiveIndices.removeFirstOccurrence(_)
  }

  def newIndices(beta: Vector): MutableList[Int] = {
    val newIndices = new MutableList[Int]
    val iterator = inactiveIndices.iterator()
    while (iterator.hasNext()) {
      val j = iterator.next()
      if (abs(beta(j)) > 0) {
        newIndices += j
        iterator.remove()
      }
    }
    newIndices //.toList
  }
}
