package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg.DenseMatrix

class XCorrelation2Scala {
  def compute(m: DenseMatrix, newColIndexes: Array[Int]): DenseMatrix = {
    //println(s"m.numRows: ${m.numRows}, m.numCols: ${m.numCols}")
    //println(s"newColIndexes: ${newColIndexes.mkString(",")}")
    val xx = DenseMatrix.zeros(m.numCols, newColIndexes.size)
    //println(s"xx.numRows: ${xx.numRows}, xx.numCols: ${xx.numCols}")
    var i = 0
    var k = 0
    var j = 0
    while (i < m.numRows) {
      while (k < newColIndexes.size) {
        while (j < m.numCols) {
          //println(s"i: $i, j: $j, k: $k")
          xx(j, k) += m(i, j) * m(i, newColIndexes(k))
          j += 1
        }
        j = 0
        k += 1
      }
      k = 0
      i += 1
    }
    xx
  }
}