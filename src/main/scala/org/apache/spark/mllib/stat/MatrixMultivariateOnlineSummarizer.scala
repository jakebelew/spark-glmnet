package org.apache.spark.mllib.stat

import org.apache.spark.mllib.linalg.Matrix

class MatrixMultivariateOnlineSummarizer extends MultivariateOnlineSummarizer_Modified {

  /**
   * Add a new sample to this summarizer, and update the statistical summary.
   *
   * @param sample The sample in dense/sparse matrix format to be added into this summarizer.
   * @return This MatrixMultivariateOnlineSummarizer object.
   */
  def add(sample: Matrix): this.type = {
    if (n == 0) {
      require(sample.numRows * sample.numCols > 0, s"Matrix should have dimension larger than zero.")
      n = sample.numCols

      currMean = Array.ofDim[Double](n)
      currM2n = Array.ofDim[Double](n)
      currM2 = Array.ofDim[Double](n)
      currL1 = Array.ofDim[Double](n)
      nnz = Array.ofDim[Double](n)
      currMax = Array.fill[Double](n)(Double.MinValue)
      currMin = Array.fill[Double](n)(Double.MaxValue)
    }

    require(n == sample.numCols, s"Dimensions mismatch when adding new sample." +
      s" Expecting $n columns but got ${sample.numCols}.")

    val localCurrMean = currMean
    val localCurrM2n = currM2n
    val localCurrM2 = currM2
    val localCurrL1 = currL1
    val localNnz = nnz
    val localCurrMax = currMax
    val localCurrMin = currMin
    sample.foreachActive { (_, index, value) =>
      if (value != 0.0) {
        if (localCurrMax(index) < value) {
          localCurrMax(index) = value
        }
        if (localCurrMin(index) > value) {
          localCurrMin(index) = value
        }

        val prevMean = localCurrMean(index)
        val diff = value - prevMean
        localCurrMean(index) = prevMean + diff / (localNnz(index) + 1.0)
        localCurrM2n(index) += (value - localCurrMean(index)) * diff
        localCurrM2(index) += value * value
        localCurrL1(index) += math.abs(value)

        localNnz(index) += 1.0
      }
    }

    totalCnt += sample.numRows
    this
  }
}