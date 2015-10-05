
// scalastyle:off println
package nonsubmit.utils

import org.apache.spark.mllib.optimization.XCorrelation2
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.optimization.XCorrelation2Scala
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.mllib.optimization.XCorrelation

object XCorrelationPerformanceComparison {

  private val ELEMENTS_PER_MiB = 131072

//  def main(args: Array[String]) {
//
//    if (args.length < 3) throw new Exception("required args: startNumRows startNumCols maxMemoryInBytes")
//
//    val startNumRows = args(0).toInt
//    val startNumCols = args(1).toInt
//    val maxMemoryInBytes = args(2).toLong
//
//    test(startNumRows, startNumCols, maxMemoryInBytes)
//    //functional correctness
//    //test(1, 1, 1000)
//
//    // square dataset
//    //test(1, 1, 1000000000)
//
//    // tall 10M x 10k dataset
//    //test(100, 1, 1000000000)
//
//    // wide 10k x 10M dataset
//    //test(1, 100, 1000000000)
//  }

  def test(startNumRows: Int, startNumCols: Int, maxMemoryInBytes: Long, growthMultiplier: Int = 2) = {
    println("format: (rows x columns)")
    println("for xx the number of columns equals the number of new indices\n")
    var numRows = startNumRows
    var numCols = startNumCols
    while (numRows * numCols * 8 <= maxMemoryInBytes) {
      val data = nonsubmit.utils.LinearDataGenerator.generateLinearInputAsMatrix(numRows, numCols)
      //val data = Array.ofDim[Double](numRows * numCols)
      //val data2 = nonsubmit.utils.LinearDataGenerator.generateLinearInputAsMatrix(numRows, numCols)
//      val rowData = nonsubmit.utils.LinearDataGenerator.generateLinearInput(numRows, numCols)
//      val data = nonsubmit.utils.LinearDataGenerator.toMatrix(rowData)
//      val rowDataFromLabelPoint = rowData.map(row => (0.0, Vectors.dense(row)))
      val memSizeInMib: Long = data.size / ELEMENTS_PER_MiB
      Thread.sleep(60000)
      println("-" * 80)
      println(s"data: ($numRows x $numCols) = $memSizeInMib MiB")
      //testNumNewIndices(numCols, rowDataFromLabelPoint, new DenseMatrix(numRows, numCols, data))
      testNumNewIndices(numCols, null, new DenseMatrix(numRows, numCols, data))
      numRows *= growthMultiplier
      numCols *= growthMultiplier
    }
  }

  def testNumNewIndices(numCols: Int, rowData: Array[(Double, Vector)], data: DenseMatrix, startNumNewIndices: Int = 1, indicesGrowthMultiplier: Int = 2) = {
    var numNewIndices = startNumNewIndices
    while (numNewIndices <= numCols) {
      //println(s"xx: $numNewIndices new indices -> $numCols rows x $numNewIndices columns = ${numCols * numNewIndices * 8 / 1048576} MiB")
      //println(s"xx:   $numCols rows x $numNewIndices columns = ${numCols * numNewIndices * 8 / BYTES_PER_MiB} MiB ($numNewIndices new indices)")
      //println(s"xx:   ($numCols x $numNewIndices) = ${numCols * numNewIndices * 8 / BYTES_PER_MiB} MiB ($numNewIndices new indices)")
      val xkMemSize: Long = (data.numRows * numNewIndices) / ELEMENTS_PER_MiB
      val xxMemSize: Long = (numCols * numNewIndices) / ELEMENTS_PER_MiB
      println(s"xk:   (${data.numRows} x $numNewIndices) = $xkMemSize MiB")
      println(s"xx:   ($numCols x $numNewIndices) = $xxMemSize MiB")
      testXcorrelation(rowData, data, numNewIndices)
      numNewIndices *= indicesGrowthMultiplier
    }
  }

  def testXcorrelation(rowData: Array[(Double, Vector)], data: DenseMatrix, numNewIndices: Int) = {
    val newColIndexes = DataGeneratorUtil.generateNewColIndexes(numNewIndices, data.numCols)

    //testScalaRowVectorsXcorrelation(rowData, newColIndexes, data.numCols)

    //much slower than ScalaRowVectorsXcorrelation - can probably be speeded up if necessary by direct indexing of array
    //testScalaMatrixXcorrelation(data, newColIndexes)

    testBlasJxKXcorrelation(data, newColIndexes)

    //same speed as JxK xcorrelation
    //testBlasKxJXcorrelation(data, newColIndexes)
  }

  def testScalaRowVectorsXcorrelation(rowData: Array[(Double, Vector)], newColIndexes: Array[Int], numFeatures: Int) = {
    val scalaXcorr = new XCorrelation(newColIndexes, numFeatures)
    Timer("xCorrelation").reset
    Timer("xCorrelation").start
    var numLambda = 0
    //    while (numLambda < 100) {
    var rowIndex = 0
    while (rowIndex < rowData.size) {
      scalaXcorr.compute(rowData(rowIndex))
      rowIndex += 1
    }
    //      numLambda += 1
    //    }
    val sxx = scalaXcorr.xx
    //println(s"sxx: ${sxx.map(r => r.mkString(",")).mkString("\n")}")
    Timer("xCorrelation").end
    printTiming("scalaRowXcorr")
  }

  def testScalaMatrixXcorrelation(data: DenseMatrix, newColIndexes: Array[Int]) = {
    val scalaXcorr = new XCorrelation2Scala
    Timer("xCorrelation").reset
    Timer("xCorrelation").start
    //for(i <- 0 until 100) {
    val sxx = scalaXcorr.compute(data, newColIndexes)
    //}
    //println(s"sxx: ${sxx}")
    Timer("xCorrelation").end
    printTiming("scalaXcorr")
  }

  def testBlasJxKXcorrelation(data: DenseMatrix, newColIndexes: Array[Int]) = {
    val xCorr = new XCorrelation2(newColIndexes, data.numCols)
    //Timer("dgemm").reset
    Timer("xCorrelation").reset
    Timer("xCorrelation").start
    //    var numLambda = 0
    //    while (numLambda < 100) {
    val xx = xCorr.computeJxK(data, newColIndexes)
    //      numLambda += 1
    //    }
    //println(s"xx: ${xx}")
    Timer("xCorrelation").end
    printTiming("blasJxKXcorr")
  }

  def testBlasKxJXcorrelation(data: DenseMatrix, newColIndexes: Array[Int]) = {
    val xCorr = new XCorrelation2(newColIndexes, data.numCols)
    //Timer("dgemm").reset
    Timer("xCorrelation").reset
    Timer("xCorrelation").start
    //for(i <- 0 until 100) {
    val xx = xCorr.computeKxJ(data, newColIndexes)
    //}
    //println(s"xx: ${xx}")
    Timer("xCorrelation").end
    printTiming("computeKxJ")
   // if (Timer("xCorrelation").totalTime > )
  }

  private def printTiming(name: String) = {
    //println(s"\t\t\t$name \t${Timer.timers.mkString("\n")}")
    println(s"$name \t${Timer.timers.mkString("\n")}")
  }
}
// scalastyle:on println
