package nonsubmit.utils

import scala.util.Random

object DataGeneratorUtil {

  def generateArray(size: Int): Array[Double] = {
    val rnd = new Random(42)
    Array.fill[Double](size)(rnd.nextDouble)
  }

  def generateMatrix(rumRows: Int, numCols: Int): Array[Array[Double]] = {
    val rnd = new Random(42)
    Array.fill[Array[Double]](rumRows)(
      Array.fill[Double](numCols)(rnd.nextDouble))
  }

  def generateNewColIndexes(numNewBeta: Int, numCols: Int): Array[Int] = {
    val rnd = new RandomSequenceOfUnique(42, numCols)
    Array.fill[Int](numNewBeta)(rnd.nextInt)
  }
}