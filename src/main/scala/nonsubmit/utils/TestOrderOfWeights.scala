package nonsubmit.utils

object TestOrderOfWeights {
  def main(args: Array[String]) {
    val weights = Array(-0.1, 0.1, 2.75, -2.55, 1.25)
    println(orderOfWeights(weights))
  }

  /** The order weights - from largest to smallest. Returns the indexes of the weights in descending order of the absolute value. */
  def orderOfWeights(weights: Seq[Double]): Seq[Int] =
    weights.map(math.abs).zipWithIndex.sortBy(-_._1).map(_._2)
}