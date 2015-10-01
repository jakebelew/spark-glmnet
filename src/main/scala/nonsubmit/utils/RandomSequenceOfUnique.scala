package nonsubmit.utils

import scala.util.Random

/**
 * Returns a pseudo-random, uniformly distributed, unique value
 *   between 0 (inclusive) and the specified value (exclusive),
 *   drawn from this random number generator's sequence.
 */
class RandomSequenceOfUnique(seed: Int, range: Int) {

  val rnd = new Random(seed)
  val unique = Array.ofDim[Int](range)
  var uniqueCount = 0

  /**
   * Returns a pseudo-random, uniformly distributed, unique int value
   *   between 0 (inclusive) and the specified value (exclusive),
   *   drawn from this random number generator's sequence.
   */
  def nextInt(): Int = {
    if (uniqueCount == range)
      throw new Exception(s"nextInt() cannot be called more than range = $range times")

    val value = rnd.nextInt(range)
    if (unique(value) == 1) nextInt else {
      uniqueCount += 1
      unique(value) = 1
      value
    }
  }
}