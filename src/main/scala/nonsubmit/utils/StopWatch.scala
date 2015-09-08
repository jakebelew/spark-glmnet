package nonsubmit.utils

/** Provides Scala-style usage for org.apache.commons.lang.time.StopWatch */
class StopWatch {

  val sw = new org.apache.commons.lang.time.StopWatch

  //var startTime: Long = 0

  def start(): StopWatch = {
    //startTime = System.currentTimeMillis
    sw.start
    this
  }

  def elapsedTime(): Long = {
    //System.currentTimeMillis - startTime
    sw.getTime
  }

  def restart() = {
    sw.reset()
    sw.start()
  }
}

object StopWatch {
  def apply() = new StopWatch
}