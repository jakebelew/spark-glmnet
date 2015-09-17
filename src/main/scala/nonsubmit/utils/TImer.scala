package nonsubmit.utils

import scala.collection.mutable.HashMap

//TODO - This can be enhanced to capture each measurement and produce statistics along with totalTime. For example, min, max and standard deviation.
class Timer(name: String) {

  var totalTime = 0L
  private var startTime = 0L

  //def enteringTimedSection() = {
  def start() = {
    startTime = System.currentTimeMillis
  }

  //def exitingTimedSection() = {
  def end() = {
    totalTime += System.currentTimeMillis - startTime
  }
  
  //TODO - Make the time units selectable and round not truncate
  override def toString() = s"$name totalTime: ${totalTime/1000} seconds"
}

object Timer {

  private val timerMap = new HashMap[String, Timer]

  def apply(name: String) = {
    timerMap.getOrElseUpdate(name, new Timer(name))
  }

  def timers() =  timerMap.values
    
  //def main(args: Array[String]) {
    //        val startTime = System.currentTimeMillis()
    //        for {
    //          i <- 0 until 1000000
    //        } {
    //          Timer("test1").enteringTimedSection
    //          Timer("test1").exitingTimedSection
    //        }
    //        println(System.currentTimeMillis() - startTime)

    //    Timer("test1").enteringTimedSection
    //    Thread.sleep(1000)
    //    Timer("test1").exitingTimedSection
    //    //println(Timer("test1").totalTime)
    //    Timer("test2").enteringTimedSection
    //    Thread.sleep(2000)
    //    Timer("test2").exitingTimedSection
    //    println(Timer("test1").totalTime)
    //    println(Timer("test2").totalTime)
  //}
}