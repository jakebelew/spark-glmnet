package nonsubmit.utils

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

object Log {

  def logRDD[T](msg: => String, rdd: RDD[T]) = {
    println(s"${msg}:\n ${rdd.toDebugString}")
  }

  //  def apply[T](name: String, t: T, suffix: String): T = {
  //    apply(name, t)
  //    println(suffix)
  //    t
  //  }

  def apply[T](t: T): T = {
    t match {
      //      case l: Seq[_] => {
      //        print(s"$name: ")
      //        l.foreach { a => println(s"{${a.mkString(", ")}} ") }
      //        //println("")
      //      }
      case aa: Array[Array[_]] => {
        aa.foreach { a => println(s"{${a.mkString(", ")}} ") }
        //println("")
      }
      case a: Array[_] => println(s"${a.mkString(", ")}")
      case v: Vector => println(s"${v.toArray.mkString(", ")}")
      case _ => println(s"$t")
    }
    t
  }

  def apply[T](name: String, t: T): T = {
    t match {
      //      case l: Seq[_] => {
      //        print(s"$name: ")
      //        l.foreach { a => println(s"{${a.mkString(", ")}} ") }
      //        //println("")
      //      }
      case aa: Array[Array[Double]] => {
        print(s"$name:\n")
        aa.foreach { a => println(s"{${a.mkString(", ")}} ") }
        //println("")
      }
      //      case a: Array[_] => println(s"$name: ${a.mkString(", ")}")
      //      case m: SparseMatrix => {
      //        println(s"$name:\n$m")
      //      }
      //      case _ => println(s"$name: $t")
    }
    t
  }
}