/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegressionWithCD
import org.apache.spark.ml.regression.LinearRegressionWithCDModel
import org.apache.spark.mllib.util.MLUtils

/**
 * A simple example demonstrating model selection using CrossValidator.
 * This example also demonstrates how Pipelines are Estimators.
 *
 * This example uses the [[LabeledDocument]] and [[Document]] case classes from
 * [[SimpleTextClassificationPipeline]].
 *
 * Run with
 * {{{
 * bin/run-example ml.CrossValidatorExample
 * }}}
 */
//From spark/examples/src/main/scala/org/apache/spark/examples/ml/CrossValidatorExample.scala
//http://spark.apache.org/docs/latest/ml-guide.html
object LinearRegressionWithCDExample {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("LinearRegressionWithCDExample").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    //val training = LinearDataGenerator.generateLinearRDD(sc, 10, 10, 10)
    val path = "data/sample_linear_regression_data.txt"
    //val path = "data/sample_linear_regression_data_fold2.txt"
    val training = MLUtils.loadLibSVMFile(sc, path)
    //println(s"training: ${training.collect.mkString("\n")}")

    val lr = new LinearRegressionWithCD("")
      .setMaxIter(100)

    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.2))

    val paramGrid = paramGridBuilder.build.flatMap(pm => Array.fill(100)(pm.copy))
    //println(s"paramGrid: ${paramGrid.mkString("\n")}")

    val models = lr.fit(training.toDF(), paramGrid)
    //logResults(models)

    sc.stop()
  }

  def logResults(multiModel: Seq[LinearRegressionWithCDModel]) = {

    //println(s"creating models ET: ${sw.elapsedTime / 1000} seconds")

    // if (args.printBeta) {
    println("Lambda & Beta for each model:")
    // multiModel.foreach(item => println(s"\n\n${item._1} ${item._2.weights.toArray.mkString(",")}"))
    multiModel.foreach(item => println(s"${item.weights.toArray.mkString(",")}"))
    // }
  }
}
// scalastyle:on println
