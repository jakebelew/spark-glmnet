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

import nonsubmit.utils.Timer
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.Logging
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.regression.{ Regressor, LinearRegressionWithCD, LinearRegressionWithCD2, LinearRegressionWithCDParams, LinearRegressionWithCDModel }
import org.apache.spark.ml.tuning.AutoGeneratedParamGridBuilder
import org.apache.spark.mllib.optimization.CoordinateDescent
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.sql.SQLContext

object LinearRegressionWithCDExample {

  def main(args: Array[String]) {
    LinearRegressionWithCDRunner.run("LinearRegressionWithCDExample", new LinearRegressionWithCD(""), 1, args)
  }
}

object LinearRegressionWithCD2Example {

  def main(args: Array[String]) {
    LinearRegressionWithCDRunner.run("LinearRegressionWithCD2Example", new LinearRegressionWithCD2(""), 2, args)
  }
}

private object LinearRegressionWithCDRunner extends Logging {

  def run(appName: String, lr: Estimator[LinearRegressionWithCDModel] with LinearRegressionWithCDParams, optimizerVersion: Int, args: Array[String]) {
    val conf = new SparkConf().setAppName(appName).setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val training = LinearDataGenerator.generateLinearRDD(sc, 790, 10, 0.1, 2, 6.2)

    lr.setOptimizerVersion(optimizerVersion)
      .setMaxIter(100)
      .setElasticNetParam(0.2)

    val paramGrid = new AutoGeneratedParamGridBuilder()
      .buildWithAutoGeneratedGrid("lambdaIndex", lr.getMaxIter)

    val models = lr.fit(training.toDF(), paramGrid)

    //val model = lr.fit(training.toDF())

    logDebug(s"${Timer.timers.mkString("\n")}")

    sc.stop()
  }
}
// scalastyle:on println
