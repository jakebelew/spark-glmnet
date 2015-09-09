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

object LinearRegressionWithCDExample {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("LinearRegressionWithCDExample")//.setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val path = "data/sample_linear_regression_data.txt"
    //val path = "data/sample_linear_regression_data_fold1.txt"
    //val path = "data/sample_linear_regression_data_fold2.txt"
    val training = MLUtils.loadLibSVMFile(sc, path)

    val lr = new LinearRegressionWithCD("")
      .setMaxIter(100)

    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.2))

    val paramGrid = paramGridBuilder.build.flatMap(pm => Array.fill(100)(pm.copy))

    val models = lr.fit(training.toDF(), paramGrid)

    sc.stop()
  }
}
// scalastyle:on println
