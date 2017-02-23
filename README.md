# spark-glmnet

### glmnet -  “Lasso and Elastic-Net Regularized Generalized Linear Models"

A Scala implementation of the "Lasso and Elastic-Net Regularized Generalized Linear Models" for Spark MLlib from "Regularization Paths for Generalized Linear Models via Coordinate Descent" by Jerome Friedman, Trevor Hastie and Rob Tibshirani of Stanford University (http://web.stanford.edu/~hastie/Papers/glmnet.pdf). The algorithm is typically referred to as “glmnet” - generalized linear model with elastic net regularization. Elastic net is the combination of the ridge and lasso regularization methods. This algorithm is generally faster than traditional methods such as linear regression and is particularly well suited for “fat” datasets (many more features than events).

### Spark MLlib

This code is fully integrated with Spark MLlib and is being submitted as an addition to MLlib. It performs K-fold cross validation, picks the best (highest accuracy) alpha/lambda combination and returns a model based on these.

Following is the process that glmnet executes:

    1. User sets up arrays of values:
      1.1 An array of alpha values.
      1.2 Number of lambda values - default is 100 (glmnet will automatically generate the series of lambda values).
      1.3 Choose number of K-folds for cross validation.
    2. On each fold:
       2.1 Using Coordinate Descent generate a model on each combination of alpha and lambda using K-fold training data.
       2.2 Test all models on K-fold test data and save accuracies.
    3. Average accuracies across the various folds of results, for each of the alpha/lambda combinations, and choose the one combination with highest accuracy.
    4. Train on all of the data using the alpha/lambda combination from step 3 and produce the final (best) model. 

### Developers
    Mike Bowles
    Jake Belew
    Ben Burford

### Instructions for setting up the project in Eclipse
	$ git clone git@github.com:jakebelew/spark-glmnet.git
	(Create an Eclipse project)
	$ cd spark-glmnet
	(Note: if this is your first time running SBT this may take a while.)
	$ sbt
	> eclipse with-source=true
	> exit
	(In Eclipse, import project)
	(To enable using the glmnet project log4j file, in order to better display linear regression information)
	Project -> Properties -> Java Build Path -> Source -> Add Folder -> src/main/resources (select, OK) -> OK

### Running the cross-validation example in Eclipse
	Run org.apache.spark.examples.ml.LinearRegressionCrossValidatorExample in eclipse.
	* It will generate training data and apply the glmnet algorithm.
	* It will run the data in K=2 folds, with alpha = 0.2 and 0.3, and 100 lambda values.
	* It will choose the “Best fit” combination of alpha and lambda and generate a model on the entire training data set using the chosen alpha and lambda.
	* It will generate test data and run the resulting model on the test data and display the accuracy of that model on the test data.
