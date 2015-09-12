# spark-glmnet

### glmnet -  “Regularization Paths for Generalized Linear Models via Coordinate Descent"

The developers coded, in Scala, the algorithm  “Regularization Paths for Generalized Linear Models via Coordinate Descent” by Jerome Friedman, Trevor Hastie and Rob Tibshirani of Stanford University (http://web.stanford.edu/~hastie/Papers/glmnet.pdf).  The algorithm is typically referred to as “glmnet” - generalized linear model with elastic net regularization.  Elastic net is the combination of the ridge and lasso regularization methods.  This algorithm is generally faster than traditional methods such as linear regression and is particularly well suited for “fat” datasets (many more features than events).

### Spark MLlib

This code is fully integrated with Spark MLlib and is being submitted as an addition to MLlib. It performs k-fold cross validation, picks the best (highest accuracy) alpha/lambda combination and returns a model based on these.

Following is the process that glmnet executes:

    1. User sets up arrays of values:
    
    * An array of alpha values.
    * Number of lambda values - default is 100 (glmnet will automatically the series of lambda values).
    * Choose number of k-folds for cross validation.
    2. On each fold:
    * Using Coordinate Descent generate a model on each combination of alpha and lambda using k-fold training data.
    * Test all models on k-fold test data and save accuracies.
    3. Average accuracies across the various folds of results, for each of the alpha/lambda combinations and choose the one combination with highest accuracy.
    4. Train on all of the data using the alpha/lambda combination from step 3 and produce the final (best) model. 

### Developers
    Mike Bowles
    Jake Belew
    Ben Burford

### Build the code (instructions are for running in Eclipse)
	$ git clone git@github.com:jakebelew/spark-glmnet.git
	(create an Eclipse project and import)
	$ cd spark-glmnet
	(Note: if this is your first time running SBT, you will be “downloading the internet” so it may take a while.)
	$ sbt
	> eclipse with-source=true
	> exit

### Run with test data
	Run org.apache.spark.examples.ml.LinearRegressionCrossValidatorExample in eclipse.
	* It will read in data/sample_linear_regression_data.txt and apply the glmnet algorithm.
	* It will run the data in k=2 folds, with alpha = 0.2 and 0.3, and 100 lambda values.
	* It will choose the “Best fit” combination of alpha and lambda and generate a model on the entire data set using the chosen alpha and lambda.
