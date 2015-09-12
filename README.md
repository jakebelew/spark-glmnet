# spark-glmnet

## glmnet -  “Regularization Paths for Generalized Linear Models via Coordinate Descent"

This project coded, in Scala, the algorithm  “Regularization Paths for Generalized Linear Models via Coordinate Descent” by Jerome Friedman, Trevor Hastie and Rob Tibshirani of Stanford University (http://web.stanford.edu/~hastie/Papers/glmnet.pdf).  The algorithm is typically referred to as “glmnet” - generalized linear model with elastic net regularization.  Elastic net is the combination of the ridge and lasso regularization methods.  This algorithm is generally faster than traditional methods such as linear regression and is particularly well suited for “fat” datasets (many more features than events).

## Developers
    Mike Bowles
    Jake Belew
    Ben Burford

## Build the code (instructions are for running in Eclipse)
(Clone the repo)
$ git clone git@github.com:jakebelew/spark-glmnet.git
(create an Eclipse project)
$ cd spark-glmnet
(If this is your first time running SBT, you will be “downloading the internet” so be prepared to take a while.)
$ sbt
> eclipse with-source=true
> exit
