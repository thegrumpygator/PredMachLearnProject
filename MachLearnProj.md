# MachineLearningProject
Brian L. Fuller  
November 16, 2015  

Read Data Files
====

Read the datafiles into two frames, one for training and one for testing. 
Convert the "classe" variable to a factor variable. In this part the 
testDat is the data for the 20 cases used for submission. FOr the trainDat
set, convert the response field to a factor.

```r
library(readr)
library(caret)
library(dplyr)
library(kernlab)

trainDat <- read_csv("pml-training.csv",
                      col_names = TRUE,
                      col_types = "ncnnccnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnc")

trainDat$classe <- as.factor(trainDat$classe)

#str(trainDat)

## these are the 20 cases for submission...
testDat <- read_csv("pml-testing.csv", 
                      col_names = TRUE,
                      col_types = "ncnnccnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
```

For model building and exploration, I will create a training set and a testing
set. The training set will have 75% of the trainDat data.

```r
# partition the data for training/validation purposes
inTrain <- createDataPartition(y = trainDat$classe, 
                               p = 0.75, 
                               list = FALSE)

training <- trainDat[inTrain, ]
testing <- trainDat[-inTrain, ]
```

Variables in the "real" testing set that are NA can't be used in predictions,
so they will be removed from the training set. Also, ID and TIME columns will
be removed from the training set since they do not factor into the way an
exercise was performed.

```r
# determine list of useless variables from testDat
nsv <- nearZeroVar(testDat, saveMetrics = TRUE)
#nsv[ nsv$nzv == TRUE,] ## not for the report
uselessvars <- row.names(nsv[ nsv$nzv == TRUE,])

# remove useless columns from training set
training <- training[, !names(trainDat) %in% uselessvars, drop = F]

# remove left 6 vars
training <- training[7:length(training)]

# are there any over-correlated vars? YES!
M <- abs(cor(training[, 1:length(training)-1])) 
diag(M) <- 0
which(M>0.8, arr.ind = T)
```

```
##                      row col
## total_accel_belt       4   1
## accel_belt_y           9   1
## accel_belt_z          10   1
## accel_belt_x           8   2
## roll_belt              1   4
## accel_belt_y           9   4
## accel_belt_z          10   4
## pitch_belt             2   8
## roll_belt              1   9
## total_accel_belt       4   9
## accel_belt_z          10   9
## roll_belt              1  10
## total_accel_belt       4  10
## accel_belt_y           9  10
## magnet_belt_z         13  12
## magnet_belt_y         12  13
## magnet_arm_z          26  24
## magnet_arm_x          24  26
## yaw_dumbbell          29  27
## roll_dumbbell         27  29
## accel_dumbbell_x      34  30
## accel_dumbbell_y      35  30
## accel_dumbbell_z      36  30
## total_accel_dumbbell  30  34
## accel_dumbbell_z      36  34
## total_accel_dumbbell  30  35
## total_accel_dumbbell  30  36
## accel_dumbbell_x      34  36
```

```r
# are there any nearzerovariance data in the training? NO!
nsvtrain <- nearZeroVar(training, saveMetrics = TRUE)
nsvtrain[ nsvtrain$zeroVar == TRUE,]
```

```
## [1] freqRatio     percentUnique zeroVar       nzv          
## <0 rows> (or 0-length row.names)
```

```r
nsvtrain[ nsvtrain$nzv == TRUE,]
```

```
## [1] freqRatio     percentUnique zeroVar       nzv          
## <0 rows> (or 0-length row.names)
```

```r
# use parallel to optimize performance
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
registerDoParallel(makeCluster(detectCores()))

# train a random forest using 5-fold cross validation
# and pca preprocessor (because of over-correlated data)
set.seed(1234)
modelFitPCA <- train(classe ~ .,
                  method = "rf",
                  preProcess = "pca",
                  trControl = trainControl(method = "cv", number = 5),
                  data = training)
```

```
## Loading required package: randomForest
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
modelFitPCA
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (52), centered
##  (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11775, 11775, 11773, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9558363  0.9440938  0.006005439  0.007605533
##   27    0.9336863  0.9160429  0.005593151  0.007083639
##   52    0.9321236  0.9140582  0.005467738  0.006929715
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modelFitPCA$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 3.87%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4132   21   15   16    1  0.01266428
## B  103 2678   55    5    7  0.05969101
## C   18   74 2443   26    6  0.04830541
## D    8    8  113 2280    3  0.05472637
## E    2   39   21   29 2615  0.03362897
```

```r
# compare with non-pca preprocess
set.seed(1234)
modelFit <- train(classe ~ .,
                  method = "rf",
                  trControl = trainControl(method = "cv", number = 5),
                  data = training)

modelFit
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11775, 11775, 11775, 11773, 11774 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9883814  0.9853008  0.004001059  0.005064280
##   27    0.9895363  0.9867627  0.002524772  0.003195479
##   52    0.9797520  0.9743797  0.003627432  0.004595090
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.8%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4174   11    0    0    0 0.002628435
## B   24 2809   12    3    0 0.013693820
## C    1   17 2540    9    0 0.010518115
## D    1    0   26 2383    2 0.012023217
## E    1    1    3    7 2694 0.004434590
```

```r
# non-pca model performs better-- use it.

# out of sample error from predicting with testing data set
rfPred <- predict(modelFit, newdata = testing)

confusionMatrix(rfPred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    8    0    0    0
##          B    2  933    3    0    0
##          C    1    5  847   12    0
##          D    0    3    5  789    2
##          E    0    0    0    3  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.991          
##                  95% CI : (0.988, 0.9935)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9887         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9831   0.9906   0.9813   0.9978
## Specificity            0.9977   0.9987   0.9956   0.9976   0.9993
## Pos Pred Value         0.9943   0.9947   0.9792   0.9875   0.9967
## Neg Pred Value         0.9991   0.9960   0.9980   0.9963   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2838   0.1903   0.1727   0.1609   0.1833
## Detection Prevalence   0.2855   0.1913   0.1764   0.1629   0.1839
## Balanced Accuracy      0.9978   0.9909   0.9931   0.9895   0.9985
```

```r
# use the model to predict cases for submission
subPred <- predict(modelFit, newdata = testDat)

subPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

---------------------------------------------------------------------
How To Build Model
===

How To Cross Validate Model
===

Expected Out of Sample Error
===

Rationale for Choices
===

20 Test Cases
===
