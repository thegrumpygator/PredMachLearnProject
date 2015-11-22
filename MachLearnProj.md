# MachineLearningProject
Brian L. Fuller  
November 16, 2015  

Read Data Files
====

Read the datafiles into two frames, one for training and one for testing.

```r
library(readr)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
trainDat <- read_csv("pml-training.csv",
                      col_names = TRUE,
                      col_types = "ncnnccnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnc")



#str(trainDat)

## these are the 20 cases for submission...
testDat <- read_csv("pml-testing.csv", 
                      col_names = TRUE,
                      col_types = "ncnnccnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
```


```r
# partition the data for training/validation purposes
inTrain <- createDataPartition(y = trainDat$classe, 
                               p = 0.75, 
                               list = FALSE)

training <- trainDat[inTrain, ]
testing <- trainDat[-inTrain, ]

# get the real training columns

# Zero Variance fields...
#  kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm, amplitude_yaw_forearm

trainingCut <- training %>% select(-user_name, -raw_timestamp_part_1, -raw_timestamp_part_2,
                                   -cvtd_timestamp, -new_window, -num_window, 
                                   -kurtosis_yaw_belt, -skewness_yaw_belt, -amplitude_yaw_belt, 
                                   -kurtosis_yaw_dumbbell, -skewness_yaw_dumbbell, 
                                   -amplitude_yaw_dumbbell, -kurtosis_yaw_forearm, 
                                   -skewness_yaw_forearm, -amplitude_yaw_forearm)
trainingCut <- trainingCut[ , 2:145]

# use 5-fold Cross Validation straight-up
cvControl <- trainControl(method = "repeatedcv",
                          repeats = 6,
                          number = 5,
                          #summaryFunction = twoClassSummary,
                          classProbs = TRUE)

modelFitCV <- train(classe ~ ., data = trainingCut,
                    method = "svmRadial",
                    tunelength = 9,
                    preProc = c("center", "scale"),
                    metric = "ROC",
                    trControl = cvControl)
```

```
## Loading required package: kernlab
```

```
## Warning in train.default(x, y, weights = w, ...): The metric "ROC" was not
## in the result set. Accuracy will be used instead.
```

```r
modelFitCV
```

```
## Support Vector Machines with Radial Basis Function Kernel 
## 
## 14718 samples
##   143 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (143), scaled (143) 
## Resampling: Cross-Validated (5 fold, repeated 6 times) 
## Summary of sample sizes: 241, 244, 241, 241, 241, 243, ... 
## Resampling results across tuning parameters:
## 
##   C     Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.25  0.5213082  0.3970912  0.06206562   0.07767216
##   0.50  0.5513445  0.4318583  0.04456702   0.05659760
##   1.00  0.5895296  0.4782515  0.05536766   0.07030469
## 
## Tuning parameter 'sigma' was held constant at a value of 0.003828279
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were sigma = 0.003828279 and C = 1.
```

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
