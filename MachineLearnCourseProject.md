# Machine Learning Course Project
mb84  
06 agosto 2016  

# Introduction #

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).  

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

**Read more:** [http://groupware.les.inf.puc-rio.br/har#ixzz4GXmbqc8w](http://groupware.les.inf.puc-rio.br/har#ixzz4GXmbqc8w)

**Read more:** [http://groupware.les.inf.puc-rio.br/har#ixzz4GXmXoGTE](http://groupware.les.inf.puc-rio.br/har#ixzz4GXmXoGTE)


# Aim #

The goal of this project is to predict the manner in which the Six participants did the exercise. The outcome variable is stored in the "classe" variable.

# Data #

Training dataset is available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Testing dataset is available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


# Preprocessing #

Before starting with prediction model building, we can perform a bit of exploration analysis on the training dataset in order to identify useless variables (variables with no variability or enriched by missing values).

Looking at the data we can notice that there are empty values, NA values and DIV/0 values that must be treat as NA strings. Hence i load the files setting na.strings accordingly.  


```r
library(caret)

train.set <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE, na.strings=c("NA", "NULL", "#DIV/0!"))

test.set <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE, na.strings=c("NA", "NULL", "#DIV/0!"))
```

In order to remove useless variable we can identify variables in which most of the values are NAs. To do this we can calculate the % of NAs values for each variables. Variables with high % (> 90) should be excluded.
Moreover we can remove variables that are completely unrelated to the outcome.


```r
na_count_perc <- sapply(train.set, function(y) sum(length(which(is.na(y))))/length(train.set$X) * 100)
na_count_perc <- data.frame(na_count_perc)

train.set.light <- train.set[,na_count_perc<90]
train.set.light <- train.set.light[, !(colnames(train.set.light) %in% c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"))]

# Accordingly i subset only remaining columns (variables) also in 20 samples test set.

test.set.light <- test.set[,colnames(test.set) %in% colnames(train.set.light)]
```

I can remove also variables that have no variability at all. 
To do this we can check and identify these variables with the nearZeroVar function in the caret package. 


```r
nearZeroVar(train.set.light, saveMetrics=TRUE)
```

```
FALSE                      freqRatio percentUnique zeroVar   nzv
FALSE user_name             1.100679    0.03057792   FALSE FALSE
FALSE roll_belt             1.101904    6.77810621   FALSE FALSE
FALSE pitch_belt            1.036082    9.37722964   FALSE FALSE
FALSE yaw_belt              1.058480    9.97349913   FALSE FALSE
FALSE total_accel_belt      1.063160    0.14779329   FALSE FALSE
FALSE gyros_belt_x          1.058651    0.71348486   FALSE FALSE
FALSE gyros_belt_y          1.144000    0.35164611   FALSE FALSE
FALSE gyros_belt_z          1.066214    0.86127816   FALSE FALSE
FALSE accel_belt_x          1.055412    0.83579655   FALSE FALSE
FALSE accel_belt_y          1.113725    0.72877383   FALSE FALSE
FALSE accel_belt_z          1.078767    1.52379982   FALSE FALSE
FALSE magnet_belt_x         1.090141    1.66649679   FALSE FALSE
FALSE magnet_belt_y         1.099688    1.51870350   FALSE FALSE
FALSE magnet_belt_z         1.006369    2.32901845   FALSE FALSE
FALSE roll_arm             52.338462   13.52563449   FALSE FALSE
FALSE pitch_arm            87.256410   15.73234125   FALSE FALSE
FALSE yaw_arm              33.029126   14.65701763   FALSE FALSE
FALSE total_accel_arm       1.024526    0.33635715   FALSE FALSE
FALSE gyros_arm_x           1.015504    3.27693405   FALSE FALSE
FALSE gyros_arm_y           1.454369    1.91621649   FALSE FALSE
FALSE gyros_arm_z           1.110687    1.26388747   FALSE FALSE
FALSE accel_arm_x           1.017341    3.95984099   FALSE FALSE
FALSE accel_arm_y           1.140187    2.73672409   FALSE FALSE
FALSE accel_arm_z           1.128000    4.03628580   FALSE FALSE
FALSE magnet_arm_x          1.000000    6.82397309   FALSE FALSE
FALSE magnet_arm_y          1.056818    4.44399144   FALSE FALSE
FALSE magnet_arm_z          1.036364    6.44684538   FALSE FALSE
FALSE roll_dumbbell         1.022388   84.20650290   FALSE FALSE
FALSE pitch_dumbbell        2.277372   81.74498012   FALSE FALSE
FALSE yaw_dumbbell          1.132231   83.48282540   FALSE FALSE
FALSE total_accel_dumbbell  1.072634    0.21914178   FALSE FALSE
FALSE gyros_dumbbell_x      1.003268    1.22821323   FALSE FALSE
FALSE gyros_dumbbell_y      1.264957    1.41677709   FALSE FALSE
FALSE gyros_dumbbell_z      1.060100    1.04984201   FALSE FALSE
FALSE accel_dumbbell_x      1.018018    2.16593619   FALSE FALSE
FALSE accel_dumbbell_y      1.053061    2.37488533   FALSE FALSE
FALSE accel_dumbbell_z      1.133333    2.08949139   FALSE FALSE
FALSE magnet_dumbbell_x     1.098266    5.74864948   FALSE FALSE
FALSE magnet_dumbbell_y     1.197740    4.30129447   FALSE FALSE
FALSE magnet_dumbbell_z     1.020833    3.44511263   FALSE FALSE
FALSE roll_forearm         11.589286   11.08959331   FALSE FALSE
FALSE pitch_forearm        65.983051   14.85577413   FALSE FALSE
FALSE yaw_forearm          15.322835   10.14677403   FALSE FALSE
FALSE total_accel_forearm   1.128928    0.35674243   FALSE FALSE
FALSE gyros_forearm_x       1.059273    1.51870350   FALSE FALSE
FALSE gyros_forearm_y       1.036554    3.77637346   FALSE FALSE
FALSE gyros_forearm_z       1.122917    1.56457038   FALSE FALSE
FALSE accel_forearm_x       1.126437    4.04647844   FALSE FALSE
FALSE accel_forearm_y       1.059406    5.11160942   FALSE FALSE
FALSE accel_forearm_z       1.006250    2.95586586   FALSE FALSE
FALSE magnet_forearm_x      1.012346    7.76679238   FALSE FALSE
FALSE magnet_forearm_y      1.246914    9.54031189   FALSE FALSE
FALSE magnet_forearm_z      1.000000    8.57710733   FALSE FALSE
FALSE classe                1.469581    0.02548160   FALSE FALSE
```

As we can see none of the variables that remain in the cleaned dataset are zero covariate variables. We can than proceed to the prediction model building process.

# Prediction model building #


```r
set.seed(9876)

inTrain = createDataPartition(train.set.light$classe, p = 0.7, list = FALSE)
training = train.set.light[ inTrain,]
testing = train.set.light[-inTrain,]
```

Once performed data partitioning i applied several machine learning algorithm to build model and then select the most accurate one.

I tested 4 models:

* Random Forest
* Stochastic Gradient Boosting
* CART
* Bagged CART

I applied cross validation with 8-fold resampling.


```r
## Model fittings ##

rf.fit <- train(training$classe ~ ., method="rf", trControl = trainControl(method="cv", number = 8, allowParallel = TRUE, verbose=FALSE), data=training)

gbm.fit <- train(training$classe ~ ., method="gbm", trControl = trainControl(method="cv", number = 8, allowParallel = TRUE, verbose=FALSE), data=training, verbose=FALSE)

cart.fit <- train(training$classe ~ ., method="rpart", trControl = trainControl(method="cv", number = 8, allowParallel = TRUE, verbose=FALSE), data=training)

treebag.fit <- train(training$classe ~ ., method="treebag", trControl = trainControl(method="cv", number = 8, allowParallel = TRUE, verbose=FALSE), data=training)

## Predictions ##

rf.pred <- predict(rf.fit, newdata = testing)
gbm.pred <- predict(gbm.fit, newdata = testing)
cart.pred <- predict(cart.fit, newdata = testing)
treebag.pred <- predict(treebag.fit, newdata = testing)
```

## Confusion matrices ##

Below the confusion matrices for each model type:

### Random Forest ###


```r
confusionMatrix(rf.pred, testing$classe)
```

```
FALSE Confusion Matrix and Statistics
FALSE 
FALSE           Reference
FALSE Prediction    A    B    C    D    E
FALSE          A 1672   10    0    0    0
FALSE          B    1 1124    2    0    0
FALSE          C    1    5 1021    9    2
FALSE          D    0    0    3  954    0
FALSE          E    0    0    0    1 1080
FALSE 
FALSE Overall Statistics
FALSE                                          
FALSE                Accuracy : 0.9942         
FALSE                  95% CI : (0.9919, 0.996)
FALSE     No Information Rate : 0.2845         
FALSE     P-Value [Acc > NIR] : < 2.2e-16      
FALSE                                          
FALSE                   Kappa : 0.9927         
FALSE  Mcnemar's Test P-Value : NA             
FALSE 
FALSE Statistics by Class:
FALSE 
FALSE                      Class: A Class: B Class: C Class: D Class: E
FALSE Sensitivity            0.9988   0.9868   0.9951   0.9896   0.9982
FALSE Specificity            0.9976   0.9994   0.9965   0.9994   0.9998
FALSE Pos Pred Value         0.9941   0.9973   0.9836   0.9969   0.9991
FALSE Neg Pred Value         0.9995   0.9968   0.9990   0.9980   0.9996
FALSE Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
FALSE Detection Rate         0.2841   0.1910   0.1735   0.1621   0.1835
FALSE Detection Prevalence   0.2858   0.1915   0.1764   0.1626   0.1837
FALSE Balanced Accuracy      0.9982   0.9931   0.9958   0.9945   0.9990
```

### Stochastic Gradient Boosting ###


```r
confusionMatrix(gbm.pred, testing$classe)
```

```
FALSE Confusion Matrix and Statistics
FALSE 
FALSE           Reference
FALSE Prediction    A    B    C    D    E
FALSE          A 1647   31    0    1    1
FALSE          B   17 1071   21    1   14
FALSE          C    7   31  984   37   11
FALSE          D    2    0   16  918   15
FALSE          E    1    6    5    7 1041
FALSE 
FALSE Overall Statistics
FALSE                                           
FALSE                Accuracy : 0.9619          
FALSE                  95% CI : (0.9567, 0.9667)
FALSE     No Information Rate : 0.2845          
FALSE     P-Value [Acc > NIR] : < 2.2e-16       
FALSE                                           
FALSE                   Kappa : 0.9519          
FALSE  Mcnemar's Test P-Value : 0.0005824       
FALSE 
FALSE Statistics by Class:
FALSE 
FALSE                      Class: A Class: B Class: C Class: D Class: E
FALSE Sensitivity            0.9839   0.9403   0.9591   0.9523   0.9621
FALSE Specificity            0.9922   0.9888   0.9823   0.9933   0.9960
FALSE Pos Pred Value         0.9804   0.9528   0.9196   0.9653   0.9821
FALSE Neg Pred Value         0.9936   0.9857   0.9913   0.9907   0.9915
FALSE Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
FALSE Detection Rate         0.2799   0.1820   0.1672   0.1560   0.1769
FALSE Detection Prevalence   0.2855   0.1910   0.1818   0.1616   0.1801
FALSE Balanced Accuracy      0.9880   0.9646   0.9707   0.9728   0.9791
```
### Classification and Regression trees (CART) ###


```r
confusionMatrix(cart.pred, testing$classe)
```

```
FALSE Confusion Matrix and Statistics
FALSE 
FALSE           Reference
FALSE Prediction    A    B    C    D    E
FALSE          A 1496  462  485  426  152
FALSE          B   35  382   26  184  124
FALSE          C  115  295  515  354  281
FALSE          D    0    0    0    0    0
FALSE          E   28    0    0    0  525
FALSE 
FALSE Overall Statistics
FALSE                                          
FALSE                Accuracy : 0.4958         
FALSE                  95% CI : (0.483, 0.5087)
FALSE     No Information Rate : 0.2845         
FALSE     P-Value [Acc > NIR] : < 2.2e-16      
FALSE                                          
FALSE                   Kappa : 0.3416         
FALSE  Mcnemar's Test P-Value : NA             
FALSE 
FALSE Statistics by Class:
FALSE 
FALSE                      Class: A Class: B Class: C Class: D Class: E
FALSE Sensitivity            0.8937  0.33538  0.50195   0.0000  0.48521
FALSE Specificity            0.6379  0.92225  0.78494   1.0000  0.99417
FALSE Pos Pred Value         0.4952  0.50866  0.33013      NaN  0.94937
FALSE Neg Pred Value         0.9378  0.85255  0.88185   0.8362  0.89554
FALSE Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
FALSE Detection Rate         0.2542  0.06491  0.08751   0.0000  0.08921
FALSE Detection Prevalence   0.5133  0.12761  0.26508   0.0000  0.09397
FALSE Balanced Accuracy      0.7658  0.62882  0.64344   0.5000  0.73969
```
### Bagged CART ###


```r
confusionMatrix(treebag.pred, testing$classe)
```

```
FALSE Confusion Matrix and Statistics
FALSE 
FALSE           Reference
FALSE Prediction    A    B    C    D    E
FALSE          A 1665   14    2    1    0
FALSE          B    5 1113    8    5    0
FALSE          C    3    7 1012   14    2
FALSE          D    1    2    4  943    1
FALSE          E    0    3    0    1 1079
FALSE 
FALSE Overall Statistics
FALSE                                           
FALSE                Accuracy : 0.9876          
FALSE                  95% CI : (0.9844, 0.9903)
FALSE     No Information Rate : 0.2845          
FALSE     P-Value [Acc > NIR] : < 2.2e-16       
FALSE                                           
FALSE                   Kappa : 0.9843          
FALSE  Mcnemar's Test P-Value : NA              
FALSE 
FALSE Statistics by Class:
FALSE 
FALSE                      Class: A Class: B Class: C Class: D Class: E
FALSE Sensitivity            0.9946   0.9772   0.9864   0.9782   0.9972
FALSE Specificity            0.9960   0.9962   0.9946   0.9984   0.9992
FALSE Pos Pred Value         0.9899   0.9841   0.9750   0.9916   0.9963
FALSE Neg Pred Value         0.9979   0.9945   0.9971   0.9957   0.9994
FALSE Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
FALSE Detection Rate         0.2829   0.1891   0.1720   0.1602   0.1833
FALSE Detection Prevalence   0.2858   0.1922   0.1764   0.1616   0.1840
FALSE Balanced Accuracy      0.9953   0.9867   0.9905   0.9883   0.9982
```

# Final model evaluation #

The random forest model emerges as the model with the best overall accuracy (99.42 %), however also Bagged CART is very accurate (almost 99% accuracy).  
I tested the prediction model on the 20 samples test set.  
Prediction is the same for both models. 


```r
# Prediction with random forest
predict(rf.fit, newdata = test.set.light)
```

```
FALSE  [1] B A B A A E D B A A B C B A E E A B B B
FALSE Levels: A B C D E
```

```r
# Prediction with Bagged CART
predict(treebag.fit, newdata = test.set.light)
```

```
FALSE  [1] B A B A A E D B A A B C B A E E A B B B
FALSE Levels: A B C D E
```


