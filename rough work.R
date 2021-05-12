library(ISLR)
library(dplyr)
library(ggplot2)
library(caret)
library(kernlab)

data(Auto)
str(Auto)
sapply(Auto, function(x) length(unique(x)))
for (i in c(2,7,8)) {
    Auto[,i] <- as.factor(Auto[,i])
}
carNames <- Auto$name
Auto <- select(Auto, -name)
summary(Auto)

train <- sample(nrow(pml), 192)
train[1:10]
length(train)

set.seed(2)
data(Auto)
str(Auto)
train <- sample(nrow(Auto), size = (0.7*nrow(Auto)))
length(Auto[train,]$mpg)
for (i in c(2,7,8)) {
    Auto[,i] <- as.factor(Auto[,i])
}

lm1 <- lm(mpg ~ horsepower, data = Auto, subset = train)
attach(Auto)
mean((mpg - predict(lm1, Auto))[-train]^2)

lm2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((mpg - predict(lm2, Auto))[-train]^2)

lm3 <- lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
mean((mpg - predict(lm3, Auto))[-train]^2)

# LOOCV
library(boot)
?cv.glm

glm1 <- glm(mpg ~ horsepower, data = Auto)
cv.err <- cv.glm(Auto, glm1)
cv.err$delta
cv.err

cv.error <- rep(0, 5)
for (i in 1:5) {
    glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
    cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
}
cv.error
plot(cv.error)
lines(cv.error)

# k-fold CV
set.seed(17)
cv.error.10 <- rep(0, 10)
for (i in 1:10) {
    glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
    cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
}
cv.error.10
plot(cv.error.10)
lines(cv.error.10, col = "blue", cex = 1)

# bootstrap
alpha.fn=function (data ,index){
    X=data$X [index]
    Y=data$Y [index]
    return ((var(Y)-cov (X,Y))/(var(X)+var(Y) -2* cov(X,Y)))
}
data(Portfolio)
boot(Portfolio, alpha.fn, R = 1000)

pml3 <- pml2[,-c(1:6)]
str(pml3)
length(unique(pml3$num_window))
install.packages("Boruta")
install.packages(c("faux", "DataExplorer"))
install.packages("corrplot")
install.packages("mlbench")

library(dplyr)
library(faux)
library(caret)
library(DataExplorer)
library(randomForest)
library(ggplot2)
library(corrplot)
library(data.table)
library(mlbench)

str(pml3)
plot_intro(pml3,title = "")
plot_bar(pml3)
plot_correlation(pml3)
length(pml3)


control = rfeControl(functions = rfFuncs, # random forests
                     method = "repeatedcv", # cross-validation
                     number = 10, # number of folds in CV
                     repeats = 5, # repetition of CV (should be 5)
                     allowParallel = TRUE
)

set.seed(9)
inBuild <- createDataPartition(pml3$classe, p = 0.8, list = FALSE)
validation <- pml3[-inBuild,]
buildData <- pml3[inBuild,]
inTrain <- createDataPartition(buildData$classe, 
                               p = 0.75, 
                               list = FALSE)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]
dim(training)
dim(testing)
dim(validation)

rfe1 <- rfe(x = training[,-53], 
            y = training$classe, 
            sizes = c(15:25), 
            rfeControl = control)
rfe1 # results
predictors(rfe1) # selected predictors
# visuals
ggplot(data = rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = rfe1, metric = "Kappa") + theme_bw()

cors <- cor(training[,-53])
corrplot(cors, method = "color")
featurePlot(training[,-53], training$classe)

control <- trainControl(method = "rf")

install.packages("doParallel")
library(doParallel)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)
stopCluster(cl)
training <- data.table(training)

modfit <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(allowParallel = TRUE))

table(predict(modfit, training[,-53]), training$classe)
table(predict(modfit, testing[,-53]), testing$classe)
confusionMatrix(predict(modfit, testing[,-53]), testing$classe)

system.time(modfit <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(allowParallel = TRUE)))

modfit
ctrl = trainControl(allowParallel = TRUE)
modfit2 <- caret::train(classe ~ ., data = training, method = "gbm", trControl = ctrl, verbose = 0)

confusionMatrix(predict(modfit2, testing), testing$classe)

# parallel preocessing testing
library(doParallel)
system.time(foreach(i = 1:10000) %do% sum(tanh(1:i)))
system.time(foreach(i = 1:10000) %dopar% sum(tanh(1:i)))
registerDoParallel()
getDoParWorkers()
registerDoSEQ()
registerDoParallel(cores = 4)

load("libraries.R")

pml <- read.csv("./Datasets/pml-training.csv", na.strings = c("NA", "", " ", "#DIV/0!"))
str(pml)
nas <- sapply(pml, function(x) sum(is.na(x)*1))
summary(nas[nas > 0])
names(pml)[nafilled]
nacols <- which(nas > 0)
names(nacols) <- NULL
summary(nacols)
pml[, nacols]

pml2 <- pml[, -nacols]
summary(sapply(pml2, function(x) sum(is.na(x)*1)))
str(pml2)

pml3 <- pml2[, -c(1:7)]
str(pml3)
pml3$classe <- as.factor(pml3$classe)

# data partition
inTrain <- createDataPartition(pml3$classe,
                               p = 0.8,
                               list = FALSE)
training <- pml3[inTrain,]
testing <- pml3[-inTrain,]

# decision tree model
modDC <- train(classe ~ ., data = training, method = "rpart")

# random forest model
modRF <- train(classe ~ ., data = training, method = "rf")

# gradient boosting model
modGBM <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE, trControl = trainControl(allowParallel = TRUE))

# logistic regression
modGLM <- train(classe ~ ., data = training, method = "multinom")

# naive bayes model
modNB <- train(classe ~ ., data = training, method = "nb")

# linear discriminant model
modLDA <- train(classe ~ ., data =  training, method = "lda")

# Accuracies
sam <- confusionMatrix(predict(modDC, testing), testing$classe)
sam$overall[1]
calc_acc <- function(predicted, actual) {
    confusionMatrix(predicted, actual)$overall[1]
}

calc_acc(predict(modDC, testing), testing$classe)
calc_acc(predict(modRF, testing), testing$classe)
calc_acc(predict(modGBM, testing), testing$classe)
calc_acc(predict(modGLM, testing), testing$classe)
calc_acc(predict(modNB, testing), testing$classe)
calc_acc(predict(modLDA, testing), testing$classe)

models <- list(modDC, modRF, modGBM, modGLM, modNB, modLDA)
accuracies <- sapply(models, function(x) {
    confusionMatrix(predict(x, testing), testing$classe)$overall[1]
})

names(accuracies) <- c("modDC", "modRF", "modGBM", "modGLM", "modNB", "modLDA")

install.packages(c("doMC", 'xgboost'))
system.time(modRANGRE <- train(classe ~ ., data = training, method = "ranger"))

pml <- read.csv("./pml-training.csv", na.strings = c("NA", "", " ", "#DIV/0!"))
str(pml)
nas <- sapply(pml, function(x) sum(is.na(x)))
summary(nas[nas > 0])
nacols <- which(nas > 0)
names(nacols) <- NULL

pml2 <- pml[, -nacols]
summary(sapply(pml2, function(x) sum(is.na(x))))

pml3 <- pml2[, -c(1:7)]
pml3$classe <- as.factor(pml3$classe)
str(pml3)
library(data.table)
pml3 <- as.data.table(pml3)

set.seed(4)
inTrain <- createDataPartition(pml3$classe,
                               p = 0.8,
                               list = FALSE)
training <- pml3[inTrain,]
testing <- pml3[-inTrain,]

calc_acc(predict(modRANGRE, testing), testing$classe)

# relatively better gbm
system.time(modXGB <- train(classe ~ ., data = training, method = "xgbTree", verbose = FALSE, trControl = trainControl(allowParallel = TRUE)))

# random forest model
system.time(modRF <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(allowParallel = TRUE)))

# gradient boosting model
system.time(modGBM <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE, trControl = trainControl(allowParallel = TRUE)))

# another randomforest (ranger)
system.time(modRANGER <- train(classe ~ ., data = training, method = "ranger", trControl = trainControl(allowParallel = TRUE)))

confusionMatrix(predict(modRANGER, training), training$classe)
class(training)

modRANGER
plot(modRANGER)
modRANGER

confusionMatrix(predict(modRANGER, training), training$classe)
confusionMatrix(predict(modRANGER, testing), testing$classe)
rangerAcc <- confusionMatrix(predict(modRANGER, testing), testing$classe)$overall[1]
names(rangerAcc) <- "Ranger (Random Forest)"

modXGB
plot(modXGB)
confusionMatrix(predict(modXGB, training), training$classe)
confusionMatrix(predict(modXGB, testing), testing$classe)
xgbAcc <- confusionMatrix(predict(modXGB, testing), testing$classe)$overall[1]
names(xgbAcc) <- "XG Boost (Tree)"


class(modRANGER$results)
modRANGER$results$Accuracy[modRANGER$results$splitrule == "gini"]

preProc = trainControl(method = "repeatedcv", number = 10, repeats = 10, allowParallel = TRUE)
modRANGER2 <- train(classe ~ ., data = training, method = "ranger", trControl = preProc, importance = 'impurity')

library(dplyr)
library(faux)
library(caret)
library(DataExplorer)
library(randomForest)
library(ggplot2)
library(corrplot)
library(data.table)
library(mlbench)

modRANGER2
confusionMatrix(predict(modRANGER2, testing), testing$classe)
plot(modRANGER2)
modRANGER2$finalModel
modRANGER2$xlevels
varImp(modRANGER2)

str(modRANGER2$control)

importance(modRANGER2)
varImp(modRANGER2)

smallTrain <- training[sample(nrow(training), size = 2000),]
str(smallTrain)

modRanS <- train(classe ~ ., 
                    data = smallTrain, 
                    method = "ranger", 
                    importance = 'impurity', 
                    trControl = trainControl(method = "repeatedcv", 
                                             number = 10, 
                                             repeats = 10, 
                                             allowParallel = TRUE))

modRanS
confusionMatrix(predict(modRanS, testing), testing$classe)$overall[1]
confusionMatrix(predict(modRANGER, testing), testing$classe)$overall[1]
confusionMatrix(predict(modRANGER2, testing), testing$classe)$overall[1]
confusionMatrix(predict(modXGB, testing), testing$classe)$overall[1]

varimp <- varImp(modRANGER2)
varimp$importance[order(varimp$importance$Overall, decreasing = TRUE),][1:10]


featurePlot(pml3[, -53], pml3$classe)
length(pml3)
?corrplot
corrplot(cor(pml3[,-53]), method = 'color')
cor(pml3)
for (i in seq_along(pml3)) {
    plot(i, pch  =19, cex = 1.5)
}
plot(pml3$roll_belt)

DataExplorer::plot_correlation(pml3)

modRANGER2$bestTune
varImpPlot(modRANGER2)
plot(modRANGER2)

varimp <- varImp(modRANGER2)$importance
head(varimp)
varimp$variable <- rownames(varimp)
rownames(varimp) <- NULL
varimp <- varimp[order(varimp$Overall, decreasing = TRUE),]
nrow(varimp)
class(varimp)
length(varimp$Overall)

g <- ggplot(varimp, aes(x = Overall, y = reorder(variable, Overall), fill = Overall))
g + geom_bar(stat = "identity", position = "dodge") + xlab("Importance") + ylab('Features') + labs(title = "Variable Importance Plot") + guides(fill = F) + scale_fill_gradient(low = "red", high = "dodgerblue")


(modRANGER2$finalModel$prediction.error)*100
confmat <- confusionMatrix(predict(modRANGER2, testing), testing$classe)
(1 - confmat$overall[1])*100
confusionMatrix(predict(modRANGER2, testing), testing$classe)
confusionMatrix(predict(modRANGER2, training), training$classe)










































































































































































































































































































































































































































































































































