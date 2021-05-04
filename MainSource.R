rm(list=ls())

library(mlr)
library(kknn)
library(MuMIn)
library(dplyr)
library(glmnet)
library(randomForest)
library(caret)

TrainA <- read.csv("Data-Source/Train.csv")
TestSet <- read.csv("Data-Source/Test.csv")#Testing set to extract full data 

# Training and Validation Sets:
#set.seed(18)
Train <- sample_frac(TrainA,size = 0.9,replace = FALSE) #Use 90% of the data 
Test  <- setdiff(TrainA,Train) # 

x <- model.matrix(Target ~ ., Train)[, -1]  #We omit the intercept
y <- Train$Target

xTest<-model.matrix(Target ~ ., Test)[, -1]
yTest<-Test$Target

xFinalPreds<-model.matrix( ~ ., TestSet)[, -1]
yFinalPreds<-Test$Target

#Create Binomial Model
fit1 <- glm(data=Train, Target~., family="binomial");fit_1_AIC <- AIC(fit1);fit_1_BIC <- BIC(fit1)
biPRed <- predict(fit1, type="response",newdata = Test)
Test$Binnomial <- biPRed

#Tree
fitControl <- trainControl(method="cv", number=10)
model.tree <- train(Target ~ ., data = Train,method = "rpart", trControl = fitControl)
tree.pred <- predict(model.tree, newdata = Test)
Test$TreeTest <- tree.pred 

#lasso
cv.out <- cv.glmnet(x,y,alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
fitlm<-lm(Target~., data=Train)
lmpred<-predict(fitlm, newdata = Test)
lasso.mod <- glmnet(x, y, alpha=1, lambda=bestlam)
lasso.pred <- predict(lasso.mod , newx=xTest,s=bestlam, exact = T)

Test$LassoTest <- lasso.pred


#Change Test Values. 
Test$Target <- factor(Test$Target,levels=c(0,1),labels = c("<50%",">50%"),ordered = FALSE)

#Change from Values to <50%
Test <- mutate(.data=Test,Binnomial =case_when(Binnomial>.5~">50%",TRUE~"<50%"))
Test <- mutate(.data=Test,TreeTest  =case_when(TreeTest>.5~">50%",TRUE~"<50%"))
Test <- mutate(.data=Test,LassoTest =case_when(LassoTest>.5~">50%",TRUE~"<50%"))

confusion1 <- table(Test$Target,Test$Binnomial, dnn=c("Actual","Predicted"))
confusion2 <- table(Test$Target,Test$TreeTest,  dnn=c("Actual","Predicted"))
confusion3 <- table(Test$Target,Test$LassoTest, dnn=c("Actual","Predicted"))

fit_1_Pred <- print(paste("Fraction of BI Correct Predictions:", round((confusion1["<50%","<50%"]+confusion1[">50%",">50%"])/sum(confusion1),4)))
fit_2_Pred <- print(paste("Fraction of Tree Correct Predictions:", round((confusion2["<50%","<50%"]+confusion2[">50%",">50%"])/sum(confusion2),4)))
fit_3_Pred <- print(paste("Fraction of Lasso Correct Predictions:", round((confusion3["<50%","<50%"]+confusion3[">50%",">50%"])/sum(confusion3),4)))

fourfoldplot(confusion1, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix Top Attributes [Binomal]")
fourfoldplot(confusion2, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix Top Attributes [Tree]")
fourfoldplot(confusion3, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix Top Attributes [Lasso]")


########Create Predections with Testing stuff########
biPred <- predict(fit1, type="response",newdata = TestSet)
lasso.predTestSet <- predict(lasso.mod, newx=xFinalPreds,s=bestlam, exact = T)
tree.predTestSet <- predict(model.tree, newdata = TestSet)

TestSet$BinomialPredications <- biPred
TestSet$lassoPredictions <- tree.predTestSet 
TestSet$TreePredictions <- lasso.predTestSet


TestSet <- mutate(.data=TestSet,BinomialPredications =case_when(BinomialPredications >.5~">50%",TRUE~"<50%"))
TestSet <- mutate(.data=TestSet,lassoPredictions     =case_when(lassoPredictions     >.5~">50%",TRUE~"<50%"))
TestSet <- mutate(.data=TestSet,TreePredictions      =case_when(TreePredictions      >.5~">50%",TRUE~"<50%"))

#####################################################

classifTask <- makeClassifTask(data = Train,target = "Target")
kfold <- makeResampleDesc("RepCV",folds = 5,reps = 20)

#### Binomial, K=10 ####
kfoldCVBIN <- resample(learner=makeLearner("classif.binomial"), classifTask,resampling = kfold, show.info = FALSE)
CMBIN <- calculateConfusionMatrix(kfoldCVBIN$pred)
AccuracyBIN <- (CMBIN$result[1]+CMBIN$result[5])/(CMBIN$result[1]+CMBIN$result[2]+CMBIN$result[4]+CMBIN$result[5])

#### TREE, K=10 ####
kfoldCVTree <- resample(learner=makeLearner("classif.rpart"), classifTask, resampling = kfold,show.info = FALSE)
CMTree        <- calculateConfusionMatrix(kfoldCVTree$pred)
AccuracyTree<- (CMTree$result[1]+CMTree$result[5])/(CMTree$result[1]+CMTree$result[2]+CMTree$result[4]+CMTree$result[5])

#### LASSO, K=10 ####
kfoldCVLAS   <- resample(learner=makeLearner("classif.glmnet"), classifTask, resampling = kfold,show.info = FALSE)
CMLAS        <- calculateConfusionMatrix(kfoldCVLAS$pred)
AccuracyLAS <- (CMLAS$result[1]+CMLAS$result[5])/(CMLAS$result[1]+CMLAS$result[2]+CMLAS$result[4]+CMLAS$result[5])

print("")
print("")

dsfdsf <-  print(paste("Binmomial Accuracy:",round(AccuracyBIN,2)))
print(paste( "Tree Accuracy:", round(AccuracyTree,2)))
print(paste("Lasso Accuracy:", round(AccuracyLAS,2)))
