rm(list=ls())

library(mlr)
library(kknn)
library(MuMIn)
library(dplyr)
library(glmnet)
#library(caret)

# Prep:
TrainA <- read.csv("Data-Source/Train.csv")

#Testing set to extract full datat 
TestSet <- read.csv("Data-Source/Test.csv")

#Switching 
TrainA$Sex <- factor(TrainA$Sex, 
                     levels=c(0,1),
                     labels=c("Female","Male"),
                     ordered=FALSE)

TrainA$ChestPain <- factor(TrainA$ChestPain,
                           levels=c(0,1,2,3),
                           labels=c("Asymptomatic","Atypical Angina","Non-Anginal Pain","Typical Angina"),
                           ordered=FALSE)

TrainA$FastingBloodSugar <- factor(TrainA$FastingBloodSugar,
                                   levels=c(0,1),labels = c("False","True"),
                                   ordered=FALSE)


TrainA$RestECG <- factor(TrainA$RestECG, levels=c(0,1,2),
                         labels=c("LV Hypertrophy","Normal","Abnormalities"),
                         ordered=FALSE)

TrainA$ExIndAng <- factor(TrainA$ExIndAng, levels=c(0,1),
                          labels=c("No","Yes"),
                          ordered=FALSE)

TrainA$Slope <- factor(TrainA$Slope, levels=c(0,1,2),
                       labels=c("Descending","Flat","Ascending"),
                       ordered=FALSE)

TrainA$ThalRate <- factor(TrainA$ThalRate, levels=c(1,2,3),
                          labels=c("Fixed Defect","Normal","Reversible Defect"),
                          ordered=FALSE)

TrainA$MajorVessels <- factor(TrainA$MajorVessels,
                              levels=c(0,1,2,3))

TrainA$Target <- factor(TrainA$Target,
                        levels=c(0,1),
                        labels = c("<50%",">50%"),
                        ordered = FALSE)

# Training and Validation Sets:
set.seed(18)

Train <- sample_frac(TrainA,size = 0.9,replace = FALSE)
Val <- setdiff(TrainA,Train)

ValB <- Val
TrainB <- Train

fit1 <- glm(data=TrainB, Target~., family="binomial")
fit_1_AIC <- AIC(fit1)
fit_1_BIC <- BIC(fit1)

ValB <- mutate(.data=ValB,Prediction1=predict(fit1, type="response",newdata = ValB))
ValB <- mutate(.data=ValB,KindPred1=case_when(Prediction1>.5~">50%",   TRUE~"<50%"))

confusion1 <- table(ValB$Target,ValB$KindPred1,dnn=c("Actual","Predicted 1"))

fit_1_Pred <- print(paste("Fraction of Fit1 Correct Predictions:", round((confusion1["<50%","<50%"]+confusion1[">50%",">50%"])/sum(confusion1),4)))
fit2 <- glm(data=TrainB, Target~Sex+ChestPain+RestingBloodPressure+MajorVessels+ThalRate, family="binomial")
Fit_2_AIC <- AIC(fit2)
Fit_2_BIC <- BIC(fit2)

ValB <- mutate(.data=ValB,Prediction2=predict(fit2, type="response",newdata = ValB))
ValB <- mutate(.data=ValB,KindPred2=case_when(Prediction2>.5~">50%", TRUE~"<50%"))
confusion2 <- table(ValB$Target,ValB$KindPred2,dnn=c("Actual","Predicted 2"))
fit_2_Pred <- print(paste("Fraction of Fit2 Correct Predictions:", round((confusion2["<50%","<50%"]+confusion2[">50%",">50%"])/sum(confusion2),4)))

###########
### MLR ###
###########

classifTask <- makeClassifTask(data = Train,target = "Target")
kfold <- makeResampleDesc("RepCV",folds = 5,reps = 20)
#### Binomial, K=10 ####

kfoldCVBIN <- resample(learner=makeLearner("classif.binomial"),classifTask,resampling = kfold,show.info = FALSE)
CMBIN <- calculateConfusionMatrix(kfoldCVBIN$pred)
AccuracyBIN <- (CMBIN$result[1]+CMBIN$result[5])/(CMBIN$result[1]+CMBIN$result[2]+CMBIN$result[4]+CMBIN$result[5])

#### LASSO, K=10 ####
kfoldCVLAS <- resample(learner=makeLearner("classif.glmnet"), classifTask, resampling = kfold,show.info = FALSE)
CMLAS <- calculateConfusionMatrix(kfoldCVLAS$pred)
AccuracyLAS <- (CMLAS$result[1]+CMLAS$result[5])/(CMLAS$result[1]+CMLAS$result[2]+CMLAS$result[4]+CMLAS$result[5])

#### Random Forest, K=10 ####
kfoldCVRF<-resample(learner=makeLearner("classif.rpart"),   classifTask,     resampling = kfold,show.info = FALSE)
CMRF <- calculateConfusionMatrix(kfoldCVRF$pred)
AccuracyRF <- (CMRF$result[1]+CMRF$result[5])/(CMRF$result[1]+CMRF$result[2]+ CMRF$result[4]+CMRF$result[5])

#### KNN, K=10 ####
kfoldCVKKNN<-resample(learner=makeLearner("classif.kknn"), classifTask, resampling = kfold,show.info = FALSE)
CMKKNN<-calculateConfusionMatrix(kfoldCVKKNN$pred)
AccuracyKKNN <- (CMKKNN$result[1]+CMKKNN$result[5])/(CMKKNN$result[1]+CMKKNN$result[2] +CMKKNN$result[4]+CMKKNN$result[5])
print('Finished :)')