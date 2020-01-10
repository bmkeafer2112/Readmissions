###HW7
###Brian Keafer

########################################################Library Loads######################################################
library(tidyverse) 
library(lubridate)
library(ggplot2)
library(forcats)
library(caret)
library(glmnet)
library(magrittr)
library(MASS)
library(dplyr)
library(randomForest)
library(rpart)
library(naivebayes)
library(car)

#########################################Hospital Readmits Classification Problem############################################

############################################Initial Dataframe Creation/Set-up/Exploration#################################

###Import Data Sets
dfTrain <- read.csv("hm7-Train.csv")

dfTest <- read.csv("hm-7-Test.csv")

####Create Unique IDs for Training and Test Data
trainIds <- unique(dfTrain$patientID)  
testIds <- unique(dfTest$patientID)

###Remove and Save "readmitted" dependent variable for later
readmitted <- dplyr::select(dfTrain, patientID, readmitted)
readmitted[readmitted==1] <- 'Yes'
readmitted[readmitted==0] <- 'No'
readmitted$readmitted <- as.factor(readmitted$readmitted)
str(readmitted)

###combine Train and Test Data Sets to perform Transformations/Feature Engineering
combined_DF <- dfTrain %>% dplyr::select(-readmitted) %>% bind_rows(dfTest)
head(combined_DF)

###View Structure
str(combined_DF) 

###Remove underscores from column names
names(combined_DF)
names(combined_DF) <- gsub("\\_","",names(combined_DF))
names(combined_DF)

names(combined_DF) <- gsub("\\.","",names(combined_DF))
names(combined_DF)

###Explore Missingness
combined_DF %>% dplyr::select(-c(patientID)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()


###Check Duplicate ids
dup <- duplicated(combined_DF$patientID)
table(dup)

###str(combined_DF)
###Convert Characters, Logical, and some numeric classes to Factors
factor_cols <- c("race", "gender", "age", "admissiontype", "dischargedisposition", "admissionsource", "payercode", 
                 "medicalspecialty", "maxgluserum", "A1Cresult", "metformin", 
                 "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
                 "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                 "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
                 "glyburidemetformin", "glipizidemetformin", "glimepiridepioglitazone", "metforminrosiglitazone",
                 "metforminpioglitazone", "diabetesMed")

combined_DF %<>% mutate_at(factor_cols, funs(factor(.)))

str(combined_DF)

###Factor Lump Medical Specialty, Race, and payer code to remove missingness and decrease factor levels
combined_DF <- combined_DF %>% 
  mutate(medicalspecialty = fct_lump(fct_explicit_na(medicalspecialty), prop = 0.01))
  
###Lump Diagnosis based upon Categories in the Disease and Injuries Tabular Index
combined_DF <- combined_DF %>% mutate(diagnosis=
ifelse(startsWith(as.character(diagnosis), "V"), "V",
ifelse(startsWith(as.character(diagnosis), "E"), "E",
ifelse(startsWith(as.character(diagnosis), "O"), "O",       
ifelse(between(as.numeric(as.character(diagnosis)), 1, 139), "1",
ifelse(between(as.numeric(as.character(diagnosis)), 140, 239), "2",
ifelse(between(as.numeric(as.character(diagnosis)), 240, 279), "3",
ifelse(between(as.numeric(as.character(diagnosis)), 280, 289), "4",
ifelse(between(as.numeric(as.character(diagnosis)), 290, 319), "5",
ifelse(between(as.numeric(as.character(diagnosis)), 320, 389), "6",
ifelse(between(as.numeric(as.character(diagnosis)), 390, 459), "7",
ifelse(between(as.numeric(as.character(diagnosis)), 460, 519), "8",
ifelse(between(as.numeric(as.character(diagnosis)), 520, 579), "9", 
ifelse(between(as.numeric(as.character(diagnosis)), 580, 629), "10",
ifelse(between(as.numeric(as.character(diagnosis)), 630, 679), "11",
ifelse(between(as.numeric(as.character(diagnosis)), 680, 709), "12",
ifelse(between(as.numeric(as.character(diagnosis)), 710, 739), "13",
ifelse(between(as.numeric(as.character(diagnosis)), 740, 759), "14",
ifelse(between(as.numeric(as.character(diagnosis)), 760, 779), "15",
ifelse(between(as.numeric(as.character(diagnosis)), 780, 799), "16",
ifelse(between(as.numeric(as.character(diagnosis)), 800, 999), "17",
"O" )))))))))))))))))))))

###Convert Diagnosis back to factor
factor_cols <- c("diagnosis")

combined_DF %<>% mutate_at(factor_cols, funs(factor(.)))         
view(combined_DF)

###Remove Variables without variance
combined_DF <- combined_DF %>% dplyr::select(-c(nearZeroVar(combined_DF)))

###Check Structure
str(combined_DF)

###Retrieve Transformed Train Data
train_trans_DF <- data.frame(combined_DF %>% filter(patientID %in% trainIds))
view(train_trans_DF)


###Join readmitted variable to transformed train data
train_trans_DF <- train_trans_DF %>% inner_join(readmitted, by = "patientID")
train_trans_DF$readmitted <- as.factor(train_trans_DF$readmitted)
head(train_trans_DF)

###Retrieve Transformed Test Data
test_trans_DF <-combined_DF %>% filter(patientID %in% testIds)
head(test_trans_DF)

###Check Structure
str(train_trans_DF)
str(test_trans_DF)

###Check Missingness
train_trans_DF %>% dplyr::select(-c(patientID)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()
test_trans_DF %>% dplyr::select(-c(patientID)) %>% mutate_all(is.na) %>% summarise_all(mean) %>% glimpse()

###Correlation Matrix & Heatmap of Numeric Variables
num_varaibles <- as.data.frame(dplyr::select(train_trans_DF, timeinhospital, numlabprocedures, numprocedures,
                                             nummedications, numberoutpatient, numberemergency, numberinpatient,
                                             numberdiagnoses, readmitted))
str(num_varaibles)
corMat<-cor(num_varaibles)
corMat
heatmap(corMat, margins = c(10,10))

################################################Logistic Regression Model################################################

###Train Control
logreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Model using caret
set.seed(21)
logreg_fit1 <- train(readmitted ~.,data = train_trans_DF %>% dplyr::select(-c(patientID)),
            method = "glm", family = binomial, metric = "Accuracy", trControl = logreg_control1)

###Summary of Model
print(logreg_fit1)
summary(logreg_fit1)
exp(coef(logreg_fit1))

###Predict Probablities
logreg_probabilities <- logreg_fit1 %>% predict(test_trans_DF, type = "prob") 
view(logreg_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
logreg_results <- cbind(ID, log_probabilities)
logreg_results <- logreg_results %>% dplyr::rename("predReadmit" = "Yes")
logreg_results <- dplyr::select(logreg_results, -c(No))


#write out results
write.csv(logreg_results, 'log_model.csv', row.names = F)

#############################################Logistic Regression Evaluation##############################################
#######################################################LogLoss#########################################################
###Train Control
logreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary) 

###Build Formula
logreg_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
logreg_fit1 <- train(logreg_form, data = train_trans_DF %>% dplyr::select(-c(patientID)),
                     method = "glm", family = binomial, metric = "ROC", trControl = logreg_control1)

###Summary of Model
print(logreg_fit1)

##############################################ROC, Sensitivity, and Specificity#########################################
###Train Control
logreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = mnLogLoss) 

###Build Formula
logreg_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
logreg_fit1 <- train(logreg_form, data = train_trans_DF %>% dplyr::select(-c(patientID)),
                     method = "glm", family = binomial, metric = "logLoss", trControl = logreg_control1)

###Summary of Model
print(logreg_fit1)

#######################################################D-Statistic####################################################
###Train Control
logreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Build Formula
logreg_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
logreg_fit1 <- glm(logreg_form, data = train_trans_DF %>% dplyr::select(-c(patientID)), family = binomial)

###Summary of Model
print(logreg_fit1)


###the "fit" glm object has a lot of useful information
names(logreg_fit1)

head(logreg_fit1$data)             # all of the data is stored
head(logreg_fit1$y)                # the "true" value of the binary target  
head(logreg_fit1$fitted.values)    # the predicted probabilities
logreg_fit1$deviance               # the residual deviance

#now let's take a look at the residuals

pearsonRes <-residuals(logreg_fit1,type="pearson")
devianceRes <-residuals(logreg_fit1,type="deviance")
rawRes <-residuals(logreg_fit1,type="response")
studentDevRes<-rstudent(logreg_fit1)
fv<-fitted(logreg_fit1)

#let's go ahead and create a classification based on the probability
train_trans_DF$pred<-as.numeric(logreg_fit1$fitted.values>0.5)


predVals <-  data.frame(trueVal=train_trans_DF$readmitted, predClass=train_trans_DF$pred, predProb=fv, 
                        rawRes, pearsonRes, devianceRes, studentDevRes)

tail(predVals)

#D statistic (2009)
readmitted.1<-predVals[predVals$trueVal==1,]
readmitted.0<-predVals[predVals$trueVal==0,]

mean(readmitted.1$predProb) - mean(readmitted.0$predProb)

###############################################Kolmogorov-Smirnov Chart#################################################
#K-S chart  (Kolmogorov-Smirnov chart) 
# measures the degree of separation 
# between the positive (y=1) and negative (y=0) distributions

predVals$group<-cut(predVals$predProb,seq(1,0,-.1),include.lowest=T)
xtab<-table(predVals$group,predVals$trueVal)

xtab

#make empty dataframe
KS<-data.frame(Group=numeric(10),
               CumPct0=numeric(10),
               CumPct1=numeric(10),
               Dif=numeric(10))

#fill data frame with information: Group ID, 
#Cumulative % of 0's, of 1's and Difference
for (i in 1:10) {
  KS$Group[i]<-i
  KS$CumPct0[i] <- sum(xtab[1:i,1]) / sum(xtab[,1])
  KS$CumPct1[i] <- sum(xtab[1:i,2]) / sum(xtab[,2])
  KS$Dif[i]<-abs(KS$CumPct0[i]-KS$CumPct1[i])
}

KS  

KS[KS$Dif==max(KS$Dif),]

maxGroup<-KS[KS$Dif==max(KS$Dif),][1,1]

#and the K-S chart
ggplot(data=KS)+
  geom_line(aes(Group,CumPct0),color="blue")+
  geom_line(aes(Group,CumPct1),color="red")+
  geom_segment(x=maxGroup,xend=maxGroup,
               y=KS$CumPct0[maxGroup],yend=KS$CumPct1[maxGroup])+
  labs(title = "K-S Chart", x= "Deciles", y = "Cumulative Percent")





#############################################Penalized Logistic Regression Model#########################################

###Train Control
penlogreg_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
penlogreg_grid <- expand.grid(alpha=seq(0.6,1,length=4), lambda=seq(0.001,0.005,length=3))

###Build Formula
penlogreg_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
penlogreg_fit1 <- train(penlogreg_form,data = train_trans_DF %>% dplyr::select(-c(patientID)),
                     method = "glmnet",  metric = "Accuracy", trControl = penlogreg_control1, 
                     tuneGrid=penlogreg_grid)

###Summary of Model
print(penlogreg_fit1)
summary(penlogreg_fit1)
exp(coef(penlogreg_fit1))


###Predict Probablities
penlogreg_probabilities <- penlogreg_fit1 %>% predict(test_trans_DF, type = "prob")
view(penlogreg_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
penlogreg_results <- cbind(ID, penlogreg_probabilities)
penlogreg_results <- penlogreg_results %>% dplyr::rename("predReadmit" = "Yes")
penlogreg_results <- dplyr::select(penlogreg_results, -c(No))


#write out results
write.csv(penlogreg_results, 'penlogreg_model.csv', row.names = F)

#############################################Random Forrest Model########################################################
###Train Control
rf_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
rf_grid <- expand.grid(mtry = seq(6,8,length=2))

###Build Formula
rf_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
rf_fit1 <- train(readmitted ~., data = train_trans_DF %>% dplyr::select(-c(patientID)),
                 method = "rf", metric = "Accuracy", trControl = rf_control1,
                 tuneGrid=rf_grid)

###Summary of Model
print(rf_fit1)
summary(rf_fit1)
exp(coef(rf_fit1))


###Predict Probablities
rf_probabilities <- rf_fit1 %>% predict(test_trans_DF, type = "prob")
view(rf_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
rf_results <- cbind(ID, rf_probabilities)
rf_results <- rf_results %>% dplyr::rename("predReadmit" = "Yes")
rf_results <- dplyr::select(rf_results, -c(No))


#write out results
write.csv(rf_results, 'rf_model.csv', row.names = F)

#############################################Decision Tree Model########################################################
###Train Control
rpart_control1 = trainControl(method = "cv", number = 5) 

###Optimize Hyperparameters
rpart_grid <- expand.grid(cp=seq(0,0.09,0.005))

###Build Formula
rpart_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed


###Model using caret
set.seed(21)
rpart_fit1 <- train(rpart_form,data = train_trans_DF %>% dplyr::select(-c(patientID)),
                 method = "rpart", metric = "Accuracy", trControl = rpart_control1,
                 tuneGrid=rpart_grid)


###Summary of Model
print(rpart_fit1)
summary(rpart_fit1)
exp(coef(rpart_fit1))


###Predict Probablities
rpart_probabilities <- rpart_fit1 %>% predict(test_trans_DF, type = "prob")
view(rpart_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
rpart_results <- cbind(ID, rpart_probabilities)
rpart_results <- rpart_results %>% dplyr::rename("predReadmit" = "Yes")
rpart_results <- dplyr::select(rpart_results, -c(No))


#write out results
write.csv(rpart_results, 'rpart_model.csv', row.names = F)

###################################################MARS Model############################################################
###Train Control
MARS_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
MARS_grid <- expand.grid(degree=3, nprune=9)

###Build Formula
MARS_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
MARS_fit1 <- train(MARS_form,data = train_trans_DF %>% dplyr::select(-c(patientID)),
                        method = "earth",  metric = "Accuracy", trControl = MARS_control1, 
                        tuneGrid=MARS_grid)

help(train)

###Summary of Model
print(MARS_fit1)
summary(MARS_fit1)
exp(coef(MARS_fit1))


###Predict Probablities
MARS_probabilities <- MARS_fit1 %>% predict(test_trans_DF, type = "prob")
view(MARS_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
MARS_results <- cbind(ID, MARS_probabilities)
MARS_results <- MARS_results %>% dplyr::rename("predReadmit" = "Yes")
MARS_results <- dplyr::select(MARS_results, -c(No))


#write out results
write.csv(MARS_results, 'MARS_model.csv', row.names = F)

############################################Naive-Bayes Classification Model#############################################
###Train Control
Naive_Bayes_control1 = trainControl(method = "cv", number = 5, classProbs = TRUE) 

###Optimize Hyperparameters
Naive_Bayes_grid <- expand.grid(laplace=1, adjust=2)

###Build Formula
Naive_Bayes_form <- readmitted ~ race + gender + age + admissiontype + dischargedisposition + admissionsource + payercode +
  medicalspecialty + numlabprocedures + numprocedures + nummedications + numberoutpatient + numberemergency +  
  numberinpatient + diagnosis + numberdiagnoses + diabetesMed

###Model using caret
set.seed(21)
Naive_Bayes_fit1 <- train(Naive_Bayes_form,data = train_trans_DF %>% dplyr::select(-c(patientID)),
                   method = "naive_bayes",  metric = "Accuracy", trControl = Naive_Bayes_control1)

###Summary of Model
print(Naive_Bayes_fit1)
summary(Naive_Bayes_fit1)
exp(coef(Naive_Bayes_fit1))


###Predict Probablities
Naive_Bayes_probabilities <- Naive_Bayes_fit1 %>% predict(test_trans_DF, type = "prob")
view(Naive_Bayes_probabilities)

###Add patientID and remove additional columns
ID <- dplyr::select(test_trans_DF, patientID)
Naive_Bayes_results <- cbind(ID, Naive_Bayes_probabilities)
Naive_Bayes_results <- Naive_Bayes_results %>% dplyr::rename("predReadmit" = "Yes")
Naive_Bayes_results <- dplyr::select(Naive_Bayes_results, -c(No))


#write out results
write.csv(Naive_Bayes_results, 'Naive_Bayes_model.csv', row.names = F)





