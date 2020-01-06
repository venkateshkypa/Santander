########## Santander Customer Transaction Prediction Problem ##################

### Set Working Directory ####
setwd('C:/Users/kyvenkat/Desktop/W0376 Backup/Documents Backup')
getwd()

#### loading libraries #######################################

x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "rpart.plot", "rpart",'xgboost','stats','xgboost','SMOTE','ROSE',
      'pROC','ROCR','C50','Matrix')

#load Packages
lapply(x, require, character.only = TRUE)
rm(x)
  
### Loading data ###
santander_train<- read.csv("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\train (1).csv", header = TRUE) 
santander_test<- read.csv("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\test.csv", header = TRUE) 

#santander_train<- read.csv("C:\\Users\\Venkatesh K\\Downloads\\train.csv", header = TRUE) 

### Exploratory Data Analysis #########

str(santander_train)

## Changing required data types

santander_train$ID_code<- as.character(santander_train$ID_code)
santander_train$target <- as.factor(santander_train$target)

### Analyzing Missing values 

Missing_Val<-data.frame(sapply(santander_train,function(x) sum(is.na(x)))) 
Missing_Val$Variables<-rownames(Missing_Val)
row.names(Missing_Val)<-NULL
colnames(Missing_Val)<-c("Missing Values","Variables")

#### There is no missing data in the given dataset

##### Outlier Analysis #######################

y<<-NULL
for(i in 3:length(santander_train)){
  z<-santander_train[,i][santander_train[,i]%in% boxplot.stats(santander_train[,i])$out]
  y<-cbind(y,which(santander_train[,i]%in% z))
}
rm(i,z)
santander_out<-apply(y,2,function(x) unique(x))
santander_out<-array(as.numeric(unlist(santander_out)))
santander_out<-unique(santander_out)



# Remove outliers from the data 

santander_train<-santander_train[-c(santander_out),]

### Check the distribution of Dependent Variables

prop.table(table(santander_train['target']))

ggplot(santander_train, aes(target))+theme_classic()+geom_bar(stat = 'count', fill = 'lightblue', show.legend = TRUE)

### Check the summary of Independent variables 

summary(santander_train)

## Check the skewness of Independent variables 

santander_train_new<-santander_train[,-c(1)]

sant_skw<- apply(santander_train_new,2, function(x) skewness(x))

qplot(sant_skw, geom = 'histogram')

### skewness is in between -0.34 to 0.3 which says that variables are more or less symmetric and follow normal distribution

### Dividing train data based on the target variable

santander_train_target0<-santander_train[which(santander_train$target==0),]
santander_train_target1<-santander_train[which(santander_train$target==1),]

plot_density(santander_train_target1[,c(3:50)], ggtheme = theme_classic(),geom_density_args = list(color='red'))


#### Features Correlation

corr_values<-cor(santander_train_new[,-c(1)], method = 'pearson')
corrgram(santander_train_new, order = NULL, lower.panel = panel.shade, upper.panel = panel.pie, text.panel = panel.txt, 
         main = "Correlation Plot")

### which shows there is no correlation between the Independent variables 


############ Outlier Analysis ##############################

y<<-NULL
for(i in 3:202){
  z<-santander_train[,i][santander_train[,i]%in% boxplot.stats(santander_train[,i])$out]
  y<-rbind(y,which(santander_train[,i]%in% z))
}
i=3
length(z)


### Split the data into Train & Test samples
idx<- sample(nrow(santander_train), size = 0.75*nrow(santander_train))
train_data<- santander_train[idx,]
test_data<-santander_train[-idx,]

#### Removing first column which is non significant 
train_data<- train_data[,-c(1)]
test_data<- test_data[,-c(1)]

############ Model Building ###############################

## Logistic Regression without class imbalance

lr_model<-glm(target~.,data = train_data, family = 'binomial')
summary(lr_model)

pred<-predict(lr_model,test_data[,-c(1)], response = 'class')
pred.resp<-ifelse(pred>0.5,1,0)
accuracy <- table(pred.resp, test_data[,"target"])
sum(diag(accuracy))/sum(accuracy)

# 1. Accuracy = 0.9106
# 2. FNR = 0.23415


### Variable Importance
var_imp_lr.model<-data.frame(varImp(lr_model))
var_imp_lr.model$Variables<-row.names(var_imp_lr.model)
row.names(var_imp_lr.model)<-NULL
var_imp_lr.model<-var_imp_lr.model[,c(2,1)]
## AUC curve

prob <- predict(lr_model, newdata=test_data[,-c(1)], type="response")
pred <- prediction(prob, test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

# auc: 0.857059



## Important variables in predicting customer transaction in linear regression without class balancing

var_imp_lr.model<-data.frame(varImp(lr_model))
var_imp_lr.model$Variables<-row.names(var_imp_lr.model)
row.names(var_imp_lr.model)<-NULL
var_imp_lr.model<-var_imp_lr.model[,c(2,1)]


prob <- predict(lr_model, newdata=test_data[,-c(1)], type="response")
pred <- prediction(prob, test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

########## Logistic Regression with class Balacing 


train.rose <- ROSE(target~., data =train_data,seed=1)$data
lr_model_cb<- glm(target~., data = train.rose, family = 'binomial')

pred<-predict(lr_model_cb,test_data[,-c(1)], response = 'class')
pred.resp<-ifelse(pred>0.5,1,0)
accuracy <- table(pred.resp, test_data[,"target"])
sum(diag(accuracy))/sum(accuracy)

# 1. Accuracy = 0.87964
# 2. FNR = 0.56857

### Variable Importance
var_imp_lr.model_cb<-data.frame(varImp(lr_model_cb))
var_imp_lr.model_cb$Variables<-row.names(var_imp_lr.model_cb)
row.names(var_imp_lr.model_cb)<-NULL
var_imp_lr.model_cb<-var_imp_lr.model_cb[,c(2,1)]
## AUC curve

prob <- predict(lr_model_cb, newdata=test_data[,-c(1)], type="response")
pred <- prediction(prob, test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

### auc:0.86132


#### Decision Tree

tree_mod <- C5.0(x = train_data[, -c(1)], y = train_data$target)
summary(tree_mod)
pred<-predict(tree_mod,test_data[,-c(1)])
accuracy <- table(pred, test_data[,"target"])
sum(diag(accuracy))/sum(accuracy)



# 1. Accuracy = 0.87688
# 2. FNR = 0.730297

### Variable Importance
var_imp_tree_mod<-data.frame(varImp(tree_mod))
var_imp_tree_mod$Variables<-row.names(var_imp_tree_mod)
row.names(var_imp_tree_mod)<-NULL
var_imp_tree_mod<-var_imp_tree_mod[,c(2,1)]

## AUC curve

prob <- predict(tree_mod, newdata=test_data[,-c(1)], type = 'prob')[,2]
pred <- prediction(prob, test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

# auc: 0.64244


#### Random Forest Classifier 

library(randomForest)

rf_model<-randomForest(target~.,data = train_data,mtry =20, ntree = 10,importance=TRUE)
summary(rf_model)
pred<-predict(rf_model,test_data[,-c(1)])
accuracy <- table(pred, test_data[,"target"])
sum(diag(accuracy))/sum(accuracy)

# 1. Accuracy = 0.89762
# 2. FNR = 0.521126

### Variable Importance
var_imp_rf_model<-data.frame(varImp(rf_model, type=2))
var_imp_rf_model$Variables<-row.names(var_imp_rf_model)
row.names(var_imp_rf_model)<-NULL
var_imp_rf_model<-var_imp_rf_model[,c(2,1)]

## AUC curve

prob <- predict(rf_model, newdata=test_data[,-c(1)], type = 'prob')
pred <- prediction(prob[,2], test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
## auc: 0.702122

#### XGBoost Classifier without class imbalance ############################



target = santander_train$target
label = as.integer(santander_train$target)-1

train.label = label[idx]
test.label = label[-idx]

dtrain  <- sparse.model.matrix(target ~ .-1, data=train_data)
dtest   <- sparse.model.matrix(target ~ .-1, data=test_data)
dim(dtrain)
dim(dtest)

param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              booster = "gbtree",
              max_depth   = 7,
              eta         = 0.1,
              gammma      = 1,
              lambda      = 1,
              colsample_bytree = 0.5,
              min_child_weight = 1)

xgb_model <- xgboost(params  = param,
                     data    = dtrain,
                     label   = train.label, 
                     nrounds = 500,
                     print_every_n = 100,
                     verbose = 1)

pred <- predict(xgb_model, dtest)
pred.resp <- ifelse(pred >= 0.5, 1, 0)

pred_table<-table(pred.resp, test_data[,"target"])
accuracy<-sum(diag(accuracy))/sum(accuracy)

# 1. Accuracy = 0.9174
# 2. FNR = 0.2115

## AUC curve

prob <- predict(xgb_model, newdata=dtest, type = 'prob')
pred <- prediction(prob, test_data$target)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col = 'red')
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
## auc: 0.88054

mat<-xgb.importance(feature_names = colnames(dtest2),model = xgb_model)
xgb.plot.importance (importance_matrix = mat[1:20])


######################## XG Boost with Class balancing ####################

prop.table(table(santander_train$target)) 

inTrain <- createDataPartition(y = santander_train$target, p = .6, list = F)
train <- santander_train[inTrain,]
testcv <- santander_train[-inTrain,]
inTest <- createDataPartition(y = testcv$target, p = .5, list = F)
test <- testcv[inTest,]
cv <- testcv[-inTest,]
santander_train <- as.factor(santander_train$target)
rm(inTrain, inTest, testcv)

train<-train[,-c(1)]
test<-test[,-c(1)]


i<- grep("target", colnames(train))

train_SMOTE<-SMOTE(target~.,data = train,perc.over = 200, perc.under=100)

table(train_SMOTE$target)

train$target<-as.numeric(levels(train$target))[train$target]
train_SMOTE$target<-as.numeric(levels(train_SMOTE$target))[train_SMOTE$target]


train_smote <- Matrix(as.matrix(train_SMOTE), sparse = TRUE)
test <- Matrix(as.matrix(test), sparse = TRUE)
cv <- Matrix(as.matrix(cv), sparse = TRUE)
train <- Matrix(as.matrix(train), sparse = TRUE)

train_smote_xgb <- xgb.DMatrix(data = train_smote[,-1], label = train_smote[,1])
test_xgb <- xgb.DMatrix(data = test[,-1], label = test[,1])
cv_xgb <- xgb.DMatrix(data = cv[,-1], label = cv[,1])
train_xgb<-xgb.DMatrix(data = train[,-1], label = train[,1])



parameters <- list(
  booster            = "gbtree",          
  silent             = 0,                 
  eta                = 0.3,               
  gamma              = 0,                 
  max_depth          = 6,                 
  min_child_weight   = 1,                 
  subsample          = 1,                 
  colsample_bytree   = 1,                 
  colsample_bylevel  = 1,                 
  lambda             = 1,                 
  alpha              = 0,                 
  objective          = "binary:logistic",   
  eval_metric        = "auc",
  seed               = 1900               
)

watchlist <- list(train  = train_xgb, cv = cv_xgb)


xgb_smote.model <- xgb.train(parameters, train_smote_xgb, nrounds = 500, watchlist)


xgb_smote.predict <- predict(xgb_smote.model, test[,-c(1)])

xgb_smote.predict.prep<-ifelse(xgb_smote.predict >= 0.5,1,0)
accuracy <- table(xgb_smote.predict.prep, test[,1])
sum(diag(accuracy))/sum(accuracy)

## 1. Accuracy = 0.7996
## 2. FNR = 0.718


roc_smote <- roc(test[,1], predict(xgb_smote.model, test[,-c(1)], type = "prob"))
roc_smote$auc

## auc : 0.8125


####################### Saving the Final model######################

RDS.model<- saveRDS(xgb_model, "C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Datasets\\Final_XG Boost model_Santander.rds")

##### Loding the saved model #############

Final_model<- readRDS("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Datasets\\Final_XG Boost model_Santander.rds")

######################### Test data #########################

## Loading the test data 


santander_test<- read.csv("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\test.csv", header = TRUE) 
head(santander_test,5)
santander_test<-santander_test[,-c(1)]
str(santander_test)
################### Finalizing the model and predicting customer transactions on test data #######################

dtest2<-Matrix(as.matrix(santander_test), sparse = TRUE)
dim(dtest2)



### Predict the test data  


xgb = predict(xgb_model,dtest2)
pred <- ifelse(xgb >= 0.5, 1, 0)

### Final test data with customer transaction predictions #############################

Customer_Transaction_predictions<- data.frame(santander_test,"predictions" = pred)

Customer_Transaction_predictions<-Customer_Transaction_predictions[c(201,1:200)]

write.csv(Customer_Transaction_predictions,"Santander_Customer_Transaction_predictions_R.csv",row.names = FALSE)



