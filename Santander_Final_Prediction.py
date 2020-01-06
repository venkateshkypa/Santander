# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:26:09 2019

@author: kyvenkat
"""

import os
import csv
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pylab as pl 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from xgboost import plot_importance
from sklearn import tree
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import graphviz
from pdpbox import pdp, get_dataset, info_plots
from matplotlib import pyplot
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve

##################### Setup Working Directory #################################

os.chdir('C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\Python Files')
os.getcwd()

#################### Load Cabfare_Train Data ##################################

santander = pd.read_csv("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\train (1).csv")
santander.head(5)
santander.dtypes
############### Converting Required Datatypes #################################

santander['target'] = pd.Categorical(santander['target'])
santander['ID_code'] = santander['ID_code'].astype(object)


################# Missing Values Analysis #####################################

missing_val = pd.DataFrame(santander.isnull().sum())
santander[santander['target'].isnull()].index
missing_val = missing_val.rename(columns = {'index':'variables',0:'Missing Values'})
missing_val_percentage = (missing_val['Missing Values']/len(santander))*100
missing_val.insert(1,"Percentage",missing_val_percentage)
missing_val  =missing_val.sort_values('Missing Values', ascending = False)

##################### Exploratory Data Analysis ###############################
################ Checking the class distribution on 'target' variable#########

target_class = santander['target'].value_counts()
percent_target_class=target_class/len(santander)*100

sns.countplot(santander['target'].values)
plt.title('Distribution of Target Class in  train dataset')

#### distribution of mean values distribution per column 
plt.figure(figsize=(16,8))
santander_attributes=santander.columns.values[1:202]
plt.title('Distribution of mean values per column in train dataset')
sns.distplot(santander[santander_attributes].mean(axis=1),color='blue',kde=True,bins=150,label='train')
###### distribution of skewness values per column 
plt.figure(figsize=(16,8))
sns.distplot(santander[santander_attributes].skew(axis=1),color='green',kde=True,bins=150,label='train')
plt.title('Distribution of skewness values per column in train dataset')

###### distribution of kurtosis values per column 
plt.figure(figsize=(16,8))
sns.distplot(santander[santander_attributes].kurtosis(axis=1),color='red',kde=True,bins=150,label='train')
plt.title('Distribution of Kurtosis values per column in train dataset')

################ Outlier Analysis #############################################

def plot_boxplot(df, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,5,figsize=(18,24))
    for feature in features:
        i += 1
        plt.subplot(10,5,i)
        sns.boxplot(df[feature])
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis = 'x', labelsize = 6, pad =-6)
        plt.tick_params(axis = 'y', labelsize = 6)
plt.show()

features = santander.columns.values[3:10]
plot_boxplot(santander,features)
############ Finding outliers and remove from the dataset######################
def outlier_calculation(x):
    ''' calculating outlier indices and replacing them with NA  '''
    #Extract quartiles
    q75, q25 = np.percentile(santander[x], [75 ,25])
    #Calculate IQR
    iqr = q75 - q25
    #Calculate inner and outer fence
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
    
    santander.loc[santander[x] < minimum,x] = np.nan
    santander.loc[santander[x] > maximum,x] = np.nan 
    
colnames = santander.columns.values[2:202]

for x in colnames:
    outlier_calculation(x)

santander.isnull().sum()

list = []
for x in colnames:
    index_list = santander[santander[x].isnull()].index.tolist()
    list.append(index_list)

list = np.unique(list)

santander = santander.dropna()

######################### Checking correlation ################################

santander_new = santander.drop(['target'], axis =1)
plt.figure(figsize =(30,10))
sns.heatmap(santander_new.corr())
plt.title('Correlation matrix ')

santander_correlations=santander_new[santander_attributes].corr()
santander_correlations=santander_correlations.values.flatten()
santander_correlations=santander_correlations[santander_correlations!=1]
plt.figure(figsize=(16,8))
sns.distplot(santander_correlations, color="Red", label="train")
plt.title('Distribution of correlation values in train dataset')


########################### Split the data ####################################
santander = santander.drop('ID_code', axis =1)

X,y=santander.drop('target',axis=1),santander['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,shuffle=True)

###################### Building Models ########################################


### Logistic Regression without class balance

lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
lr_score=lr_model.score(X_train,y_train)

cv_predict=cross_val_predict(lr_model,X_test,y_test)

cm=confusion_matrix(y_test,cv_predict)

## plotting confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
### Finding AUC value 
roc_score=roc_auc_score(y_test,cv_predict)
print('ROC score :',roc_score)

false_positive_rate,recall,thresholds=roc_curve(y_test,cv_predict)
roc_auc=auc(false_positive_rate,recall)
## 1. Accuracy : 0.9162
## 2. FNR : 0.73
## 3. AUC: 0.612

################# Logistic Regression with Class balance using SMOTE ##############################

sm = SMOTE(random_state=42, ratio=1.0)
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_test,y_test)
smote_lr=LogisticRegression()
smote_lr.fit(X_smote,y_smote)
cv_pred=cross_val_predict(smote_lr,X_smote_v,y_smote_v)
cm=confusion_matrix(y_smote_v,cv_pred)

## plotting confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);

smote_lr_score=smote_lr.score(X_smote,y_smote)
print('Accuracy of the smote_model :',smote_lr_score)

roc_score=roc_auc_score(y_smote_v,cv_pred)
print('ROC score :',roc_score)

false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)
roc_auc=auc(false_positive_rate,recall)

## 1. Accuracy :0.79762
## 2. FNR: 0.1885
## 3. AUC: 0.80055



###################### Decision Tree Classifier ##############################

tree_model = DecisionTreeClassifier(max_depth=10,min_samples_split = 500)

tree_model.fit(X_smote,y_smote)

cv_predict=cross_val_predict(tree_model,X_smote_v,y_smote_v)

cm=confusion_matrix(y_smote_v,cv_predict)

## plotting confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);

tree_model_score=tree_model.score(X_smote,y_smote)
print('Accuracy of the tree_model :',tree_model_score)

false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_predict)
roc_auc=auc(false_positive_rate,recall)



## 1. Accuracy :0.6838
## 2. FNR: 0.2753
## 3. AUC: 0.6753

##################### RandomForest Classifier ###############################
forest_model = RandomForestClassifier(max_depth = 90, n_estimators = 500, random_state=0)
forest_model.fit(X_train,y_train)

cv_predict=cross_val_predict(forest_model,X_test,y_test)
cm=confusion_matrix(y_test,cv_predict)

## plotting confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);

forest_model_score=forest_model.score(X_test,y_test)
print('Accuracy of the forest_model :',forest_model_score)

false_positive_rate,recall,thresholds=roc_curve(y_test,cv_predict)
roc_auc=auc(false_positive_rate,recall)


## 1. Accuracy : 0.90307
## 2. FNR : 0.5211
## 3. AUC : 0.7021


######################## XGBoost Classifier ##################################


train_data = xgb.DMatrix(X_smote, y_smote)
test_data  = xgb.DMatrix(X_smote, y_smote)

watchlist = [(train_data, 'train'), (test_data, 'valid')]


params = {"eta":0.3, "gamma":10, "max_depth":5, 
          "min_child_weight":2,"booster":"gbtree", "subsample":0.5, 'objective': 'binary:logistic'}

xgb_model = xgb.train(param_distributions = params, max_depth=6, n_estimators=100, learning_rate=0.3, gamma=3, objective='binary:logistic',watchlist) 
xgb_model = XGBClassifier(params = params, max_depth=6, n_estimators=100, learning_rate=0.3, gamma=3, objective='binary:logistic')
xgb_model.fit(X_smote,y_smote)

print(xgb_model.feature_importances_)
cv_predict=cross_val_predict(xgb_model,X_smote_v,y_smote_v)
cm=confusion_matrix(y_smote_v,cv_predict)

## plotting confusion matrix
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);

xgb_model_score=xgb_model.score(X_smote_v,y_smote_v)
print('Accuracy of the xgb_model :',xgb_model_score)

false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_predict)
roc_auc=auc(false_positive_rate,recall)

## 1. Accuracy :0.931
## 2. FNR : 0.0681
## 3. AUC :0.9304
############## plotting Important Features ###############################

plot_importance(xgb_model,color='blue',max_num_features=10)
pyplot.show()

##################### Finalizing the model ####################################

---------+-------------------------------------------------+------------+-----------+
|  S.No  |              Model                              |    AUC     |     FNR   |
+--------+-------------------------------------------------+------------+-----------+
|    1   |   Logistic Regression(without class balance)    |   0.612    |   73%     |          
|--------+-------------------------------------------------+------------+-----------+
|    2   |   Logistic Regression(with class balance)       |   0.800    |   18.85%  |      
|--------+-------------------------------------------------+------------+-----------+
|    3   |    Decision Tree Classification Model           |   0.675    |   27.53%  |
|--------+-------------------------------------------------+------------+-----------+
|    4   |    Random Forest Classification Model           |   0.702    |   52.11%  |
|--------+-------------------------------------------------+------------+-----------+
|    5   |    XGBoost Model (Gradient Boosting Method)     |   0.930    |    6.81%  |
 --------+-------------------------------------------------+------------+-----------+

####################### Loading the Test data ################################

santander_test = pd.read_csv("C:\\Users\\kyvenkat\\Desktop\\W0376 Backup\\Documents Backup\\test.csv")
santander_test.head(5)
santander_test.dtypes
santander_test = santander_test.drop(['ID_code'], axis =1)
####################### Predict Customer Transaction on Test Data #############

santander_test_predict = pd.DataFrame(xgb_model.predict(santander_test.values))

santander_test_predict.to_csv("santander_customer_predictions.csv")


plot_importance(xgb_model,color='red',max_num_features=10)
pyplot.show()
