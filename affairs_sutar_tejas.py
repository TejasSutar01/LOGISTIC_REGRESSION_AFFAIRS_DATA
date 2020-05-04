# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:34:36 2020

@author: tejas
"""

pwd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=affairscsv
df.isnull().sum()
df1=pd.get_dummies(df[["gender","children"]])
affair=pd.concat([df,df1],axis=1)
df_affairs=affair.drop(["gender","children","Unnamed: 0"],axis=1)
df_affairs["AF"]=1
df_affairs.loc[df_affairs.affairs==0,"AF"]=0
df_affairs=df_affairs.drop(["affairs"],axis=1)
df_affairs=df_affairs.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]

df.isnull().sum()

############Checking with Boxplot###################
plt.boxplot(df_affairs["age"])     #outlier is present
plt.boxplot(df_affairs["yearsmarried"])    # No outlier is present
plt.boxplot(df_affairs["religiousness"])    # No outlier is present
plt.boxplot(df_affairs["education"])   # No outlier is present
plt.boxplot(df_affairs["occupation"])   # No outlier is present
plt.boxplot(df_affairs["rating"])   # No outlier is present

######Need log transform age for outlier###########
x=np.log(df_affairs["age"])
plt.boxplot(x)
 #####Split the data into train and test#############
 from sklearn.model_selection import train_test_split 
train_data,test_data=train_test_split(df_affairs,test_size=0.3)
train_data=train_data.reset_index()
test_data=test_data.reset_index()
train_data=train_data.drop(["index"],axis=1)
test_data=test_data.drop(["index"],axis=1)
########Building the model############
import statsmodels.formula.api as sm
train_data.isnull().sum()
m1=sm.logit("AF~np.log(age)+yearsmarried+religiousness+education+occupation+rating+gender_female+gender_male+children_no+children_yes", data=train_data).fit()
m1.summary()
m1.summary2()
#AIC=486

train_pred=m1.predict(train_data)


from scipy import stats
import scipy.stats as st
st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

train_data["train_pred"]=np.zeros(420)

train_data.loc[train_pred>=0.5,"train_pred"]=1

from sklearn.metrics import classification_report

train_classification=classification_report(train_data["AF"],train_data["train_pred"])

##########confusion matrix###########
confusion_matrix=pd.crosstab(train_data["AF"],train_data.train_pred)

train_accuracy=(305+9)/420  ###74%

#######ROC Curve#########
from sklearn  import metrics 
fpr,tpr,threshold=metrics.roc_curve(train_data["AF"],train_pred)
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_auc=metrics.auc(fpr,tpr) #70%


##################Test Model#########
test_pred=m1.predict(test_data)

#SToring the values########
test_data["test_pred"]=np.zeros(181)
test_data.loc[test_pred>=0.50,"test_pred"]=1

classification_test=classification_report(test_data["AF"],test_data["test_pred"])
confusion_matrix_test=pd.crosstab(test_data["AF"],test_data.test_pred)
accuracy_test=(130+11)/181    ##77%
 
###########ROC Curve################
fpr,tpr,threshold=metrics.roc_curve(test_data["AF"],test_pred)
plt.plot(fpr,tpr);plt.xlabel("FALSE POSITIVE RATE");plt.ylabel("TRUE POSITIVE RATE")
roc_au=metrics.auc(fpr,tpr)         
