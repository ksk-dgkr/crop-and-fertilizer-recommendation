#!/usr/bin/env python
# coding: utf-8

# In[412]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[413]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='ldhfIqhcgh3tRCy5PolYy_gR7loWakDDvxJBmrS58XqC',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'croprecommendation-donotdelete-pr-wbu2e1ya1mvtrm'
object_key = 'Crop_recommendation (1).csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# In[414]:


df.head()


# In[415]:


# DATA DICTIONARY & EDA( Exploratory Data Analysis )


# In[416]:


df.info()


# In[417]:


df.head()


# In[418]:


df.shape


# In[419]:


round(df.describe(),2).T


# In[420]:


df['label'].value_counts()


# In[421]:


# bivariate analysis


# In[422]:


df.head()


# In[423]:


df.corr()


# In[424]:


# OBSERVATION


# In[425]:


df['label'].unique()


# In[426]:


df.head()


# In[427]:


df.info()


# In[428]:


dum=pd.get_dummies(df['label'])


# In[429]:


x=pd.concat([df.drop('label',axis=1),dum],axis=1)


# In[430]:


y=df['label']


# In[431]:


df['label'].unique()


# In[432]:


df.groupby(df['label']).agg(np.mean)


# In[433]:


plt.scatter(df['label'],df['temperature'])


# In[434]:


sns.distplot(df['temperature'])


# In[435]:


# OUTLIERS 


# In[436]:


sns.heatmap(df.corr(),annot=True)


# In[437]:


sns.boxplot(y=df['humidity'])
# 1 outlier --> humidity --> chickpea 


# In[438]:


sns.boxplot(y=df['ph'])


# In[439]:


sns.boxplot(y=df['temperature'])


# In[440]:


sns.boxplot(y=df['rainfall'])


# In[441]:


sns.boxplot(y=df['N'])


# In[442]:


sns.boxplot(y=df['P'])


# In[443]:


sns.boxplot(y=df['K'])


# **Humidity  outliers replacing**

# In[466]:


sns.boxplot(df.humidity)


# In[467]:


q1=df.humidity.quantile(0.25)  #(Q1)
q3=df.humidity.quantile(0.75)  #(Q3)


# In[468]:


IQR=q3-q1


# In[469]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[470]:


lower_limit


# In[471]:


upper_limit


# In[472]:


df['humidity']= np.where(df['humidity']<lower_limit,lower_limit,df['humidity'])


# In[473]:


sns.boxplot(df.humidity)


# **ph outliers replace**

# In[474]:


sns.boxplot(y=df['ph'])


# In[483]:


q1=df.ph.quantile(0.25)  #(Q1)
q3=df.ph.quantile(0.75)  #(Q3)


# In[484]:


IQR=q3-q1


# In[485]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[486]:


lower_limit


# In[487]:


upper_limit


# In[488]:


df['ph']= np.where(df['ph']<lower_limit,lower_limit,df['ph'])


# In[489]:


sns.boxplot(df.ph)


# **temperature outliers replace**

# In[505]:


sns.boxplot(df.temperature)


# In[499]:


q1=df.temperature.quantile(0.25)  #(Q1)
q3=df.temperature.quantile(0.75)  #(Q3)


# In[500]:


IQR=q3-q1


# In[501]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[502]:


lower_limit


# In[503]:


upper_limit


# In[504]:


df['temperature']= np.where(df['temperature']<lower_limit,lower_limit,df['temperature'])


# **Rainfall replace outliers**

# In[513]:


sns.boxplot(df.rainfall)


# In[507]:


q1=df.rainfall.quantile(0.25)  #(Q1)
q3=df.rainfall.quantile(0.75)  #(Q3)


# In[508]:


IQR=q3-q1


# In[509]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[510]:


lower_limit


# In[511]:


upper_limit


# In[512]:


df['rainfall']= np.where(df['rainfall']>upper_limit,upper_limit,df['rainfall'])


# **P ouliers replace**

# In[521]:


sns.boxplot(df.P)


# In[515]:


q1=df.P.quantile(0.25)  #(Q1)
q3=df.P.quantile(0.75)  #(Q3)


# In[516]:


IQR=q3-q1


# In[517]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[518]:


lower_limit


# In[519]:


upper_limit


# In[520]:


df['P']= np.where(df['P']>upper_limit,upper_limit,df['P'])


# **K outliers replace**

# In[529]:


sns.boxplot(df.K)


# In[523]:


q1=df.K.quantile(0.25)  #(Q1)
q3=df.K.quantile(0.75)  #(Q3)


# In[524]:


IQR=q3-q1


# In[525]:


upper_limit= q3 + 1.5*IQR

lower_limit= q1 - 1.5*IQR


# In[526]:


lower_limit


# In[527]:


upper_limit


# In[528]:


df['K']= np.where(df['K']>upper_limit,upper_limit,df['K'])


# **CHECKING OUTLIERS**

# In[530]:


sns.boxplot(df.N)


# In[531]:


sns.boxplot(df.P)


# In[532]:


sns.boxplot(df.K)


# In[533]:


sns.boxplot(df.temperature)


# In[534]:


sns.boxplot(df.humidity)


# In[535]:


sns.boxplot(df.ph)


# In[536]:


sns.boxplot(df.rainfall)


# In[537]:


df.head()


# # X and y split

# In[538]:


X=df.drop(columns=['label'],axis = 1)
X.head()


# In[539]:


y = df.label
df['label'].unique()


# In[540]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[541]:


scaled_x = pd.DataFrame(scale.fit_transform(X),columns=X.columns)
scaled_x.head()


# # train test split

# In[542]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(scaled_x,y,test_size=0.2,random_state = 1)


# In[543]:


x_train.shape


# In[544]:


x_test.shape


# ## Model building
# 

# In[545]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[546]:


lr = LogisticRegression()
dtc= DecisionTreeClassifier()
rf = RandomForestClassifier()
knn=KNeighborsClassifier()
nb=GaussianNB()
svm = SVC()
xgb=XGBClassifier()


# In[547]:


lr.fit(x_train,y_train)


# In[548]:


dtc.fit(x_train,y_train)


# In[549]:


rf.fit(x_train,y_train)


# In[550]:


knn.fit(x_train,y_train)


# In[551]:


nb.fit(x_train,y_train)


# In[552]:


# svm.fit(x_train,y_train)


# In[553]:


# xgb.fit(x_train,y_train)


# In[554]:


y_test


# In[ ]:





# model_id

# In[555]:


lrpred_test= lr.predict(x_test)
lrpred_test


# In[ ]:





# In[556]:


lrpred_test= lr.predict(x_test)
dtcpred_test= dtc.predict(x_test)
rfpred_test= rf.predict(x_test)
knnpred_test= knn.predict(x_test)
nbpred_test= nb.predict(x_test)
# svmpred_test= svm.predict(x_test)
# xgbpred_test= xgb.predict(x_test)


# In[557]:


lrpred_train = lr.predict(x_train)
dtcpred_train = dtc.predict(x_train)
rfpred_train = rf.predict(x_train)
knnpred_train = knn.predict(x_train)
nbpred_train = nb.predict(x_train)
# svmpred_train = svm.predict(x_train)
# xgbpred_train = xgb.predict(x_train)


# In[ ]:





# #Accuracy Score

# In[558]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve


# #Logistic regression is noly for 2 categories
# not only consider accuracy but also all performance metrics
# -> optimal one should be choosen from over and underfitting
# overfitting - test>train
# underfitting - test<<<<<train ex: 72 & 97
# optimal - test<train ex: 97 & 99

# In[559]:


accuracy_score(y_test,lrpred_test) # test accuracy


# In[560]:


accuracy_score(y_train,lrpred_train) # train accuracy


# In[561]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score


# In[562]:


pd.crosstab(y_test,lrpred_test)


# In[563]:


print(classification_report(y_test,lrpred_test))


# In[564]:


probability = lr.predict_proba(x_test)[:,1]


# In[565]:


fpr,tpr,thresholds = roc_curve(y_test,probability)


# #Decision tree

# In[566]:


accuracy_score(y_test,dtcpred_test) 


# In[567]:


accuracy_score(y_train,dtcpred_train) 


# In[568]:


probability = dtc.predict_proba(x_test)[:,1]
probability


# In[569]:


roc_auc_score(y_test,probability)


# In[570]:


score = cross_val_score(dtc, X, y,cv=5)
score


# In[571]:


print(classification_report(y_test,dtcpred_test))


# #Random forest

# In[572]:


accuracy_score(y_test,rfpred_test) 


# In[573]:


accuracy_score(y_train,rfpred_train) 


# In[574]:


probability = rf.predict_proba(x_test)


# In[575]:


score = cross_val_score(rf, X, y,cv=5)
score


# #KNN

# In[576]:


accuracy_score(y_test,knnpred_test) 


# In[577]:


accuracy_score(y_train,knnpred_train) 


# In[578]:


probability = knn.predict_proba(x_test)[:,1]
probability


# In[579]:


score = cross_val_score(knn, X, y,cv=5)
score


# In[580]:


print(classification_report(y_test,knnpred_test))


# #Naive Bayes

# In[581]:


accuracy_score(y_test,nbpred_test) 


# In[582]:


accuracy_score(y_train,nbpred_train) 


# In[583]:


probability = nb.predict_proba(x_test)[:,1]
probability


# In[ ]:





# In[584]:


score = cross_val_score(nb, X, y,cv=5)
score


# #SVM

# In[585]:


# accuracy_score(y_test,svmpred_test) 


# In[586]:


# accuracy_score(y_train,svmpred_train) 


# In[587]:


# score = cross_val_score(svm, X, y,cv=5)
score


# #XGB

# In[588]:


# accuracy_score(y_test,xgbpred_test) 


# In[589]:


# accuracy_score(y_train,xgbpred_train) 


# In[590]:


# score = cross_val_score(xgb, X, y,cv=5)
# score


# cross validation score is nothing but if we take 5 as cv=5 then 4 different samples are trained and 1 sample is tested

# #Hyper parameter tunning

# In[591]:


from sklearn.model_selection import RandomizedSearchCV


# In[592]:


param_dist = {"max_depth": range(1,20) ,
              "min_samples_leaf": range(1,10),
              "criterion": ["gini"]}


# In[593]:


tree_cv = RandomizedSearchCV(dtc, param_dist, cv=5)


# In[594]:


tree_cv.fit(x_train,y_train)


# In[595]:


print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[596]:


tree_cv.best_estimator_.fit(x_train,y_train)


# In[597]:


mod_rs=tree_cv.best_estimator_


# In[598]:


mod_rs.fit(x_train,y_train)


# In[599]:


pred_rs=mod_rs.predict(x_test)


# In[600]:


print(classification_report(y_test,pred_rs))


# In[614]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[615]:


from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url" :"https://us-south.ml.cloud.ibm.com",
    "apikey":"WezMsA_u5_AkYOA-tEcRDl16kJHorqyB3n8KB7sytask"
}
client = APIClient(wml_credentials)


# In[616]:


client.spaces.list()


# In[617]:


space_uid="79da55aa-9549-4413-b7ca-ba7e681316e6"


# In[618]:


client.set.default_space(space_uid)


# In[619]:


client.software_specifications.list()


# In[620]:


software_spec_uid=client.software_specifications.get_uid_by_name("runtime-22.1-py3.9")
software_spec_uid


# In[621]:


model_details = client.repository.store_model(model=mod_rs,meta_props={
    client.repository.ModelMetaNames.NAME:"Crop_recommending",
    client.repository.ModelMetaNames.TYPE:"scikit-learn_1.0",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid}
                                             )
model_id=client.repository.get_model_uid(model_details)


# In[622]:


model_id


# In[623]:


type(mod_rs)


# In[640]:


client.repository.get_model_id(model_details)


# In[638]:


deployment_props={
    client.deployments.ConfigurationMetaNames.NAME:'deployment',
    client.deployments.ConfigurationMetaNames.ONLINE:{}
}


# In[641]:


deployment=client.deployments.create('b99eedae-48a6-436e-a97c-cdf8870c1890',meta_props=deployment_props)


# In[631]:


import pickle
with open('mod.pkl','wb') as files:
    pickle.dump(mod_rs,files)


# In[632]:


with open('mod.pkl','rb') as f:
    lr = pickle.load(f)


# In[ ]:




