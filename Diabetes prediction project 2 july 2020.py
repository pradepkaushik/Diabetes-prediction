#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[111]:


df = pd.read_csv('diabetes_data_upload.csv')


# In[112]:


df.head()


# In[113]:


df.info()


# In[114]:


df['class'].value_counts()


# In[115]:


pd.crosstab(df['Age'],df['class']).head()


# In[116]:


df.columns


# In[117]:


df['Gender'].value_counts()


# In[118]:


df['Obesity'].value_counts()


# In[119]:


df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df['Polyuria'] = df['Polyuria'].map({'No':0,'Yes':1})
df['Polydipsia'] = df['Polydipsia'].map({'No':0,'Yes':1})
df['sudden weight loss'] = df['sudden weight loss'].map({'No':0,'Yes':1})
df['weakness'] = df['weakness'].map({'No':0,'Yes':1})
df['Polyphagia'] = df['Polyphagia'].map({'No':0,'Yes':1})
df['Genital thrush'] = df['Genital thrush'].map({'No':0,'Yes':1})
df['visual blurring'] = df['visual blurring'].map({'No':0,'Yes':1})
df['Itching'] = df['Itching'].map({'No':0,'Yes':1})
df['Irritability'] = df['Irritability'].map({'No':0,'Yes':1})
df['delayed healing'] = df['delayed healing'].map({'No':0,'Yes':1})
df['partial paresis'] = df['partial paresis'].map({'No':0,'Yes':1})
df['muscle stiffness'] = df['muscle stiffness'].map({'No':0,'Yes':1})
df['Alopecia'] = df['Alopecia'].map({'No':0,'Yes':1})
df['Obesity'] = df['Obesity'].map({'No':0,'Yes':1})


# In[120]:


df.head()


# In[121]:


df.isnull().sum()


# In[122]:


df['class'] = df['class'].map({'Negative':0,'Positive':1})


# In[123]:


df.describe()


# In[124]:


sns.distplot(df['Age'])


# In[125]:


sns.boxplot(df['Age'])


# In[126]:


iqr = df['Age'].quantile(.75) - df['Age'].quantile(.25)


# In[127]:


df['Age'].quantile(.75) + (1.5*iqr)


# In[128]:


df['Age'] = np.where(df['Age']>84.0,84.0,df['Age'])


# In[129]:


sns.boxplot(df['Age'])


# In[130]:


sns.violinplot(x=df['Obesity'],y=df['class'],data=df,hue=df['Gender'])


# In[131]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(),annot=True)


# In[132]:


bar= df.corr()
bar1 = abs(bar['class'])
imp_features = bar1[bar1>0.1]
imp_features


# In[133]:


df.columns


# In[134]:


df.drop(['Genital thrush','Itching','delayed healing','muscle stiffness','Obesity'],axis=1,inplace=True)


# In[135]:


from sklearn.model_selection import train_test_split


# In[154]:


X = df.drop('class',axis=1)
y=df['class']


# In[166]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[167]:


from sklearn.preprocessing import StandardScaler


# In[168]:


scaling = StandardScaler()


# In[169]:


X_train = scaling.fit_transform(X_train)
X_test = scaling.transform(X_test)


# In[170]:


from sklearn.pipeline import Pipeline


# In[171]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
import xgboost


# In[172]:


pipeline_linear_reg=Pipeline([('Logistic_regression',LogisticRegression())])
pipeline_rfr=Pipeline([('random_forest_class',RandomForestClassifier())])
pipeline_svm=Pipeline([('scm',SVC())])
pipeline_dtr=Pipeline([('dsc_classifier',DecisionTreeClassifier())])
pipeline_knn=Pipeline([('knn',KNeighborsClassifier())])
pipeline_dtr=Pipeline([('ridge_classifier',RidgeClassifier())])
pipeline_lasso=Pipeline([('lasso_classifier',LassoCV())])
pipeline_XGB=Pipeline([('xgb',xgboost.XGBClassifier())])


# In[173]:


pipe_dictonary={0:'LogisticRegression',
                1 :'RandomForestclassifier',
                2 :'Svm' , 
                3:'Decision_tree', 
                4 : 'KNN',
               5:'Ridge',
               6:'Lasso',
               7:'XGB'}


# In[174]:


pipelines=[pipeline_linear_reg,pipeline_rfr,pipeline_svm,pipeline_dtr,
           pipeline_knn,pipeline_dtr,pipeline_lasso,pipeline_XGB]
for pipes in pipelines:
    pipes.fit(X_train,y_train)


# In[175]:


# print the Accuracy on test data of all the algorithm
for i,model in enumerate(pipelines):
    print('R-Sq Value {}  test data : {}'.format(pipe_dictonary[i],model.score(X_test,y_test)))


# In[176]:


# print the Accuracy on train data of all the algorithm to check the model is overfitted or not.
for i,model in enumerate(pipelines):
    print('R-Sq Value {} of train dataset : {}'.format(pipe_dictonary[i],model.score(X_train,y_train)))


# In[183]:


import xgboost as xgb


# In[184]:


xgb = xgb.XGBClassifier()


# In[185]:


xgb.fit(X_train,y_train)


# In[186]:


y_pred = xgb.predict(X_test)


# In[187]:


import pickle
filename='diabetes_prediction_2aug.pkl'
pickle.dump(xgb,open(filename,'wb'))


# In[ ]:




