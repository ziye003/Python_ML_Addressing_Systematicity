# Databricks notebook source
pip install xgboost

# COMMAND ----------

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import random
from random import sample

# Data processing
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# COMMAND ----------

# Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,average_precision_score,classification_report,roc_auc_score

# COMMAND ----------

# MAGIC %md #load data

# COMMAND ----------

readme=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/readme.csv')
readme

# COMMAND ----------

data=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/data.csv')
print(data.shape)
data.head()

# COMMAND ----------

# MAGIC %md # data cleaning

# COMMAND ----------

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

# COMMAND ----------

clean_df=data.copy()
labels = clean_df['application_status']
labels[labels != 'hired'] = 0
labels[labels == 'hired'] = 1
labels = labels.astype(int)

features_to_drop = ['application_status', 'candidate_id', 'occupation_id', 'company_id', 'application_attribute_1']
clean_df = clean_df.drop(features_to_drop, axis=1)

feature_names = np.array(clean_df.keys().to_list())

# COMMAND ----------

encode_df = clean_df.copy()
features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
for feature in features_to_encode:
    encode_df = encode_and_bind(encode_df, feature)

data = encode_df.values
data = np.nan_to_num(data.astype(float))

# COMMAND ----------

# MAGIC %md # split data

# COMMAND ----------

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.005, random_state=0)
for train_index, test_index in sss.split(data, labels):
    X_train, Y_train = data[train_index], labels[train_index]
    X_test, Y_test = data[test_index], labels[test_index]

# COMMAND ----------

# MAGIC %md # model: unit test

# COMMAND ----------

# model = LogisticRegression(random_state=0).fit(train_data, train_labels)
# model = HistGradientBoostingClassifier().fit(train_data, train_labels)
model = XGBClassifier().fit(X_train, Y_train)

# COMMAND ----------

f1 = f1_score(Y_test,y_pred, pos_label = 1)
acc = accuracy_score(Y_test,y_pred)
print(f'Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

# COMMAND ----------

# MAGIC %md #Introduce randomness by withdrawing features

# COMMAND ----------

# MAGIC %md ## predict

# COMMAND ----------

def predict(clean_df,feature_names):
    
  
    # prints a random value from the list
    fnumber = list(np.arange(len(feature_names)))
    feature_to_mask = feature_names[random.choice(fnumber)]
    
    encode_df = clean_df.copy()
    encode_df = encode_df.drop(feature_to_mask, axis=1)

    features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
    for feature in features_to_encode:
        if feature == feature_to_mask:
            continue
        encode_df = encode_and_bind(encode_df, feature)
    
    data = encode_df.values
    data = np.nan_to_num(data.astype(float))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.005, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]


    model = XGBClassifier().fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
    
    predict_outcome.loc[:,feature_to_mask]=y_pred
    evaluation_outcome.loc['Accuracy',feature_to_mask]=acc
    evaluation_outcome.loc['F1',feature_to_mask]=f1
    evaluation_outcome.loc['AUC_ROC',feature_to_mask]=auc_roc

    print(f'Masking feature {i_feature}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

predict(clean_df,feature_names)

# COMMAND ----------

# MAGIC %md ## Systematicity

# COMMAND ----------

predict_outcome=pd.DataFrame(columns=feature_names,index=np.array(range(len(Y_test))))
evaluation_outcome=pd.DataFrame(columns=feature_names,index=['Accuracy','F1','AUC_ROC'])

# COMMAND ----------

for i_feature, feature_to_mask in enumerate(feature_names):
    encode_df = clean_df.copy()
    encode_df = encode_df.drop(feature_to_mask, axis=1)

    features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
    for feature in features_to_encode:
        if feature == feature_to_mask:
            continue
        encode_df = encode_and_bind(encode_df, feature)

    data = encode_df.values
    data = np.nan_to_num(data.astype(float))
    for train_index, test_index in sss.split(data, labels):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]


    model = XGBClassifier().fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
    
    predict_outcome.loc[:,feature_to_mask]=y_pred
    evaluation_outcome.loc['Accuracy',feature_to_mask]=acc
    evaluation_outcome.loc['F1',feature_to_mask]=f1
    evaluation_outcome.loc['AUC_ROC',feature_to_mask]=auc_roc

#     print(f'Masking feature {i_feature}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}')

# COMMAND ----------

predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_predict_outcome.csv')
evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_evaluation_outcome.csv')

# COMMAND ----------

# MAGIC %md ## Evaluation

# COMMAND ----------

evaluation_outcome
print(min(evaluation_outcome.loc['Accuracy',:]))
print(np.mean(evaluation_outcome.loc['Accuracy',:]))
print(np.mean(evaluation_outcome.loc['F1',:]))
print(np.mean(evaluation_outcome.loc['AUC_ROC',:]))

# COMMAND ----------

all_wrong=list()
all_right=list()
for i_result, result in enumerate(Y_test):
    if abs(result-np.mean(predict_outcome.loc[i_result,:]))==1:
        all_wrong.append(i_result)
    elif abs(result-np.mean(predict_outcome.loc[i_result,:]))==0:
        all_right.append(i_result)

# COMMAND ----------

# number of prediction who were consistently wrong for all models
print((len(all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(all_right)/len(Y_test)))

# COMMAND ----------

Y_test.index=predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(predict_outcome.loc[i,:])

# COMMAND ----------

plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='pink')

# COMMAND ----------

# MAGIC %md 4.8% percent of hired population were wrongly predicted by all models. 50.8% of hired population were correctly predicted all the time.

# COMMAND ----------

# MAGIC %md #Introduce randomness by withdrawing 10% of traning data

# COMMAND ----------

# MAGIC %md ## predict

# COMMAND ----------

def predict(clean_df,feature_names):
    
    encode_df = clean_df.copy()
    encode_df = encode_df.drop(feature_to_mask, axis=1)

    encode_df = clean_df.copy()
    features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
    for feature in features_to_encode:
        encode_df = encode_and_bind(encode_df, feature)

    data = encode_df.values
    data = np.nan_to_num(data.astype(float))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.005, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]
    Y_train=pd.DataFrame(Y_train)
    X_train=pd.DataFrame(X_train)
    Y_train.index=X_train.index

    
    # sample 90% of training data 
    n_employee=int(X_train.shape[0]*0.9)
    employee_list=list(X_train.index)
    sampled_emp=random.sample(employee_list,n_employee)
    X_train, Y_train = X_train.loc[sampled_emp,:], Y_train.loc[sampled_emp,:]

    model = XGBClassifier().fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
    
    predict_outcome.loc[:,feature_to_mask]=y_pred
    evaluation_outcome.loc['Accuracy',feature_to_mask]=acc
    evaluation_outcome.loc['F1',feature_to_mask]=f1
    evaluation_outcome.loc['AUC_ROC',feature_to_mask]=auc_roc

    print(f'Masking feature {i_feature}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

predict(clean_df,feature_names)

# COMMAND ----------

# MAGIC %md ## Systematicity

# COMMAND ----------

predict_outcome=pd.DataFrame(columns=np.arange(10),index=np.array(range(len(Y_test))))
evaluation_outcome=pd.DataFrame(columns=np.arange(10),index=['Accuracy','F1','AUC_ROC'])

# COMMAND ----------

for i in range(10):
    encode_df = clean_df.copy()
    encode_df = encode_df.drop(feature_to_mask, axis=1)

    encode_df = clean_df.copy()
    features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
    for feature in features_to_encode:
        encode_df = encode_and_bind(encode_df, feature)

    data = encode_df.values
    data = np.nan_to_num(data.astype(float))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.005, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]
    Y_train=pd.DataFrame(Y_train)
    X_train=pd.DataFrame(X_train)
    Y_train.index=X_train.index

    
    # sample 90% of training data 
    n_employee=int(X_train.shape[0])
    sampled_emp=np.arange(int(n_employee*i/10),int(n_employee*(i+1)/10),1)

    X_train, Y_train = X_train.loc[sampled_emp,:], Y_train.loc[sampled_emp,:]

    model = XGBClassifier().fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
    
    predict_outcome.loc[:,i]=y_pred
    evaluation_outcome.loc['Accuracy',i]=acc
    evaluation_outcome.loc['F1',i]=f1
    evaluation_outcome.loc['AUC_ROC',i]=auc_roc

# COMMAND ----------

evaluation_outcome

# COMMAND ----------

predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_predict_outcome.csv')
evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_evaluation_outcome.csv')

# COMMAND ----------

# MAGIC %md ##Evaluation

# COMMAND ----------

evaluation_outcome
print(min(evaluation_outcome.loc['Accuracy',:]))
print(np.mean(evaluation_outcome.loc['Accuracy',:]))
print(np.mean(evaluation_outcome.loc['F1',:]))
print(np.mean(evaluation_outcome.loc['AUC_ROC',:]))

# COMMAND ----------

all_wrong=list()
all_right=list()
for i_result, result in enumerate(Y_test):
    if abs(result-np.mean(predict_outcome.loc[i_result,:]))==1:
        all_wrong.append(i_result)
    elif abs(result-np.mean(predict_outcome.loc[i_result,:]))==0:
        all_right.append(i_result)
        
# number of prediction who were consistently wrong for all models
print((len(all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(all_right)/len(Y_test)))

# COMMAND ----------

Y_test.index=predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(predict_outcome.loc[i,:])
plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='pink')

# COMMAND ----------

# MAGIC %md 5.6% percent of hired population were wrongly predicted by all models. 54.4% of hired population were correctly predicted all the time.

# COMMAND ----------


