# Databricks notebook source
pip install xgboost

# COMMAND ----------

pip install catboost

# COMMAND ----------

pip install lightgbm

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,average_precision_score,classification_report,roc_auc_score

# COMMAND ----------

#plot
plt.rcParams['figure.dpi']=150
%config InlineBackend.figure_format = 'png2x'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x')

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

# MAGIC %md # data exploration

# COMMAND ----------

import matplotlib.pyplot as plt
feature=labels

l = feature.unique()
sizes = feature.value_counts()
explode = (0, 0.1)  # only "explode" the 2nd slice

def func(pct, allvalues):
    absolute = round(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} )".format(pct, absolute)

fig, ax = plt.subplots(figsize =(10, 10))
colors = ( "#E9967A", "#7AC5CD") 
wedges, texts, autotexts = ax.pie(sizes,
                                  autopct = lambda pct: func(pct, sizes),
                                  explode = explode,
                                  labels = l,
                                  shadow = False,
                                  colors = colors,
                                  startangle = 90,
#                                   wedgeprops = wp,
                                  textprops = dict(color ="black"))

plt.setp(autotexts, size = 15)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
feature=labels

l = ['numerical features','dummy features']
sizes = [len(feature_names)-3,data.shape[1]-(len(feature_names)-3)]
explode = (0, 0.1)  # only "explode" the 2nd slice

def func(pct, allvalues):
    absolute = round(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} )".format(pct, absolute)

fig, ax = plt.subplots(figsize =(10, 10))
colors = ( "#8EE5EE", "#CDAA7D") 
wedges, texts, autotexts = ax.pie(sizes,
                                  autopct = lambda pct: func(pct, sizes),
                                  explode = explode,
                                  labels = l,
                                  shadow = False,
                                  colors = colors,
                                  startangle = 90,
#                                   wedgeprops = wp,
                                  textprops = dict(color ="black"))

plt.setp(autotexts, size = 15)
plt.show()

# COMMAND ----------

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
scaler = StandardScaler()
df_subset=scaler.fit_transform(data)
tsne_results = tsne.fit_transform(df_subset)

# COMMAND ----------

df_subset=pd.DataFrame(df_subset)
tsne_results=pd.DataFrame(tsne_results)
df_subset['tsne-2d-one'] = tsne_results.iloc[:,0]
df_subset['tsne-2d-two'] = tsne_results.iloc[:,1]
df_subset['hired'] = labels
# plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue='hired',
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)


# COMMAND ----------

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# COMMAND ----------

sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="hired",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)

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

# model = LogisticRegression(random_state=0,solver='lbfgs', max_iter=1000).fit(X_train, Y_train) #Accuracy = 0.69%. F1-score = 0.0000. AUC_ROC = 0.4943
# model = HistGradientBoostingClassifier().fit(X_train, Y_train) #Accuracy = 0.87%. F1-score = 0.7647. AUC_ROC = 0.8191
model = XGBClassifier().fit(X_train, Y_train) #Accuracy = 0.86%. F1-score = 0.7518. AUC_ROC = 0.8142
# model = LGBMClassifier().fit(X_train, Y_train) # Accuracy = 0.87%. F1-score = 0.7626. AUC_ROC = 0.8199
# model = GradientBoostingClassifier().fit(X_train, Y_train) #Accuracy = 0.82%. F1-score = 0.6563. AUC_ROC = 0.7476
# model = CatBoostClassifier(silent=True).fit(X_train, Y_train) #Accuracy = 0.88%. F1-score = 0.7857. AUC_ROC = 0.8360


y_pred = model.predict(X_test)

# COMMAND ----------

y_proba = model.predict_proba(X_test)

# COMMAND ----------

f1 = f1_score(Y_test,y_pred, pos_label = 1)
acc = accuracy_score(Y_test,y_pred)
auc_roc = roc_auc_score(Y_test,y_pred)
print(f'Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

# COMMAND ----------

# MAGIC %md #Introduce arbitrariness by withdrawing 10% data

# COMMAND ----------

# MAGIC %md 
# MAGIC Because the population size is large enough, we can achieve some extent of arbitrariness by random sampling 90% people from the population. 

# COMMAND ----------

# MAGIC %md ## predict

# COMMAND ----------

def predict(clean_df):
    
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

    print(f'Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')
#     return(f1,acc,auc_roc,y_pred)

# COMMAND ----------

predict(clean_df)

# COMMAND ----------

# MAGIC %md ## Prediction outcome for 55 runs

# COMMAND ----------

predict_outcome=pd.DataFrame(columns=np.arange(55),index=np.array(range(len(Y_test))))
evaluation_outcome=pd.DataFrame(columns=np.arange(55),index=['Accuracy','F1','AUC_ROC'])

# COMMAND ----------

for i in range(55):
    f1,acc,auc_roc,y_pred=predict(clean_df)
    predict_outcome.loc[:,i]=y_pred
    evaluation_outcome.loc['Accuracy',i]=acc
    evaluation_outcome.loc['F1',i]=f1
    evaluation_outcome.loc['AUC_ROC',i]=auc_roc

# COMMAND ----------

predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_predict_outcome.csv',index_label=False)
evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_evaluation_outcome.csv',index_label=False)


# COMMAND ----------

# MAGIC %md ##Evaluation metric

# COMMAND ----------

employee_predict_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_predict_outcome.csv')
employee_evaluation_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/employee_evaluation_outcome.csv')

# COMMAND ----------

# systematic error across all models
e_all_wrong=list()
e_all_false_pos=list()
e_all_false_neg=list()
e_all_right=list()
for i_result, result in enumerate(Y_test):
    if (result-np.mean(employee_predict_outcome.loc[i_result,:]))==-1:
        e_all_false_pos.append(i_result)
    if (result-np.mean(employee_predict_outcome.loc[i_result,:]))==1:
        e_all_false_neg.append(i_result)
    if abs(result-np.mean(employee_predict_outcome.loc[i_result,:]))==1:
        e_all_wrong.append(i_result)
    elif abs(result-np.mean(employee_predict_outcome.loc[i_result,:]))==0:
        e_all_right.append(i_result)
        
# number of prediction who were consistently wrong for all models
print((len(e_all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(e_all_right)/len(Y_test)))

# COMMAND ----------

employee_evaluation_outcome['mean']=employee_evaluation_outcome.mean(axis=1)
employee_evaluation_outcome['std']=employee_evaluation_outcome.std(axis=1)
employee_evaluation_outcome
# evaluation_outcome

# COMMAND ----------

Y_test.index=predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(predict_outcome.loc[i,:])
plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='#9FB6CD')
plt.xlabel('Probability of correct prediction for hired employee')
plt.ylabel('Frequency')

# COMMAND ----------

# MAGIC %md 3.6% percent of hired population were wrongly predicted by all models. 69.4% of hired population were correctly predicted all the time.

# COMMAND ----------

# MAGIC %md #Introduce arbitrariness by withdrawing features

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

    print(f'Masking feature {feature_to_mask}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

predict(clean_df,feature_names)

# COMMAND ----------

# MAGIC %md ## Prediction outcome for eliminating each feature

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

predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_predict_outcome.csv',index_label=False)
evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_evaluation_outcome.csv',index_label=False)


# COMMAND ----------

# MAGIC %md ##Evaluation metric

# COMMAND ----------


feature_predict_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_predict_outcome.csv')
feature_evaluation_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/feature_evaluation_outcome.csv')

# COMMAND ----------

# systematic error across all models
f_all_wrong=list()
f_all_false_pos=list()
f_all_false_neg=list()
f_all_right=list()
for i_result, result in enumerate(Y_test):
    if (result-np.mean(feature_predict_outcome.loc[i_result,:]))==-1:
        f_all_false_pos.append(i_result)
    if (result-np.mean(feature_predict_outcome.loc[i_result,:]))==1:
        f_all_false_neg.append(i_result)
    if abs(result-np.mean(feature_predict_outcome.loc[i_result,:]))==1:
        f_all_wrong.append(i_result)
    elif abs(result-np.mean(feature_predict_outcome.loc[i_result,:]))==0:
        f_all_right.append(i_result)
        
# number of prediction who were consistently wrong for all models
print((len(f_all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(f_all_right)/len(Y_test)))

# COMMAND ----------

feature_evaluation_outcome['mean']=feature_evaluation_outcome.mean(axis=1)
feature_evaluation_outcome['std']=feature_evaluation_outcome.std(axis=1)
# feature_evaluation_outcome['min']=feature_evaluation_outcome.min(axis=1)
feature_evaluation_outcome

# COMMAND ----------

Y_test.index=predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(predict_outcome.loc[i,:])
plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='#D8BFD8')
plt.xlabel('Probability of correct prediction for hired employee')
plt.ylabel('Frequency')

# COMMAND ----------

# MAGIC %md 4.8% percent of hired population were wrongly predicted by all models. 50.8% of hired population were correctly predicted all the time.

# COMMAND ----------

# MAGIC %md #Introduce arbitrariness by different classifier

# COMMAND ----------

# MAGIC %md ## predict

# COMMAND ----------

def predict(clean_df):
    
    classifier_list=[LGBMClassifier(random_state=0),
                     CatBoostClassifier(silent=True),
                     HistGradientBoostingClassifier(random_state=0),
                     XGBClassifier(random_state=0)]
  
    # prints a random value from the list
    fnumber = list(np.arange(len(classifier_list)))
    classifier = classifier_list[random.choice(fnumber)]
    
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


    model = classifier.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)

    print(f'Masking feature {classifier}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')

# COMMAND ----------

predict(clean_df)

# COMMAND ----------

# MAGIC %md ## Prediction outcome for each classifier

# COMMAND ----------

classifier_list=[LGBMClassifier(random_state=0),
                     CatBoostClassifier(silent=True),
                     HistGradientBoostingClassifier(random_state=0),
                     XGBClassifier(random_state=0)]
predict_outcome=pd.DataFrame(columns=np.array(range(len(classifier_list))),index=np.array(range(len(Y_test))))
evaluation_outcome=pd.DataFrame(columns=np.array(range(len(classifier_list))),index=['Accuracy','F1','AUC_ROC'])

# COMMAND ----------

encode_df = clean_df.copy()
features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
for feature in features_to_encode:
    encode_df = encode_and_bind(encode_df, feature)
data = encode_df.values
data = np.nan_to_num(data.astype(float))
    
for train_index, test_index in sss.split(data, labels):
    X_train, Y_train = data[train_index], labels[train_index]
    X_test, Y_test = data[test_index], labels[test_index]

# COMMAND ----------

for i_classifier, classifier in enumerate(classifier_list):

    model = classifier.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
    
    predict_outcome.loc[:,i_classifier]=y_pred
    evaluation_outcome.loc['Accuracy',i_classifier]=acc
    evaluation_outcome.loc['F1',i_classifier]=f1
    evaluation_outcome.loc['AUC_ROC',i_classifier]=auc_roc

#     print(f'Masking feature {i_feature}. Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}')

# COMMAND ----------

evaluation_outcome

# COMMAND ----------

predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/classifier_predict_outcome.csv',index_label=False)
evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/classifier_evaluation_outcome.csv',index_label=False)


# COMMAND ----------

# MAGIC %md ##Evaluation metric

# COMMAND ----------


classifier_predict_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/classifier_predict_outcome.csv')
classifier_evaluation_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/classifier_evaluation_outcome.csv')

# COMMAND ----------

# systematic error across all models
c_all_wrong=list()
c_all_false_pos=list()
c_all_false_neg=list()
c_all_right=list()
for i_result, result in enumerate(Y_test):
    if (result-np.mean(classifier_predict_outcome.loc[i_result,:]))==-1:
        c_all_false_pos.append(i_result)
    if (result-np.mean(classifier_predict_outcome.loc[i_result,:]))==1:
        c_all_false_neg.append(i_result)
    if abs(result-np.mean(classifier_predict_outcome.loc[i_result,:]))==1:
        c_all_wrong.append(i_result)
    elif abs(result-np.mean(classifier_predict_outcome.loc[i_result,:]))==0:
        c_all_right.append(i_result)
        
# number of prediction who were consistently wrong for all models
print((len(c_all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(c_all_right)/len(Y_test)))

# COMMAND ----------


classifier_evaluation_outcome['mean']=classifier_evaluation_outcome.mean(axis=1)
classifier_evaluation_outcome['std']=classifier_evaluation_outcome.std(axis=1)
classifier_evaluation_outcome

# COMMAND ----------

Y_test.index=predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(predict_outcome.loc[i,:])
plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='#66CDAA')
plt.xlabel('Probability of correct prediction for hired employee')
plt.ylabel('Frequency')

# COMMAND ----------

# MAGIC %md 10% percent of hired population were wrongly predicted by all models. 82.8% of hired population were correctly predicted all the time.

# COMMAND ----------

# MAGIC %md #Introduce arbitrariness by withdrawing features

# COMMAND ----------

# MAGIC %md ## predict

# COMMAND ----------

def predict_prob(clean_df):    
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
    y_predict_probas = model.predict_proba(X_test)[:, 1] # only get the probablility of label 1
    return(y_predict_probas)

# COMMAND ----------

predict_prob_outcome=pd.DataFrame(columns=np.arange(55),index=np.array(range(len(Y_test))))
for i in range(55):
    y_predict_probas=predict_prob(clean_df)
    predict_prob_outcome.loc[:,i]=y_predict_probas
# predict_prob_outcome

# COMMAND ----------

# predict_prob_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/predict_prob_outcome.csv',index_label=False)

# COMMAND ----------

# get the var of 55 prediction prob, then get the average of var of all label 1 samples.
estimated_var = np.mean(np.var(predict_prob_outcome, axis=1))
estimate_std = estimated_var**0.5

# COMMAND ----------

def predict(clean_df,estimate_std):
    
    encode_df = clean_df.copy()

    features_to_encode=['gender','ethnicity','candidate_demographic_variable_5']
    for feature in features_to_encode:
        encode_df = encode_and_bind(encode_df, feature)

    data = encode_df.values
    data = np.nan_to_num(data.astype(float))
    for train_index, test_index in sss.split(data, labels):
        X_train, Y_train = data[train_index], labels[train_index]
        X_test, Y_test = data[test_index], labels[test_index]

    model = XGBClassifier().fit(X_train, Y_train)
    
    # add a random generated number centered at 1 with std from previous analysis
    y_pred = (model.predict_proba(X_test)[:, 1] + np.random.normal(0, estimate_std, size=(len(X_test),)) > 0.5).astype(int)

    f1 = f1_score(Y_test,y_pred, pos_label = 1)
    acc = accuracy_score(Y_test,y_pred)
    auc_roc = roc_auc_score(Y_test,y_pred)
#     print(f'Accuracy = {acc%100:.2f}%. F1-score = {f1:.4f}. AUC_ROC = {auc_roc:.4f}')
    return(f1,acc,auc_roc,y_pred)

# COMMAND ----------

predict(clean_df,estimate_std)

# COMMAND ----------

# MAGIC %md ## Prediction outcome for 55 runs

# COMMAND ----------

r_predict_outcome=pd.DataFrame(columns=np.arange(55),index=np.array(range(len(Y_test))))
r_evaluation_outcome=pd.DataFrame(columns=np.arange(55),index=['Accuracy','F1','AUC_ROC'])

# COMMAND ----------

for i in range(55):
    f1,acc,auc_roc,y_pred=predict(clean_df,estimate_std)
    r_predict_outcome.loc[:,i]=y_pred
    r_evaluation_outcome.loc['Accuracy',i]=acc
    r_evaluation_outcome.loc['F1',i]=f1
    r_evaluation_outcome.loc['AUC_ROC',i]=auc_roc

# COMMAND ----------

r_predict_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/randomness_predict_outcome.csv',index_label=False)
r_evaluation_outcome.to_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/randomness_evaluation_outcome.csv',index_label=False)


# COMMAND ----------

# MAGIC %md ## Evaluation metric

# COMMAND ----------

r_predict_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/randomness_predict_outcome.csv')
r_evaluation_outcome=pd.read_csv('/dbfs/mnt/client-002sap21p010-pasca/04_data_analysis/trial/randomness_evaluation_outcome.csv')

# COMMAND ----------

# systematic error across all models
r_all_wrong=list()
r_all_false_pos=list()
r_all_false_neg=list()
r_all_right=list()
for i_result, result in enumerate(Y_test):
    if (result-np.mean(r_predict_outcome.loc[i_result,:]))==-1:
        r_all_false_pos.append(i_result)
    if (result-np.mean(r_predict_outcome.loc[i_result,:]))==1:
        r_all_false_neg.append(i_result)
    if abs(result-np.mean(r_predict_outcome.loc[i_result,:]))==1:
        r_all_wrong.append(i_result)
    elif abs(result-np.mean(r_predict_outcome.loc[i_result,:]))==0:
        r_all_right.append(i_result)
        
# number of prediction who were consistently wrong for all models
print((len(r_all_wrong)/len(Y_test)))

# number of prediction who were consistently right for all models
print((len(r_all_right)/len(Y_test)))

# COMMAND ----------

r_evaluation_outcome['mean']=r_evaluation_outcome.mean(axis=1)
r_evaluation_outcome['std']=r_evaluation_outcome.std(axis=1)
r_evaluation_outcome

# COMMAND ----------

r_evaluation_outcome
print(min(r_evaluation_outcome.loc['Accuracy',:]))
print(np.mean(r_evaluation_outcome.loc['Accuracy',:]))
print(np.mean(r_evaluation_outcome.loc['F1',:]))
print(np.mean(r_evaluation_outcome.loc['AUC_ROC',:]))

# COMMAND ----------

Y_test.index=r_predict_outcome.index
hired_accuracy=pd.DataFrame(columns=Y_test[Y_test==1].index,index=['mean_prediction'])
for i in Y_test[Y_test==1].index.to_list():
    hired_accuracy.loc['mean_prediction',i]=np.mean(r_predict_outcome.loc[i,:])
plt.hist(hired_accuracy.loc['mean_prediction',:].values,color='#EEC900')
plt.xlabel('Probability of correct prediction for hired employee')
plt.ylabel('Frequency')

# COMMAND ----------

# MAGIC %md 7.2% percent of hired population were wrongly predicted by all models. 75.6% of hired population were correctly predicted all the time.

# COMMAND ----------

# MAGIC %md # conclusion

# COMMAND ----------

evaluation_df=pd.concat([employee_evaluation_outcome['mean'],feature_evaluation_outcome['mean'],classifier_evaluation_outcome['mean'],r_evaluation_outcome['mean']],axis=1)
evaluation_df.columns=['Population_sampling','Featrue_sampling','Algorithm_sampling','Randomne_to_prediction']
evaluation_df

# COMMAND ----------

systematic_error_all=[(len(e_all_wrong)/len(Y_test)),(len(f_all_wrong)/len(Y_test)),(len(c_all_wrong)/len(Y_test)),(len(r_all_wrong)/len(Y_test))]
systematic_error_false_pos=[(len(e_all_false_pos)/len(Y_test)),(len(f_all_false_pos)/len(Y_test)),(len(c_all_false_pos)/len(Y_test)),(len(r_all_false_pos)/len(Y_test))]
systematic_error_false_neg=[(len(e_all_false_neg)/len(Y_test)),(len(f_all_false_neg)/len(Y_test)),(len(c_all_false_neg)/len(Y_test)),(len(r_all_false_neg)/len(Y_test))]
systematic_correct_all=[(len(e_all_right)/len(Y_test)),(len(f_all_right)/len(Y_test)),(len(c_all_right)/len(Y_test)),(len(r_all_right)/len(Y_test))]

systematic_error=pd.DataFrame([systematic_error_all,systematic_error_false_pos,systematic_error_false_neg])
systematic_error.columns=['Population_sampling','Featrue_sampling','Algorithm_sampling','Randomne_to_prediction']
systematic_error.index=['systematic error','systematic false positive','systematic false negative']
systematic_error

systematic=pd.DataFrame([systematic_error_all,systematic_error_false_pos,systematic_error_false_neg,systematic_correct_all])
systematic.columns=['Population_sampling','Featrue_sampling','Algorithm_sampling','Randomne_to_prediction']
systematic.index=['systematic error','systematic false positive','systematic false negative','systematic correct']
systematic

# COMMAND ----------

systematic_error

# COMMAND ----------

# MAGIC %md ##conclusion
# MAGIC The four different method gives similar accuracy performance.
# MAGIC 
# MAGIC To minimize false negative, i.e all hired employee were predicted as hired in some model, population sampling has the best performance.
# MAGIC 
# MAGIC Population sampling also achieve lowest overall systematic error.

# COMMAND ----------


