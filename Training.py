# -*- coding: utf-8 -*-
"""
======================================================
Out-of-core classification of  Avazu data
======================================================
wc count for train.csv 40428968
wc count for test.csv   4577465
 Features engineerig
 Features Hasher
 SGD classifier with partial fit
 invscaling with eta0 =4-8
l2 penalty
labels changed in -1,1 as vowpal wabbit

warm on a shuffled file
"""

# Authors: Elena Cuoco <elena.cuoco@gmail.com>
#
#Avazu competitor usign pandas and scikit library
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn import  cross_validation
from sklearn.feature_extraction import FeatureHasher
from sklearn import preprocessing 
from sklearn.pipeline import Pipeline

#joblib library for serialization
from sklearn.externals import joblib
#json library for settings file
import json

##Read configuration parameters
file_dir = '/home/cuoco/workspace/git/avazu/src/Avazu-settings.json'
config = json.loads(open(file_dir).read())
train_file=config["HOME"]+config["TRAIN_DATA_PATH"]+'train.csv'
MODEL_PATH=config["HOME"]+config["MODEL_PATH"]
warm_file=config["HOME"]+config["TRAIN_DATA_PATH"]+'start.csv'
seed= int(config["SEED"])


###############################################################################
# Main
###############################################################################
chunk_size=int(config["CHUNK_SIZE"])
header=['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id'\
        ,'device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
reader = pd.read_table(train_file, sep=',', chunksize=chunk_size, names=header,header=0)
#classes for partial fit
all_classes = np.array([-1, 1])

#classifier
cls= SGDClassifier(loss='log', n_iter=200, alpha=.0000001, penalty='l2',\
learning_rate='invscaling',power_t=0.5,eta0=4.0,shuffle=True,n_jobs=-1,random_state=seed)

 
#preprocessing
preproc =Pipeline([('fh',FeatureHasher( n_features=2**27,input_type='string', non_negative=False))])
# 
def clean_data(data):
    y_train=data['click'].values +data['click'].values-1##for Vowpal Wabbit
    data['app']=data['app_id'].values+data['app_domain'].values+data['app_category'].values
    data['site']=data['site_id'].values+data['site_domain'].values+data['site_category'].values
    data['device']= data['device_id'].values+data['device_ip'].values+data['device_model'].values+(data['device_type'].values.astype(str))+(data['device_conn_type'].values.astype(str))
    data['type']=data['device_type'].values +data['device_conn_type'].values 
    data['iden']=data['app_id'].values +data['site_id'].values +data['device_id'].values
    data['domain']=data['app_domain'].values +data['site_domain'].values 
    data['category']=data['app_category'].values+data['site_category'].values
    data['pS1']=data['C1'].values.astype(str)+data['app_id']
    data['pS2']= data['C14'].values+data['C15'].values+data['C16'].values+data['C17'].values
    data['pS3']=data['C18'].values+data['C19'].values+data['C20'].values+data['C21'].values
    data['sum']=data['C1'].values+data['C14'].values+data['C15'].values+data['C16'].values+data['C17'].values\
    +data['C18'].values+data['C19'].values+data['C20'].values+data['C21'].values
    data['pos']= data['banner_pos'].values.astype(str)+data['app_category'].values+data['site_category'].values 
    data['pS4']=data['C1'].values-data['C20'].values
    data['ps5']=data['C14'].values-data['C21'].values 
    
    data['hour']=data['hour'].map(lambda x: datetime.strptime(x.astype(str),"%y%m%d%H"))
    data['dayoftheweek']=data['hour'].map(lambda x:  x.weekday())
    data['day']=data['hour'].map(lambda x:  x.day)
    data['hour']=data['hour'].map(lambda x:  x.hour)
    day=data['day'].values[len(data)-1]
    clean=data.drop(['id','click'], axis=1)#remove id and click columns
    X_dict=np.asarray(clean.astype(str))
    y_train = np.asarray(y_train).ravel()
   ######## preprocessing
    
    X_train=preproc.fit_transform(X_dict)
    
    
    return day,y_train,X_train
###################################################################################
# start warming
start = datetime.now()
warmer = pd.read_table(warm_file, sep=',', chunksize=chunk_size, names=header,header=None)
for data in  warmer:
 day,y_train,X=clean_data(data )
 cls.partial_fit(X, y_train,classes = all_classes)
 y_pred=cls.predict_proba(X)
#
####estimate log_loss
 LogLoss=log_loss(y_train, y_pred)
 print('elapsed time: %s, log_loss:%f' % (str(datetime.now() - start), LogLoss))


###################################################################################
# start training

i=0
temp=21
for data in reader:
    i+=1
    day,y_train,X_train=clean_data(data )
    cls.partial_fit(X_train, y_train,classes = all_classes)
    y_pred=cls.predict_proba(X_train)
    #
    ###estimate log_loss
    LogLoss=log_loss(y_train, y_pred)

    if i%10==0:
     print('iter:%s' %i)
     print(LogLoss)
    if day!=temp:
      print('day:%s' %day)
      LogLoss1= cross_validation.cross_val_score(cls, X_train, y_train, scoring='log_loss')
      print LogLoss1
      print('elapsed time: %s, log_loss:%f' % (str(datetime.now() - start), LogLoss))
    temp=day
print
print ('Ended training')
print('latest iter:%s' %i)
LogLoss1= cross_validation.cross_val_score(cls, X_train, y_train, scoring='log_loss')
print LogLoss1
print('elapsed time: %s, log_loss:%f' % (str(datetime.now() - start), LogLoss))

#serialize training
model_file=MODEL_PATH+'model-avazu-sgd.pkl'
joblib.dump(cls, model_file)
#serialize training
preproc_file=MODEL_PATH+'model-avazu-preproc.pkl'
joblib.dump(preproc, preproc_file)
