"""
======================================================
Out-of-core classification of  Avazu data
======================================================
wc count for train.csv 40428968
wc count for test.csv   4577465
This file read the model from disk.
It reads the parameters from the file json Avazu-settings.json
It produces the prediction and the submission file on disk.
"""

# Authors: Elena Cuoco <elena.cuoco@gmail.com>
#

from __future__ import print_function
import time
import string
from operator import itemgetter
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import  DataFrame
from patsy import dmatrix
from sklearn.utils import resample

#joblib library for serialization
from sklearn.externals import joblib

#json library for settings file
import json
##Read configuration parameters
file_dir = './Avazu-settings.json'
config = json.loads(open(file_dir).read())

MODEL_PATH=config["HOME"]+config["MODEL_PATH"]
test=config["HOME"]+config["TEST_DATA_PATH"]+'test.csv'
SUBMISSION_PATH=config["HOME"]+config["SUBMISSION_PATH"]
seed= int(config["SEED"])
chunk_size=int(config["CHUNK_SIZE"])

###############################################################################
# Main
###############################################################################

header_test=['id','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id'\
        ,'device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']
reader_test = pd.read_table(test, sep=',', chunksize=chunk_size,header=0,names=header_test)
#serialize training
model_file=MODEL_PATH+'model-avazu-sgd.pkl'
cls = joblib.load(model_file)
preproc_file=MODEL_PATH+'model-avazu-preproc.pkl'
preproc = joblib.load(preproc_file)

submission=SUBMISSION_PATH+'prediction-avazu.csv'
# prepare data 
def clean_data(data):
    data['app']=data['app_id'].values+data['app_domain'].values+data['app_category'].values
    data['site']=data['site_id'].values+data['site_domain'].values+data['site_category'].values
    data['device']= data['device_id'].values+data['device_ip'].values+data['device_model'].values+(data['device_type'].values.astype(str))+(data['device_conn_type'].values.astype(str))
    data['type']=data['device_type'].values +data['device_conn_type'].values 
    data['iden']=data['app_id'].values +data['site_id'].values +data['device_id'].values
    data['domain']=data['app_domain'].values +data['site_domain'].values 
    data['category']=data['app_category'].values+data['site_category'].values
    data['sum']=data['C1'].values+data['C14'].values+data['C15'].values+data['C16'].values+data['C17'].values\
     +data['C18'].values+data['C19'].values+data['C20'].values+data['C21'].values
    
    data['pos']= data['banner_pos'].values.astype(str)+data['app_category'].values+data['site_category'].values 
   ##
     
   
    data['hour']=data['hour'].map(lambda x: datetime.strptime(x.astype(str),"%y%m%d%H"))
    data['dayoftheweek']=data['hour'].map(lambda x:  x.weekday())
    data['day']=data['hour'].map(lambda x:  x.day)
    data['hour']=data['hour'].map(lambda x:  x.hour)
    day=data['day'].values[len(data)-1]
    
    
    
    clean=data.drop(['id','day'], axis=1)#remove id and click columns
     
    X_dict=np.asarray(clean.astype(str))

    ######## preprocessing

    X=preproc.transform(X_dict)
   

    return X


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

with open(submission, 'a') as outfile:
    outfile.write('id,click\n')
    for data in reader_test:
     ID=data['id'].values
     x=clean_data(data)
     p =cls.predict_proba(x)[:,1]
     dfjo = DataFrame(dict(ID=ID,click=p), columns=['ID','click'])
     dfjo.to_csv(outfile,header=None,index_label=None,index_col=False,index=False)
