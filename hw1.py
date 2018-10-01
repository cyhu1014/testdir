###for ntu_ml 2018 hw1

import sys
import math
import pandas as pd
import numpy as np
import data_process
import func
train  = pd.read_csv("train.csv",encoding="big5" ,header=None)
test   = pd.read_csv("test.csv",encoding="big5" ,header=None)

####train_using only pm25
(train_unmodify,train_modify)=data_process.data_processing_pm25(train)

'''
print(len(train_unmodify[0]),len(train_modify[0]))

(tf,tl)= data_process.train_dataset_xd(train_modify,5)
data_process.create_model(tf,tl)

y=func.best_function_xd(tf,tl)
np.save('hw1_model_pm25', y)
print(len(y))
print(y)
data_process.create_test_submission_xd(test,y)

'''


###train using all feature
train_all_feat=data_process.data_processing_allfeat(train)
print(len(train_all_feat[0]))

(tf,tl)=data_process.train_all_feat_9hr(train_all_feat)
print(len(tf[100]))
print(len(tl))
y=func.best_function_xd(tf,tl)
print(len(y))
