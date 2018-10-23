###for ntu_ml 2018 hw1

import sys
import math
import pandas as pd
import numpy as np
import data_process
import func
train  = pd.read_csv("train.csv",encoding="big5" ,header=None)
test   = pd.read_csv("test.csv",encoding="big5" ,header=None)


(train_unmodify,train_modify)=data_process.data_processing(train)

print(len(train_unmodify[0]),len(train_modify[0]))

(tf,tl)= data_process.train_dataset_2d(train_modify)
data_process.create_model(tf,tl)

y=func.best_function_2d(tf,tl)

data_process.create_test_submission_2d(test,y)


















