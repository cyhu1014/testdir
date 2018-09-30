###for ntu_ml 2018 hw1

import sys
import math
import pandas as pd
import numpy as np
import data_process
import func
train  = pd.read_csv(sys.argv[1],encoding="big5" ,header=None)
test   = pd.read_csv(sys.argv[2],encoding="big5" ,header=None)


(train_unmodify,train_modify)=data_process.data_processing(train)

print(len(train_unmodify[0]),len(train_modify[0]))

(tf,tl)= data_process.train_dataset_2_x(train_modify,1)

print(len(tf),len(tl))

(w,b)=func.best_function(tf,tl)

df = data_process.create_test_submission(test,w,b)

df.to_csv(sys.argv[3],header=False)

(x,y)=np.load("hw1_model.npy")

(w,b)=func.best_function(x,y)
















