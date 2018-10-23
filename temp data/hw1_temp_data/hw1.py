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
####all feat 
train_all_feat=data_process.data_processing_allfeat(train)

###plot 4 learning rate for question
'''
(tf,tl)= data_process.train_dataset_1(train_modify)
print(len(tf))
(w,b,loss1)=func.loss_function_2(tf,tl,10000,1e-7)
(w,b,loss2)=func.loss_function_2(tf,tl,10000,1e-9)
(w,b,loss3)=func.loss_function_2(tf,tl,10000,1e-11)
(w,b,loss4)=func.loss_function_2(tf,tl,10000,1e-13 )
plt.plot(loss1,label="1e-7")
plt.plot(loss2,label="1e-9")
plt.plot(loss3,label="1e-11")
plt.plot(loss4,label="1e-13")
plt.xlabel("iteration")
plt.ylabel("RMSE")
plt.legend()
plt.savefig('loss.jpg')
'''
#train use nine hour pm25 and spuare of pm25

(tf,tl)=data_process.train_dataset_pm25_9hr_overlap_square(train_modify)
print(len(tl))
y=func.best_function_xd(tf,tl)
#np.save('hw1_9feat_overlap', y)
#print(len(y))
print(y)
data_process.create_test_submission_pm25_square(test,y)




###train using all feature

print(len(train_all_feat[0]))

(tf,tl)=data_process.train_all_feat_9hr_overlap(train_all_feat)
print(len(tf))

y=func.best_function_xd(tf,tl)

#print(len(y))
data_process.create_test_submission_allfeat(test,y)


###train use pm25 and pm25 square and pm10, co

(tf,tl) = data_process.train_dataset_some_feat(train_all_feat)
y=func.best_function_xd(tf,tl)
print(len(y))
np.save('3feat',y)
data_process.create_test_submission_somefeat(test,y)



###train with regularization
'''
(tf,tl)= data_process.train_dataset_1(train_modify)
itr =100
(w,b,loss1,lam)=func.loss_function_reg(tf,tl,itr,1e-7,0)
plt.plot(loss1,label=str(lam))
data_process.create_test_submission_with_lam(test,w,b,lam)
(w,b,loss2,lam)=func.loss_function_reg(tf,tl,itr,1e-7,4000)
plt.plot(loss2,label=str(lam))

data_process.create_test_submission_with_lam(test,w,b,lam)
(w,b,loss5,lam)=func.loss_function_reg(tf,tl,itr,1e-7,10000)
plt.plot(loss5,label=str(lam))
data_process.create_test_submission_with_lam(test,w,b,lam)
(w,b,loss5,lam)=func.loss_function_reg(tf,tl,itr,1e-7,100000)
plt.plot(loss5,label=str(lam))
data_process.create_test_submission_with_lam(test,w,b,lam)
(w,b,loss6,lam)=func.loss_function_reg(tf,tl,itr,1e-7,1000000)
plt.plot(loss6,label=str(lam))
data_process.create_test_submission_with_lam(test,w,b,lam)
(w,b,loss7,lam)=func.loss_function_reg(tf,tl,itr,1e-7,3000000)
plt.plot(loss7,label=str(lam))
data_process.create_test_submission_with_lam(test,w,b,lam)
plt.legend()
plt.savefig("different_iteration1.jpg")
'''