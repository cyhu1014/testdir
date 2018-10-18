##ntu-machine learning hw2
##predict 
import pandas as pd
import numpy  as np
import math
import sys
import matplotlib.pyplot as plt
def sigmoid(z):
    if(1.0+np.exp(-z)==0):
        return
    res = 1.0 / (1.0 + np.exp(-z))
    return res
tf = pd.read_csv(sys.argv[1],dtype=int)
tl = pd.read_csv(sys.argv[2],dtype=int)
###estimate how many people vaild
valid_num=0
unvalid_num=0
for i in range (len(tl)):
    if(tl['Y'][i]==True):
        valid_num+=1
    else:
        unvalid_num+=1
print(valid_num,unvalid_num)

w=[0.0001,0.00001,0.00001]
w_change=  [0,0,0]
lr_w=[0.0000000000005,0.0000000000005,0.0000000000005]
lr_b=0.00000001
b=0
b_change = 0
fx=[]
for i in range (len(tf)):
    fx.append(0)
cross_entropy_line = []
for iter in range (20):
    w_change=  [0,0,0]
    for i in range (20000):
        fx[i]=w[0]*tf['SEX'][i]+w[1]*tf['AGE'][i]+w[2]*tf['LIMIT_BAL'][i]+b
    for i in range (len(tf)):
        fx[i]=sigmoid(fx[i])
    

    cross_entropy = 0
    for i in range (1000):
        tempfx  = np.log(fx[i]) * tl['Y'][i] + np.log(1-fx[i]) * (1-tl['Y'][i])
        cross_entropy-=tempfx
        w_change[0]+= -(tl['Y'][i]-fx[i])*tf['SEX'][i]
        w_change[1]+= -(tl['Y'][i]-fx[i])*tf['AGE'][i]
        w_change[2]+= -(tl['Y'][i]-fx[i])*tf['LIMIT_BAL'][i]
        b_change   += -(tl['Y'][i]-fx[i])
    w[0] = w[0] - lr_w[0]*w_change[0]
    w[1] = w[1] - lr_w[1]*w_change[1]
    w[2] = w[2] - lr_w[2]*w_change[2]
    b = b - lr_b*b_change
    cross_entropy_line.append(cross_entropy)
    print(iter,w,b,cross_entropy)

plt.plot(cross_entropy_line)
plt.savefig("loss_change.jpg")
test = pd.read_csv(sys.argv[3],dtype=int)
ans_list = []
for i in range (10000):
    temp=w[0]*test['SEX'][i]+w[1]*test['AGE'][i]+w[2]*test['LIMIT_BAL'][i]+b
    temp=sigmoid(temp)
    print(temp)
    if(temp>0.5):
        ans_list.append(1)
    else:
        ans_list.append(0)
print(ans_list)
x=1

test_label=[]
test_title =[]
test_title.append("id")
test_label.append("value")
for i in range (10000):
    test_title.append("id_"+str(i))
    test_label.append(ans_list[i])

df =pd.DataFrame(test_label,test_title)
df.to_csv("testsummit.csv",header=False)
    

    

