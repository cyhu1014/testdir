
import pandas as pd
import numpy  as np
import math

def sigmoid(X):
    res= 1 / (1 + np.exp(-X))
    return np.clip(res, 0.00000000000001, 0.99999999999999)
def train(feat,Y,data_num,iter):
    feat_num=len(feat[0])
    w=[]
    b=0
    lr_w=1e-18
    lr_b=1e-18
    change_b=0
    change_w=[]
    cross_entropy_line=[]
    fx = []
    for i in range (feat_num):
        w.append(1e-10)
        change_w.append(0)
    for i in range (data_num):
        fx.append(0)
    for i in range(iter):
        for j in range(data_num):
            temp=0
            for k in range(feat_num):
                temp+=feat[j][k]*w[k]
            temp+=b
            temp=sigmoid(temp)
            fx[j]=temp
        cross_entropy = 0
        for j in range (data_num):
            tempfx  = np.log(fx[j]) * Y[j] + np.log(1-fx[j]) * (1-Y[j])
            cross_entropy-=tempfx
            for k in range(feat_num):
                change_w[k]+= -(Y[j]-fx[j])*feat[j][k]
            change_b   += -(Y[j]-fx[j])
        for j in range (feat_num):
            w[j] = w[j] - lr_w*change_w[j]
            change_w[j]=0
       
        b = b - lr_b*change_b
        print(w[0],w[1]) 
        cross_entropy_line.append(cross_entropy)
    print(len(fx))
    return w ,b ,cross_entropy_line