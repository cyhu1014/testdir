
import numpy as np
import pandas as pd
import sys
import hw2_func_GD as func

x = [[1,2,3],[3,5,7],[4,7,10]]
m =np.matrix(x)
x = np.std (m ,axis=0)
print(x)
np.std

tf = pd.read_csv(sys.argv[1],header=None)
tl = pd.read_csv(sys.argv[2],dtype=int)
X = func.create_train_dataset(tf)
Y = func.create_train_label(tl)

A = [] ##valid
B = [] ##unvalid
data_num = len(X)
feat_num = len(X[0])

A_num=0
B_num=0
for i in range (data_num):
    if(Y[i]==1):
        A.append([])
        for j in range (feat_num):
            A[A_num].append(X[i][j])
        A_num+=1
    else:
        B.append([])
        for j in range (feat_num):
            B[B_num].append(X[i][j])
        B_num+=1

A_mean = []
B_mean = []
for i in range (feat_num):
    A_mean.append(0)
    B_mean.append(0)
for i in range (A_num):
    for j in range (feat_num):
        A_mean[j]+=A[i][j]
for i in range (B_num):
    for j in range (feat_num):
        B_mean[j]+=B[i][j]
for i in range(feat_num):
    A_mean[i]/=A_num
    B_mean[i]/=B_num


A_sigma = []
B_sigma = []
sigma = []
sigmaI = []
x_temp = []
for i in range (feat_num):
    A_sigma.append([])
    B_sigma.append([])
    sigma.append([])
    sigmaI.append([])
    x_temp.append(0)
    for j in range (feat_num):
        A_sigma[i].append(0)
        B_sigma[i].append(0)
        sigma[i].append(0)
        sigmaI[i].append(0)
print(len(A_sigma))
for k in range (A_num):
    for l in range(feat_num):
        x_temp[l] = A[k][l]-A_mean[l]
    for i in range(feat_num):
        for j in range(feat_num):
            A_sigma[i][j]+=x_temp[i]*x_temp[j]
for k in range (B_num):
    for l in range(feat_num):
        x_temp[l] = B[k][l]-B_mean[l]
    for i in range(feat_num):
        for j in range(feat_num):
            B_sigma[i][j]+=x_temp[i]*x_temp[j]

for i in range(feat_num):
    for j in range(feat_num):
        A_sigma[i][j]/=A_num  
        B_sigma[i][j]/=B_num


for i in range(feat_num):
    for j in range(feat_num):
        sigma[i][j]=(A_num*A_sigma[i][j] +B_num*B_sigma[i][j])/data_num

x = np.matrix(sigma)
x = x.getI()
for i in range (feat_num):
    for j in range(feat_num):
        sigmaI[i][j]=x[i,j]
w = []
b= 0


meanAsB = []
for i in range(feat_num):
    meanAsB.append(A_mean[i]-B_mean[i])

for i in range(feat_num):
    temp = 0
    for j in range(feat_num):
        temp+=meanAsB[j]*sigmaI[j][i]
    w.append(temp)
print(w)

sI = np.matrix(sigmaI)
u1 = np.matrix(A_mean)
u1T = u1.getT()
u2 = np.matrix(B_mean)
u2T = u2.getT()


    
         
