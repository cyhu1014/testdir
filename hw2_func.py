
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

def train_2 (feat,Y):
    X = np.matrix(feat)
    Y = np.matrix(Y)
    Y = np.matrix.getT(Y)
    w = np.zeros(23)
    w = np.matrix(w)
    w = np.matrix.getT(w)
    b = np.zeros((1,))
    
    
    
    epoch_num =100 
    batch_size = 100
    batch_num = 20000//batch_size
   
    l_rate = 0.003
    w_grad = []
    for i in range (23):
        w_grad.append(0)
    
    for epoch in range(100):
        epoch_loss = 0.0
        for idx in range(100):
            Xin = X[idx*batch_size:(idx+1)*batch_size]
            Yin = Y[idx*batch_size:(idx+1)*batch_size]
            Y_predict = Xin*w+b
            
            for i in range(batch_size):
                Y_predict[i,0] = sigmoid(Y_predict[i,0])
            
            for i in range(batch_size):
                cross_entropy = - ( Yin[i,0]*Y_predict[i,0] + (1-Yin[i,0])*(1-Y_predict[i,0])    )
                
                epoch_loss += cross_entropy
            
            for i in range (23):
                w_grad[i]=0
            
            
            b_grad = 0
            for i in range(batch_size):
                for j in range (23):
                    w_grad[j] += -1 * (Yin[i,0] - Y_predict[i,0] * Xin[i,j])
                b_grad += -1 * (Yin[i,0] - Y_predict[i,0])
            

            for i in range (23):
                w[i] = w[i] - l_rate * w_grad[i]
            
            b = b - l_rate * b_grad
        if (epoch+1) % 1== 0:
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / 20000)))
            #print(w,b)
            
    return w, b

def feature_normalize_min_max(X_train):
    #need_normalize = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    need_normalize = []
    for i in range (23):
        need_normalize.append(i)
    normalize_min  = []
    normalize_max  = []
    for i  in range (len(need_normalize)):
        normalize_max.append(-9999999999)
        normalize_min.append(9999999999)
        for j in range (len(X_train)):
            normalize_max[i]=max(normalize_max[i],X_train[j][need_normalize[i]])
            normalize_min[i]=min(normalize_min[i],X_train[j][need_normalize[i]])
    for i in range (len(need_normalize)):
        for j in range (len(X_train)): 
            X_train[j][need_normalize[i]] = (X_train[j][need_normalize[i]]-normalize_min[i]) / (normalize_max[i] - normalize_min[i])
            
    return X_train
def feature_normalize_mean(X_train):
    #need_normalize = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    need_normalize = []
    for i in range (23):
        need_normalize.append(i)
    normalize_mean  = []
    normalize_min  = []
    normalize_max  = []
    for i  in range (len(need_normalize)):
        normalize_max.append(-9999999999)
        normalize_min.append(9999999999)
        for j in range (len(X_train)):
            normalize_max[i]=max(normalize_max[i],X_train[j][need_normalize[i]])
            normalize_min[i]=min(normalize_min[i],X_train[j][need_normalize[i]])
    for i  in range (len(need_normalize)):
        normalize_mean.append(0)
        for j in range (len(X_train)):
            normalize_mean[i]+=X_train[j][need_normalize[i]]
        normalize_mean[i]/=len(X_train)
            
    for i in range (len(need_normalize)):
        for j in range (len(X_train)): 
            X_train[j][need_normalize[i]] = (X_train[j][need_normalize[i]]-normalize_mean[i]) / (normalize_max[i] - normalize_min[i])
            
    return X_train

def predict (test,w,b):
    X = np.matrix(test)
    Y = X*w+b
    ans_list = []
    for i in range (len(test)):
        Y[i]=sigmoid(Y[i])
        if(Y[i]>=0.5):
            ans_list.append(1)
        else:
            ans_list.append(0)
    test_label=[]
    test_title =[]
    test_title.append("id")
    test_label.append("value")
    for i in range (10000):
        test_title.append("id_"+str(i))
        test_label.append(ans_list[i])

    df =pd.DataFrame(test_label,test_title)
    df.to_csv("testsummit.csv",header=False)

def train_3 (feat,Y):
    X = feat
    
    w = []
    w_grad = []
    
    for i in range (24):
        w.append(0)
        w_grad.append(0)
    

    epoch_num =100 
    batch_size = 100
    batch_num = 20000//batch_size
   
    l_rate = 0.03
   
    Y_predict = []
    for i in range (batch_size):
        Y_predict.append(0)
    
    for epoch in range(10000):
        epoch_loss = 0.0
        for idx in range(200):
            Xin = X[idx*batch_size:(idx+1)*batch_size]
            Yin = Y[idx*batch_size:(idx+1)*batch_size]
            for i in range (batch_size):
                Y_predict[i]=0
                for j in range (23):
                    Y_predict[i]+=Xin[i][j]*w[j]
                Y_predict[i]+=w[23]
            for i in range(batch_size):
                Y_predict[i] = sigmoid(Y_predict[i])
            
            for i in range(batch_size):
                cross_entropy = - ( Yin[i]*np.log(Y_predict[i]) + (1-Yin[i])*np.log(1-Y_predict[i])   )
                
                epoch_loss += cross_entropy
            
            for i in range (24):
                w_grad[i]=0
            
        
            for i in range(batch_size):
                for j in range (23):
                    w_grad[j] += -1 * (Yin[i] - Y_predict[i] ) * Xin[i][j]
                w_grad[23] += -1 * (Yin[i] - Y_predict[i])
            

            for i in range (23):
                w[i] = w[i] - l_rate * w_grad[i]
            w[23] = w[23] - l_rate * w_grad[23]
            
        if (epoch+1) % 10== 0:
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / 20000)))
            #print(w,b)
        
    return w