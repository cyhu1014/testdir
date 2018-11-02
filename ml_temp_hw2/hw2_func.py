
import pandas as pd
import numpy  as np
import math
import random as r
def sigmoid(X):
    res= 1 / (1 + np.exp(-X))
    return np.clip(res, 0.000000000001, 0.999999999999)


def shuffle(X, Y):
    feat_num = len(X[0])
    data_num = len(X) 
    
    for i in range (2000):
        X.append([])
        for j in range(feat_num):
            X[feat_num].append(X[0][j])
        X.remove(X[0])
        
        Y.append(Y[0])
        Y.remove(Y[0])
        
    return X,Y
            
def shuf (X, Y):
    data_num=len(X)
    
    newX=[]
    newY=[]
    for i in range (0,len(X)):
        rand =r.randint(0,data_num-1)
        newX.append([])
        for j in range (len(X[0])):
            newX[i].append(X[rand][j])
        newY.append(Y[rand])
        X.remove(X[rand])
        Y.remove(Y[rand])
        data_num-=1
    return newX ,newY
def feature_normalize_mean_covariance(X_train):
    feat_num = len(X_train[0])
    need_normalize = []
    for i in range (feat_num):
        need_normalize.append(i)
    print(len(X_train))
    '''
    not_to_normalize = [2]
    for i in range (len(not_to_normalize)):
        need_normalize.remove(not_to_normalize[i])
    '''
    normalize_mean  = []
    covariance = []
   
    for i  in range (len(need_normalize)):
        normalize_mean.append(0)
        for j in range (len(X_train)):
            
            normalize_mean[i]+=X_train[j][need_normalize[i]]
        normalize_mean[i]/=len(X_train)
    for i in range (len(need_normalize)):
        covariance.append(0)
        for j in range (len(X_train)):
            temp =  (X_train[j][need_normalize[i]]-normalize_mean[i])**2
            covariance[i]+=temp
        covariance[i]/=len(X_train)
        covariance[i] = np.sqrt(covariance[i])
    for i in range (len(need_normalize)):
        for j in range (len(X_train)): 
            X_train[j][need_normalize[i]] = (X_train[j][need_normalize[i]]-normalize_mean[i]) / covariance[i]
    
    return X_train

def predict (X,w,text):
    
    Y = []
    feat_num = len(X[0])
    for i in range (len(X)):
        Y.append(0)
        for j in range (len(X[0])):
            Y[i]+=w[j]*X[i][j]
        Y[i]+=w[feat_num]    
    ans_list = []
    for i in range (len(X)):
        Y[i]=sigmoid(Y[i])
        if(Y[i]>0.4):
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
    df.to_csv(text,header=False)

def train_3 (X,Y):
    train_set_num = 20000
    valid_set_num = 2000
    v_index=18000
    feat_num = len(X[0])
    w = []
    w_grad = []
    w_grad_ada = []
    
    for i in range (feat_num+1):
        w.append(1)
        w_grad.append(0) 
        w_grad_ada.append(0)

    batch_size = 100
    l_rate = 0.5
   
    Y_predict = []
    for i in range (batch_size):
        Y_predict.append(0)
    


    for epoch in range(1,1001):
        epoch_loss = 0.0
        
        for idx in range(train_set_num//100):
            
            Xin = X[idx*batch_size:(idx+1)*batch_size]
            Yin = Y[idx*batch_size:(idx+1)*batch_size]
            for i in range (batch_size):
                Y_predict[i]=0
                for j in range (feat_num):
                    Y_predict[i]+=Xin[i][j]*w[j]
                Y_predict[i]+=w[feat_num]
            for i in range(batch_size):
                Y_predict[i] = sigmoid(Y_predict[i])
            
            for i in range(batch_size):
                cross_entropy = - ( Yin[i]*np.log(Y_predict[i]) + (1-Yin[i])*np.log(1-Y_predict[i])   )
                
                epoch_loss += cross_entropy
            
            for i in range (feat_num+1):
                w_grad[i]=0
            
        
            for i in range(batch_size):
                for j in range (feat_num):
                    w_grad[j] += -1 * (Yin[i] - Y_predict[i] ) * Xin[i][j]
                w_grad[feat_num] += -1 * (Yin[i] - Y_predict[i])
            for i in range (feat_num+1):
                x = w_grad[i]**2
                w_grad_ada[i]+= x
            
            
            for i in range (feat_num):
                x = (1/epoch) * w_grad_ada[i]
                w[i] = w[i] - l_rate * w_grad[i] * (1/x)
                
            x = (1/epoch) * w_grad_ada[feat_num]
            w[feat_num] = w[feat_num] - l_rate * w_grad[feat_num] *(1/x)
            
        if (epoch+1) % 5== 0:
            epoch_loss_v = 0.0
            Y_predict_v = []
            
            for i in range(v_index,len(X)):
                Y_predict_v.append(0)
                Y_predict_v[i-v_index]=0
                for j in range (feat_num):
                    Y_predict_v[i-v_index]+=X[i][j]*w[j]
                Y_predict_v[i-v_index]+=w[feat_num]
                Y_predict_v[i-v_index] = sigmoid(Y_predict_v[i-v_index])
                cross_entropy = - ( Y[i]*np.log(Y_predict_v[i-v_index]) + (1-Y[i])*np.log(1-Y_predict_v[i-v_index])   )
                epoch_loss_v += cross_entropy
            
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / train_set_num)),(epoch_loss_v/(len(X)-v_index)))
            #print(w,b)
        
    return w



def create_train_dataset(tf):
    feat=[]

    for i in range (1,len(tf)):
        feat.append([])
        for j in range (23):
            feat[i-1].append(float(tf[j][i]))
    return feat


def create_train_label(tl):
    Y= []
    for i in range(len(tl)):
        Y.append(float(tl['Y'][i]))
    return Y

def feat_expand (X):
    
    num = len(X)
    square = [0,5,11,12,13,14,15,16,17,18,19,20,21,22]
    cube   = [0,5,11]
    quada  = []
    panta  = []
    six  = []
    seven = []
    
    for i in range (num):
        for j in range (len(square)):
            X[i].append(X[i][square[j]]**2)
        for j in range (len(cube)):
            X[i].append(X[i][cube[j]]**3)
        
        
        
        
        
            
        
            
            
    return X
def age_one_hot (X):
    data_num = len(X)
    for i in range (data_num):
        if(X[i][4]<30):
            X[i].append(0)
            X[i].append(0)
            X[i].append(0)
            X[i].append(1)
        elif(X[i][4]>=30 and X[i][4]<50):
            X[i].append(0)
            X[i].append(0)
            X[i].append(1)
            X[i].append(0)
        elif(X[i][4]>=50 and X[i][4]<70):
            X[i].append(0)
            X[i].append(1)
            X[i].append(0)
            X[i].append(0)
        else:
            X[i].append(1)
            X[i].append(0)
            X[i].append(0)
            X[i].append(0)
    return X

def feat_onehot (X):
    data_num = len(X)
    
    
    for i in range (data_num):
        ##sex
        if(X[i][1]==1):
            X[i].append(0)
            X[i].append(1)
        else:
            X[i].append(1)
            X[i].append(0)
        ##education
        if(X[i][2]==1):
            X[i].append(0)
            X[i].append(0)
            X[i].append(0)
            X[i].append(1)
        elif(X[i][2]==2):
            X[i].append(0)
            X[i].append(0)
            X[i].append(1)
            X[i].append(0)
        elif(X[i][2]==3):
            X[i].append(0)
            X[i].append(1)
            X[i].append(0)
            X[i].append(0)
        else:
            X[i].append(1)
            X[i].append(0)
            X[i].append(0)
            X[i].append(0)
        ##marriage
        if(X[i][3]==1):
            X[i].append(0)
            X[i].append(1)
        else:
            X[i].append(1)
            X[i].append(0)
        
    return X

def feat_delete (X):
    delete_feat = [1,2,3]
    for i in range (len(X)):
        for j in range (len(delete_feat)):
            X[i].remove(X[i][delete_feat[j]])
    return X

def oversample(X,Y):
    data_num=len(X)
    index =len(X)
    for i in range (data_num):
        
        if(Y[i]==1):
            X.append([])
            for j in range (len(X[0])):
                X[index].append(X[i][j])
            Y.append(Y[i])
            index+=1
            X.append([])
            for j in range (len(X[0])):
                X[index].append(X[i][j])
            Y.append(Y[i])
            index+=1
    
    return X ,Y

def feat_process (feat):
    feat = feat_expand(feat)
    feat = feat_onehot(feat) 
    feat = feat_delete(feat)
    feat = feature_normalize_mean_covariance(feat)
    
    print("feat_num:" ,len(feat[0]))
    return feat

