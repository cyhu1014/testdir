
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



def train_GD (X,Y):
    data_num = 20000
    valid_num = 2000
    feat_num = len(X[0])
    valid_num = 0
    unvalid_num = 0
    mean_valid = []
    mean_unvalid = []
    for i in range (data_num):
        if(Y[i]==0):
            valid_num+=1
        else:
            unvalid_num+=1

    for i in range (feat_num):
        mean_valid.append(0)
        mean_unvalid.append(0)
    for i in range (data_num):
        if (Y[i]==1):
            for j in range(feat_num):
                mean_valid[j]+=X[i][j]
        else:
            for j in range(feat_num):
                mean_unvalid[j]+=X[i][j]
    for i in range (feat_num):
        mean_valid[i]/=valid_num
        mean_unvalid[i]/=unvalid_num
    
    ###create valid mean and covariance matrix
    uvalid  = np.matrix(mean_valid)
    uvalidT = uvalid.getT()
    uunvalid  = np.matrix(mean_unvalid)
    uunvalidT = uunvalid.getT()
    cm = []
    for i in range (feat_num):
        cm.append([])
        for j in range (feat_num):
            cm[i].append(0.0)
    c_matrix_valid = np.matrix(cm)
    c_matrix_unvalid = np.matrix(cm)
    for i in range (data_num):
        x = np.matrix(X[i])
        if (Y[i]==1):
            a = x-uvalid
            aT = a.getT()
            y=aT*a
  
            c_matrix_valid+=y
        else:
            a = x-uunvalid
            aT = a.getT()
            y=aT*a
            c_matrix_unvalid+=y
    for i in range (feat_num):
        for j in range (feat_num):
            c_matrix_valid[i,j]/=valid_num
            c_matrix_unvalid[i,j]/=unvalid_num
    c_matrix = np.matrix(cm)
    for i in range (feat_num):
        for j in range (feat_num):
            c_matrix[i,j]= c_matrix[i,j] + c_matrix_valid[i,j]*valid_num/data_num + c_matrix_unvalid[i,j] *unvalid_num/data_num
    c_matrixI = np.linalg.pinv(c_matrix)
    
    w = (uvalid-uunvalid) *c_matrixI
    b = -0.5 * uvalid *c_matrixI*uvalidT - 0.5 * uunvalid *c_matrixI * uunvalidT - np.log(valid_num/unvalid_num)
    
    w_return = []
    for i in range (feat_num):
        w_return.append(w[0,i])
    w_return.append(b[0,0])
    epoch_loss = 0.0
    for i in range(data_num):
        Y_predict = 0.0
        for j in range (feat_num):
            Y_predict +=w_return[j]*X[i][j]
        Y_predict+=w_return[feat_num]
        Y_predict = sigmoid(Y_predict)
        cross_entropy = - ( Y[i]*np.log(Y_predict) + (1-Y[i])*np.log(1-Y_predict)   )
        epoch_loss += cross_entropy
    epoch_loss_train = epoch_loss
    epoch_loss = 0.0
    for i in range(valid_num):
        Y_predict = 0.0
        for j in range (feat_num):
            Y_predict +=w_return[j]*X[i][j]
        Y_predict+=w_return[feat_num]
        Y_predict = sigmoid(Y_predict)
        cross_entropy = - ( Y[i]*np.log(Y_predict) + (1-Y[i])*np.log(1-Y_predict)   )
        epoch_loss += cross_entropy
    print("epl: ",(epoch_loss_train/data_num),"epv:" ,(epoch_loss/valid_num))
    return w_return



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
    delete_feat = [22]
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



def feat_process_GD (feat):
    #feat = feat_expand(feat)
    #feat = feat_onehot(feat) 
    feat = feat_delete(feat)
    feat = feature_normalize_mean_covariance(feat)
    print("feat_num:" ,len(feat[0]),"data_num:",len(feat))
    return feat