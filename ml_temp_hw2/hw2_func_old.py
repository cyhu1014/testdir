
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

def shuffle(X, Y):
    feat_num = len(X[0])
    data_num = len(X) 
    for i in range (2000):
        temp  = X[0]
        X.remove(X[0])
        X.append(temp)
        temp  = Y[0]
        Y.remove(Y[0])
        Y.append(temp)
    return X,Y
            
def feature_normalize_min_max(X_train):
    #need_normalize = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    feat_num = len(X_train[0])
    need_normalize = []
    for i in range (feat_num-8):
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
    need_normalize = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
    '''
    need_normalize = []
    for i in range (23):
        need_normalize.append(i)
    '''
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

def feature_normalize_mean_covariance(X_train):
    feat_num = len(X_train[0])
    need_normalize = []
    for i in range (feat_num):
        need_normalize.append(i)

    
    not_to_normalize = [1,2,3,5,6,7,8,9,10]
    for i in range (len(not_to_normalize)):
        need_normalize.remove(not_to_normalize[i])
    
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
        if(Y[i]>0.5):
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
    train_set_num = 18000
    valid_set_num = 2000
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
    


    for epoch in range(1000):
        epoch_loss = 0.0
        
        #X,Y = shuffle(X,Y)
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
            
            
            for i in range (feat_num):
                w[i] = w[i] - l_rate * w_grad[i]
            w[feat_num] = w[feat_num] - l_rate * w_grad[feat_num]
            
        if (epoch+1) % 10== 0:
            
            epoch_loss_v = 0.0
            Y_predict_v = []
            for i in range(18000,20000):
                Y_predict_v.append(0)
                Y_predict_v[i-18000]=0
                for j in range (feat_num):
                    Y_predict_v[i-18000]+=X[i][j]*w[j]
                Y_predict_v[i-18000]+=w[feat_num]
                Y_predict_v[i-18000] = sigmoid(Y_predict_v[i-18000])
                cross_entropy = - ( Y[i]*np.log(Y_predict_v[i-18000]) + (1-Y[i])*np.log(1-Y_predict_v[i-18000])   )
                epoch_loss_v += cross_entropy
            
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / train_set_num)),(epoch_loss_v/2000))
            #print(w,b)
        
    return w

def train_GD (X,Y):
    data_num = 18000
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
    c_matrixI = c_matrix.getI()
    
    w = (uvalid-uunvalid) *c_matrixI
    b = -0.5 * uvalid *c_matrixI*uvalidT + -0.5 * uunvalid *c_matrixI * uunvalidT +np.log(valid_num/unvalid_num)
    
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


def train_5 (X,Y):
    feat_num = len(X[0])
    w = []
    w_grad = []
    for i in range (feat_num):
        w.append(0)
        w_grad.append(0)    
    b=0.0
    b_grad = 0.0
    batch_size = 100
    l_rate = 0.0003
    
    Y_predict = []
    for i in range (batch_size):
        Y_predict.append(0)
    w = np.matrix(w)
    for epoch in range(200):
        epoch_loss = 0.0
        X,Y = shuffle(X,Y)
        for idx in range(180):
            Xin = X[idx*batch_size:(idx+1)*batch_size]
            Yin = Y[idx*batch_size:(idx+1)*batch_size]
            Xin = np.matrix(Xin)

            
            Y_predict=Xin * w.getT()
            Y_predict+=b
           
            Y_predict = sigmoid(Y_predict)
            
            for i in range(batch_size):
                cross_entropy = - ( Yin[i]*np.log(Y_predict[i,0]) + (1-Yin[i])*np.log(1-Y_predict[i,0])   )
                
                epoch_loss += cross_entropy
            
            for i in range (feat_num):
                w_grad[i]=0
            b_grad
        
            for i in range(batch_size):
                for j in range (feat_num):
                    w_grad[j] += -1 * (Yin[i] - Y_predict[i,0] ) * Xin[i,j]
                b_grad += -1 * (Yin[i] - Y_predict[i,0])
            

            for i in range (feat_num):
                w[0,i] = w[0,i] - l_rate * w_grad[i]
            b = b - l_rate * b_grad
            
        if (epoch+1) % 10== 0:
            
            epoch_loss_v = 0.0
            Y_predict_v = []
            for i in range(18000,20000):
                Y_predict_v.append(0)
                Y_predict_v[i-18000]=0
                for j in range (feat_num):
                    Y_predict_v[i-18000]+=X[i][j]*w[0,j]
                Y_predict_v[i-18000]+=b
                Y_predict_v[i-18000] = sigmoid(Y_predict_v[i-18000])
                cross_entropy = - ( Y[i]*np.log(Y_predict_v[i-18000]) + (1-Y[i])*np.log(1-Y_predict_v[i-18000])   )
                epoch_loss_v += cross_entropy
            
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / 18000)),(epoch_loss_v/2000))
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
    cube   = [0,5,11,12,13,14,15,16,17,18,19,20,21,22]
    quada  = []
    panta  = []
    six  = []
    seven = []
    for i in range (num):
        for j in range (len(square)):
            X[i].append(X[i][square[j]]**2)
        for j in range (len(cube)):
            X[i].append(X[i][square[j]]**3)
        for j in range (len(quada)):
            X[i].append(X[i][square[j]]**4)
        for j in range (len(panta)):
            X[i].append(X[i][square[j]]**5)
            X[i].append(X[i][square[j]]**6)
            
        
        
            
        #for j in range (len(seven)):
           # X[i].append(X[i][square[j]]**9)
            
            
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

def feat_delelte (X):
    delete_feat = [1,2,3]
    for i in range (len(X)):
        for j in range (len(delete_feat)):
            X[i].remove(X[i][delete_feat[j]])
    return X
def feat_process (feat):
    feat = feat_expand(feat)
    feat = feat_onehot(feat)
    feat = feat_delelte(feat)
    feat = feature_normalize_mean_covariance(feat)
    print("feat_num:" ,len(feat[0]))
    return feat