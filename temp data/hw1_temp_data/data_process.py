###for ntu_ml 2018 hw1
import pandas as pd
import  numpy as np

def data_processing_pm25 (train):
    train_unmodify = []
    index=10
    train_mean=0
    train_mean_num=0
    for i in range (12):
        train_unmodify.append([])
        for k in range (20):
            for j in range (3,27):
                if(float(train[j][index])>0 and float(train[j][index])<300):
                    train_mean+=float(train[j][index])
                    train_mean_num+=1
                train_unmodify[i].append(float(train[j][index]))
            index+=18
    train_mean/=train_mean_num
    ############train data set modify 移除過於大或過小的資料
    train_modify =[]
    for i in range (12):
        train_modify.append([])
        for k in range (480):
            if(train_unmodify[i][k]>0 and  train_unmodify[i][k]<300):
                train_modify[i].append(train_unmodify[i][k])
            else:
                train_modify[i].append(train_mean)
    
    return train_unmodify,train_modify

def data_processing_allfeat (train):
    train_unmodify = []
    row=1
    for i in range (12):
        train_unmodify.append([])
        j=0
        for n in range (20):
            for k in range (3,27):
                train_unmodify[i].append([])
                for l in range (18):
                    if(train[k][row]=='NR'):
                        train_unmodify[i][j].append(0)
                    else:        
                        train_unmodify[i][j].append(float(train[k][row]))
                    row+=1
                row-=18
                j+=1
            row+=18

    return train_unmodify           


def train_dataset_1(train_data):
    ############ train data set 1 連續九小時當data 不重疊
    train_feat=[]
    train_label=[]
    for i in range (12):
        index=0
        while(index<len(train_data[i])):
            temp_mean=0
            for j in range (9):
                temp_mean+=train_data[i][index]
                index+=1
            temp_mean/=9
            train_feat.append(temp_mean)
            train_label.append(train_data[i][index])
            index+=1
    print("tf_1: ",len(train_feat),", tl_1: ",(len(train_label)))
    return train_feat,train_label

def train_dataset_2(train_data):
    ############ train data set 2 連續九小時當資料 重疊
    train_feat=[]
    train_label=[]
    for i in range (12):
        index=0
        for j in range (471):
            temp_mean=0
            for k in range (j,j+9):
                temp_mean+=train_data[i][k]
            temp_mean/=9.0
            train_feat.append(temp_mean)
            train_label.append(train_data[i][j+9])
    print("tf_2: ",len(train_feat),", tl_2: ",(len(train_label)))

    return train_feat,train_label

def train_dataset_2_x(train_data,x):
    ############ train data set 3_m 連續5小時當資料 重疊 使用modify的資料
    train_feat=[]
    train_label=[]
    for i in range (12):
        index=0
        for j in range (480-x):
            temp_mean=0
            for k in range (j,j+x):
                temp_mean+=train_data[i][k]
            temp_mean/=x
            train_feat.append(temp_mean)
            train_label.append(train_data[i][j+x])
    print("tf_3_m: ",len(train_label),", tl_3_m: ",(len(train_label)))
    return train_feat,train_label

def create_test_submission(test,w,b):
    x=1
    test_feat= []
    test_label=[]
    test_title =[]
    row=9
    for i in range(260):
        pm25_mean=0
        for j in range(11-x,11):
            pm25_mean+=float(test[j][row])
            test_feat.append(pm25_mean/x)
        row+=18
    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        test_label.append(test_feat[i]*w+b)

    df =pd.DataFrame(test_label,test_title)
    w=format(w, '.5g')
    b=format(b, '.5g')
    df.to_csv("submit_w"+str(w)+"b"+str(b)+".csv",header=False)
    return df 

def create_model(tf,tl):
    length = len(tf)*9//10
    x=np.array(tf[:length])
    y=np.array(tl[:length])
    np.save('hw1_model_x_2', x)
    np.save('hw1_model_y_2', y)    

def train_dataset_2d (train_data):
    train_feat=[]
    train_label=[]
    l_index=0
    for i in range (12):
        index=0
        for j in range (480//3):
            train_feat.append([])
            train_feat[l_index].append(train_data[i][index])
            train_feat[l_index].append(train_data[i][index+1])
            train_feat[l_index].append(1)
            train_label.append(train_data[i][index+2])
            index+=3
            l_index+=1 
    print("tf_2d: ",len(train_feat),", tl_2d: ",(len(train_label)))
    return train_feat,train_label

def train_dataset_xd (train_data,x):
    train_feat=[]
    train_label=[]
    length=len(train_data[0])
    l_index=0
    for i in range (12):
        index=0
        for j in range (length//(x+1)):
            train_feat.append([])
            for k in range (x):
                train_feat[l_index].append(train_data[i][index])
                index+=1
            train_feat[l_index].append(1)
            train_label.append(train_data[i][index])
            index+=1
            l_index+=1 
    print("tf_",str(x),len(train_feat),", tl_",str(x),(len(train_label)))
    return train_feat,train_label

def train_dataset_pm25_9hr_overlap (train_data):
    train_feat=[]
    train_label=[]
    index_tabel=0
    for i in range (12):
        print(i)
        for j in range (471):
            train_feat.append([])
            
            for k in range (j,j+9):
            
                train_feat[index_tabel].append(train_data[i][k])
            train_feat[index_tabel].append(1)
            train_label.append(train_data[i][j+9])
            index_tabel+=1
    return train_feat ,train_label

def train_dataset_pm25_9hr_overlap_square (train_data):
    train_feat=[]
    train_label=[]
    index_tabel=0
    for i in range (12):
        print(i)
        for j in range (471):
            train_feat.append([])
            
            for k in range (j,j+9):       
                train_feat[index_tabel].append(train_data[i][k])
                train_feat[index_tabel].append(train_data[i][k]**2)
            train_feat[index_tabel].append(1)
            train_label.append(train_data[i][j+9])
            index_tabel+=1
    return train_feat ,train_label



def train_all_feat_9hr(train_data):
    train_feat=[]
    train_label=[]
    index_l=0
    for i in range (12):
        index=0

        for j in range (48):
            train_feat.append([])
            for l in range(9):
                for k in range (18):
                    train_feat[index_l].append(train_data[i][index][k])
                index+=1
            train_feat[index_l].append(1)   
            train_label.append(train_data[i][index][9])
            index+=1
            index_l+=1
    return train_feat,train_label

def train_all_feat_9hr_overlap(train_data):
    train_feat=[]
    train_label=[]
    index_l=0
    for i in range (12):
        for j in range (471):
            train_feat.append([])
            for l in range(j,j+9):
                for k in range (18):
                    train_feat[index_l].append(train_data[i][j][k])
            train_feat[index_l].append(1)   
            train_label.append(train_data[i][j+9][9])
            index_l+=1
    return train_feat,train_label


def create_test_submission_2d(test,y):

    w1=y[0]
    w2=y[1]
    b=y[2]
    test_feat_1= []
    test_feat_2= []
    test_label=[]
    test_title =[]
    row=9 
    for i in range(260):
        test_feat_1.append(float(test[9][row]))
        test_feat_2.append(float(test[10][row]))
        
        row+=18

    
    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=test_feat_1[i]*w1+test_feat_2[i]*w2+b
        test_label.append(temp[0,0])

    df =pd.DataFrame(test_label,test_title)
   
    df.to_csv("submit_w1.csv",header=False)
    return df

def create_test_submission_xd(test,y):
    
    feat_num = len(y)-1
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=9 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            test_feat[i].append(float(test[j][row]))
        row+=18


    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num):
            temp+=y[j]*test_feat[i][j]
        temp+=y[9]
        test_label.append(temp[0,0])

    df =pd.DataFrame(test_label,test_title)
   
    df.to_csv("submit_9feat.csv",header=False)
    return df 

def create_test_submission_allfeat(test,y):
    
    feat_num = len(y)-1
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=9 

    ###create test feat
    row=0
    for i in range(260):
        test_feat.append([])
        for j in range (2,11):
            for k in range (18):
                if(test[j][row]=='NR'):
                    test_feat[i].append(0)
                else:
                    test_feat[i].append(float(test[j][row]))
                row+=1
            row-=18
        row+=18
    print(len(test_feat[1]))

    ###create answer file
    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num):
            temp+=y[j]*test_feat[i][j]
        temp+=y[feat_num]
        test_label.append(temp[0,0])

    df =pd.DataFrame(test_label,test_title)
   
    df.to_csv("submit_allfeat_overlap.csv",header=False)
    return df 


def create_test_submission_xd_use_model(test,y):
    
    feat_num = len(y)-1
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=9 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            test_feat[i].append(float(test[j][row]))
        row+=18


    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num):
            temp+=y[j]*test_feat[i][j]
        temp+=y[9]
        test_label.append(temp[0])

    df =pd.DataFrame(test_label,test_title)
   
    return df 


def create_test_submission_pm25_square(test,y):
    
    feat_num = (len(y)-1)//2
    print(feat_num)
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=9 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            test_feat[i].append(float(test[j][row]))
            test_feat[i].append(float(test[j][row])**2)

        row+=18


    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num*2):
            temp+=y[j]*test_feat[i][j]
        temp+=y[18]
        test_label.append(temp[0,0])

    df =pd.DataFrame(test_label,test_title)
   
    df.to_csv("submit_9feat_square.csv",header=False)
    return df 

def train_dataset_some_feat (train_data) :
    tf =[]
    tl =[]
    pm25_mean=0
    pm25_mean_num=0
    pm10_mean=0
    pm10_mean_num=0
    co_mean=0
    co_mean_num=0
    nox_mean =0
    nox_mean_num =0
    so2_mean=0
    so2_mean_num =0
    ##calculate mean
    for i in range (12):
        for j in range (20):
            if(train_data[i][j][2]>0):
                co_mean+=train_data[i][j][2]
                co_mean_num+=1
            if(train_data[i][j][8]>0):
                pm10_mean+=train_data[i][j][8]
                pm10_mean_num+=1
            if(train_data[i][j][9]>0 and train_data[i][j][9]<200):
                pm25_mean+=train_data[i][j][9]
                pm25_mean_num+=1
            ##another two parameters
            if(train_data[i][j][6]>0):
                nox_mean+=train_data[i][j][6]
                nox_mean_num+=1
            if(train_data[i][j][12]>0):
                so2_mean+=train_data[i][j][12]
                so2_mean_num+=1
    co_mean/=co_mean_num
    pm10_mean/=pm10_mean_num
    pm25_mean/=pm25_mean_num
    nox_mean/=nox_mean_num
    so2_mean/=so2_mean_num 
    #print(co_mean,pm10_mean,pm25_mean,nox_mean,so2_mean)
    ####replace bad data by mean
    for i in range (12):
        for j in range (20):
            if(train_data[i][j][2]<=0):
                train_data[i][j][2]=co_mean
            if(train_data[i][j][8]<=0):
                train_data[i][j][8]=pm10_mean
            if(train_data[i][j][9]<=0 or train_data[i][j][9]>=200):
                train_data[i][j][9]=pm25_mean
            if(train_data[i][j][6]<=0):
                train_data[i][j][6]=nox_mean
            if(train_data[i][j][12]<=0):
                train_data[i][j][12]=so2_mean
    index=0
    for i in range (12):
        for j in range (471):
            tf.append([])
            tl.append([])
            for k in range(j,j+9):
                tf[index].append(train_data[i][k][2])
                tf[index].append(train_data[i][k][8])
                tf[index].append(train_data[i][k][9])
                #tf[index].append(train_data[i][k][6])
                #tf[index].append(train_data[i][k][12])
            tf[index].append(1)
            tl[index].append(train_data[i][j+9][9])
            index+=1
    return tf,tl

def create_test_submission_somefeat(test,y):
    feat_num = (len(y)-1)//3
  
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=2 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            if(float(test[j][row])>0):
                test_feat[i].append(float(test[j][row]))
            else:
                test_feat[i].append(0.547)
            if(float(test[j][row+6])>0):
                test_feat[i].append(float(test[j][row+6]))
            else:
                test_feat[i].append(64.4)
            if(float(test[j][row+7])>0 and float(test[j][row+7])<200):
                test_feat[i].append(float(test[j][row+7]))
            else:
                test_feat[i].append(36.5)
            '''
            if(float(test[j][row+4])>0):
                test_feat[i].append(float(test[j][row+4]))
            else:
                test_feat[i].append(20.8)
            if(float(test[j][row+10])>0):
                test_feat[i].append(float(test[j][row+10]))
            else:
                test_feat[i].append(3.1)
            '''

        row+=18


    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num*3):
            temp+=y[j]*test_feat[i][j]
        temp+=y[27]
        test_label.append(temp[0,0])

    df =pd.DataFrame(test_label,test_title)
   
    df.to_csv("3feat.csv",header=False)
    return df 
def create_test_submission_somefeat_use_model(test,y):

    feat_num = (len(y)-1)//3
  
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=2 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            if(float(test[j][row])>0):
                test_feat[i].append(float(test[j][row]))
            else:
                test_feat[i].append(0.547)
            if(float(test[j][row+6])>0):
                test_feat[i].append(float(test[j][row+6]))
            else:
                test_feat[i].append(64.4)
            if(float(test[j][row+7])>0 and float(test[j][row+7])<200):
                test_feat[i].append(float(test[j][row+7]))
            else:
                test_feat[i].append(36.5)
            

        row+=18


    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num*3):
            temp+=y[j]*test_feat[i][j]
        temp+=y[27]
        test_label.append(temp[0])

    df =pd.DataFrame(test_label,test_title)
   
    
    return df 

def create_test_submission_somefeat_use_model_int(test,y):

    feat_num = (len(y)-1)//3
  
    test_feat  =[]
    test_label =[]
    test_title =[]
    row=2 

    ###create test feat
    
    for i in range(260):
        test_feat.append([])
        for j in range (11-feat_num,11):
            if(float(test[j][row])>0):
                test_feat[i].append(float(test[j][row]))
            else:
                test_feat[i].append(0.547)
            if(float(test[j][row+6])>0):
                test_feat[i].append(float(test[j][row+6]))
            else:
                test_feat[i].append(64.4)
            if(float(test[j][row+7])>0 and float(test[j][row+7])<200):
                test_feat[i].append(float(test[j][row+7]))
            else:
                test_feat[i].append(36.5)
        row+=18
    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        temp=0
        for j in range (feat_num*3):
            temp+=y[j]*test_feat[i][j]
        temp+=y[27]
        if(temp[0]-int(temp[0])>0.5):
            test_label.append(int(temp[0])+1)
        else:
            test_label.append(int(temp[0]))

    df =pd.DataFrame(test_label,test_title)
    return df 

def create_test_submission_with_lam(test,w,b,lam):
    x=1
    test_feat= []
    test_label=[]
    test_title =[]
    row=9
    for i in range(260):
        pm25_mean=0
        for j in range(11-x,11):
            pm25_mean+=float(test[j][row])
            test_feat.append(pm25_mean/x)
        row+=18
    test_title.append("id")
    test_label.append("value")
    for i in range (260):
        test_title.append("id_"+str(i))
        test_label.append(test_feat[i]*w+b)

    df =pd.DataFrame(test_label,test_title)
    df.to_csv("submit_lambda:"+str(lam)+".csv",header=False)
    return df 

