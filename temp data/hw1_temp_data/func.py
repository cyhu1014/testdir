###for ntu_ml 2018 hw1
import math
import numpy as np
import matplotlib.pyplot as plt

def validation (tl,tf,w,b):
    loss_train=0
    loss_val=0
    length=len(tl)*9//10
    num_data=len(tl)-length
    for i in range (length):
        y=w*tf[i]+b
        loss_train+=(y-tl[i])**2
    loss_train=math.sqrt(loss_train/length)
    for i in range (length,len(tl)):
        y=w*tf[i]+b
        loss_val+=(y-tl[i])**2
    loss_val=math.sqrt(loss_val/num_data)
    return loss_train,loss_val

def best_function(tf,tl):
    ###  best function
    length = len(tf)*9//10
    x=np.array(tf[:length])
    y=np.array(tl[:length])
    A = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(A, y)[0]
    print(w,",", b)
    print(validation(tl,tf,w,b))
    ###delete when done
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o', label='Original data', markersize=1)
    plt.plot(x, w*x + b, 'r', label='Fitted line')
    plt.legend()
    plt.savefig("figure.png")
    #####
    return w,b

def best_function_use_model(x,y):
    
    
    A = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(A, y)[0]
    print(w,",", b)

    ###delete when done
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o', label='Original data', markersize=1)
    plt.plot(x, w*x + b, 'r', label='Fitted line')
    plt.legend()
    plt.savefig("figure1.png")
    #####
    return w,b

def best_function_xd(tf,tl):
    ###  best function
    x=np.matrix(tf)
    
    y=np.matrix(tl)
    
    xT = np.matrix.getT(x)

    yT = np.matrix.getT(y)

    A=xT*x
    
    invA=np.linalg.pinv(A)
    y=invA*xT*y
    return y    

def loss_function (tf,tl,iteration):
    p_set = []
    lr=0.0000001
    loss_change =[]
    for i in range (0,len(tf[0])):
        p_set.append(1)
    for i in range (iteration):
        (p_set,loss)=lf_2(tf,tl,p_set,lr)
        loss = math.sqrt(loss)
        loss_change.append(loss)
    plt.plot(loss_change,color="orange",label="loss_train")
    #plt.plot(validation_list,label="loss_val")
    plt.legend()
    plt.savefig('loss.jpg')

def lf(tf,tl,p_set,lr):
    length =len(tf) 
    print(length)
    print((len(p_set)))
    p_len =len(p_set)
    ans = []
    loss = 0
    p_set_change = []
    for i in range (p_len):
        p_set_change.append(0)
    for i in range (length):
        ans.append(0)
        for j in range (p_len):
            ans[i]+=tf[i][j]*p_set[j]
        loss+=(ans[i]-tl[i])**2
    for i in range (length):
        for j in range(p_len-1):
            print(i,j)
            p_set_change[j]+=2*(tl[i]-ans[i])*(-1)*(p_set[j])
        print("ser")
        p_set_change[p_len-1]+=2*(tl[i]-ans[i])*(-1)
    for i in range (len(p_set)):
        p_set_change[i]=p_set_change[i]*(-1)*lr+p_set[i]
    print(p_set_change,loss)
    return p_set_change ,loss

def loss_function_2 (tf,tl,iteration,lr):
    w=3
    b=3

    loss_change =[]
    val_change  =[]
    for i in range (iteration):
        (w,b,loss)=lf_2(tf,tl,w,b,lr)
        
        loss_change.append(loss)         
        loss_val = val (tf,tl,w,b)
        val_change.append(loss_val)

    
    print("w:",w)
    print("b:",b)
    print("loss:",loss)
    print("val",loss_val)
    return w,b,loss_change

def lf_2(tf,tl,w,b,lr):
    length=len(tf)
    length=length*9//10
    y=0
    loss = 0
    w_next=0
    b_next=0
    for i in range (length):
        y=w*tf[i]+b
        loss+=(y-tl[i])**2
        w_next+=2*(tl[i]-y)*(-tl[i])
        b_next+=2*(tl[i]-y)*(-1)
    w_next=w_next*(-1)*lr+w
    b_next=b_next*(-1)*lr+b
    loss = math.sqrt(loss/length)
    
    return (w_next,b_next,loss)

def val (tf,tl,w,b):
    start = len(tf)*9//10
    end   = len(tf)
    loss_val=0
    for i in range (start,end):
        y=w*tf[i]+b
        loss_val+= (y-tl[i])**2
    loss_val = math.sqrt(loss_val/(end-start))

    return loss_val

def loss_function_reg (tf,tl,iteration,lr,reg):
    w=3
    b=3

    loss_change =[]
    val_change  =[]
    for i in range (iteration):
        (w,b,loss)=lf_reg(tf,tl,w,b,lr,reg)
        
        loss_change.append(loss)         
        loss_val = val (tf,tl,w,b)
        val_change.append(loss_val)

    print("lambda: ",reg)
    print("w:",w)
    print("b:",b)
    print("loss:",loss)
    print("val",loss_val)
    return w,b,loss_change,reg

def lf_reg(tf,tl,w,b,lr,reg):
    length=len(tf)
    length=length*9//10
    y=0
    loss = 0
    w_next=0
    b_next=0
    for i in range (length):
        y=w*tf[i]+b
        loss+=(y-tl[i])**2
        w_next+=2*(tl[i]-y)*(-tl[i])
        b_next+=2*(tl[i]-y)*(-1)
    w_next+=reg*(w**2)
    w_next=w_next*(-1)*lr+w
    b_next=b_next*(-1)*lr+b
    loss = math.sqrt(loss/length)
    
    return (w_next,b_next,loss)