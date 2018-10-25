###for ntu_ml 2018 hw1
import math
import numpy as np
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

def best_function_2d(tf,tl):
    ###  best function
    x=np.matrix(tf)
    y=np.matrix(tl)
    xT = np.matrix.getT(x)
    yT = np.matrix.getT(y)
    A=xT*x
    invA=np.matrix.getI(A)
    y=invA*xT*yT
    return y    