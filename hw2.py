##ntu-machine learning hw2
##predict 
import pandas as pd
import numpy  as np
import math
import sys
import hw2_func as func
import hw2_data_process as dp
#import matplotlib.pyplot as plt  
def main(args):
    
    tf = pd.read_csv(sys.argv[1],header=None)
    tl = pd.read_csv(sys.argv[2],dtype=int)
    feat = dp.create_train_dataset(tf)
    Y    = dp.create_train_label(tl)
    feat = func.feature_normalize(feat)
    w= func.train_3(feat,Y)
    
    '''
    test = pd.read_csv(sys.argv[3],header=None)
    test = dp.create_train_dataset(test)
    test = func.feature_normalize(test)
    func.predict (test,w,b)
    '''
    '''
    (w,b,cel)=func.train(feat,Y,20000,10000)
    plt.plot(cel)
    plt.savefig("crossentropy.jpg")
    '''
if __name__ == '__main__':
    main(sys.argv)