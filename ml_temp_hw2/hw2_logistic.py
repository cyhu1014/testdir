##ntu-machine learning hw2
##predict 
import pandas as pd
import numpy  as np
import math
import sys
import hw2_func as func


def main(args):
    
    tf = pd.read_csv(sys.argv[1],header=None)
    tl = pd.read_csv(sys.argv[2],dtype=int)
    feat = func.create_train_dataset(tf)
    Y    = func.create_train_label(tl)
    feat= func.feat_process(feat)
    w= func.train_3(feat,Y)
    #np.save("noev",w)
    #w=np.load("best.npy")
    
    test = pd.read_csv(sys.argv[3],header=None)
    test = func.create_train_dataset(test)
    test= func.feat_process(test)
    func.predict (test,w,sys.argv[4])
    
if __name__ == '__main__':
    main(sys.argv)