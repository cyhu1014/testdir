##ntu-machine learning hw2
##predict GD

import numpy  as np
import math
import sys
import hw2_func_GD as func
import pandas as pd
def main(args):
    tf = pd.read_csv(sys.argv[1],header=None)
    tl = pd.read_csv(sys.argv[2],dtype=int)
    feat = func.create_train_dataset(tf)
    Y    = func.create_train_label(tl)
    feat = func.feat_process_GD(feat)
    w=func.train_GD(feat,Y)
    
    test = pd.read_csv(sys.argv[3],header=None)
    test = func.create_train_dataset(test)
    test = func.feat_process_GD(test)
    func.predict (test,w,sys.argv[4])

if __name__ == '__main__':
    main(sys.argv)