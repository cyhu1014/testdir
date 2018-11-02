##ntu-machine learning hw2
##predict 
import pandas as pd
import numpy  as np
import math
import sys
import hw2_func as func

def main(args):
    x=sys.argv[4]
    w=np.load("best.npy")
    test = pd.read_csv(sys.argv[3],header=None)
    test = func.create_train_dataset(test)
    test= func.feat_process(test)
    func.predict (test,w,x)
    
if __name__ == '__main__':
    main(sys.argv)