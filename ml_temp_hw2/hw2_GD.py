##ntu-machine learning hw2
##predict GD

import numpy  as np
import math
import sys
import hw2_func as func
import pandas as pd
def main(args):
    tf = pd.read_csv(sys.argv[1],header=None)
    tl = pd.read_csv(sys.argv[2],dtype=int)
    feat = func.create_train_dataset(tf)
    Y    = func.create_train_label(tl)
    feat = func.feat_expand(feat)
    feat = func.feat_onehot(feat)
    feat = func.feature_normalize_mean_covariance(feat)
    w=func.train_GD(feat,Y)

    test = pd.read_csv(sys.argv[3],header=None)
    test = func.create_train_dataset(test)
    test = func.feat_expand(test)
    test = func.feat_onehot(test)
    test = func.feature_normalize_mean_covariance(test)
    func.predict (test,w,"Gaussion_o_e.csv")

if __name__ == '__main__':
    main(sys.argv)