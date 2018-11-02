##predict GD

import numpy  as np
import math
import sys
import hw2_func_GD as func
import pandas as pd
def main(args):
    test = pd.read_csv(sys.argv[3],header=None)
    w = np.load("GD.npy")
    test = func.create_train_dataset(test)
    test = func.feat_process_GD(test)
    func.predict (test,w,sys.argv[4])

if __name__ == '__main__':
    main(sys.argv)