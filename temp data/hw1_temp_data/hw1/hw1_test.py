###hw1_for test

import sys
import math
import pandas as pd
import numpy as np
import data_process
import func
x=np.load(sys.argv[1])
y=np.load(sys.argv[2])
(w,b)=func.best_function_use_model(x,y)

test   = pd.read_csv(sys.argv[3],encoding="big5" ,header=None)
df = data_process.create_test_submission(test,w,b)
df.to_csv(sys.argv[4],header=False)