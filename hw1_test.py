###hw1_for test

import sys
import math
import pandas as pd
import numpy as np
import data_process
import func

y=np.load(sys.argv[1])


test   = pd.read_csv(sys.argv[2],encoding="big5" ,header=None)
df = data_process.create_test_submission_somefeat_use_model(test,y)
df.to_csv(sys.argv[3],header=False)
