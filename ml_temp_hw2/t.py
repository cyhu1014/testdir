#test

import numpy as np

x = [[1,5]]
x = np.matrix(x)
a = [[1,2],[3,4]]


print(a+x.getT()*x)