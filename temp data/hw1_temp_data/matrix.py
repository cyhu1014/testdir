import math
import numpy as np
import matplotlib.pyplot as plt


x=np.matrix([[2,5,5],[3,1,6]])
t=np.matrix([0,10,5])
A = np.matrix([[2,0,0],[0,1,0],[0,0,3]])
xT =x.getT()
tT=t.getT()
B=x*A*xT
B =np.linalg.inv(B)
print(B)
B=B*x*A*tT
print(B)