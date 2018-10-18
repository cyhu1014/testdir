import pandas as pd
import numpy  as np
import math
import sys
import hw2_func as func

def create_train_dataset(tf):
    feat=[]

    for i in range (1,len(tf)):
        feat.append([])
        for j in range (23):
            feat[i-1].append(float(tf[j][i]))
    return feat


def create_train_label(tl):
    Y= []
    for i in range(len(tl)):
        Y.append(float(tl['Y'][i]))
    return Y