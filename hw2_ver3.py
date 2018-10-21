import sys
import numpy as np
from math import log, floor

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.99999999999999)

def load_data():
	X_train = np.delete(np.genfromtxt(sys.argv[1], delimiter=','), 0, 0)
	Y_train = np.genfromtxt(sys.argv[2], delimiter=',')
	X_test = np.delete(np.genfromtxt(sys.argv[3], delimiter=','), 0, 0)
	return X_train, Y_train, X_test

def main():
	X_train, Y_train, X_test = load_data()
    '''
	X_train_normed, X_test_normed = feature_normalize(X_train, X_test)
	w, b = train(X_train_normed, Y_train)
	predict(w, b, X_test_normed)
	return
    '''
if __name__ == '__main__':
	main()