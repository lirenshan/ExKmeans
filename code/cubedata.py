import numpy as np
import math
import random

random.seed()
np.random.seed()

def positive_or_negative():
    if random.random() < 0.5:
        return 1
    else:
        return -1

def cube_random(k , d, eps):
	m = 2
	C = np.random.rand(k,d)
	X = np.ndarray(shape=(k*m,d), dtype=float)
	t = 0
	for i in range(0,k):
		for l in range(0,d):
			X[t][l] = C[i][l] + eps
			X[t+1][l] = C[i][l] - eps
		t = t+2
	return X,C

# def cube_random(k , d, eps):
# 	m = 10*d
# 	C = np.random.rand(k,d)
# 	X = np.ndarray(shape=(k*m,d), dtype=float)
# 	t = 0
# 	for i in range(0,k):
# 		for j in range(0,m):
# 			for l in range(0,d):
# 				X[t][l] = C[i][l] + positive_or_negative()*eps
# 			t = t+1
# 	return X


# X,C = cube_random(1,2,0.1)
# print(C)
# print(" ")
# print(X)



