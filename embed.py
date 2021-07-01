import numpy as np

def sgn(x):
    if x > 0:
        return 1
    else:
        return -1

def terminal_embed(X,C):
	n = X.shape[0]
	d = X.shape[1]
	k = C.shape[0]
	data = np.ndarray(shape=(n,d), dtype=float)
	center = np.ndarray(shape=(k,d), dtype=float)
	for i in range(0,d):
		y = []
		for j in range(0,k):
			y.append((j,C[j][i]))
		y.sort(key = lambda x: x[1])
		z = np.zeros(k)
		for j in range(0,k-1):
			z[j+1] = z[j]+(y[j+1][1]-y[j][1])*(y[j+1][1]-y[j][1])/2
		for j in range(0,k):
			center[y[j][0]][i] = z[j]

		x = []
		for j in range(0,n):
			x.append((j,X[j][i]))
		x.sort(key = lambda x: x[1])

		t = 0
		for j in range(0,n):
			while (t+1 < k) and (y[t+1][1] < x[j][1]):
				t = t+1
			mindis = abs(x[j][1]-y[t][1])
			if (t+1 < k) and (abs(x[j][1] - y[t+1][1]) < mindis):
				t = t+1
			data[j][i] = center[t][i] + sgn(x[j][1] - y[t][1])*(x[j][1] - y[t][1])*(x[j][1] - y[t][1])

	return data, center


