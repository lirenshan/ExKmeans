from embed import terminal_embed
from ExTree3 import ExTree
from sklearn.cluster import KMeans

def ExKmeans(X, k, C = None, numr = 10):

	if C is None:
		print('Finding k-means')
		kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
		print(kmeans.inertia_)

	data,center = terminal_embed(X,C)


	tree = ExTree(k=k) 
	threshold_tree = tree.fit(data, center,X,C)
	cost=threshold_tree.score
	for i in range(0, numr):
		tree = ExTree(k=k) 
		threshold_tree = tree.fit(data, center ,X,C)
		#print(threshold_tree.score)
		if threshold_tree.score < cost:
			cost = threshold_tree.score
	# print(cost)
	return cost


# k = 1000
# d = 300*round(np.log(k))
# eps = 3*np.log(k)/k

# X,C = cube_random(k,d,eps)

# ExKmeans(X,k,C,numr = 10)