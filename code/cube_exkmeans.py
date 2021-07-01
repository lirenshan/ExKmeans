from ExKMC.Tree import Tree
from cubedata import cube_random
from embed import terminal_embed
import numpy as np
from ExKmeans import ExKmeans
from ExTree import ExTree
from sklearn.cluster import KMeans

# Create dataset

k = 1000
d = 300*round(np.log(k))

eps = 3*np.log(k)/k

X,C = cube_random(k,d,eps)


# ExKMC

tree = Tree(k=k, max_leaves=k) 

# Construct the tree, and return cluster labels
prediction = tree.fit_predict(X)

print(tree.score(X))
# Tree plot saved to filename
# tree.plot('result')



#our algorithm

cost = ExKmeans(X,k,C,numr = 10)
print(cost)


#data,center = terminal_embed(X,C)

# cost = 0
# for i in range(0,k):
# 	cost = cost + np.abs(data[2*i]-center[i]).sum()
# 	cost = cost + np.abs(data[2*i+1]-center[i]).sum()

# print(cost)


# kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
# print(kmeans.inertia_)

# kmedians = KMedians(k=k)
# kmedians.fit(data)

# print(kmedians.cost)
#print(kmedians.cluster_centers_)

# tree = ExTree(k=k) 
# threshold_tree = tree.fit(data, center,X,C)
# cost=threshold_tree.score
# for i in range(0,10):
# 	tree = ExTree(k=k) 
# 	threshold_tree = tree.fit(data, center ,X,C)
# 	#print(threshold_tree.score)
# 	if threshold_tree.score < cost:
# 		cost = threshold_tree.score
# print(cost)

