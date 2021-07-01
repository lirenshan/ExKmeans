#from kmedians import KMeans
from kmedians import KMedians
from ExTree4 import ExTree
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits, load_wine
import random
from IMM import IMM_Tree
from ExKMC.Tree import Tree


random.seed()

#data, target = load_breast_cancer(return_X_y = True)
#data, target = load_digits(return_X_y = True)
data, target = load_wine(return_X_y = True)
k = 20

kmedians = KMedians(k=k)
kmedians.fit(data)

print("regular kmedians:(kmedians++)")
print(kmedians.cost)
#print(kmedians.cluster_centers_)


print("our algorithm:")
tree = ExTree(k=k) 
threshold_tree = tree.fit(data, kmedians)
cost=threshold_tree.score
for i in range(0,1000):
	tree = ExTree(k=k) 
	threshold_tree = tree.fit(data, kmedians)
	#print(threshold_tree.score)
	if threshold_tree.score < cost:
		cost = threshold_tree.score
print(cost)


print("IMM:")
IMM_tree = IMM_Tree(k=k, max_leaves=k) 

# Construct the tree, and return cluster labels
prediction = IMM_tree.fit_predict(data)

print(IMM_tree.surrogate_score(data))

# tree = Tree(k=k, max_leaves=k) 

# # Construct the tree, and return cluster labels
# prediction = tree.fit_predict(data)

# print(tree.score(data))