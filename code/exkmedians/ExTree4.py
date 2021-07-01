from kmedians import KMedians
import numpy as np
import random
from sympy import Interval, Union
from scipy.spatial import distance


class Node:
    def __init__(self, dim=None, val=None, valid_centers=None):
        self.left = None
        self.right = None
        self.dim = dim
        self.val = val
        self.valid_centers = valid_centers

class ExTree:

    def __init__(self, k, verbose=0, light=True, n_jobs=None):
        #Constructor for explainable k-medians tree.
        #:param k: Number of clusters.
        #:param verbose: Verbosity mode.
        #:param light: If False, the object will store a copy of the input examples associated with each leaf.
        #:param n_jobs: The number of jobs to run in parallel.
        self.k=k
        self.tree=None
        self._leaves_data=[]
        self.score=None
        self.cluster_centers = None
        self.all_centers = None
        self.verbose=verbose
        self.light=light
        self.n_jobs=n_jobs if n_jobs is not None else 1
        self._feature_importance = None

    def sample_coordinate(self,cut_range,total_range,d):
        t = random.uniform(0,total_range)
        tot = 0
        for i in range(0,d):
            if tot + cut_range[i][1] - cut_range[i][0] >= t:
                return i
            tot = tot + cut_range[i][1] - cut_range[i][0]

    def split(self, node):
        d = np.shape(self.all_centers)[1]
        intervals = []
        total_range = 0
        
        for j in range(0,d):
            c_ind = node.valid_centers[0]
            left = self.all_centers[c_ind][j]
            right = self.all_centers[c_ind][j]
            for i in node.valid_centers:
                if self.all_centers[i][j] < left:
                    left = self.all_centers[i][j]
                if self.all_centers[i][j] > right:
                    right = self.all_centers[i][j]
            intervals.append([left,right])
            total_range = total_range + right - left

        d_ind = self.sample_coordinate(intervals,total_range,d)

        cut = random.uniform(intervals[d_ind][0],intervals[d_ind][1])
        
        #if (cut == None): return 
        left_centers = []
        right_centers = []
        for i in node.valid_centers:
            if self.all_centers[i][d_ind] <= cut:
                left_centers.append(i) 
            if self.all_centers[i][d_ind] > cut:
                right_centers.append(i)
        left_child = Node(valid_centers = left_centers)
        right_child = Node(valid_centers = right_centers)
        node.left = left_child
        node.right = right_child
        node.dim = d_ind
        node.val = cut

        if len(left_centers) > 1:
            self.split(left_child)
        if len(right_centers) > 1:
            self.split(right_child)
        # self._leaves_data.remove(node)
        # self._leaves_data.append(left_child)
        # self._leaves_data.append(right_child)



    def _build_tree(self, centers):
       
        # Build a tree.
        # :param x_data: The input samples.
        # :param y: Clusters of the input samples, according to the kmeans classifier given (or trained) by fit method.
        # :param valid_centers: Boolean array specifying which centers should be considered for the tree creation.
        # :param valid_cols: Boolean array specifying which columns should be considered fot the tree creation.
        # :return: The root of the created tree.
        
        #if self.verbose > 1:
        #    print('build node (samples=%d)' % x_data.shape[0])
        centers_index = list(range(0,self.k))
        root = Node(valid_centers = centers_index)
       
        self._leaves_data.append(root)

        self.split(root)

        return root


    def _cluster_(self,x_data):
        n = x_data.shape[0]
        d = x_data.shape[1]
        cost = 0
        self.cluster_centers=np.zeros(n)
        for i in range(0,n-1):
            node = self.tree
            while (node.left != None):
                if x_data[i][node.dim] < node.val:
                    node = node.left
                else:
                    node = node.right
            c_ind = node.valid_centers[0]
            self.cluster_centers[i] = c_ind
            cost = cost + np.abs(x_data[i]-self.all_centers[c_ind]).sum()

        self.score = cost


    def fit(self, x_data, kmedians=None):
        """
        Build a threshold tree from the training set x_data.
        :param x_data: The training input samples.
        :param kmeans: Trained model of k-means clustering over the training data.
        :return: Fitted threshold tree.
        """

        #x_data = convert_input(x_data)

        #Compute the reference centers with kmedians++
        random.seed()

        if kmedians is None:
            print('Finding %d-means' % self.k)
            kmedians = KMedians(self.k)
            kmedians.fit(x_data)

        #y = np.array(kmeans.predict(x_data), dtype=np.int32)

        self.all_centers = np.array(kmedians.cluster_centers_, dtype=np.float64)

        self.tree = self._build_tree(self.all_centers)
        self._cluster_(x_data)
        

        return self