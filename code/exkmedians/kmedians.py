# Copyright Mathieu Blondel December 2011
# License: BSD 3 clause

import numpy as np
import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeansGood
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets import make_blobs
from random import randint,uniform,seed
 
##############################################################################
# Generate sample data
seed()


# Create dataset
# n = 100
# d = 10
# k = 3
# X, _ = make_blobs(n, d, k, cluster_std=3.0)
# n_clusters = k

class KMeans(BaseEstimator):

    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.n_clusters = k


    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)
        self.cost_list = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).min(axis=1)
        self.cost = np.sum(self.cost_list) 

    def _average(self, X):
        return X.mean(axis=0)

    def _m_step(self, X):
        X_center = None
        for center_id in range(0,self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = self._average(X[center_mask])

    def sample_center(self,n):
        t = uniform(0,self.cost)
        tot = 0
        for i in range(0,n):
            if tot+self.cost_list[i] >= t:
                return i
            tot = tot+self.cost_list[i]

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        #vdata = np.mean(np.var(X, 0))

        # random_state = check_random_state(self.random_state)
        # self.labels_ = random_state.permutation(n_samples)[:self.k]
        # self.cluster_centers_ = X[self.labels_]

        # kmeans++
        c_ind = randint(0,n_samples-1)
        self.cluster_centers_ = []
        self.cluster_centers_.append(X[c_ind])
        self._e_step(X)
        for i in range(1,self.k):
            c_ind = self.sample_center(n_samples)
            self.cluster_centers_.append(X[c_ind])
            self._e_step(X)

        self._e_step(X)

        for i in range(self.max_iter):
            cost_old = self.cost

            self._m_step(X)
            self._e_step(X)

            if (cost_old - self.cost) < self.tol:
                break

        return self

class KMedians(KMeans):

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)
        self.cost_list = manhattan_distances(X, self.cluster_centers_).min(axis=1)
        self.cost = np.sum(self.cost_list) 

    def _average(self, X):
        return np.median(X, axis=0)



# kmeans = KMeans(k=3)
# res = kmeans.fit(X)

# kmedians = KMedians(k=3)
# kmedians.fit(X)

# print(res.cost)