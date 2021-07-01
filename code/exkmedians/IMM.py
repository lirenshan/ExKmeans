import ExKMC
from ExKMC.Tree import Tree
from ExKMC.Tree import convert_input
import numpy as np

class IMM_Tree(Tree):

    def surrogate_score(self, x_data):
        x_data = convert_input(x_data)
        clusters = self.predict(x_data)
        cost = 0
        for c in range(self.k):
            cluster_data = x_data[clusters == c, :]
            if cluster_data.shape[0] > 0:
                center = self.all_centers[c]
                cost += np.abs(cluster_data - center).sum()
        return cost

    


    # def fit(self, x_data, kmeans=None):
    #     x_data = convert_input(x_data)

    #     if kmeans is None:
    #         if self.verbose > 0:
    #             print('Finding %d-means' % self.k)
    #         kmeans = KMeans(self.k, n_jobs=self.n_jobs, verbose=self.verbose, n_init=1, max_iter=40)
    #         kmeans.fit(x_data)
    #     else:
    #         assert kmeans.n_clusters == self.k

    #     #y = np.array(kmeans.predict(x_data), dtype=np.int32)
    #     y = np.array(kmeans.labels_, dtype=np.int32)

    #     self.all_centers = np.array(kmeans.cluster_centers_, dtype=np.float64)

    #     if self.base_tree == "IMM":
    #         self.tree = self._build_tree(x_data, y,
    #                                      np.ones(self.all_centers.shape[0], dtype=np.int32),
    #                                      np.ones(self.all_centers.shape[1], dtype=np.int32))
    #         leaves = self.k
    #     else:
    #         self.tree = Node()
    #         self.tree.value = 0
    #         leaves = 1

    #     if self.max_leaves > leaves:
    #         self.__gather_leaves_data__(self.tree, x_data, y)
    #         all_centers_norm_sqr = (np.linalg.norm(self.all_centers, axis=1) ** 2).astype(np.float64, copy=False)
    #         self.__expand_tree__(leaves, all_centers_norm_sqr)
    #         if self.light:
    #             self._leaves_data = {}

    #     self._feature_importance = np.zeros(x_data.shape[1])
    #     self.__fill_stats__(self.tree, x_data, y)

    #     return self
