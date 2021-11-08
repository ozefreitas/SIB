# k = z
# gerar obrigatoriamente k centroides
# Enquanto os clustaers mudam ou até ser atingido o numero máximo de iterações
#   distancia de cada ponto a cada centroide
#   identificar para cada ponto o centroide mais proximo
#   definir os clusters e calcular os novos centroides

# 1º - ou centralizar: X.np.mean(k,) ou usar o standardScaler

from src.si.util.util import euclidean, manhattan
from src.si.data import Dataset
import numpy as np


class KMeans:
    def __init__(self, k: int, max_iterations=100, measure="euclidean"):
        self.k = k
        self.n = max_iterations
        self.centroides = None
        self.measure = measure

    #   def distance_12(self, x, y):
    #       """
    #       Distancia euclidiana distance
    #       :param x:
    #       :param y:
    #       :return:
    #       """
    #       dist = np.absolute((x - y) ** 2).sum(axis=1)
    #       return dist

    def fit(self, dataset):
        self.min = np.min(dataset.X, axis=0)  # fazer a média em relação às features
        self.max = np.max(dataset.X, axis=0)

    def init_centroids(self, dataset):
        x = dataset.X
        centroides = []
        for i in range(x.shape[1]):
            centroides.append(np.random.uniform(low=self.min[i], high=self.max[i], size=(self.k,)))
        self.centroides = np.array(centroides).transpose()

    def get_closest_centroid(self, x):
        if self.measure is "euclidean":
            dist = euclidean(x, self.centroides)
            closest_centroid_index = np.argmin(dist, axis=0)
        else:
            dist = manhattan(x, self.centroides)
            closest_centroid_index = np.argmin(dist, axis=0)
        return closest_centroid_index

    def transform(self, dataset):
        self.init_centroids(dataset)
        x = dataset.X
        changed = True
        count = 0
        old_idxs = np.zeros(x.shape[0])
        while changed is True or count < self.n:
            idxs = np.apply_along_axis(self.get_closest_centroid, axis=0, arr=x.transpose())
            cent = [np.mean(x[idxs == i]) for i in range(x.shape[0])]
            self.centroides = np.array(cent)
            changed = np.all(old_idxs == idxs)
            old_idxs = idxs
            count += 1
        return self.centroides, old_idxs

    def fit_transform(self, dataset):
        self.fit(dataset)
        centroides, indices = self.transform(dataset)
        return centroides, indices
