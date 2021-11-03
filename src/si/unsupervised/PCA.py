import numpy as np
from src.si.util.scale import StandardScaler


class PCA:
    def __init__(self, num_components=2):
        self.numcomps = num_components

    def fit(self, dataset):  # objeto Dataset
        pass

    def transform(self, dataset):  # objeto Dataset
        x_scaled = StandardScaler.fit_transform(dataset)  # standardização dos dados usando o StandardScaler

        matriz_cov = np.cov(x_scaled, rowvar=False)  #

        eigen_values, eigen_vectors = np.linalg.eigh(matriz_cov)

        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]  # np.argsort returns an array of indices of the same shape.
        sorted_eigenvalue = eigen_values[sorted_index]
        # similarly sort the eigenvectors
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        # select the first n eigenvectors, n is desired dimension of our final reduced data.
        # you can select any number of components.
        eigenvector_subset = sorted_eigenvectors[:, 0:self.numcomps]

        x_reduced = np.dot(eigenvector_subset.transpose(), x_scaled.transpose()).transpose()
        return x_reduced
