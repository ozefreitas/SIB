import numpy as np
from copy import copy
import warnings
from ..data import Dataset
import scipy.stats as stats


class VarianceThreshold:
    def __init__(self, threshold=0):
        """
        the variance threshold is a simple baseline approach to feature selection
        it removes all features which variance doesn't meet some threshold limit
        it removes all zero-variance features, i.e..
        """
        self.var = None
        if threshold < 0:
            raise Exception('Threshold must be a non negative value')
        else:
            self.threshold = threshold

    def fit(self, dataset):  # guarda a variância das colunas num vetor
        X = dataset.X
        self.var = np.var(X, axis=0)

    def transform(self, dataset, inline=False):
        X = dataset.X
        cond = self.var > self.threshold  # vetor de booleanos, onde a variancia for maior que o threshold definido, será True
        ind = []  # lista que irá ter os indices das features que têm uma variância superior ao threshold
        for i in range(len(cond)):
            if cond[i]:
                ind.append(i)
        X_trans = X[:, ind]  # vai buscar todas as linhas, mas apenas as colunas onde se verifica a condição
        xnames = [dataset._xnames[i] for i in ind]
        if inline:
            dataset.X = X_trans
            dataset._xnames = xnames
            return dataset
        else:
            return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))

    def fit_transform(self, dataset, inline=False):  # faz tod o processo de verificação da variância e eliminação de features
        self.fit(dataset)
        return self.transform(dataset, inline)


class SelectKBest:
    def __init__(self, k, funcao_score = "f_regress"):
        self.feat_num = k
        self.function = funcao_score

    def fit(self, dataset):
        pass

    def transform(self, dataset):
        pass

    def fit_transform(self, dataset):
        pass


def f_regress(self):
    pass
