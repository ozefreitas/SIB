from .Modelo import Model
from src.si.util.util import euclidean
from src.si.util.metrics import accuracy_score
import numpy as np


class KNN(Model):

    def __init__(self, num_neigbours, classification=True):  # numero de pontos aos quais serão calculadas as distancias
        super().__init__()
        self.k = num_neigbours
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fited = True

    def get_neigbours(self, x):  # calcula as distancias entre cada ponto de teste em relação a todos os pontos do dataset
        # de treino
        dist = euclidean(x, self.dataset.X)
        indx_sort = np.argsort(dist)
        return indx_sort[:self.k]

    def predict(self, x):
        """

        :param x: array de teste
        :return: predicted labels
        """
        neigbours = self.get_neigbours(x)  # vizinhos mais proximos dos pontos do dataset de teste
        values = self.dataset.Y[neigbours].tolist()  # lista
        if self.classification:
            prediction = max(set(values), key=values.count)
        else:
            prediction = sum(values)/len(values)
        return prediction

    def cost(self):
        pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.transpose())
        return accuracy_score(pred, self.dataset.Y)
