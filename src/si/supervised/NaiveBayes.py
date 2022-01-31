import numpy as np
import pandas as pd
from .Modelo import Model


class NaiveBayes(Model):
    def __init__(self):
        super().__init__()
        self.mean = 0
        self.sd = 0
        self.sumario = []
        self.by_label = {}

    def sep_by_class(self, dataset):
        X, y = dataset.getXy()
        for row in range(len(y)):
            if y[row] not in self.by_label:
                self.by_label[y[row]] = []
            self.by_label[y[row]].append(X[row])
        return self.by_label

    def mean_sd(self, dataset):
        from src.si.util.scale import StandardScaler
        scaler = StandardScaler()
        info = scaler.fit(dataset)
        self.mean = scaler.mean
        self.sd = np.sqrt(scaler.var)
        for m, sd in zip(self.mean, self.sd):
            self.sumario.append((m, sd))
        return self.sumario
