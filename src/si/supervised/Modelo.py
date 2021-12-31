from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self.is_fited = False

    @abstractmethod
    def fit(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    def cost(self):
        raise NotImplementedError
