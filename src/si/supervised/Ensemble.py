from .Modelo import Model
import numpy as np


def majority(values):
    return max(set(values), key=values.count)  # retorna o value que aparece mais vezes


def average(values):
    return sum(values)/len(values)


class Ensemble(Model):
    def __init__(self, models, fvote, score):
        super().__init__()
        self.models = models  # modelos já inicializados
        self.fvote = fvote  # função para ranking das predictions
        self.score = score

    def fit(self, dataset):  # vai fazer fit dos modelos
        self.dataset = dataset
        for model in self.models:
            model.fit(dataset)  # faz o fitting do dataset para todos os modelos dados
        self.is_fited = True

    def predict(self, x):
        assert self.is_fited
        preds = []
        for model in self.models:
            preds.append(model.predict(x))  # faz o predict para cada modelo
        vote = self.fvote(preds)  # retorna o maior valor
        return vote

    def cost(self, X=None, Y=None):
        X = X if X is not None else self.dataset.X
        Y = Y if Y is not None else self.dataset.Y
        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(Y, y_pred)
