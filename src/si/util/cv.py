from src.si.util.util import train_test_split, add_intersect
import numpy as np
import itertools
import pandas as pd


class CrossValidation:
    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model  # qual o algoritmo a ser corrido e feito o fit, ja inicializado
        self.dataset = dataset  # dataset completo que vai levar o fit e prever os valores
        self.score = score  # não vai ser usada aqui
        self.cv = kwargs.get('cv', 3)  # parametros retirados da lista kwargs, numero de cv's default será 3
        self.split = kwargs.get('split', 0.8)  # split feito com 80% dos dados para treino como default
        self.train_scores = None
        self.test_scores = None
        self.ds = None

    def run(self):
        train_scores = []  # vai guardar o score do fitting do treino que resulta de cada ciclo de cv
        test_scores = []  # o mesmo para os scores do teste
        ds = []  # guarda ambo datasets divididos de treino e teste na forma de tuplo
        for _ in range(self.cv):
            train, test = train_test_split(self.dataset, self.split)  # faz o split
            ds.append((train, test))  # adiciona os dois na forma de tuplo
            self.model.fit(train)  # faz o fitting de acordo com o modelo escolhido
            train_scores.append(self.model.cost())  # corre a função de custo que cada modelo tem implementada, para o treino
            test_scores.append(self.model.cost(test.X, test.Y))  # faz o mesmo, mas desta vez com os dados do dataset de teste
        self.train_scores = train_scores  # guarda os scores de treino
        self.test_scores = test_scores  # guarda os scores de teste
        self.ds = ds  # guarda ambos datasets
        return train_scores, test_scores

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, 'need to run model'
        return pd.DataFrame({'train Scores': self.train_scores, 'Test Scores': self.test_scores})


class GridSearchCV:

    def __init__(self, model, dataset, parameters, score=None, **kwargs):
        self.model = model  # modelo inicializado à priori
        self.dataset = dataset  # dataset a ser feito implementado o modelo
        self.score = score  # função de score (facultativo)
        hasparam = (hasattr(self.model, param) for param in parameters)  # vê se o modelo que se deu, tem os atributos
        # que se pôs na lista de parametros
        if np.all(hasparam):  # se o modelo tiver todos esses atributos
            self.parameters = parameters  # na forma de dicionário
        else:
            index = hasparam.index(False)
            keys = list(parameters.keys())
            raise ValueError(f'Wrong parameters: {keys[index]}')  # se não tiver, dá raise e diz quais os atributos o modelo não possui
        self.kwargs = kwargs
        self.results = None

    def run(self):
        self.results = []
        attrs = list(self.parameters.keys())  # o nome dos atributos são as keys do dicionário, que ficam em lista
        values = list(self.parameters.values())  # e os valores que queremos fazer o grid desses atributos são os values
        # do dicionário, que ficam em lista
        for conf in itertools.product(*values):
            for i in range(len(attrs)):  # vai atualizando os atributos
                setattr(self.model, attrs[i], conf[i])  # muda os valores dos atributos do modelo, para os diferentes
                # valores que se deu
            scores = CrossValidationScore(self.model, self.dataset, self.score, **self.kwargs)  # inicializa a cv com
            # cada conjunto de atributos
            self.results.append((scores.run()))  # corre a cv para cada conjunto de atributos
        return self.results

    def toDataframe(self):
        assert self.results, "Need to run training before hand"
        n_cv = len(self.results[0][0])
        data = np.hstack((np.array([res[0] for res in self.results]), np.array([res[1] for res in self.results])))
        return pd.DataFrame(data=data, columns=[f"CV_{i+1} train" for i in range(n_cv)]+
                                               [f"CV_{i+1} test" for i in range(n_cv)])


class CrossValidationScore:  # igual à CrossValidation, mas com uma função de score específica

    def __init__(self, model, dataset, score=None, **kwargs):
        self.model = model  # qual o algoritmo a ser corrido e feito o fit, ja inicializado
        self.dataset = dataset  # dataset completo que vai levar o fit e prever os valores
        self.score = score  # função de score (facultativa)
        self.cv = kwargs.get('cv', 3)  # parametros retirados da lista kwargs, numero de cv's default será 3
        self.split = kwargs.get('split', 0.8)  # split feito com 80% dos dados para treino como default
        self.train_score = None
        self.test_score = None
        self.ds = None

    def run(self):
        train_score = []  # vai guardar o score do fitting do treino que resulta de cada ciclo de cv
        test_score = []  # o mesmo para os scores do teste
        ds = []  # guarda ambo datasets divididos de treino e teste na forma de tuplo
        for _ in range(self.cv):  # um ciclo para cada cv
            train, test = train_test_split(self.dataset, self.split)  # faz o split
            ds.append((train, test))  # adiciona os dois na forma de tuplo
            self.model.fit(train)  # faz o fitting de acordo com o modelo escolhido
            if not self.score:  # se não for dada função de score
                train_score.append(self.model.cost())  # corre a função de custo que cada modelo tem implementada, para o treino
                test_score.append(self.model.cost(test.X, test.Y))  # faz o mesmo, mas desta vez com os dados do dataset de teste
            else:  # quando se dá uma função de score
                y_train = np.ma.apply_along_axis(self.model.predict, axis=1, arr=train.X)  # faz a prediction das labels
                # para os dados de treino e depois calcular o score com a função dada
                train_score.append(self.score(train.Y, y_train))
                y_test = np.ma.apply_along_axis(self.model.predict, axis=1, arr=test.X)  # o mesmo para o dataset teste
                test_score.append(self.score(test.Y, y_test))
        self.train_score = train_score  # guarda os scores de treino
        self.test_score = test_score  # guarda os scores de teste
        self.ds = ds  # guarda ambos datasets
        return train_score, test_score

    def toDataframe(self):
        import pandas as pd
        assert self.train_score and self.test_score, 'Need to run function'
        return pd.DataFrame({'Train Scores:': self.train_score, 'Test Scores:': self.test_score})
