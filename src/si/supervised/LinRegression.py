import numpy as np
from .Modelo import Model
from src.si.data.dataset import Dataset
from src.si.util.metrics import mse


class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd = gd  # gradient descendent
        self.theta = None
        self.num_iterations = epochs
        self.lr = lr  # learning rate, velocidade de atualização de parametros

    def fit(self, dataset):
        X, Y = dataset.getXy()
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.X = X
        self.Y = Y
        if self.gd:  # se quiser com gradiente
            self.train_gd(X, Y)
        else:
            self.train_closed(X, Y)
        self.is_fited = True

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}  # vai guardar o resultado dos thetas e dos erros (custos) a cada iteração
        self.theta = np.zeros(n)
        for epoch in range(self.num_iterations):  # método iterativo para estimação dos parametros
            grad = 1/m*(X.dot(self.theta)-Y).dot(X)  # função diferenciável
            self.theta -= self.lr*grad  # atualização do theta a cada iteração
            self.history[epoch] = [self.theta[:], self.cost()]

    def train_closed(self, X, Y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)  # método dos minimos quadrados e que minimiza a função de erro

    def predict(self, x):
        assert self.is_fited
        _X = np.hstack(([1], x))
        return np.dot(self.theta, _X)

    def cost(self):  # função de erro
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2


class LinearRegressionReg(LinearRegression):  # para sobreajustamento
    def __init__(self, gd=False, epochs=1000, lbd=1):
        super(LinearRegressionReg, self).__init__()
        self.gd = gd
        self.num_iterations = epochs
        self.lbd = lbd  # parametro de regularização,

    def train_closed(self, X, Y):
        n = X.shape[1]
        identity = np.eye(n)  # matriz identidade
        identity[0, 0] = 0  # mudar o primeiro elemento para 0, para nao dar biased
        self.theta = np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(Y)  # método analítico
        self.is_fited = True

    def train_gd(self, X, Y):
        m = X.shape[0]
        n = X.shape[1]
        self.history = {}
        self.theta = np.zeros(n)
        lbds = np.full(m, self.lbd)
        lbds[0] = 0
        for epoch in range(self.num_iterations):
            grad = (X.dot(self.theta)-Y).dot(X)  # mesma que a de cima
            self.theta -= (self.lr/m)*(lbds+grad)  # atualização dos valores de theta
            self.history[epoch] = [self.theta[:], self.cost()]  # guardar atualizações e erros a cada iteração

    def predict(self, X):
        assert self.is_fited
        _x = np.hstack(([1], X))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred = np.dot(self.X, self.theta)
        return mse(self.Y, y_pred)/2
