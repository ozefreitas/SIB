import itertools
import numpy as np
import pandas as pd


# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary', "euclidean", "manhattan", "train_test_split", "sigmoide", "add_intersect", "confusion_matrix"]


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1
    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s
    return [gen() for _ in range(n)]


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)
    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    if dataset.hasLabel():
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y))))
        names= [dataset._xnames,dataset._yname]
    else:
        data = np.hstack((dataset.X, dataset.Y.reshape(len(dataset.Y))))
        names = [dataset._xnames]
    mean = np.mean(data, axis=0)
    var = np.var(data, axis=0)
    maxi = np.max(data, axis=0)
    mini = np.min(data, axis=0)
    stats = {}
    for i in range(data.shape[1]):
        stat = {'mean': mean[i]
                , 'var': var[i]
                , 'max': maxi[i]
                , 'min': mini[i]}
        stats[names[i]] = stat
    if format == 'df':
        import pandas as pd
        df= pd.DataFrame(stats)
        return df
    else:
        return stats


def euclidean(x, y):  # distancia de um ponto a um conjunto de pontos
    dist = np.sqrt(np.sum((x-y)**2, axis=1))
    return dist


def manhattan(x, y):  # mesmo que euclidean
    dist = np.abs(x - y)
    dist = np.sum(dist)
    return dist


def train_test_split(dataset, split=0.8):
    x = dataset.X
    n = x.shape[0]  # tamanho do dataset
    m = int(split*n)  # número da samples a ficar no train
    # print(m)
    arr = np.arange(n)  # em forma de array
    # print(arr)
    np.random.shuffle(arr)  # randomize dos indices
    # print("depois de aplicado o random:", arr)
    from src.si.data.dataset import Dataset
    train = Dataset(x[arr[:m]], dataset.Y[arr[:m]], dataset._xnames, dataset._yname)
    test = Dataset(x[arr[m:]], dataset.Y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test


def sigmoide(z):
    return 1/(1+np.exp(-z))


def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, batchsize=256, shuffle=True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize: (i + 1) * batchsize]

    return mb_generator(),


def confusion_matrix(y_true, y_pred):
    """
    Computes a dataframe of predicted and true values
    Parameters
    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :return: pandas.DataFrame conf_matrix: dataframe-like of shape (2,2)
        Countdown of how many predicted values correspond to the true values
    """
    conf_matrix = pd.crosstab(y_true, y_pred, rownames=["True"], colnames=["Predicted"], margins=True)
    return conf_matrix
