
import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Classification performance metric that computes the accuracy of y_true
    and y_pred.
    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :returns: C (float) Accuracy score.
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    accuracy = correct / len(y_true)
    return accuracy


def precision_score(y_true, y_pred, binary=True):
    """
    Classification performance metric that computes the precision of y_true
    and y_pred.
    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :param bool binary: for problems with binary labels as positives (1) and negatives (0)
    :returns: C (float) Precision score.
    """
    true_pos, false_pos = 0, 0
    if binary:
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                true_pos += 1
            if pred == 1 and true == 0:  # falsos positivos
                false_pos += 1
        precision = true_pos / (true_pos+false_pos)
    else:
        raise TypeError("Problem must be binary classification 0 and 1")
    return precision


def recall(y_true, y_pred, binary=True):
    """
    Classification performance metric that computes the recall of y_true
    and y_pred.
    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :param bool binary: for problems with binary labels as positives (1) and negatives (0)
    :returns: C (float) Recall score.
    """
    true_pos, false_neg = 0, 0
    if binary:
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                true_pos += 1
            if pred == 0 and true == 1:  # falsos positivos
                false_neg += 1
        rec = true_pos / (true_pos+false_neg)
    else:
        raise TypeError("Problem must be binary classification 0 and 1")
    return rec


def f1_score(y_true, y_pred, binary=True):
    """
    Classification performance metric that computes the F1 score of y_true
    and y_pred.
    :param numpy.array y_true: array-like of shape (n_samples,) Ground truth correct labels.
    :param numpy.array y_pred: array-like of shape (n_samples,) Estimated target values.
    :param bool binary: for problems with binary labels as positives (1) and negatives (0)
    :returns: C (float) Precision score.
    """
    if binary:
        precision = precision_score(y_true, y_pred, binary=binary)
        rec = recall(y_true, y_pred, binary=binary)
        f1 = 2 * (precision * rec) / (precision + rec)
    else:
        raise TypeError("Problem must be binary classification 0 and 1")
    return f1


def mse(y_true, y_pred, squared=True):
    """
    Mean squared error regression loss function.
    Parameters
    :param numpy.array y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    :param numpy.array y_pred: array-like of shape (n_samples,)
        Estimated target values.
    :param bool squared: If True returns MSE, if False returns RMSE. Default=True
    :returns: loss (float) A non-negative floating point value (the best value is 0.0).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.average((y_true - y_pred) ** 2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)


def r2_score(y_true, y_pred):
    """
    R^2 regression score function.
        R^2 = 1 - SS_res / SS_tot
    where SS_res is the residual sum of squares and SS_tot is the total
    sum of squares.
    :param numpy.array y_true : array-like of shape (n_samples,) Ground truth (correct) target values.
    :param numpy.array y_pred : array-like of shape (n_samples,) Estimated target values.
    :returns: score (float) R^2 score.
    """
    # Residual sum of squares.
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    # Total sum of squares.
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    # R^2.
    score = 1 - numerator / denominator
    return score


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred)).sum()


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
