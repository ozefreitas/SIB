import math
import numpy as np


class NaiveBayes:
    def __init__(self, dataset):
        # self.by_label = {}
        self.dataset = dataset

    # def fit(self):
    #     self.sep_by_class(self.dataset)

    def sep_by_class(self, dataset):  # separa as linhas do dataset de acordo com a classe Y
        by_label = {}
        X, y = dataset.getXy()
        for row in range(len(y)):  # para todas as linhas
            if y[row] not in by_label:  # se essa label ainda nao estiver no dicionário
                by_label[y[row]] = []  # adiciona-se como key
            by_label[y[row]].append(X[row])  # adiciona-se a linha à lista dessa label
        return by_label

    def mean(self, numbers):  # faz a média de um conjunto de números em lista
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):  # faz o desvio padrao de uma conjunto de números em lista
        avg = self.mean(numbers)
        var = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(var)

    def summarize_dataset(self, data):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*data)]
        del (summaries[-1])
        return summaries

    def summarize_by_class(self, dataset):
        separated = self.sep_by_class(dataset)
        summ_per_class = {}
        for class_value, rows in separated.items():  # para cada label e todas as linhas de dados
            summ_per_class[class_value] = self.summarize_dataset(rows)
        return summ_per_class

    def calculate_probability(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, sum_per_class, row):
        total_rows = sum([sum_per_class[label][0][2] for label in sum_per_class])
        probabilities = dict()
        for class_value, class_summaries in sum_per_class.items():
            probabilities[class_value] = sum_per_class[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def run(self, train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        X, y = test.getXy()
        for row in X:
            output = self.predict(summarize, row)
            predictions.append(output)
        return np.array(predictions)
