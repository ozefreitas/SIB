import math


class NaiveBayes:
    def __init__(self, dataset):
        self.by_label = {}
        self.dataset = dataset

    # def fit(self):
    #     self.sep_by_class(self.dataset)

    def sep_by_class(self, dataset):
        X, y = dataset.getXy()
        for row in range(len(y)):
            if y[row] not in self.by_label:
                self.by_label[y[row]] = []
            self.by_label[y[row]].append(X[row])
        return self.by_label

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        var = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(var)

    def summarize_dataset(self, dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    def summarize_by_class(self, dataset):
        summ_per_class = self.sep_by_class(dataset)
        for class_value, rows in self.by_label.items():
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

    def naive_bayes(self, train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return predictions
