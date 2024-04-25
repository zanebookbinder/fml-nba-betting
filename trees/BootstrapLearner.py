import numpy as np

class BootstrapLearner:
    def __init__(self, constituent, kwargs, bags):
        self.bags = bags
        self.constituents = []
        for i in range(bags):
            self.constituents.append(constituent(**kwargs))

    def train(self, x, y):
        for i in range(len(self.constituents)):
            sample_size = np.random.choice(len(x), len(x), replace=True)
            training_x = x[sample_size]
            training_y = y[sample_size]
            learner = self.constituents[i]
            learner.train(training_x, training_y)

    def test(self, x):
        preds = []
        for i in range(len(self.constituents)):
            learner = self.constituents[i]
            preds.append(learner.test(x))
        return np.mean(preds, axis=0)