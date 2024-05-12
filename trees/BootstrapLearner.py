import numpy as np

class BootstrapLearner:
    def __init__(self, constituent, model_kwargs, bags):
        self.bags = bags
        self.constituents = []
        for i in range(bags):
            self.constituents.append(constituent(**model_kwargs))

    def train(self, x, y):
        x, y = x.copy(), y.copy()
        x.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        for i in range(len(self.constituents)):
            sample_size = np.random.choice(len(x), len(x), replace=True)
            training_x = x.iloc[sample_size]
            training_y = y.iloc[sample_size]
            learner = self.constituents[i]
            learner.train(training_x, training_y)

    def test(self, x):
        preds = []
        for i in range(len(self.constituents)):
            learner = self.constituents[i]
            preds.append(learner.test(x))
        return np.mean(preds, axis=0)