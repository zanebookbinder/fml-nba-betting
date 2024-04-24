import numpy as np

class Node:
        def __init__(self, y=None, decision=None, left=None, right=None, feature=None, split_value=None):
            # Value (only for leaf nodes)
            self.y = y
            # Left and right subtrees
            self.left = left
            self.right = right
            # Index of feature that is considered
            self.feature = feature
            # Threshold value that is split on
            self.split_value = split_value

class CARTLearner:

    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        root = None
        
    def build_tree(self, x, y):
        node = Node()
        features = x
        # if all the y values are the same, return a leaf with that y value
        if len(np.unique(y)) == 1:
            node.y = y[0]
            return node
        # else if it is not possible to split (all X values same), return a leaf
        elif np.all(x == x[0]):
            node.y = np.mean(y)
            return node
        # else if another stopping condition is met, return a leaf
        elif len(y) <= self.leaf_size:
            node.y = np.mean(y)
            return node
        else:
        #   determine the best X feature to split on
            correlations = []
            for i in range(len(features[0])):
                feature = features[:, i]
                correlation = np.abs(np.corrcoef(feature, y)[0, 1])
                correlations.append(correlation)
            best_feature = np.argmax(correlations)

        #   split on the median value of the selected X feature
        #   be sure to record the selected feature and value for this decision node
            median = np.mean(x[:, best_feature])
            node.split_value = median
            node.feature = best_feature
        #   recursively build the left child (w/ data where selected feature value <= split value)
        #   recursively build the right child (w/ data where selected feature value > split value)
            left_list = x[np.where(x[:, best_feature] <= median)]
            right_list = x[np.where(x[:, best_feature] > median)]
            # case where all the x values are on one side
            if len(right_list) == 0 or len(left_list) == 0:
                node.y = np.mean(y)
                return node
            left_labels = y[np.where(x[:, best_feature] <= median)]
            right_labels = y[np.where(x[:, best_feature] > median)]

            left = self.build_tree(left_list, left_labels)
            right = self.build_tree(right_list, right_labels)

            node.left = left
            node.right = right
            
            return node

    def train(self, x, y):
        self.root = self.build_tree(x, y)
        # induce a decision tree based on this training data

    def test(self, x):
        node = self.root
        preds = np.zeros(len(x))
        for i in range(len(x)):
            row = x[i, :]
            while node.y is None:
                if row[node.feature] <= node.split_value:
                    node = node.left
                elif row[node.feature] > node.split_value:
                    node = node.right
            preds[i] = node.y
            node = self.root
        return preds
