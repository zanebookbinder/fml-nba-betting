
import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, split_feature=None, split_value=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = None
        self.right = None

    def p(self):
        print(self.split_feature, self.split_value, self.left, self.right)

    def is_leaf(self):
        return not self.left and not self.right

class CARTLearner:
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.tree_root = None

    def build_tree(self, data):
        if np.all(data['Y'] == data['Y'].iloc[0]):
            return TreeNode(None, data['Y'].iloc[0])
        
        if not data.loc[:, data.columns != 'Y'].std().max():
            return TreeNode(None, data['Y'].mean())
                
        if len(data) <= self.leaf_size:
            return TreeNode(None, data['Y'].mean())
        
        corr_matrix = data.corr()['Y'][:-1].abs()
        best_corr_feature = corr_matrix.idxmax()
        median = data[best_corr_feature].median()

        left = data.loc[data[best_corr_feature] <= median]
        right = data.loc[data[best_corr_feature] > median]

        if not len(left) or not len(right):
            return TreeNode(None, data['Y'].mean())

        my_node = TreeNode(best_corr_feature, median)
        my_node.left = self.build_tree(left)
        my_node.right = self.build_tree(right)
        return my_node

    def train(self, x, y):
        data = pd.DataFrame(x)
        data['Y'] = y
        self.tree_root = self.build_tree(data)

    def test(self, x):
        output = np.empty(x.shape[0])

        for i in range(x.shape[0]):
            cur_node = self.tree_root

            while not cur_node.is_leaf():
                row_split_val = x.iloc[i][cur_node.split_feature]
                if row_split_val <= cur_node.split_value:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right

            output[i] = cur_node.split_value

        return output