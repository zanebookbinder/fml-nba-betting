import numpy as np
import pandas as pd
import random

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

class PERTLearner:
	def __init__(self, leaf_size=1, allowable_fails=10):
		self.leaf_size = leaf_size
		self.tree_root = None
		self.allowable_fails = allowable_fails

	def build_tree(self, data):
		if len(data) < self.leaf_size:
			return TreeNode(None, data['Y'].mean())
		for _ in range(self.allowable_fails):
			random_rows = data.sample(n=2, replace=False)
			a, b = random_rows.iloc[0], random_rows.iloc[1]
			j = random.choice(data.columns.drop('Y'))
			if a[j] == b[j]:
				continue

			alpha = random.random()
			split_value = alpha * a[j] + (1-alpha) * b[j]

			data2 = data.copy()
			left = data2.loc[data2[j] <= split_value]
			right = data2.loc[data2[j] > split_value]

			if not len(left) or not len(right):
				continue

			my_node = TreeNode(data.columns.get_loc(j), split_value)
			my_node.left = self.build_tree(left)
			my_node.right = self.build_tree(right)
			return my_node

		return TreeNode(None, data['Y'].mean())

	def train(self, x, y):
		data = pd.DataFrame(x)
		data["Y"] = y
		self.tree_root = self.build_tree(data)

	def test(self, x):
		output = np.empty(x.shape[0])

		for i in range(x.shape[0]):
			cur_node = self.tree_root

			while not cur_node.is_leaf():
				row_split_val = x.iloc[i, cur_node.split_feature]
				if row_split_val <= cur_node.split_value:
					cur_node = cur_node.left
				else:
					cur_node = cur_node.right

			output[i] = cur_node.split_value

		return output
