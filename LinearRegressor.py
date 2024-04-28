from sklearn.linear_model import LinearRegression

class LinearRegressor():
	def __init__(self):
		self.model = LinearRegression()

	def train(self, X, y):
		self.model.fit(X, y)

	def test(self, X):
		return self.model.predict(X)
