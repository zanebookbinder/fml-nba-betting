import xgboost as xgb

class XGBoostRegressor:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.params = {
            'objective': 'reg:squarederror',  
            'n_estimators': n_estimators, 
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'eval_metric': 'rmse'  
        }
        self.model = None  

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.model = xgb.train(self.params, dtrain)

    def test(self, x_test):
        dtest = xgb.DMatrix(x_test)
        return self.model.predict(dtest) 