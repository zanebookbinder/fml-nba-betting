import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from CARTLearner import CARTLearner
from trees.BootstrapLearner import BootstrapLearner
import matplotlib.pyplot as plt

data_folder = './data'
df = pd.read_csv(data_folder + f"/all_stat_data.csv")
df = df.iloc[:1000]
df = df.fillna(method='bfill')
# features_df = df[['last_8_pts', 'last_8_pts_opp', 'last_8_ast', 'last_8_ast_opp', 'last_8_reb_opp', 'last_8_reb_opp', 'last_8_tov', 'last_8_tov_opp', 'fg']]
features_df = df.drop(columns=['Season', 'Date', 'Location', 'Team', 'Result'])
target_df = df['pts'] - df['pts_opp']

x_train, x_test, y_train, y_test = train_test_split(features_df.values, target_df.values, test_size=0.4)


# Shuffle the rows and partition some data for testing.
# x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)

# # Construct our learner.
# lrn = CARTLearner(leaf_size=1)
# lrn.train(x_train, y_train)

# # Test in-sample.
# y_pred = lrn.test(x_train)
# print("pred spread:", y_pred)
# rmse_is = mean_squared_error(y_train, y_pred, squared=False)
# corr_is = np.corrcoef(y_train, y_pred)[0,1]

# # Test out-of-sample.
# y_pred = lrn.test(x_test)
# rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
# corr_oos = np.corrcoef(y_test, y_pred)[0,1]

# # Print summary.
# print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
# print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")

# CART Learner
iterations = 50
in_sample = []
out_of_sample = []
for i in range(1, iterations):
    # Plot 1
    lrn = CARTLearner(leaf_size=i)
    lrn.train(x_train, y_train)

    # Test in-sample.
    y_pred = lrn.test(x_train)
    rmse_is = root_mean_squared_error(y_train, y_pred)
    corr_is = np.corrcoef(y_train, y_pred)[0,1]
    in_sample.append(rmse_is)

    # Test out-of-sample.
    y_pred = lrn.test(x_test)
    rmse_oos = root_mean_squared_error(y_test, y_pred)
    corr_oos = np.corrcoef(y_test, y_pred)[0,1]
    out_of_sample.append(rmse_oos)

    # Print summary.
    # print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
    # print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")

plt.plot(in_sample, label="IS")
plt.plot(out_of_sample, label="OOS")
plt.xlabel('Leaf Size')
plt.ylabel('RMSE')
plt.legend()
plt.title('CARTLearner Leaf Size and RMSE')
plt.show()

# CART Learner w/ Bags
in_sample = []
out_of_sample = []
for i in range(1, iterations):
    # Plot 1
    lrn = BootstrapLearner(constituent = CARTLearner, kwargs = {"leaf_size": i}, bags = 10)
    lrn.train(x_train, y_train)

    # Test in-sample.
    y_pred = lrn.test(x_train)
    rmse_is = root_mean_squared_error(y_train, y_pred)
    in_sample.append(rmse_is)

    # Test out-of-sample.
    y_pred = lrn.test(x_test)
    rmse_oos = root_mean_squared_error(y_test, y_pred)
    out_of_sample.append(rmse_oos)

    # # Print summary.
    # print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
    # print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")

plt.plot(in_sample, label="IS")
plt.plot(out_of_sample, label="OOS")
plt.xlabel('Leaf Size')
plt.ylabel('RMSE')
plt.legend()
plt.title('CARTLearner Ensemble Leaf Size and RMSE')
plt.show()



