from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from LinearRegressor import LinearRegressor
from trees.PERTLearner import PERTLearner
from trees.CARTLearner import CARTLearner
import matplotlib.pyplot as plt
from useful_functions import get_odds_data
from indicators.indicator_data import IndicatorData

class ModelTester():
	def __init__(self, model_class=LinearRegressor, start_date='2013-10-29', end_date='2023-04-09', predict_type='Spread', odds_type='best', betting_threshold=5, **kwargs):		
		if predict_type not in ['Spread', 'OU', 'Both']:
			raise ValueError('predict_type must be either "spread" or "OU" or "Both')
		if odds_type not in ['best', 'worst', 'average']:
			raise ValueError('odds_type must be either "best", "worst", or "average"')	
			
		self.predict_type = predict_type
		self.odds_type = odds_type
		self.betting_threshold = betting_threshold
		self.model_class = model_class

		self.load_data(start_date, end_date)
		self.model = model_class(**kwargs)

		if model_class == IndicatorData:
			self.indicator_df_result = self.test_indicator_systems()
		else:
			self.train_df_result, self.test_df_result = self.train_model()

		# self.graph_betting_threshold(self.test_df_result)

	def graph_betting_threshold(self, test_df_result, plot=True):
		# graph betting threshold vs. win percentage and unit gain/loss

		thresholds = np.arange(1, 15, .1)
		win_rates = []
		unit_gains = []
		bets_made_list = []

		for threshold in thresholds:
			bets_made, win_rate, unit_gain = self.bet_with_predictions(test_df_result, betting_threshold=threshold)

			win_rates.append(win_rate)
			unit_gains.append(unit_gain)
			bets_made_list.append(bets_made)

		if not plot:
			return thresholds, win_rates, unit_gains, bets_made_list

		plt.plot(thresholds, win_rates, label='Win Rate')
		plt.xlabel('Betting Threshold')
		plt.ylabel('Win Rate')
		plt.title('Betting Threshold vs. Win Rate for ' + self.predict_type + ' Predictions')
		plt.show()

		plt.plot(thresholds, unit_gains, label='Units Gained/Lost')
		plt.xlabel('Betting Threshold')
		plt.ylabel('Units Gained/Lost')
		plt.title('Betting Threshold vs. Unit Gain/Loss for ' + self.predict_type + ' Predictions')
		plt.show()

	def load_data(self, start_date, end_date, stat_path='./data/all_stat_data.csv'):
		# read csv and filter by date
		stat_data = pd.read_csv(stat_path)
		stat_data = stat_data[(stat_data['Date'] >= start_date) & (stat_data['Date'] <= end_date)]

		if self.predict_type in ['Spread', 'Both']:
			# team points - opponent points = margin	 (120-105 win --> 15 margin prediction)
			stat_data['predict_col_Spread'] = stat_data['pts'] - stat_data['pts_opp']

		if self.predict_type in ['OU', 'Both']:
			# team points + opponent points = total		 (120-105 game --> 225 total prediction)
			stat_data['predict_col_OU'] = stat_data['pts_opp'] + stat_data['pts']

		# get odds data and merge with stat data
		if self.predict_type == 'Both':
			odds_data_spread = get_odds_data('Spread', start_date, end_date)
			odds_data_OU = get_odds_data('OU', start_date, end_date)
			odds_data = pd.merge(odds_data_spread, odds_data_OU, on=['Date', 'Team', 'OppTeam'], suffixes=('_Spread', '_OU'))
			self.final_df = pd.merge(stat_data, odds_data, on=['Date', 'Team', 'OppTeam'])
		else:
			odds_data = get_odds_data(self.predict_type, start_date, end_date)
			self.final_df = pd.merge(stat_data, odds_data, on=['Date', 'Team', 'OppTeam'], suffixes=('', '_'+self.predict_type))

		# limit columns
		final_df_columns = (
			['home_game', 'days_of_rest', 'travel_miles', 'days_of_rest_opp', 'travel_miles_opp'] + 
			[col for col in stat_data.columns if 'last_8' in col or 'predict_col' in col] +
			[col for col in odds_data.columns if 'Best_Line' in col or 'Best_Odds' in col]
		)
		self.final_df = self.final_df[final_df_columns]

		if self.model_class == IndicatorData:
			start_column = self.final_df.columns.get_loc('predict_col_Spread')
			self.final_df = self.final_df.iloc[:, start_column:]

	def bet_with_predictions(self, df, print_results=False, betting_threshold=None):
		# df columns are: 'Odds', 'Prediction', 'Results', 'Line'
		# bet on games where the model's prediction differs from the line by more than the betting threshold
  
		if not betting_threshold:
			betting_threshold = self.betting_threshold

		df['bet'] = (df['Prediction'] - df['Line']).abs() > betting_threshold
		df['win_bet'] = ((df['Prediction'] - df['Line']) * (df['Results'] - df['Line'])) > 0

		df['gain/loss'] = np.zeros(len(df), dtype='float')
		df.loc[(df['bet']) & (~df['win_bet']), 'gain/loss'] = -1
		df.loc[(df['bet']) & (df['win_bet']), 'gain/loss'] = df['Odds']

		if not df['bet'].sum():
			return 0, 0, 0

		bets_made = df['bet'].sum()
		win_rate = round(df.loc[df["bet"] == True, "win_bet"].sum() / df["bet"].sum(), 3)
		gain_or_loss = df["gain/loss"].sum()

		if print_results:
			print(f'Testing model with {len(df)} total games')
			print(f'Number of bets: {bets_made}')
			print(f'Number of wins: {df.loc[df["bet"] == True, "win_bet"].sum()}')
			print(f'Win rate: {win_rate}')
			print(f'Total gain/loss: {gain_or_loss}')

		return bets_made, win_rate, gain_or_loss

	def train_model(self):
		some_columns = ['Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2', 'predict_col']

		test = self.final_df.sample(frac=0.35)
		train = self.final_df.drop(test.index)

		x_train = train.drop(some_columns, axis=1)
		y_train = train['predict_col']

		x_test = test.drop(some_columns, axis=1)

		print('Training model...')
		self.model.train(x_train, y_train)
		print('Done training model')

		is_predictions = self.model.test(x_train)
		oos_predictions = self.model.test(x_test)

		train_df = train[some_columns]
		train_df.loc[:, 'Prediction'] = np.array(is_predictions, dtype='float')

		test_df = test[some_columns]
		test_df.loc[:, 'Prediction'] = np.array(oos_predictions, dtype='float')

		for df in [train_df, test_df]:
			if self.odds_type == 'best':
				df['Line'] = df.apply(lambda x: x['Best_Line_Option_1'] if abs(x['Best_Line_Option_1'] - x['Prediction']) > abs(x['Best_Line_Option_2'] - x['Prediction']) else x['Best_Line_Option_2'], axis=1)
				df['Odds'] = df.apply(lambda x: max(x['Best_Odds_Option_1'], x['Best_Odds_Option_2']), axis=1)
			elif self.odds_type == 'worst':
				df['Line'] = df.apply(lambda x: x['Best_Line_Option_1'] if abs(x['Best_Line_Option_1'] - x['Prediction']) < abs(x['Best_Line_Option_2'] - x['Prediction']) else x['Best_Line_Option_2'], axis=1)
				df['Odds'] = df.apply(lambda x: min(x['Best_Odds_Option_1'], x['Best_Odds_Option_2']), axis=1)
			elif self.odds_type == 'average':
				df['Line'] = df.apply(lambda x: (x['Best_Line_Option_1'] + x['Best_Line_Option_2']) / 2, axis=1)
				df['Odds'] = df.apply(lambda x: (x['Best_Odds_Option_1'] + x['Best_Odds_Option_2']) / 2, axis=1)

			df.drop(['Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2'], axis=1, inplace=True)
			df.columns = ['Results', 'Prediction', 'Line', 'Odds']

		return train_df, test_df
	
	def test_indicator_systems(self):

		indicator_cols = self.model.df[self.model.indicator_cols]

		indicator_df = self.final_df.copy()
		indicator_df = pd.concat([indicator_df, indicator_cols], axis=1)

		indicator_df['best_odds_Spread'] = indicator_df.apply(lambda x: max(x['Best_Odds_Option_1_Spread'], x['Best_Odds_Option_2_Spread']), axis=1)

		indicator_df['best_odds_OU'] = indicator_df.apply(lambda x: max(x['Best_Odds_Option_1_OU'], x['Best_Odds_Option_2_OU']), axis=1)
		indicator_df['worst_odds_OU'] = indicator_df.apply(lambda x: min(x['Best_Odds_Option_1_OU'], x['Best_Odds_Option_2_OU']), axis=1)

		indicator_df['best_line_Spread'] = indicator_df.apply(lambda x: min(x['Best_Line_Option_1_Spread'], x['Best_Line_Option_2_Spread']), axis=1)
		indicator_df['worst_line_Spread'] = indicator_df.apply(lambda x: max(x['Best_Line_Option_1_Spread'], x['Best_Line_Option_2_Spread']), axis=1)

		indicator_df['best_line_OU'] = indicator_df.apply(lambda x: min(x['Best_Line_Option_1_OU'], x['Best_Line_Option_2_OU']), axis=1)
		indicator_df['worst_line_OU'] = indicator_df.apply(lambda x: max(x['Best_Line_Option_1_OU'], x['Best_Line_Option_2_OU']), axis=1)

		bets_to_make = []
		for _, row in indicator_df.iterrows():

			# this contributes 10.6 units
			if row['fadeATS']:
				bets_to_make.append([row['best_odds_Spread'], -1000, row['predict_col_Spread'], row['worst_line_Spread']])

			# this contributes 5.45 units
			if row['betOver']:
				bets_to_make.append([row['best_odds_OU'], 1000, row['predict_col_OU'], row['best_line_OU']])

			# this contributes 394.8 units, win rate of 0.563
			if row['useTunnel']:
				bets_to_make.append([.93, 1000, row['predict_col_OU'], row['best_line_OU']])
				bets_to_make.append([.93, -1000, row['predict_col_OU'], row['worst_line_OU']])

			# this contributes 5.56 units
			if row['fadeATS_2']:
				bets_to_make.append([row['best_odds_Spread'], -1000, row['predict_col_Spread'], row['worst_line_Spread']])

		output_df = pd.DataFrame(bets_to_make, columns=['Odds', 'Prediction', 'Results', 'Line'])
		return output_df		

def compare_odd_types(predict_types=['Spread' ,'OU'], graph_type='win_rate', plot=True):
	fig, axs = plt.subplots(2, 2)

	# compare best, worst, and average odds
	for odd_type in ['best', 'worst', 'average']:
		for i, predict_type in enumerate(predict_types):
			m = ModelTester(predict_type=predict_type, odds_type=odd_type)
			thresholds, win_rates, unit_gains, bets_made = m.graph_betting_threshold(m.test_df_result, plot=False)

			to_graph = win_rates if graph_type == 'win_rate' else unit_gains
			axs[1, i].plot(thresholds, to_graph, label=odd_type.capitalize() + ' Odds')
			axs[0, i].plot(
				thresholds,
				bets_made,
				label='Bets Made with ' + odd_type.capitalize() + ' Odds',
				linestyle='--'
			)
			axs[0, i].set(title=predict_type + ' Predictions')

	for i in range(len(predict_types)):
		axs[1, i].plot(thresholds, [0.53] * len(thresholds), label='Breakeven Win Rate', linestyle='--', color='gray')
		axs[1, i].legend()
		axs[1, i].set(xlabel='Betting Threshold')

	axs[0,0].set(ylabel='Bets Placed')

	if graph_type == 'win_rate':
		axs[1,0].set(ylabel='Win Rate')
		fig.suptitle('Betting Threshold vs. Win Rate for Predictions With Various Odds Types')
	else:
		axs[1,0].set(ylabel='Units Gained/Lost')
		fig.suptitle('Betting Threshold vs. Unit Gain/Loss for Predictions With Various Odds Types')

	plt.show()

def compare_PERT_leaf_sizes(predict_type='Spread', graph_type='win_rate'):
	# compare leaf sizes
	results = []
	leaf_sizes = [5, 10, 20, 30, 40, 50]
	for leaf_size in leaf_sizes:
		print('Leaf size:', leaf_size)
		m = ModelTester(model_class=PERTLearner, predict_type=predict_type, leaf_size=leaf_size)
		bets_made, win_rate, unit_gains = m.bet_with_predictions(m.test_df_result, print_results=False)

		if graph_type == 'win_rate':
			results.append(win_rate)
		else:
			results.append(unit_gains)

	plt.plot(leaf_sizes, results)
	plt.xlabel('Leaf Size')

	if graph_type == 'win_rate':
		plt.ylabel('Win Rate')
		plt.title('Leaf Size vs. Win Rate for PERT Learner' + predict_type + ' Predictions')
	else:
		plt.ylabel('Units Gained/Lost')
		plt.title('Leaf Size vs. Win Rate for PERT Learner' + predict_type + ' Predictions')

	plt.show()


# comapare_odd_types = compare_odd_types()

# m = ModelTester(model_class=CARTLearner, predict_type='OU', odds_type='best', betting_threshold=10, leaf_size=10)
# m.bet_with_predictions(m.test_df_result, print_results=True)

m = ModelTester(model_class=IndicatorData, predict_type='Both')
m.bet_with_predictions(m.indicator_df_result, print_results=True)
