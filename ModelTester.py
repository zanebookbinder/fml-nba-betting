from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from LinearRegressor import LinearRegressor
from trees.PERTLearner import PERTLearner as PERTLearner
import matplotlib.pyplot as plt
import time

class ModelTester():
	def __init__(self, model_class=LinearRegressor, start_date='2013-10-29', end_date='2023-04-09', predict_type='Spread', odds_type='best', betting_threshold=5, **kwargs):		
		if predict_type not in ['Spread', 'OU']:
			raise ValueError('predict_type must be either "spread" or "OU"')
		if odds_type not in ['best', 'worst', 'average']:
			raise ValueError('odds_type must be either "best", "worst", or "average"')	
			
		self.predict_type = predict_type
		self.odds_type = odds_type
		self.betting_threshold = betting_threshold
		
		self.load_data(start_date, end_date)
		self.model = model_class(**kwargs)

		self.train_df_result, self.test_df_result = self.train_model()

		# self.graph_betting_threshold(self.test_df_result)

	def graph_betting_threshold(self, test_df_result, plot=True):
		# graph betting threshold vs. win percentage and unit gain/loss

		thresholds = np.arange(1, 20, .1)
		win_rates = []
		unit_gains = []

		for threshold in thresholds:
			bets_made, win_rate, unit_gain = self.bet_with_predictions(test_df_result, betting_threshold=threshold)

			win_rates.append(win_rate)
			unit_gains.append(unit_gain)

		if not plot:
			return thresholds, win_rates, unit_gains

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

	def get_odds_data(self, start_date, end_date, odds_path='./data/all_odds_data.csv'):
		# read csv and filter by date
		self.odds_data = pd.read_csv(odds_path)
		self.odds_data = self.odds_data[(self.odds_data['Date'] >= start_date) & (self.odds_data['Date'] <= end_date)]
		
		# get best and worst lines for each game
		self.odds_data['Best_Line_Option_1'] = self.odds_data['Best_Line_' + self.predict_type]
		self.odds_data['Best_Line_Option_2'] = self.odds_data['Worst_Line_' + self.predict_type]

		# fill in missing lines
		self.odds_data.loc[self.odds_data['Best_Line_Option_2'] == 0, 'Best_Line_Option_2'] = self.odds_data.loc[self.odds_data['Best_Line_Option_2'] == 0, 'Best_Line_Option_1']

		# if predicting spread, make negative lines positive (margin of victory)
		if self.predict_type == 'Spread':
			self.odds_data['Best_Line_Option_1'] = -1 * self.odds_data['Best_Line_Option_1']
			self.odds_data['Best_Line_Option_2'] = -1 * self.odds_data['Best_Line_Option_2']

		# get best and worst odds for each game
		self.odds_data['Best_Odds_Option_1'] = self.odds_data['Best_Odds_' + self.predict_type]
		self.odds_data['Best_Odds_Option_2'] = self.odds_data['Worst_Odds_' + self.predict_type]

		for odds_col in ['Best_Odds_Option_1', 'Best_Odds_Option_2']:
			# fill in missing odds
			self.odds_data.loc[self.odds_data[odds_col] == 0, odds_col] = -110

			# calculate decimal odds from American odds
			self.odds_data.loc[self.odds_data[odds_col] > 0, odds_col] = self.odds_data[odds_col] / 100
			self.odds_data.loc[self.odds_data[odds_col] < 0, odds_col] = 100 / abs(self.odds_data[odds_col])

		columns_we_need = ['Date', 'Team', 'OppTeam', 'Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2']
		return self.odds_data[columns_we_need]

	def load_data(self, start_date, end_date, stat_path='./data/all_stat_data.csv'):
		# read csv and filter by date
		stat_data = pd.read_csv(stat_path)
		stat_data = stat_data[(stat_data['Date'] >= start_date) & (stat_data['Date'] <= end_date)]

		if self.predict_type == 'Spread':
			# team points - opponent points = margin	 (120-105 win --> 15 margin prediction)
			stat_data['predict_col'] = stat_data['pts'] - stat_data['pts_opp']

		elif self.predict_type == 'OU':
			# team points + opponent points = total		 (120-105 game --> 225 total prediction)
			stat_data['predict_col'] = stat_data['pts_opp'] + stat_data['pts']

		# get odds data
		odds_data = self.get_odds_data(start_date, end_date)

		# merge stat and odds data
		self.final_df = pd.merge(stat_data, odds_data, on=['Date', 'Team', 'OppTeam'])

		# limit columns
		final_df_columns = (
			['home_game', 'days_of_rest', 'travel_miles', 'days_of_rest_opp', 'travel_miles_opp'] + 
			[col for col in stat_data.columns if 'last_8' in col] +
			['predict_col', 'Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2']
		)
		self.final_df = self.final_df[final_df_columns]

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
			print(f'Total games: {len(df)}')
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

		self.model.train(x_train, y_train)

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
	
def compare_odd_types(predict_type='Spread', graph_type='win_rate'):
	# compare best, worst, and average odds
	for odd_type in ['best', 'worst', 'average']:
		m = ModelTester(predict_type=predict_type, odds_type=odd_type)
		thresholds, win_rates, unit_gains = m.graph_betting_threshold(m.test_df_result, plot=False)

		to_graph = win_rates if graph_type == 'win_rate' else unit_gains
		plt.plot(thresholds, to_graph, label=odd_type.capitalize() + ' Odds')

	plt.xlabel('Betting Threshold')

	if graph_type == 'win_rate':
		plt.ylabel('Win Rate')
		plt.title('Betting Threshold vs. Win Rate for ' + predict_type + ' Predictions With Various Odds Types')
	else:
		plt.ylabel('Units Gained/Lost')
		plt.title('Betting Threshold vs. Unit Gain/Loss for ' + predict_type + ' Predictions With Various Odds Types')

	plt.legend()
	plt.show()
	
compare_odd_types(predict_type='OU')