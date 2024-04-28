from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from LinearRegressor import LinearRegressor
from trees.PERTLearner import PERTLearner as PERTLearner

class ModelTester():
	def __init__(self, model_class=LinearRegressor, start_date='2013-10-29', end_date='2023-04-09', predict_type='Spread', odds_type='best', betting_threshold=5, **kwargs):		
		if predict_type not in ['Spread', 'OU']:
			raise ValueError('predict_type must be either "spread" or "OU"')
		if odds_type not in ['best', 'worst', 'average']:
			raise ValueError('odds_type must be either "best", "worst", or "average"')	
			
		self.predict_type = predict_type
		self.odds_type = odds_type
		self.betting_threshold = betting_threshold

		self.line_column_title = self.odds_type.capitalize() + '_Line_' + self.predict_type
		self.odds_column_title = self.odds_type.capitalize() + '_Odds_' + self.predict_type
		
		self.load_data(start_date, end_date)
		self.model = model_class(**kwargs)

		train_df_result, test_df_result = self.train_and_evaluate_model()

		print('In Sample Results:')
		self.bet_with_predictions(train_df_result)

		print('')

		print('Out of Sample Results:')
		self.bet_with_predictions(test_df_result)

	def get_odds_data(self, start_date, end_date, odds_path='./data/all_odds_data.csv'):
		self.odds_data = pd.read_csv(odds_path)
		self.odds_data = self.odds_data[(self.odds_data['Date'] >= start_date) & (self.odds_data['Date'] <= end_date)]
		
		self.odds_data.loc[self.odds_data[self.line_column_title] == 0, self.line_column_title] = self.odds_data.loc[self.odds_data[self.line_column_title] == 0, 'Best_Line_' + self.predict_type]
		self.odds_data.loc[self.odds_data[self.odds_column_title] == 0, self.odds_column_title] = -110

		# odds_type options: 'best', 'worst', 'average'
		columns_we_need = ['Date', 'Team', 'OppTeam']
		columns_we_need += [self.line_column_title, self.odds_column_title]

		return self.odds_data[columns_we_need]

	def load_data(self, start_date, end_date, stat_path='./data/all_stat_data.csv'):
		
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
		final_df_columns = ['home_game', 'days_of_rest', 'travel_miles', 'days_of_rest_opp', 'travel_miles_opp'] + [col for col in stat_data.columns if 'last_8' in col] + ['predict_col', self.line_column_title, self.odds_column_title]
		self.final_df = self.final_df[final_df_columns]

	def bet_with_predictions(self, df):
		# df columns are: 'Line', 'Odds', 'Prediction', 'Results'
		# bet on games where the model's prediction differs from the line by more than the betting threshold
		
		if self.predict_type == 'Spread':
			df['Line'] = -1 * df['Line']

		df.loc[df['Odds'] > 0, 'Odds'] = df['Odds'] / 100
		df.loc[df['Odds'] < 0, 'Odds'] = 100 / abs(df['Odds'])

		df['bet'] = (df['Prediction'] - df['Line']).abs() > self.betting_threshold

		df['win_bet'] = ((df['Prediction'] - df['Line']) * (df['Results'] - df['Line'])) > 0

		df['gain/loss'] = np.zeros(len(df), dtype='float')
		df.loc[(df['bet']) & (~df['win_bet']), 'gain/loss'] = -1
		df.loc[(df['bet']) & (df['win_bet']), 'gain/loss'] = df['Odds']

		print(f'Total games: {len(df)}')
		print(f'Number of bets: {df["bet"].sum()}')
		print(f'Number of wins: {df.loc[df["bet"] == True, "win_bet"].sum()}')
		print(f'Win rate: {round(df.loc[df["bet"] == True, "win_bet"].sum() / df["bet"].sum(), 3)}')
		print(f'Total gain/loss: {df["gain/loss"].sum()}')

	def train_and_evaluate_model(self):
		test = self.final_df.sample(frac=0.35)
		train = self.final_df.drop(test.index)

		x_train = train.drop(['predict_col', self.line_column_title, self.odds_column_title], axis=1)
		y_train = train['predict_col']

		x_test = test.drop(['predict_col', self.line_column_title, self.odds_column_title], axis=1)

		self.model.train(x_train, y_train)

		is_predictions = self.model.test(x_train)
		oos_predictions = self.model.test(x_test)

		train_df = train[[self.line_column_title, self.odds_column_title, 'predict_col']]
		train_df.loc[:, 'Prediction'] = np.array(is_predictions, dtype='float')
		train_df.columns = ['Line', 'Odds', 'Results', 'Prediction']

		test_df = test[[self.line_column_title, self.odds_column_title, 'predict_col']]
		test_df.loc[:, 'Prediction'] = np.array(oos_predictions, dtype='float')
		test_df.columns = ['Line', 'Odds', 'Results', 'Prediction']

		return train_df, test_df

	
ModelTester(predict_type='Spread', odds_type='best', betting_threshold=15)