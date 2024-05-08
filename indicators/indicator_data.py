import pandas as pd
import sys
sys.path.append('..')
from useful_functions import get_odds_data

class IndicatorData():
	def __init__(self, predict_type='Spread'):
		self.predict_type = predict_type

		self.df = self.load_data()

		self.df.to_csv('temp_df.csv')
		
		# print(self.df.columns)

	def load_data(self, stat_path='../data/all_stat_data.csv', odds_path='../data/all_odds_data.csv'):
		stat_data = pd.read_csv(stat_path)
		# stat_data = stat_data[(stat_data['Date'] >= start_date) & (stat_data['Date'] <= end_date)]

		# team points - opponent points = margin	 (120-105 win --> 15 margin prediction)
		stat_data['predict_col_spread'] = stat_data['pts'] - stat_data['pts_opp']

		# team points + opponent points = total		 (120-105 game --> 225 total prediction)
		stat_data['predict_col_OU'] = stat_data['pts_opp'] + stat_data['pts']


		odds_data_spread = get_odds_data('Spread', '1900-01-01', '2025-01-01', odds_path=odds_path)
		odds_data_OU = get_odds_data('OU', '1900-01-01', '2025-01-01', odds_path=odds_path)

		# merge stat and odds data
		final_df = pd.merge(stat_data, odds_data_spread, on=['Date', 'Team', 'OppTeam'], suffixes=('', '_Spread'))
		final_df = pd.merge(final_df, odds_data_OU, on=['Date', 'Team', 'OppTeam'], suffixes=('', '_OU'))

		final_df.rename(columns={'Best_Line_Option_1': 'Best_Line_Option_1_Spread', 'Best_Line_Option_2': 'Best_Line_Option_2_Spread'}, inplace=True)
		return final_df

		# final_df_columns = (
		# 	['home_game', 'days_of_rest', 'travel_miles', 'days_of_rest_opp', 'travel_miles_opp'] + 
		# 	[col for col in stat_data.columns if 'last_8' in col] +
		# 	['predict_col', 'Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2']
		# )
		# return final_df[final_df_columns]

	def bounce_back_after_bad_shooting(self):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(['Season', 'Team'])
		temp_df['prev_game_shooting'] = grouped['fg_pct'].shift()
		temp_df['prev_game_shooting_3'] = grouped['fg3_pct'].shift()
		temp_df['bad_shooting_last_game'] = (temp_df['prev_game_shooting'] < 0.19) | (temp_df['prev_game_shooting_3'] < 0.13)

		temp_df['winOver'] = temp_df.apply(lambda row: True if row['predict_col_OU'] > min(row['Best_Line_Option_1_OU'], row['Best_Line_Option_2_OU']) else False, axis=1)

		print(f'Number of games that fit bad shooting in previous game: {temp_df["bad_shooting_last_game"].sum()}')
		print(f'Over hits for those teams next game: {temp_df[temp_df["bad_shooting_last_game"]]["winOver"].sum()}')
		print(f'Win rate Over for those teams: {temp_df[temp_df["bad_shooting_last_game"]]["winOver"].sum() / temp_df["bad_shooting_last_game"].sum()}')

		self.df['betOver'] = False
		self.df.loc[temp_df[temp_df['bad_shooting_last_game']].index, 'betOver'] = True

	def fade_favorites_after_blowout_ats(self):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(['Season', 'Team'])
		temp_df['pts_diff_last_game'] = (grouped['pts'].shift() - grouped['pts_opp'].shift()).fillna(0)

		temp_df['blowout_last_game'] = temp_df['pts_diff_last_game'] >= 15
		temp_df['big_favorite'] = temp_df.apply(lambda row: True if max(row['Best_Line_Option_1_Spread'], row['Best_Line_Option_2_Spread']) > 14 else False, axis=1)
		temp_df['favorite_after_blowout_ats'] = temp_df['blowout_last_game'] & temp_df['big_favorite']

		temp_df['winATS'] = temp_df['predict_col_spread'] > temp_df['Best_Line_Option_1_Spread']

		print(f'Number of games that fit home favorites after blowout: {temp_df["favorite_after_blowout_ats"].sum()}')
		print(f'Wins ATS for those teams: {temp_df[temp_df["favorite_after_blowout_ats"]]["winATS"].sum()}')
		print(f'Win rate ATS for those teams: {temp_df[temp_df["favorite_after_blowout_ats"]]["winATS"].sum() / temp_df["favorite_after_blowout_ats"].sum()}')

		self.df['fadeATS'] = False
		self.df.loc[temp_df[temp_df['favorite_after_blowout_ats']].index, 'fadeATS'] = True

	def three_games_in_four_nights(self):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(['Season', 'Team'])

		temp_df['game_date_two_games_ago'] = pd.to_datetime(grouped['Date'].shift(2))
		temp_df['date_diff'] = pd.to_datetime(temp_df['Date']) - temp_df['game_date_two_games_ago']
		temp_df['three_games_in_four_days'] = (temp_df['date_diff'].dt.days <= 3) #& (temp_df['travel_miles'] > 1000) & (~temp_df['home_game'])

		temp_df['winOver'] = temp_df['predict_col_OU'] > temp_df['Best_Line_Option_1_OU']

		print(f'Number of games that fit home three games in four nights: {temp_df["three_games_in_four_days"].sum()}')
		print(f'Over hits for those teams: {temp_df[temp_df["three_games_in_four_days"]]["winOver"].sum()}')
		print(f'Over hit rate for those teams: {temp_df[temp_df["three_games_in_four_days"]]["winOver"].sum() / temp_df["three_games_in_four_days"].sum()}')
		
		print("DONT USE THIS ONE")
		# self.df['betOver'] = False
		# self.df.loc[temp_df[temp_df['three_games_in_four_days']].index, 'betOver'] = True


id = IndicatorData()
# id.fade_favorites_after_blowout_ats()
# id.bounce_back_after_bad_shooting()
# id.three_games_in_four_nights()