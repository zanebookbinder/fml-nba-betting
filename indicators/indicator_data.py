import pandas as pd
import sys

sys.path.append("..")
from useful_functions import get_odds_data


class IndicatorData:
	def __init__(self, stat_path="./data/all_stat_data.csv", odds_path="./data/all_odds_data.csv"):

		self.df = self.load_data(stat_path, odds_path)

		self.indicator_cols = []

		for func in [
			self.fade_favorites_after_blowout_ats,
			self.bounce_back_after_bad_shooting,
			self.three_games_in_four_nights,
			self.tunnel,
			self.fade_ats_after_scoring_135_points,
		]:
			res = func()
			# print()

			if res:
				self.indicator_cols.append(res)
				
	def load_data(
		self,
		stat_path,
		odds_path,
	):
		stat_data = pd.read_csv(stat_path)
		# stat_data = stat_data[(stat_data['Date'] >= start_date) & (stat_data['Date'] <= end_date)]

		# team points - opponent points = margin	 (120-105 win --> 15 margin prediction)
		stat_data["predict_col_spread"] = stat_data["pts"] - stat_data["pts_opp"]

		# team points + opponent points = total		 (120-105 game --> 225 total prediction)
		stat_data["predict_col_OU"] = stat_data["pts_opp"] + stat_data["pts"]

		odds_data_spread = get_odds_data(
			"Spread", "1900-01-01", "2025-01-01", odds_path=odds_path
		)
		odds_data_OU = get_odds_data(
			"OU", "1900-01-01", "2025-01-01", odds_path=odds_path
		)

		# merge stat and odds data
		final_df = pd.merge(
			stat_data,
			odds_data_spread,
			on=["Date", "Team", "OppTeam"],
			suffixes=("", "_Spread"),
		)
		final_df = pd.merge(
			final_df, odds_data_OU, on=["Date", "Team", "OppTeam"], suffixes=("", "_OU")
		)

		final_df.rename(
			columns={
				"Best_Line_Option_1": "Best_Line_Option_1_Spread",
				"Best_Line_Option_2": "Best_Line_Option_2_Spread",
			},
			inplace=True,
		)
		return final_df

		# final_df_columns = (
		# 	['home_game', 'days_of_rest', 'travel_miles', 'days_of_rest_opp', 'travel_miles_opp'] +
		# 	[col for col in stat_data.columns if 'last_8' in col] +
		# 	['predict_col', 'Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2']
		# )
		# return final_df[final_df_columns]

	def bounce_back_after_bad_shooting(self, print_results=False):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(["Season", "Team"])
		temp_df["prev_game_shooting"] = grouped["fg_pct"].shift()
		temp_df["prev_game_shooting_3"] = grouped["fg3_pct"].shift()
		temp_df["bad_shooting_last_game"] = (temp_df["prev_game_shooting"] < 0.19) | (
			temp_df["prev_game_shooting_3"] < 0.13
		)

		temp_df["winOver"] = temp_df.apply(
			lambda row: (
				True
				if row["predict_col_OU"]
				> min(row["Best_Line_Option_1_OU"], row["Best_Line_Option_2_OU"])
				else False
			),
			axis=1,
		)

		if print_results:
			print(
				f'Number of games that fit bad shooting in previous game: {temp_df["bad_shooting_last_game"].sum()}'
			)
			print(
				f'Over hits for those teams next game: {temp_df[temp_df["bad_shooting_last_game"]]["winOver"].sum()}'
			)
			print(
				f'Win rate Over for those teams: {temp_df[temp_df["bad_shooting_last_game"]]["winOver"].sum() / temp_df["bad_shooting_last_game"].sum()}'
			)

		self.df["betOver"] = False
		self.df.loc[temp_df[temp_df["bad_shooting_last_game"]].index, "betOver"] = True
		return "betOver"

	def fade_favorites_after_blowout_ats(self, print_results=False):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(["Season", "Team"])
		temp_df["pts_diff_last_game"] = (
			grouped["pts"].shift() - grouped["pts_opp"].shift()
		).fillna(0)

		temp_df["blowout_last_game"] = temp_df["pts_diff_last_game"] >= 15
		temp_df["big_favorite"] = temp_df.apply(
			lambda row: (
				True
				if max(
					row["Best_Line_Option_1_Spread"], row["Best_Line_Option_2_Spread"]
				)
				> 14
				else False
			),
			axis=1,
		)
		temp_df["favorite_after_blowout_ats"] = (
			temp_df["blowout_last_game"] & temp_df["big_favorite"]
		)

		temp_df["winATS"] = (
			temp_df["predict_col_spread"] > temp_df["Best_Line_Option_1_Spread"]
		)

		if print_results:
			print(
				f'Number of games that fit home favorites after blowout: {temp_df["favorite_after_blowout_ats"].sum()}'
			)
			print(
				f'Wins ATS for those teams: {temp_df[temp_df["favorite_after_blowout_ats"]]["winATS"].sum()}'
			)
			print(
				f'Win rate ATS for those teams: {temp_df[temp_df["favorite_after_blowout_ats"]]["winATS"].sum() / temp_df["favorite_after_blowout_ats"].sum()}'
			)

		self.df["fadeATS"] = False
		self.df.loc[temp_df[temp_df["favorite_after_blowout_ats"]].index, "fadeATS"] = (
			True
		)
		return "fadeATS"

	def three_games_in_four_nights(self, print_results=False):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(["Season", "Team"])

		temp_df["game_date_two_games_ago"] = pd.to_datetime(grouped["Date"].shift(2))
		temp_df["date_diff"] = (
			pd.to_datetime(temp_df["Date"]) - temp_df["game_date_two_games_ago"]
		)
		temp_df["three_games_in_four_days"] = (
			temp_df["date_diff"].dt.days <= 3
		)  # & (temp_df['travel_miles'] > 1000) & (~temp_df['home_game'])

		temp_df["winOver"] = (
			temp_df["predict_col_OU"] > temp_df["Best_Line_Option_1_OU"]
		)

		if print_results:
			print(
				f'Number of games that fit home three games in four nights: {temp_df["three_games_in_four_days"].sum()}'
			)
			print(
				f'Over hits for those teams: {temp_df[temp_df["three_games_in_four_days"]]["winOver"].sum()}'
			)
			print(
				f'Over hit rate for those teams: {temp_df[temp_df["three_games_in_four_days"]]["winOver"].sum() / temp_df["three_games_in_four_days"].sum()}'
			)

			print("DONT USE THIS ONE")
		return None
		# self.df['betOver'] = False
		# self.df.loc[temp_df[temp_df['three_games_in_four_days']].index, 'betOver'] = True

	def tunnel(self, print_results=False):
		temp_df = self.df.copy()
		temp_df["low_OU"] = temp_df.apply(
			lambda row: min(row["Best_Line_Option_1_OU"], row["Best_Line_Option_2_OU"]),
			axis=1,
		)
		temp_df["high_OU"] = temp_df.apply(
			lambda row: max(row["Best_Line_Option_1_OU"], row["Best_Line_Option_2_OU"]),
			axis=1,
		)

		temp_df["tunnel"] = temp_df.apply(
			lambda row: (row["high_OU"] - row["low_OU"]) > 4, axis=1
		)

		temp_df["winInTunnel"] = temp_df.apply(
			lambda row: (
				True
				if row["predict_col_OU"] > row["low_OU"]
				and row["predict_col_OU"] < row["high_OU"]
				else False
			),
			axis=1,
		)

		temp_df["profit"] = 0
		temp_df["profit"] = temp_df.apply(
			lambda row: (
				row["Best_Odds_Option_1_OU"] + row["Best_Odds_Option_2_OU"] - 1
				if row["winInTunnel"]
				else row["Best_Odds_Option_1_OU"] - 1
			),
			axis=1,
		)

		if print_results:
			print(f'Number of games that fit tunnel: {temp_df["tunnel"].sum()}')
			print(
				f'Tunnel hits for those teams: {temp_df[temp_df["tunnel"]]["winInTunnel"].sum()}'
			)
			print(
				f'Tunnel hit rate for those teams: {temp_df[temp_df["tunnel"]]["winInTunnel"].sum() / temp_df["tunnel"].sum()}'
			)
			print(f'Profit: {temp_df.loc[temp_df["tunnel"], "profit"].sum()} units')

		self.df["useTunnel"] = False
		self.df.loc[temp_df[temp_df["tunnel"]].index, "useTunnel"] = True
		return "useTunnel"

	def fade_ats_after_scoring_135_points(self, print_results=False):
		temp_df = self.df.copy()
		grouped = temp_df.groupby(["Season", "Team"])

		temp_df["prev_game_pts"] = grouped["pts"].shift()
		temp_df['scored_135_last_game'] = temp_df['prev_game_pts'] >= 135

		temp_df["winATS"] = temp_df.apply(
			lambda row: (
				True
				if row["predict_col_spread"]
				> row["Best_Line_Option_1_Spread"]
				else False
			),
			axis=1,
		)

		if print_results:
			print(
				f'Number of games that fit fade after scoring 135 points: {temp_df["scored_135_last_game"].sum()}'
			)
			print(
				f'ATS wins for those teams next game: {temp_df[temp_df["scored_135_last_game"]]["winATS"].sum()}'
			)
			print(
				f'Win rate ATS for those teams: {temp_df[temp_df["scored_135_last_game"]]["winATS"].sum() / temp_df["scored_135_last_game"].sum()}'
			)

		self.df["fadeATS_2"] = False
		self.df.loc[temp_df[temp_df["scored_135_last_game"]].index, "fadeATS_2"] = True
		return "fadeATS_2"

# id = IndicatorData()
# id.fade_ats_after_scoring_135_points()
