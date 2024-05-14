from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from NeuralNetRegressor import NeuralNetRegressor
from LinearRegressor import LinearRegressor
from trees.PERTLearner import PERTLearner
from trees.CARTLearner import CARTLearner
from trees.BootstrapLearner import BootstrapLearner
from trees.XGBoostRegressor import XGBoostRegressor
import matplotlib.pyplot as plt
from useful_functions import get_odds_data
from indicators.indicator_data import IndicatorData
import itertools


class ModelTester:
	def __init__(
		self,
		model_class=LinearRegressor,
		start_date="2013-10-29",
		end_date="2023-04-09",
		predict_type="Spread",
		odds_type="best",
		betting_threshold=5,
		**kwargs,
	):
		if predict_type not in ["Spread", "OU", "Both"]:
			raise ValueError('predict_type must be either "spread" or "OU" or "Both')
		if odds_type not in ["best", "worst", "average"]:
			raise ValueError('odds_type must be either "best", "worst", or "average"')

		self.predict_type = predict_type
		self.odds_type = odds_type
		self.betting_threshold = betting_threshold
		self.model_class = model_class

		self.load_data(start_date, end_date)
		self.model = model_class(**kwargs)

		if model_class == IndicatorData:
			self.test_df_result = self.test_indicator_systems()
		else:
			self.train_df_result, self.test_df_result = self.train_model()

		# self.graph_betting_threshold(self.test_df_result)

	def graph_betting_threshold(self, test_df_result, plot=True):
		# graph betting threshold vs. win percentage and unit gain/loss

		max_threshold = 8 if self.predict_type == "Spread" else 16

		thresholds = np.arange(1, max_threshold, 0.1)
		win_rates = []
		unit_gains = []
		bets_made_list = []

		for threshold in thresholds:
			res = self.bet_with_predictions(test_df_result, betting_threshold=threshold)
			# return bets_made, win_rate, kelly_gain_or_loss, normal_gain_or_loss, df['cumulative_total'], df['kelly_cumulative_total']

			win_rates.append(res[1])
			unit_gains.append(res[4].iloc[-1])
			bets_made_list.append(res[0])

		if not plot:
			return thresholds, win_rates, unit_gains, bets_made_list

		plt.plot(thresholds, win_rates, label="Win Rate")
		plt.xlabel("Betting Threshold")
		plt.ylabel("Win Rate")
		plt.title(
			"Betting Threshold vs. Win Rate for " + self.predict_type + " Predictions"
		)
		plt.show()

		plt.plot(thresholds, unit_gains, label="Units Gained/Lost")
		plt.xlabel("Betting Threshold")
		plt.ylabel("Units Gained/Lost")
		plt.title(
			"Betting Threshold vs. Unit Gain/Loss for "
			+ self.predict_type
			+ " Predictions"
		)
		plt.show()

	def load_data(self, start_date, end_date, stat_path="./data/all_stat_data.csv"):
		# read csv and filter by date
		stat_data = pd.read_csv(stat_path)
		stat_data = stat_data[
			(stat_data["Date"] >= start_date) & (stat_data["Date"] <= end_date)
		]

		if self.predict_type in ["Spread", "Both"]:
			# team points - opponent points = margin	 (120-105 win --> 15 margin prediction)
			stat_data["predict_col_Spread"] = stat_data["pts"] - stat_data["pts_opp"]

		if self.predict_type in ["OU", "Both"]:
			# team points + opponent points = total		 (120-105 game --> 225 total prediction)
			stat_data["predict_col_OU"] = stat_data["pts_opp"] + stat_data["pts"]

		# get odds data and merge with stat data
		if self.predict_type == "Both":
			odds_data_spread = get_odds_data("Spread", start_date, end_date)
			odds_data_OU = get_odds_data("OU", start_date, end_date)
			odds_data = pd.merge(
				odds_data_spread,
				odds_data_OU,
				on=["Date", "Team", "OppTeam"],
				suffixes=("_Spread", "_OU"),
			)
			self.final_df = pd.merge(
				stat_data, odds_data, on=["Date", "Team", "OppTeam"]
			)
		else:
			odds_data = get_odds_data(self.predict_type, start_date, end_date)
			self.final_df = pd.merge(
				stat_data,
				odds_data,
				on=["Date", "Team", "OppTeam"],
				suffixes=("", "_" + self.predict_type),
			)

		# limit columns
		final_df_columns = (
			[
				"home_game",
				"days_of_rest",
				"travel_miles",
				"days_of_rest_opp",
				"travel_miles_opp",
			]
			+ [
				col
				for col in stat_data.columns
				if "last_8" in col or "predict_col" in col
			]
			+ [
				col
				for col in odds_data.columns
				if "Best_Line" in col or "Best_Odds" in col
			]
		)
		self.final_df = self.final_df[final_df_columns]

		if self.model_class == IndicatorData:
			start_column = self.final_df.columns.get_loc("predict_col_Spread")
			self.final_df = self.final_df.iloc[:, start_column:]

	def bet_with_predictions(self, df, print_results=False, betting_threshold=None):
		# df columns are: 'Odds', 'Prediction', 'Results', 'Line'
		# bet on games where the model's prediction differs from the line by more than the betting threshold

		STARTING_CASH = 100

		df = df.reset_index()

		if not betting_threshold:
			betting_threshold = self.betting_threshold

		df["bet"] = (df["Prediction"] - df["Line"]).abs() > betting_threshold
		df["win_bet"] = (
			(df["Prediction"] - df["Line"]) * (df["Results"] - df["Line"])
		) > 0
		df['lose_bet'] = (
			(df["Prediction"] - df["Line"]) * (df["Results"] - df["Line"])
		) < 0

		df["prediction_diff"] = (df["Prediction"] - df["Line"]).abs()
		df["win_probability"] = 0.5 + 0.5 * np.tanh(df["prediction_diff"] / 50)
		df["kelly_wager_size"] = (
			df["win_probability"] - (1 - df["win_probability"]) / df["Odds"]
		)
		df["kelly_wager_size"] = df["kelly_wager_size"].apply(
			lambda x: 0 if x < 0 else x
		)

		# print("Kelly stake when bet wins:", df.loc[df["bet"] & df["win_bet"], "kelly_wager_size"].mean())
		# print("Kelly stake when bet loses:", df.loc[df["bet"] & ~df["win_bet"], "kelly_wager_size"].mean())
		# print('Normal betting average stake:', df.loc[df["bet"], "kelly_wager_size"].median() * STARTING_CASH)
		# print("Total kelly money staked:", df.loc[df["bet"], "kelly_wager_size"].sum() * STARTING_CASH)
		# print("Total normal money staked:", df["bet"].sum() * df["kelly_wager_size"].median() * STARTING_CASH)

		df["kelly_gain_loss"] = np.zeros(len(df), dtype="float")
		df["normal_gain_loss"] = np.zeros(len(df), dtype="float")

		df.loc[(df["bet"]) & (df["win_bet"]), "kelly_gain_loss"] = (
			df["kelly_wager_size"] * STARTING_CASH * df["Odds"]
		)
		df.loc[(df["bet"]) & (df["lose_bet"]), "kelly_gain_loss"] = (
			-1 * df["kelly_wager_size"] * STARTING_CASH
		)

		df.loc[(df["bet"]) & (df["win_bet"]), "normal_gain_loss"] = (
			df["kelly_wager_size"].median() * STARTING_CASH * df["Odds"]
		)
		df.loc[(df["bet"]) & (df["lose_bet"]), "normal_gain_loss"] = (
			-1 * df["kelly_wager_size"].median() * STARTING_CASH
		)



		df["kelly_cumulative_return"] = df["kelly_gain_loss"].cumsum() #/ STARTING_CASH
		df["cumulative_return"] = df["normal_gain_loss"].cumsum() #/ STARTING_CASH

		if not df["bet"].sum():
			return 0, 0, 0

		bets_made = df["bet"].sum()
		bets_won = df.loc[df["bet"] & df["win_bet"], "bet"].sum()
		bets_lost = df.loc[df["bet"] & df["lose_bet"], "bet"].sum()
		win_rate = round(
			bets_won / (bets_won + bets_lost), 3
		)
		kelly_gain_or_loss = df["kelly_gain_loss"].sum()
		normal_gain_or_loss = df["normal_gain_loss"].sum()

		if print_results:
			print(f"Testing model with {len(df)} total games")
			print(f"Number of bets: {bets_made}")
			print(f'Number of wins: {df.loc[df["bet"] == True, "win_bet"].sum()}')
			print(f"Win rate: {win_rate}")
			print(f"Normal gain/loss: {normal_gain_or_loss}")
			print(f"Kelly gain/loss: {kelly_gain_or_loss}")
			plt.plot(df.index, df["cumulative_return"])
			plt.plot(df.index, df["kelly_cumulative_return"])
			plt.show() 
	
		return (
			bets_made,
			win_rate,
			kelly_gain_or_loss,
			normal_gain_or_loss,
			df["kelly_cumulative_return"],
			df["cumulative_return"],
		)


	def train_model(self, test_split=.35):
		predict_col_name = "predict_col_" + self.predict_type
		some_columns = [
			"Best_Line_Option_1",
			"Best_Line_Option_2",
			"Best_Odds_Option_1",
			"Best_Odds_Option_2",
			predict_col_name,
		]

		test = self.final_df.sample(frac=test_split)
		train = self.final_df.drop(test.index)

		x_train = train.drop(some_columns, axis=1)
		y_train = train[predict_col_name]

		x_test = test.drop(some_columns, axis=1)

		print("Training model...")
		self.losses = self.model.train(x_train, y_train)
		print("Done training model")

		is_predictions = self.model.test(x_train)
		oos_predictions = self.model.test(x_test)

		train_df = train[some_columns]
		train_df.loc[:, "Prediction"] = np.array(is_predictions, dtype="float")

		test_df = test[some_columns]
		test_df.loc[:, "Prediction"] = np.array(oos_predictions, dtype="float")

		for df in [train_df, test_df]:
			if self.odds_type == "best":
				df["Line"] = df.apply(
					lambda x: (
						x["Best_Line_Option_1"]
						if abs(x["Best_Line_Option_1"] - x["Prediction"])
						> abs(x["Best_Line_Option_2"] - x["Prediction"])
						else x["Best_Line_Option_2"]
					),
					axis=1,
				)
				df["Odds"] = df.apply(
					lambda x: max(x["Best_Odds_Option_1"], x["Best_Odds_Option_2"]),
					axis=1,
				)
			elif self.odds_type == "worst":
				df["Line"] = df.apply(
					lambda x: (
						x["Best_Line_Option_1"]
						if abs(x["Best_Line_Option_1"] - x["Prediction"])
						< abs(x["Best_Line_Option_2"] - x["Prediction"])
						else x["Best_Line_Option_2"]
					),
					axis=1,
				)
				df["Odds"] = df.apply(
					lambda x: min(x["Best_Odds_Option_1"], x["Best_Odds_Option_2"]),
					axis=1,
				)
			elif self.odds_type == "average":
				df["Line"] = df.apply(
					lambda x: (x["Best_Line_Option_1"] + x["Best_Line_Option_2"]) / 2,
					axis=1,
				)
				df["Odds"] = df.apply(
					lambda x: (x["Best_Odds_Option_1"] + x["Best_Odds_Option_2"]) / 2,
					axis=1,
				)

			df.drop(
				[
					"Best_Line_Option_1",
					"Best_Line_Option_2",
					"Best_Odds_Option_1",
					"Best_Odds_Option_2",
				],
				axis=1,
				inplace=True,
			)
			df.columns = ["Results", "Prediction", "Line", "Odds"]

		return train_df, test_df

	def test_indicator_systems(self):

		indicator_cols = self.model.df[self.model.indicator_cols]

		indicator_df = self.final_df.copy()
		indicator_df = pd.concat([indicator_df, indicator_cols], axis=1)

		indicator_df["best_odds_Spread"] = indicator_df.apply(
			lambda x: max(
				x["Best_Odds_Option_1_Spread"], x["Best_Odds_Option_2_Spread"]
			),
			axis=1,
		)

		indicator_df["best_odds_OU"] = indicator_df.apply(
			lambda x: max(x["Best_Odds_Option_1_OU"], x["Best_Odds_Option_2_OU"]),
			axis=1,
		)
		indicator_df["worst_odds_OU"] = indicator_df.apply(
			lambda x: min(x["Best_Odds_Option_1_OU"], x["Best_Odds_Option_2_OU"]),
			axis=1,
		)

		indicator_df["best_line_Spread"] = indicator_df.apply(
			lambda x: min(
				x["Best_Line_Option_1_Spread"], x["Best_Line_Option_2_Spread"]
			),
			axis=1,
		)
		indicator_df["worst_line_Spread"] = indicator_df.apply(
			lambda x: max(
				x["Best_Line_Option_1_Spread"], x["Best_Line_Option_2_Spread"]
			),
			axis=1,
		)

		indicator_df["best_line_OU"] = indicator_df.apply(
			lambda x: min(x["Best_Line_Option_1_OU"], x["Best_Line_Option_2_OU"]),
			axis=1,
		)
		indicator_df["worst_line_OU"] = indicator_df.apply(
			lambda x: max(x["Best_Line_Option_1_OU"], x["Best_Line_Option_2_OU"]),
			axis=1,
		)

		bets_to_make = []
		for _, row in indicator_df.iterrows():

			# this contributes 10.6 units
			if row["fadeATS"]:
				bets_to_make.append(
					[
						row["best_odds_Spread"],
						-1000,
						row["predict_col_Spread"],
						row["worst_line_Spread"],
					]
				)

			# this contributes 5.45 units
			if row["betOver"]:
				bets_to_make.append(
					[
						row["best_odds_OU"],
						1000,
						row["predict_col_OU"],
						row["best_line_OU"],
					]
				)

			# this contributes 394.8 units, win rate of 0.563
			if row["useTunnel"]:
				bets_to_make.append(
					[0.93, 1000, row["predict_col_OU"], row["best_line_OU"]]
				)
				bets_to_make.append(
					[0.93, -1000, row["predict_col_OU"], row["worst_line_OU"]]
				)

			# this contributes 5.56 units

			if row["fadeATS_2"]:
				bets_to_make.append(
					[
						row["best_odds_Spread"],
						-1000,
						row["predict_col_Spread"],
						row["worst_line_Spread"],
					]
				)

		output_df = pd.DataFrame(
			bets_to_make, columns=["Odds", "Prediction", "Results", "Line"]
		)
		return output_df

	def graph_training_losses(self):
		max_shown_loss = min(self.losses) * 10
		show_losses = [
			loss if loss < max_shown_loss else max_shown_loss for loss in self.losses
		]
		plt.plot(show_losses)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title("Training Loss Over Time")
		plt.show()
        

def compare_feature_importance(model, feature_names):
   
    if hasattr(model, 'feature_importance_values'):  # CART
        importances = model.feature_importance_values
        indices = np.argsort(importances)[::-1]

        # Plot Feature Importance (tree-based)
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices], color='b', align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.show()
        
    elif hasattr(model, 'get_score'):  # XGBoost
        importances = model.model.get_score(importance_type='weight')  # Can also use 'gain' or 'cover'
        # XGBoost get_score() might return a dictionary where features are keys and importance are values
        # Make sure features from the model are matched by order and existence in feature_names
        sorted_importances = np.array([importances.get(f, 0) for f in feature_names])
        indices = np.argsort(sorted_importances)[::-1]

    elif hasattr(model, 'coef'):  # LinReg models
        importances = np.abs(model.model.coef_)
        indices = np.argsort(importances)[::-1]

        # Plot Feature Importance (linear model)
        plt.figure(figsize=(10, 6))
        plt.title('Feature Coefficients')
        plt.bar(range(len(importances)), importances[indices], color='r', align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.show()

    else:
        print("The model provided does not have an attribute to infer feature importance.")
        
def compare_models(self, model_classes, plot=True):
       
        model_results = {}
        
        win_rates = []
        model_names = []

        for model_name, model_info in model_classes.items():
            print(f"Training and testing model: {model_name}")
            model_class = model_info['class']
            model_params = model_info.get('params', {})
            self.model = model_class(**model_params)
            self.train_df_result, self.test_df_result = self.train_model(self.test_split)
            bets_made, win_rate, kelly_gain_or_loss, normal_gain_or_loss = self.bet_with_predictions(self.test_df_result, print_results=False)

            model_results[model_name] = (bets_made, win_rate, kelly_gain_or_loss, normal_gain_or_loss)
            win_rates.append(win_rate)
            model_names.append(model_name)

            if plot:
                model_win_rates = model_results[model_name][1]
                plt.plot(model_win_rates, label=model_name)

        if plot:
            fig, ax = plt.subplots()
            ax.bar(model_names, win_rates, color='blue')
            ax.set_xlabel('Models')
            ax.set_ylabel('Win Rate')
            ax.set_title('Comparison of Model Win Rates')
            ax.set_xticklabels(model_names)
            plt.show()

        return model_results
   
def compare_network_params():
    
    param_dict = {
        'lr': [0.0001],
        'dropout_prob': [0.26],
        'epochs': [165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215]
    }
    # Create all combinations of parameters from the parameter grid
    keys, values = zip(*param_dict.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    for params in param_combinations:
        print(f"Testing with parameters: {params}")
        m = ModelTester(model_class=NeuralNetRegressor, predict_type='OU', odds_type='best', betting_threshold=10, input_features=43, **params)
        bets_made, win_rate, kelly_gain_or_loss, normal_gain_or_loss = m.bet_with_predictions(m.test_df_result, print_results=True)
        m.graph_training_losses()
        results.append((params, bets_made, win_rate, kelly_gain_or_loss, normal_gain_or_loss))
    
    # Find the best parameter set based on win rate
    best_params = max(results, key=lambda x: x[2])
    print(f"Best parameters based on win rate: {best_params[0]}")
    return best_params


def compare_odd_types(
	model_class=LinearRegressor,
	predict_types=["Spread", "OU"],
	graph_type="win_rate",
	plot=True,
):
	fig, axs = plt.subplots(len(predict_types), 2)

	# compare best, worst, and average odds
	for odd_type in ["best", "worst", "average"]:
		print(odd_type)
		for i, predict_type in enumerate(predict_types):
			bets_made_copies = []
			win_rates_copies = []
			unit_gains_copies = []
			for _ in range(10):
				m = ModelTester(
					model_class=model_class,
					predict_type=predict_type,
					odds_type=odd_type,
				)
				thresholds, win_rates, unit_gains, bets_made = (
					m.graph_betting_threshold(m.test_df_result, plot=False)
				)

				bets_made_copies.append(bets_made)
				win_rates_copies.append(win_rates)
				unit_gains_copies.append(unit_gains)

			bets_made = np.array(bets_made_copies).mean(axis=0)
			win_rates = np.array(win_rates_copies).mean(axis=0)
			unit_gains = np.array(unit_gains_copies).mean(axis=0)

			to_graph = win_rates if graph_type == "win_rate" else unit_gains
			axs[1, i].plot(thresholds, to_graph, label=odd_type.capitalize() + " Odds")
			axs[0, i].plot(
				thresholds,
				bets_made,
				label="Bets Made with " + odd_type.capitalize() + " Odds",
				linestyle="--",
			)
			axs[0, i].set(title=predict_type + " Predictions")

			if odd_type == "best":
				axs[1, i].plot(
					thresholds,
					[0.525] * len(thresholds),
					label="Breakeven Win Rate",
					linestyle="--",
					color="gray",
				)

	for i in range(len(predict_types)):
		# axs[1, i].plot(thresholds, [0.53] * len(thresholds), label='Breakeven Win Rate', linestyle='--', color='gray')
		axs[1, i].legend()
		axs[1, i].set(xlabel="Betting Threshold")

	axs[0, 0].set(ylabel="Bets Placed")

	if graph_type == "win_rate":
		axs[1, 0].set(ylabel="Win Rate")
		fig.suptitle(
			"Betting Threshold vs. Win Rate for Predictions With Various Odds Types"
		)
	else:
		axs[1, 0].set(ylabel="Percent Gain/Loss")
		fig.suptitle(
			"Betting Threshold vs. Percent Gain/Loss for Predictions With Various Odds Types"
		)
		plt.gca().set_yticklabels([f"{x:.0%}" for x in plt.gca().get_yticks()])

	plt.show()


def compare_PERT_leaf_sizes(predict_type="Spread", graph_type="win_rate"):
	# compare leaf sizes
	results = []
	leaf_sizes = [5, 10, 20, 30, 40, 50]
	for leaf_size in leaf_sizes:
		print("Leaf size:", leaf_size)
		m = ModelTester(
			model_class=PERTLearner, predict_type=predict_type, leaf_size=leaf_size
		)
		bets_made, win_rate, unit_gains = m.bet_with_predictions(
			m.test_df_result, print_results=False
		)

		if graph_type == "win_rate":
			results.append(win_rate)
		else:
			results.append(unit_gains)

	plt.plot(leaf_sizes, results)
	plt.xlabel("Leaf Size")

	if graph_type == "win_rate":
		plt.ylabel("Win Rate")
		plt.title(
			"Leaf Size vs. Win Rate for PERT Learner" + predict_type + " Predictions"
		)
	else:
		plt.ylabel("Units Gained/Lost")
		plt.title(
			"Leaf Size vs. Win Rate for PERT Learner" + predict_type + " Predictions"
		)

	plt.show()


def graph_odd_types_with_all_bets():
	odd_types = ["best", "average", "worst"]
	win_rates = []
	gains = []
	for odd_type in odd_types:
		m = ModelTester(
			model_class=LinearRegressor, predict_type="Spread", odds_type=odd_type
		)
		# print(m.test_df_result)
		bets_made, win_rate, gain_or_loss = m.bet_with_predictions(
			m.test_df_result, betting_threshold=-1
		)
		win_rates.append(win_rate)
		gains.append(gain_or_loss / 100)

	stats = {
		"Win Rate": win_rates,
		"Unit Gain/Loss (divided by 100)": gains,
	}

	x = np.arange(len(odd_types))  # the label locations
	width = 0.25  # the width of the bars
	multiplier = 0

	fig, ax = plt.subplots(layout="constrained")

	for attribute, measurement in stats.items():
		offset = width * multiplier
		rects = ax.bar(x + offset, measurement, width, label=attribute)
		ax.bar_label(rects, padding=3)
		multiplier += 1

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel("Length (mm)")
	ax.set_title("Penguin attributes by species")
	ax.set_xticks(x + width, odd_types)
	ax.legend(loc="upper left", ncols=3)
	# ax.set_ylim(0, 250)

	plt.show()



def compare_kelly_to_normal():
	colors = {"best": "green", "worst": "red", "average": "orange"}
	for odds in ["best", "worst", "average"]:
		average_totals = []
		average_kelly_totals = []

		for i in range(1):
			m = ModelTester(
				model_class=LinearRegressor,
				predict_type="OU",
				odds_type=odds,
				betting_threshold=10,
			)
			res = m.bet_with_predictions(m.test_df_result)
			cumulative_total = res[5]
			kelly_cumulative_total = res[4]
			average_totals.append(cumulative_total)
			average_kelly_totals.append(kelly_cumulative_total)

		average_totals = np.array(average_totals).mean(axis=0)
		average_kelly_totals = np.array(average_kelly_totals).mean(axis=0)

		plt.plot(
			average_totals,
			label="Normal Betting, " + odds.capitalize() + " Odds",
			linestyle=(0, (1, 5)),
			color=colors[odds],
		)
		plt.plot(
			average_kelly_totals,
			label="Kelly Criterion Betting, " + odds.capitalize() + " Odds",
			color=colors[odds],
		)

	plt.xlabel("Bet number")
	plt.ylabel("$ Gain")
	plt.title("Out of Sample Betting Results for OU Predictions (Betting Threshold=10, Starting Cash=100)")
	plt.legend()
	plt.show()


compare_kelly_to_normal()

# comapare_odd_types = compare_odd_types(graph_type="win_rate")

# m = ModelTester(model_class=CARTLearner, predict_type='OU', odds_type='best', betting_threshold=10, leaf_size=10)
# m.bet_with_predictions(m.test_df_result, print_results=True)

# m = ModelTester(model_class=XGBoostRegressor, predict_type='OU', odds_type='best', betting_threshold=10)
# m.bet_with_predictions(m.test_df_result, print_results=True)

# m = ModelTester(model_class=BootstrapLearner, predict_type='OU', odds_type='best', betting_threshold=10, constituent=PERTLearner, bags = 10, model_kwargs={"leaf_size": 10})
# m.bet_with_predictions(m.test_df_result, print_results=True)

# m = ModelTester(model_class=NeuralNetRegressor, predict_type='OU', odds_type='best', betting_threshold=10, input_features=43)
# m.bet_with_predictions(m.test_df_result, print_results=True)
# m.graph_training_losses()

# m = ModelTester(model_class=IndicatorData, predict_type='Both')
# m.bet_with_predictions(m.test_df_result, print_results=True)

# compare_odd_types(model_class=IndicatorData, predict_types=['Both'])


# graph_odd_types_with_all_bets()

win_rates = []
bests_over_trials = []


for i in range(12):
    best_network_params = compare_network_params()
    print("Best model parameters found:", best_network_params)
    bests_over_trials.append(best_network_params)
    win_rates.append(best_network_params[2])

print(f'Win rates over 10 trials: {win_rates}')
print(f'List of bests: {bests_over_trials}')


# model_classes = {
#     'Neural Network': {'class': NeuralNetRegressor, 'params': {'input_features': 43, 'dropout_prob': 0.3, 'lr': 0.0001}},
#     'Linear Regression': {'class': LinearRegressor, 'params': {}},
#     'CART Learner': {'class': CARTLearner, 'params': {'leaf_size': 10}},
#     # Add other models here
# }

# model_evaluation = compare_models(model_classes)
# print("Model performance comparison:", model_evaluation)
