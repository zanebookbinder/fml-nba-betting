import pandas as pd

def get_odds_data(predict_type, start_date, end_date, odds_path='./data/all_odds_data.csv'):
		# read csv and filter by date
		odds_data = pd.read_csv(odds_path)
		odds_data = odds_data[(odds_data['Date'] >= start_date) & (odds_data['Date'] <= end_date)]
		
		# get best and worst lines for each game
		odds_data['Best_Line_Option_1'] = odds_data['Best_Line_' + predict_type]
		odds_data['Best_Line_Option_2'] = odds_data['Worst_Line_' + predict_type]

		# fill in missing lines
		odds_data.loc[odds_data['Best_Line_Option_2'] == 0, 'Best_Line_Option_2'] = odds_data.loc[odds_data['Best_Line_Option_2'] == 0, 'Best_Line_Option_1']

		# if predicting spread, make negative lines positive (margin of victory)
		if predict_type == 'Spread':
			odds_data['Best_Line_Option_1'] = -1 * odds_data['Best_Line_Option_1']
			odds_data['Best_Line_Option_2'] = -1 * odds_data['Best_Line_Option_2']

		# get best and worst odds for each game
		odds_data['Best_Odds_Option_1'] = odds_data['Best_Odds_' + predict_type]
		odds_data['Best_Odds_Option_2'] = odds_data['Worst_Odds_' + predict_type]

		for odds_col in ['Best_Odds_Option_1', 'Best_Odds_Option_2']:
			# fill in missing odds
			odds_data.loc[odds_data[odds_col] == 0, odds_col] = -110

			# calculate decimal odds from American odds
			odds_data.loc[odds_data[odds_col] > 0, odds_col] = odds_data[odds_col] / 100
			odds_data.loc[odds_data[odds_col] < 0, odds_col] = 100 / abs(odds_data[odds_col])

		columns_we_need = ['Date', 'Team', 'OppTeam', 'Best_Line_Option_1', 'Best_Line_Option_2', 'Best_Odds_Option_1', 'Best_Odds_Option_2']
		return odds_data[columns_we_need]