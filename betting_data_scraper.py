#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:33 2024

@author: pmullin
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
from datetime import datetime

CUMULATIVE_GAME_STATS = [
    "PTS",
    "AST",
    "REB",
    "TOV",
    "FG_PCT",
    "FT_PCT",
    "FG3_PCT",
]

GAME_STATS = [
    "PTS",
    "FG_PCT",
    "FT_PCT",
    "FG3_PCT",
    "AST",
    "REB",
    "TOV",
    "TEAM_WINS",
    "TEAM_LOSSES",
]


class OddsDataScraper:
    def __init__(self):
        # {abbreviation: city name}   ,  {city name: abbreviation}
        self.abbreviation_dict, self.city_name_dict = self.create_abbreviation_dict()

    def create_abbreviation_dict(
        self, path="./data/kaggle_odds_data/2012-13/raw_scores.txt"
    ):
        abbrev = {}
        df = pd.read_csv(path)

        for i, row in df.iterrows():
            if row["TEAM_ABBREVIATION"] not in abbrev:
                abbrev[row["TEAM_ABBREVIATION"]] = row["TEAM_CITY_NAME"]

        abbrev["NOP"] = "New Orleans"
        abbrev['LAC'] = 'LA'
        abbrev['LAL'] = 'Los Angeles'
        reverse_abbrev = {value: key for key, value in abbrev.items()}
        return abbrev, reverse_abbrev

    def read_kaggle_odds_data(self, path="./data/kaggle_odds_data"):
        odds_data = pd.DataFrame()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "vegas.txt":
                    year_data = pd.read_csv(os.path.join(root, file))
                    year_data.drop(["TeamId", "GameId"], axis=1, inplace=True)
                    odds_data = pd.concat([odds_data, year_data], ignore_index=True)

        odds_data = odds_data.sort_values(by="Date").reset_index(drop=True)
        return odds_data

    def read_kaggle_game_data(self, path="./data/kaggle_odds_data"):
        game_data = pd.DataFrame()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "raw_scores.txt":
                    year_data = pd.read_csv(os.path.join(root, file))
                    year_data["PTS_QTR4"] = (
                        year_data["PTS_OT1"]
                        + year_data["PTS_OT2"]
                        + year_data["PTS_OT3"]
                        + year_data["PTS_OT4"]
                        + year_data["PTS_OT5"]
                        + year_data["PTS_OT6"]
                        + year_data["PTS_OT7"]
                        + year_data["PTS_OT8"]
                        + year_data["PTS_OT9"]
                        + year_data["PTS_OT10"]
                    )
                    year_data["TEAM_WINS"] = (
                        year_data["TEAM_WINS_LOSSES"].str.split("-").str[0].astype(int)
                    )
                    year_data["TEAM_LOSSES"] = (
                        year_data["TEAM_WINS_LOSSES"].str.split("-").str[1].astype(int)
                    )
                    year_data.drop(
                        [
                            "GAME_SEQUENCE",
                            "GAME_ID",
                            "TEAM_ID",
                            "PTS_OT1",
                            "PTS_OT2",
                            "PTS_OT3",
                            "PTS_OT4",
                            "PTS_OT5",
                            "PTS_OT6",
                            "PTS_OT7",
                            "PTS_OT8",
                            "PTS_OT9",
                            "PTS_OT10",
                            "TEAM_WINS_LOSSES",
                        ],
                        axis=1,
                        inplace=True,
                    )
                    year_data.rename(
                        columns={"GAME_DATE_EST": "Date", "TEAM_CITY_NAME": "Team"},
                        inplace=True,
                    )
                    game_data = pd.concat([game_data, year_data], ignore_index=True)

        game_data.loc[game_data["TEAM_ABBREVIATION"] == "LAC", "Team"] = "LA"
        game_data.loc[game_data["TEAM_ABBREVIATION"] == "LAL", "Team"] = "Los Angeles"
        game_data = game_data.sort_values(by="Date").reset_index(drop=True)
        return game_data

    def get_all_kaggle_odds_data(self):
        odds = self.read_kaggle_odds_data()
        odds["Team"] = odds["Team"].replace(
            {"L.A. Clippers": "LA", "L.A. Lakers": "Los Angeles"}
        )
        odds["OppTeam"] = odds["OppTeam"].replace(
            {"L.A. Clippers": "LA", "L.A. Lakers": "Los Angeles"}
        )
        game = self.read_kaggle_game_data()

        result = pd.merge(odds, game, on=["Date", "Team"], how="inner")
        result.drop(["Pts", "Spread", "Result", "Total", 'TEAM_ABBREVIATION'], axis=1, inplace=True)

        result.insert(0, "Season", "")
        result.insert(5, "Result", "")
        result["Date"] = pd.to_datetime(result["Date"], format="%Y-%m-%d")

        result["Season"] = result["Date"].apply(
            lambda x: (
                str(x.year) + "-" + str(x.year + 1)
                if x.month >= 6
                else str(x.year - 1) + "-" + str(x.year)
            )
        )

        result = pd.merge(
            result,
            result[["Date", "OppTeam"] + GAME_STATS],
            left_on=["Date", "Team"],
            right_on=["Date", "OppTeam"],
            suffixes=("", "_opp"),
        ).drop(["OppTeam_opp"], axis=1)

        result["Result"] = np.where(result["PTS"] > result["PTS_opp"], 1, 0)

        betting_data = result.iloc[:, :54].drop(columns=['Season','Location','Result'])
        result = pd.concat([result.iloc[:, :6], result.iloc[:, 54:]], axis=1)

        return betting_data

    def get_rotowire_data(self, path='https://www.rotowire.com/betting/nba/tables/games-archive.php'):
        data = requests.get(path).json()
        df = pd.DataFrame(data, columns=['game_date', 'home_team_abbrev', 'visit_team_abbrev',
                                        'line', 'game_over_under'])

        df = df.sort_values(by='game_date').reset_index(drop=True)
        df = df.rename(columns={'home_team_abbrev': 'Team', 'game_date': 'Date',
                                'visit_team_abbrev': 'OppTeam', 'line': 'Spread', 'game_over_under': 'O/U'})
        
        df['Location'] = 'home'
        df['Date'] = df['Date'].str.split(' ').str[0]
        df['Date'] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

        new_df = df.copy()
        new_df['Location'] = 'away'
        new_df['Team'], new_df['OppTeam'] = new_df['OppTeam'], new_df['Team']
        new_df['Spread'] = -new_df['Spread']

        df = pd.concat([df, new_df], axis=0).sort_values(by='Date').reset_index(drop=True)
        # df.to_csv('./data/rotowire_odds_data.csv', index=False)
        return df

    def combine_all_odds_data(self, kaggle, rotowire):
        rotowire['Team'] = rotowire['Team'].apply(lambda x: self.abbreviation_dict[x])
        rotowire['OppTeam'] = rotowire['OppTeam'].apply(lambda x: self.abbreviation_dict[x])

        max_kaggle_date = kaggle['Date'].max()
        rotowire = rotowire.loc[rotowire['Date'] > max_kaggle_date]

        rotowire = rotowire.rename(columns={'Spread': 'Best_Line_Spread', 'O/U': 'Best_Line_OU'})
        rotowire.drop(['Location'], axis=1, inplace=True)

        merged_df = pd.concat([kaggle, rotowire]).fillna(0)
        return merged_df

    def get_all_data(self):
        kaggle = self.get_all_kaggle_odds_data()
        rotowire = self.get_rotowire_data()
        merged_df = self.combine_all_odds_data(kaggle, rotowire)
        merged_df.to_csv('./data/compiled_odds_data.csv', index=False)

    # break up kaggle data into team and season dataframes
    def divide_df_by_team_and_year(self, df=pd.DataFrame(), path="./data/compiled_stat_data.csv"):
        if not df.shape[0]:
            df = pd.read_csv(path)
        teams = df["Team"].unique()
        years = df["Season"].unique()

        season_dict = {year: {team: None for team in teams} for year in years}

        for team in teams:
            for year in years:
                this_season = df.loc[
                    (df["Team"] == team) & (df["Season"] == year)
                ].reset_index(drop=True)
                season_dict[year][team] = this_season

        return season_dict

    # set up year-long and last 8 game stats (opponent stats should be compared to before-game average)
    def create_season_stats(self, season_df):
        # season_df[CUMULATIVE_GAME_STATS + ['PTS_opp']].to_csv('test_before.csv')
        fgm_column = season_df.columns.to_list().index('fgm')
        cumulative_stats = season_df.columns[fgm_column:]
        for stat in cumulative_stats:
            season_df["last_8_" + stat] = (
                season_df[stat].shift().rolling(window=8, min_periods=1).mean()
            )
            # season_df["last_8" + stat + "_opp"] = season_df[stat + "_opp"].rolling(window=8, min_periods=1).mean()
        season_df["last_8_wins"] = (
            season_df["Result"].shift().rolling(window=8, min_periods=1).sum()
        )
        return season_df

    def create_stats_df(self, odds_data_path='./data/all_odds_data.csv'):
        df = pd.read_csv('./data/kaggle_game_data/game.csv').rename(
            columns={
                'game_date': 'Date',
                'team_abbreviation_home': 'Team',
                'team_abbreviation_away': 'OppTeam',
                'wl_home': 'Result',
            }
        )
        df.insert(0, 'Location', 'home')
        odds_data = pd.read_csv(odds_data_path)

        min_odds_data_date = odds_data['Date'].min()
        df = df.loc[(df['season_type'] == 'Regular Season') & (df['Date'] >= min_odds_data_date) & (df['Team'].isin(self.abbreviation_dict.keys())) & (df['OppTeam'].isin(self.abbreviation_dict.keys()))]
        
        df['Team'] = df['Team'].apply(lambda x: self.abbreviation_dict[x])
        df['OppTeam'] = df['OppTeam'].apply(lambda x: self.abbreviation_dict[x])
        df['Result'] = df['Result'].apply(lambda x: 1 if x == 'W' else 0)

        df = df.drop(df.columns[df.columns.str.contains('_id_|matchup|wl|video|season_type')], axis=1)
        df = df.drop(['season_id', 'team_name_home', 'game_id', 'team_name_away', 'min', 'plus_minus_home', 'plus_minus_away'], axis=1)
        df.reset_index(drop=True, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"].str.split(' ').str[0], format="%Y-%m-%d")
        df.insert(0, "Season", "")
        df["Season"] = df["Date"].apply(
            lambda x: (
                str(x.year) + "-" + str(x.year + 1)
                if x.month >= 6
                else str(x.year - 1) + "-" + str(x.year)
            )
        )

        df.columns = df.columns.str.replace("_home", "")
        df.columns = df.columns.str.replace("_away", "_opp")

        df.insert(3, 'OppTeam', df.pop('OppTeam'))
        df.insert(1, 'Date', df.pop('Date'))

        opposite_df = df.copy()
        opposite_df['Location'] = 'away'
        opposite_df['Team'], opposite_df['OppTeam'] = opposite_df['OppTeam'], opposite_df['Team']
        opposite_df['Result'] = 1 - opposite_df['Result']
        opposite_df_columns = []
        for col in opposite_df.columns[6:]:
            if '_opp' in col:
                opposite_df_columns.append(col.replace('_opp', ''))
            else:
                opposite_df_columns.append(col + '_opp')

        opposite_df.columns = df.columns[:6].to_list() + opposite_df_columns
        opposite_df = opposite_df[df.columns]

        df = pd.concat([df, opposite_df], axis=0).sort_values(by='Date').reset_index(drop=True)
        return df

    def get_all_stats(self):
        df = self.create_stats_df()
        season_dict = self.divide_df_by_team_and_year(df, path=None)
        
        final_df = pd.DataFrame()
        for year in season_dict.keys():
            print(year)
            for team in season_dict[year].keys():
                final_df = pd.concat([final_df, self.create_season_stats(season_dict[year][team])], ignore_index=True)

        final_df.sort_values(by=['Date', 'Team', 'OppTeam'], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        final_df.to_csv('./data/all_stat_data.csv', index=False)

def main():
    scraper = OddsDataScraper()

    # scraper.get_all_data()
    scraper.get_all_stats()


if __name__ == "__main__":
    main()
