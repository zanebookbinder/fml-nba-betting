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
        self.abbreviation_dict = self.create_abbreviation_dict()

    def create_abbreviation_dict(
        self, path="./data/kaggle_data/2012-13/raw_scores.txt"
    ):
        abbrev = {}
        df = pd.read_csv(path)

        for i, row in df.iterrows():
            if row["TEAM_ABBREVIATION"] not in abbrev:
                abbrev[row["TEAM_ABBREVIATION"]] = row["TEAM_CITY_NAME"]

        abbrev["NOP"] = "New Orleans"
        return abbrev

    def read_kaggle_odds_data(self, path="./data/kaggle_data"):
        odds_data = pd.DataFrame()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "vegas.txt":
                    year_data = pd.read_csv(os.path.join(root, file))
                    year_data.drop(["TeamId", "GameId"], axis=1, inplace=True)
                    odds_data = pd.concat([odds_data, year_data], ignore_index=True)

        odds_data = odds_data.sort_values(by="Date").reset_index(drop=True)
        return odds_data

    def read_kaggle_game_data(self, path="./data/kaggle_data"):
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

    def get_all_kaggle_data(self):
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
        betting_data.to_csv("./data/compiled_odds_data.csv", index=False)

        result = pd.concat([result.iloc[:, :6], result.iloc[:, 54:]], axis=1)
        result.to_csv("./data/compiled_stat_data.csv", index=False)

        return result

    # break up kaggle data into team and season dataframes
    def divide_df_by_team_and_year(self, path="./data/compiled_stat_data.csv"):
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
        for stat in CUMULATIVE_GAME_STATS:
            season_df["LAST_8_" + stat] = (
                season_df[stat].shift().rolling(window=8, min_periods=1).mean()
            )
            season_df["LAST_8_" + stat + "_opp"] = season_df[stat + "_opp"].rolling(window=8, min_periods=1).mean()
        season_df["LAST_8_WINS"] = (
            season_df["Result"].shift().rolling(window=8, min_periods=1).sum()
        )
        return season_df

    def add_stats(self):
        season_dict = self.divide_df_by_team_and_year()
        df_list = []
        for teams in season_dict.values():
            for team_df in teams.values():
                df_list.append(self.create_season_stats(team_df))

        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        df.to_csv('./data/all_games_with_recent_stats.csv')

def main():
    scraper = OddsDataScraper()

    # scraper.get_all_kaggle_data()
    scraper.add_stats()


if __name__ == "__main__":
    main()
