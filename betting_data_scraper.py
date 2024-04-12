#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:33 2024

@author: pmullin
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import os


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
                    year_data.drop(
                        ["TeamId", "GameId"], axis=1, inplace=True
                    )
                    odds_data = pd.concat([odds_data, year_data], ignore_index=True)

        odds_data = odds_data.sort_values(by="Date").reset_index(drop=True)
        return odds_data

    def read_kaggle_game_data(self, path="./data/kaggle_data"):
        game_data = pd.DataFrame()

        for root, dirs, files in os.walk(path):
            for file in files:
                if file == "raw_scores.txt":
                    year_data = pd.read_csv(os.path.join(root, file))
                    year_data.drop(
                        [
                            "GAME_SEQUENCE",
                            "GAME_ID",
                            "TEAM_ID",
                            "PTS_OT4",
                            "PTS_OT5",
                            "PTS_OT6",
                            "PTS_OT7",
                            "PTS_OT8",
                            "PTS_OT9",
                            "PTS_OT10",
                        ],
                        axis=1,
                        inplace=True,
                    )
                    year_data.rename(
                        columns={"GAME_DATE_EST": 'Date', 'TEAM_CITY_NAME': 'Team'},
                        inplace=True,
                    )
                    game_data = pd.concat([game_data, year_data], ignore_index=True)

        game_data.loc[game_data['TEAM_ABBREVIATION'] == 'LAC', 'Team'] = 'LA'
        game_data.loc[game_data['TEAM_ABBREVIATION'] == 'LAL', 'Team'] = 'Los Angeles'
        game_data = game_data.sort_values(by="Date").reset_index(drop=True)
        return game_data

    def get_all_kaggle_data(self, print_details=False):
        odds = self.read_kaggle_odds_data()
        odds['Team'] = odds['Team'].replace({'L.A. Clippers': 'LA', 'L.A. Lakers': 'Los Angeles'})
        odds['OppTeam'] = odds['OppTeam'].replace({'L.A. Clippers': 'LA', 'L.A. Lakers': 'Los Angeles'})
        game = self.read_kaggle_game_data()
        
        result = pd.merge(odds, game, on=['Date', 'Team'], how='inner')
        result.drop(['Pts', 'Spread', 'Result','Total'], axis=1, inplace=True)
        result.to_csv('./data/compiled_kaggle_data.csv')

        if print_details:
            print(odds.head())
            print(game.head())
            print(len(odds), len(game), len(result))
            print(result.head())

        return result


def main():
    scraper = OddsDataScraper()
    # scraper.scrape_rotowire_odds()
    res = scraper.get_all_kaggle_data()
    


if __name__ == "__main__":
    main()
