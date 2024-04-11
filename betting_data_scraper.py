#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:33 2024

@author: pmullin
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

class OddsDataScraper():
	# def __init__(self):
	# 	return
	
    def scrape_rotowire_odds(self, url='https://www.rotowire.com/betting/nba/tables/games-archive.php'):
        # Fetch the webpage content
        r = requests.get(url)
        data = r.json()

        # Read the json data into a pandas df
        df = pd.DataFrame(data, columns=['game_date', 'home_team_abbrev', 'visit_team_abbrev',
                                        'line', 'home_team_score', 'visit_team_score'])
        df['Final Score Spread'] = df['visit_team_score'] - df['home_team_score']
        df = df.sort_values(by='home_team_abbrev')
        df = df.rename(columns={'home_team_abbrev': 'Home Team', 'game_date': 'Date',
                                'visit_team_abbrev': 'Away Team', 'line': 'Betting Score Spread'})
        df = df.drop(columns=['home_team_score', 'visit_team_score'])
        df['Betting Line Error'] = df['Betting Score Spread'] - df['Final Score Spread']
        df.to_csv('nba_betting_lines.csv')

    def scrape_sportsbookreviews_online_odds(self, url='https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-'):
        df = pd.DataFrame(columns=['game_date', 'home_team_abbrev', 'visit_team_abbrev',
                                        'line', 'home_team_score', 'visit_team_score'])
        

        for year in range(2007, 2023):
            next_year = year + 1
            url_ending = str(year) + '-' + str(next_year)[2:]
            r = requests.get(url + url_ending)
            print(url + url_ending)
            soup = BeautifulSoup(r.content, 'html5lib')
            print(soup)
        # /nba-odds-2007-08/

def main():
    scraper = OddsDataScraper()
    # scraper.scrape_rotowire_odds()
    scraper.scrape_sportsbookreviews_online_odds()

if __name__ == '__main__':
    main()