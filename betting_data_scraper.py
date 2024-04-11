#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:42:33 2024

@author: pmullin
"""

import requests
import pandas as pd


# URL of the page containing NBA game spreads
url = 'https://www.rotowire.com/betting/nba/tables/games-archive.php'

# Fetch the webpage content
r = requests.get(url)
data = r.json()


# Read the json data into a pandas df
df = pd.DataFrame(data, columns=['season', 'home_team_abbrev', 'visit_team_abbrev',
                                 'line', 'home_team_score', 'visit_team_score'])
df['Final Score Spread'] = df['visit_team_score'] - df['home_team_score']
df = df.sort_values(by='home_team_abbrev')
df = df.rename(columns={'home_team_abbrev': 'Home Team', 'season': 'Season',
                        'visit_team_abbrev': 'Away Team', 'line': 'Betting Score Spread'})
df = df.drop(columns=['home_team_score', 'visit_team_score'])
df['Betting Line Error'] = df['Betting Score Spread'] - df['Final Score Spread']
df.to_csv('nba_betting_lines.csv')