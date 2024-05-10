from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscorefourfactorsv3

# Anthony Davis
adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id='0021800283')
ff = boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id='0021800283')
print(adv.get_data_frames()[0])
print(ff.get_data_frames()[0])
