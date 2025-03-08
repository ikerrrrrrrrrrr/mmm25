# %% [markdown]
# #  Notes
# 
# - I know nothing about Basketball so let's see how this goes
# - The feature engineering seriously needs to be worked on
# - Seeds do not change during the season
# - Predict the probability of the lower TeamID winning
# - Only rely on given data (just to make it easier)
# - Possibly make a df for team statistics throughout the years (like if a team has been has been active since 1990, you do 1 row for that team per year). If you do this, make a separate df for women.
# - Possibly make a df that's just for games where it shows who won.
# - Somehow combine these into something a model can use to train on? This I need to think a lot about. If I have 3 different dataframes, with each giving me different kinds of data, how do I make a ML model that can train on those and predict the mashups of March Madness in the correct format?
# - Because of the last bullet, try to not do any feature engineering or creating dataframes until you think about how a model can use it.
# - After making dataframes, check dtypes
# 
# # Current game plan for features to attempt
# 
# ### MTeams and WTeams
# - **TeamID:** this is the unique team ID (does not repeat in this csv)
#   - 4 digit number
# - **TeamName:** short spelling of the team's name. this can be good just for a reference (not for prediction)
#   - 16 character name
# - **FirstD1Season:** the year that they became D1. these start at 1985
#   - 4 digit number (1985-2025)
# - **LastD1Season:** the last year they were D1.
#   - 4 digit number (1985-2025)
# 
# ### MNCAATourneySeeds and WNCAATourneySeeds
# - **Season:** year of the season
#   - 4 digit number (1985-2024)
# - **Seed:** seed of the team
#   - last 2 characters are the seed
# - **TeamID:** team id of the season and seed
#   - 4 digit number
# 
# ### MNCAATourneySlots and WNCAATourneySlots
# - **Season:** year of season
#   - 4 digit number (1985-2024)
# - **Slot:** slot that gives information on the seed from the winning or favored team
#   - for play-ins, it is a 3 character string where the last 2 numbers are the winning seed. for regular tournaments, it is a 4 character string where the second character tells you the round, and the last character (or 2) tells you the expected seed of the favored team
# - **StrongSeed:** seed that's strong
#   - last 2 characters are the seed
# - **WeakSeed:** weaker seed
#   - last 2 characters are the seed
# 
# ### MRegularSeasonDetailedResults and WRegularSeasonDetailedResults
# - **Season:** year of the season
#   - 4 digit number (2023-2025)
# - **DayNum**: higher the number, the better the team is because they lasted longer
#   - number
# - **WTeamID:** winning team id
#   - 4 digit number
# - **WScore:** winning score
#   - number
# - **LTeamID:** losting team id
#   - 4 digit number
# - **LScore:** losing score
#   - number
# - **WLoc:** location of winning team
#   - character (H, A, or N)
# - **NumOT:** number of overtime periods
#   - number
# - **WFGM:** field goals made by winning team
#   - number
# 
# - **OOPS! There are way more columns that are good, I just didn't see them before. Use them all...**
# 
# ### MNCAATourneyDetailedResults and WNCAATourneyDetailedResults
# - same as (M/W)RegularSeasonDetailedResults
# 
# ### MSecondaryTourneyTeams and WSecondaryTournamentTeams
# - **Season:** year
#   - 4 digit number (1985-2024)
# - **SecondaryTourney:** abbreviation of tournament
#   - 3 characters
# - **TeamID:** team that played in it
#   - 4 digit number
# 
# ### MSecondaryTourneyCompactResults and WSecondaryTourneyCompactResults
# - all of this is the same as (M/W)RegularSeasonDetailedResults except there is no WFGM and instead it has the SecondaryTourney column and no team box scores
# 
# ### Cities
# - **CityID:** unique city id
#   - number
# - **State:** state abbreviation
#   - 2 characters
# 
# ### MGameCities and WGameCities
# - **Season:** year
#   - 4 digit number (2010-2025)
# - **DayNum**: higher is better
#   - number
# - **WTeamID:** winning team
#   - 4 digit number
# - **LTeamID:** losing team
#   - 4 digit number
# - **CRType:** says whether the game is regular, ncaa, or secondary
#   - characters
# - **CityID:** unique city id
#   - 4 digit number
# 
# ### Other features to possibly make
# - Win/lose ratio (for each team)
# - Score of left team - score of right team difference (for that game)
# - Calculate ELO rating per team
# 
# ### Target
# - I think a good target would be to make it so if each row is for 2 competing teams, show a 1 if the left team won, and a 0 if they lost. Then, the ML model will predict this and show a probability for it.
# - I saw someone did it where instead of it being a 1 if they actually won, they did it so that the smaller TeamID is 1 and the other is 0. Not sure which route to go.
# 
# 
# - Actually, maybe just make it into 1 df.
# 
# ### Dataframe Structure
# - **ID:** of team mashup (YYYY_T1ID_T2ID) from SampleSubmissionStage1
# - **Year:** YYYY taken from SampleSubmissionStage1
# 
# - **LTeam:** taken from SampleSubmissionStage1
# - **LTeamName:** taken from TeamName from (M/W)Teams (just to see team name, not for training)
# 
# - **RTeam:** taken from SampleSubmissionStage1
# - **RTeamName:** taken from TeamName from (M/W)Teams (just to see team name, not for training)
# 
# - **D1Diff:** taken from subtracting the left LastD1Season - FirstD1Season from the right from (M/W)Teams
# - **SeedDiff:** taken from MNCAATourneySeeds from subtracting the left seed from the right
# - **FavoredSeed:** taken from MNCAATourneySlots where it's 1 if the left team is the favored seed (if it's all 1 then remove this)
# 
# - **RegELODiff:** which uses the data from (M/W)RegularSeasonDetailedResults to calculate the difference in ELO
# - **TourneyELODiff:** which uses the data from (M/W)NCAATourneyDetailedResults to calculate the difference in ELO
# 
# - **Pred:** where 1 is if the left team won the game or 0 if they didn't.
# 
# # WHAT?????????????
# - Please review all of the code and figure out exactly how the feature selection works
# - why are you clipping predictions
# - why use a regressor instead of a classifier
# - why are missing values filled with -1
# - is the standard scaler scaling any categorical columns??
# - why are we fitting on X and then predicting with X? is this not the same as fitting with training data and then predicting with training data?

# %%
# Libraries
import pandas as pd # I wanted to use fireducks instead, but it was giving me issues
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import glob
import os

# %%
# Getting all files
path = "./kaggle/input/march-machine-learning-mania-2025/**"
data = {p.split('/')[-1].split('.')[0].split('\\')[1] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}
#data = {os.path.splitext(os.path.basename(p))[0]: pd.read_csv(p, encoding='latin-1') for p in glob.glob(path, recursive=True)}
#data

# %%
# Loading data
"""
MTeams = data["MTeams"]
WTeams = data["WTeams"]

MNCAATourneySeeds = data["MNCAATourneySeeds"]
WNCAATourneySeeds = data["WNCAATourneySeeds"]

MNCAATourneySlots = data["MNCAATourneySlots"]
WNCAATourneySlots = data["WNCAATourneySlots"]

MRegularSeasonDetailedResults = data["MRegularSeasonDetailedResults"]
WRegularSeasonDetailedResults = data["WRegularSeasonDetailedResults"]

MNCAATourneyDetailedResults = data["MNCAATourneyDetailedResults"]
WNCAATourneyDetailedResults = data["WNCAATourneyDetailedResults"]

MSecondaryTourneyTeams = data["MSecondaryTourneyTeams"]
WSecondaryTourneyTeams = data["WSecondaryTourneyTeams"]

MSecondaryTourneyCompactResults = data["MSecondaryTourneyCompactResults"]
WSecondaryTourneyCompactResults = data["WSecondaryTourneyCompactResults"]

Cities = data["Cities"]

MGameCities = data["MGameCities"]
WGameCities = data["WGameCities"]
"""

# To be used for storing data, and later to take the IDs
df = data["SampleSubmissionStage2"]
#data['MTeams']
df


# %% [markdown]
# # Feature Engineering

# %%
# Creating year, left team, and right team columns
"""
df['Year'] = [int(yr[0:4]) for yr in df['ID']]
df['LTeam'] = [int(L[5:9]) for L in df['ID']]
df['RTeam'] = [int(R[10:14]) for R in df['ID']]
"""

# %%
# Lots of feature selecting and engineering
teams = pd.concat([data['MTeams'], data['WTeams']])
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
teams_spelling.columns = ['TeamID', 'TeamNameCount']
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
del teams_spelling

season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
gcities = pd.concat([data['MGameCities'], data['WGameCities']])
seasons = pd.concat([data['MSeasons'], data['WSeasons']])

seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}
cities = data['Cities']
sub = data['SampleSubmissionStage2']
del data

season_cresults['ST'] = 'S'
season_dresults['ST'] = 'S'
tourney_cresults['ST'] = 'T'
tourney_dresults['ST'] = 'T'
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

games = games[games['ST']=='T']

sub['WLoc'] = 3
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']
sub = sub.fillna(-1)

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm',
                                             'WLoc'] + c_score_col]

# %% [markdown]
# # Model

# %%
# Selecting training data
X = games[col].fillna(-1)
sub_X = sub[col].fillna(-1)

# %%
# GridSearchCV

# Do some grid search here for the parameters
# After doing the grid search, save the best parameters and create the param_grid in this cell

# %%
# XGB parameters
param_grid = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'max_depth': 6
}

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(**param_grid, device="gpu", random_state=42))
])

# Fitting pipeline
pipeline.fit(X, games['Pred'])

# Predicting games and submissions
pred = pipeline.predict(X).clip(0.001, 0.999)
sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

# Cross validation (for the MSE)
cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")

# %% [markdown]
# # Results

# %%
# Results
print(f'Log Loss: {log_loss(games["Pred"], pred):.5f}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred):.5f}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred):.5f}')
print(f'Cross-validated MSE: {-cv_scores.mean():.5f}')

# %% [markdown]
# # Submission

# %%
# Creating submission dataframe
submission_df = pd.DataFrame({
    'ID': df['ID'],
    'Pred': sub_pred
})

# Shape and head/tail of submission
print(f"{submission_df.shape} \n")
print(f"{submission_df.head().to_string(index=False)} \n")
print(submission_df.tail().to_string(index=False))

# Saving to csv
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file saved! Good luck!!! :)")

# %% [markdown]
# # What I Learned

# %% [markdown]
# Write some things that you learned, struggled with, etc later.


