# %% [markdown]
# **Introduction:**  
# In this notebook we integrated insights from our comprehensive EDA and prior winning solutions into a competitive submission pipeline for NCAA March Machine Learning Mania 2025. We computed Elo ratings (trained on seasons before 2024), prepared training data for margin modeling, trained separate KNN regressors for men's and women's data, and converted margin predictions to win probabilities.
#  
# **Validation:**  
# We simulated a hold‐out validation on the 2024 regular season detailed results by predicting game outcomes and calculating the Brier score. This gave us an indication of our model’s predictive performance using the same competition metric.
#  
# **Key Takeaways:**  
#  - The refreshed data (up to DayNum 106 for regular season and updated Massey Ordinals) is now reliable with fixed issues.  
#  - Separately modeling men's and women's games is essential due to inherent scoring differences.  
#  - Our two‑stage approach—computing Elo ratings and then modeling margin differences using KNN—translates well into win probability predictions.  
#  - Validation on 2024 data provides a realistic measure of model performance before final submissions.
#  
# **Conclusion:**  
# This complete pipeline not only generates competitive submissions (with the ability to produce 100 variants for ensembling) but also validates performance on recent data. Future improvements might include incorporating ensemble methods, alternate models such as XGBoost with GPU support, or additional features inspired by past winning solutions.
# 

# %%
import glob
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import time

from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook", font_scale=1.1)

print("Libraries imported and default styles set.")

# %%
# Define the input folder path
input_folder = r"/kaggle/input/march-machine-learning-mania-2025"

# Find all CSV files in the input folder.
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Create a dictionary to store DataFrames.
dataframes = {}

# Loop through each CSV file, read it, and store it in the dictionary using the filename (without extension) as the key.
for file in csv_files:
    key = os.path.splitext(os.path.basename(file))[0]
    try:
        dataframes[key] = pd.read_csv(file, low_memory=False, encoding="latin-1")
        print(f"Loaded {key} with shape {dataframes[key].shape}")
    except Exception as e:
        print(f"Error loading {file}: {e}")

print("\nAll CSV files loaded automatically.")

# %%
# Use Stage 2 sample submission if available; otherwise, use Stage1.
if 'SampleSubmissionStage2' in dataframes:
    df_sub = dataframes['SampleSubmissionStage2'].copy()
    print("Using Stage 2 sample submission.")
else:
    df_sub = dataframes['SampleSubmissionStage1'].copy()
    print("Stage 2 not found. Using Stage 1 sample submission.")

def parse_id(match_id):
    season, t1, t2 = match_id.split('_')
    return int(season), int(t1), int(t2)

df_sub['Season'] = df_sub['ID'].apply(lambda x: parse_id(x)[0])
df_sub['Team1'] = df_sub['ID'].apply(lambda x: parse_id(x)[1])
df_sub['Team2'] = df_sub['ID'].apply(lambda x: parse_id(x)[2])
print("Sample submission file prepared:")
display(df_sub.head(3))

# %%
def initialize_elo(team_ids, start_elo=1500):
    return {tid: start_elo for tid in team_ids}

def update_elo(elo_dict, teamA, teamB, scoreA, scoreB, k=20):
    ra = elo_dict[teamA]
    rb = elo_dict[teamB]
    ea = 1.0 / (1 + 10 ** ((rb - ra) / 400))
    # Actual result: 1 if teamA wins, 0 if teamB wins.
    sa = 1 if scoreA > scoreB else 0
    elo_dict[teamA] = ra + k * (sa - ea)
    elo_dict[teamB] = rb + k * ((1 - sa) - (1 - ea))

def compute_elo(df_games, teams_df):
    df_sorted = df_games.sort_values(by=['Season','DayNum'])
    team_ids = teams_df['TeamID'].unique()
    elo_dict = initialize_elo(team_ids)
    for idx, row in df_sorted.iterrows():
        update_elo(elo_dict, row['WTeamID'], row['LTeamID'], row['WScore'], row['LScore'])
    return elo_dict

# %%
# Filter men's and women's regular season detailed results:
df_MReg = dataframes['MRegularSeasonDetailedResults']
df_WReg = dataframes['WRegularSeasonDetailedResults']

# Use only seasons before 2024 for training Elo ratings.
df_MReg_train = df_MReg[df_MReg['Season'] < 2024].copy()
df_WReg_train = df_WReg[df_WReg['Season'] < 2024].copy()

# Get team lists.
df_MTeams = dataframes['MTeams']
df_WTeams = dataframes['WTeams']

elo_m_train = compute_elo(df_MReg_train, df_MTeams)
elo_w_train = compute_elo(df_WReg_train, df_WTeams)
print("Elo ratings computed using seasons before 2024 for men's and women's data.")

# %%
def prepare_training_data(df, elo_dict):
    elo_diffs = []
    margins = []
    for idx, row in df.iterrows():
        diff = elo_dict.get(row['WTeamID'], 1500) - elo_dict.get(row['LTeamID'], 1500)
        elo_diffs.append(diff)
        margins.append(row['WScore'] - row['LScore'])
    return pd.DataFrame({'EloDiff': elo_diffs, 'Margin': margins})

train_m = prepare_training_data(df_MReg_train, elo_m_train)
train_w = prepare_training_data(df_WReg_train, elo_w_train)
print("Training data prepared for men's and women's margin models.")

# %%
def train_margin_model(df_train):
    X = df_train[['EloDiff']].values
    y = df_train['Margin'].values
    knn = KNeighborsRegressor()
    param_grid = {'n_neighbors': [5, 7, 10, 13, 20, 27, 37, 40]}
    gscv = GridSearchCV(knn, param_grid, cv=3, scoring='neg_mean_squared_error')
    gscv.fit(X, y)
    print("Best n_neighbors:", gscv.best_params_)
    return gscv.best_estimator_

knn_m = train_margin_model(train_m)
knn_w = train_margin_model(train_w)
print("KNN margin models trained for men's and women's data.")

# %%
def margin_to_probability(margin, scale=10.0):
    return 1.0 / (1 + 10 ** (-margin / scale))

def predict_match(row, elo_m, elo_w, knn_m, knn_w):
    # For competition submissions, the prediction is for the team with the lower TeamID.
    team1, team2 = row['Team1'], row['Team2']
    if team1 < team2:
        lower = team1
        higher = team2
    else:
        lower = team2
        higher = team1
        
    # Determine bracket based on team IDs.
    if lower < 2000 and higher < 2000:
        e_lower = elo_m.get(lower, 1500)
        e_higher = elo_m.get(higher, 1500)
        elo_diff = e_lower - e_higher
        margin_pred = knn_m.predict(np.array([[elo_diff]]))[0]
        prob = margin_to_probability(margin_pred)
    elif lower >= 3000 and higher >= 3000:
        e_lower = elo_w.get(lower, 1500)
        e_higher = elo_w.get(higher, 1500)
        elo_diff = e_lower - e_higher
        margin_pred = knn_w.predict(np.array([[elo_diff]]))[0]
        prob = margin_to_probability(margin_pred)
    else:
        prob = 0.5  # Default case (should not occur)
    return prob

# %%
def get_actual_outcome(row):
    # Outcome is 1 if the team with lower ID won (i.e. equals WTeamID), else 0.
    team1, team2 = row['WTeamID'], row['LTeamID']
    lower = min(team1, team2)
    return 1 if lower == row['WTeamID'] else 0

# For men's validation: Use MRegularSeasonDetailedResults from season 2024.
df_MReg_val = df_MReg[df_MReg['Season'] == 2024].copy()
# For women's validation: Use WRegularSeasonDetailedResults from season 2024.
df_WReg_val = df_WReg[df_WReg['Season'] == 2024].copy()

# Compute predictions for validation games.
def validate_games(df_val, elo_dict, knn_model):
    preds = []
    actuals = []
    for idx, row in df_val.iterrows():
        # For validation, always predict for the team with lower ID.
        team1, team2 = row['WTeamID'], row['LTeamID']
        lower = min(team1, team2)
        # Get Elo ratings from training Elo dictionary (which did NOT use 2024)
        if lower < 2000:
            e_lower = elo_m_train.get(lower, 1500)
            # For the opponent, use the rating for the higher ID.
            e_higher = elo_m_train.get(max(team1, team2), 1500)
            elo_diff = e_lower - e_higher
            margin_pred = knn_m.predict(np.array([[elo_diff]]))[0]
        else:
            e_lower = elo_w_train.get(lower, 1500)
            e_higher = elo_w_train.get(max(team1, team2), 1500)
            elo_diff = e_lower - e_higher
            margin_pred = knn_w.predict(np.array([[elo_diff]]))[0]
        prob = margin_to_probability(margin_pred)
        preds.append(prob)
        # Actual outcome: 1 if the lower team wins.
        outcome = 1 if lower == row['WTeamID'] else 0
        actuals.append(outcome)
    return np.array(preds), np.array(actuals)

# %%
preds_m, actuals_m = validate_games(df_MReg_val, elo_m_train, knn_m)
preds_w, actuals_w = validate_games(df_WReg_val, elo_w_train, knn_w)

# Compute Brier scores for men's and women's validation sets.
brier_m = mean_squared_error(actuals_m, preds_m)
brier_w = mean_squared_error(actuals_w, preds_w)

print("Men's 2024 Regular Season Brier Score:", brier_m)
print("Women's 2024 Regular Season Brier Score:", brier_w)

# %%
# # Create a folder to save submissions.
# submission_folder = "ensemble_submissions"
# os.makedirs(submission_folder, exist_ok=True)

# num_submissions = 100

# for i in range(1, num_submissions + 1):
#     seed_val = 1000 + i
#     print(f"\n=== Iteration {i} using seed {seed_val} ===")
    
#     # Retrain KNN models with the new seed on the same training data.
#     # (In a more advanced approach, you might retrain with additional hyperparameter variations.)
#     knn_m_i = train_margin_model(train_m)
#     knn_w_i = train_margin_model(train_w)
    
#     # Generate predictions for each matchup in the submission file.
#     preds = []
#     for idx, row in df_sub.iterrows():
#         p = predict_match(row, elo_m_train, elo_w_train, knn_m_i, knn_w_i)
#         # Optionally add a small random perturbation.
#         p += np.random.normal(0, 0.005)
#         p = np.clip(p, 0.001, 0.999)
#         preds.append(p)
#     df_sub['Pred'] = preds
    
#     submission_filename = os.path.join(submission_folder, f"submission_{i}.csv")
#     df_sub[['ID', 'Pred']].to_csv(submission_filename, index=False)
#     print(f"Saved submission file: {submission_filename}")

# print("\nEnsemble submission generation complete. 100 submission files created.")

# %%
# Generate predictions for each matchup in df_sub.
predictions = []
for idx, row in df_sub.iterrows():
    p = predict_match(row, elo_m_train, elo_w_train, knn_m, knn_w)
    # Optionally, add a small perturbation for uncertainty.
    p += np.random.normal(0, 0.005)
    p = np.clip(p, 0.001, 0.999)
    predictions.append(p)
df_sub['Pred'] = predictions

print("Predictions generated for all matchups. Here are a few examples:")
print(df_sub.head(3))

# %%
# %% [code]
submission_filename = "submission.csv"
df_sub[['ID', 'Pred']].to_csv(submission_filename, index=False)
print(f"Submission file '{submission_filename}' created successfully.")

# %%



