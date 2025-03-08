# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib
import xgboost as xgb

DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/**'

def load_data(data_path):
    files = glob.glob(data_path, recursive=True)
    files = [f for f in files if os.path.isfile(f)]  # Filter out directories
    data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}
    return data

data = load_data(DATA_PATH)

def preprocess_data(data):
    # Combine men's and women's data
    teams = pd.concat([data['MTeams'], data['WTeams']])
    teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])
    teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
    teams_spelling.columns = ['TeamID', 'TeamNameCount']
    teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
    del teams_spelling

    # Combine regular season and tournament results
    season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])
    season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])
    tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])
    tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])

    # Process seeds
    seeds_df = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])
    seeds = {
        '_'.join(map(str, [int(k1), k2])): int(v[1:3])
        for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values
    }

    # Load submission file
    sub = data['SampleSubmissionStage2']  # Load the correct sample submission file
    del seeds_df

    # Combine all games and preprocess
    season_cresults['ST'] = 'S'
    season_dresults['ST'] = 'S'
    tourney_cresults['ST'] = 'T'
    tourney_dresults['ST'] = 'T'

    games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
    games.reset_index(drop=True, inplace=True)
    games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

    # Create unique IDs for games and teams
    games['ID'] = games.apply(
        lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1
    )
    games['IDTeams'] = games.apply(
        lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
    )
    games['Team1'] = games.apply(
        lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1
    )
    games['Team2'] = games.apply(
        lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1
    )
    games['IDTeam1'] = games.apply(
        lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
    )
    games['IDTeam2'] = games.apply(
        lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
    )
    games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
    games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

    # Calculate additional features
    games['ScoreDiff'] = games['WScore'] - games['LScore']
    games['Pred'] = games.apply(
        lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1
    )
    games['ScoreDiffNorm'] = games.apply(
        lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1)
    games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
    games = games.fillna(-1)

    # Aggregate statistics
    c_score_col = [
        'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
        'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
        'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
    ]
    c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
    gb = games.groupby("IDTeams").agg({k: c_score_agg for k in c_score_col}).reset_index()
    gb.columns = ["".join(c) + "_c_score" for c in gb.columns]

    # Filter tournament games
    games = games[games["ST"] == "T"]

    # Preprocess submission data
    sub["WLoc"] = 3
    sub["Season"] = sub["ID"].map(lambda x: x.split("_")[0]).astype(int)
    sub["Team1"] = sub["ID"].map(lambda x: x.split("_")[1])
    sub["Team2"] = sub["ID"].map(lambda x: x.split("_")[2])
    sub["IDTeams"] = sub.apply(
        lambda r: "_".join(map(str, [r["Team1"], r["Team2"]])), axis=1)
    sub["IDTeam1"] = sub.apply(
        lambda r: "_".join(map(str, [r["Season"], r["Team1"]])), axis=1)
    sub["IDTeam2"] = sub.apply(
        lambda r: "_".join(map(str, [r["Season"], r["Team2"]])), axis=1)
    sub["Team1Seed"] = sub["IDTeam1"].map(seeds).fillna(0)
    sub["Team2Seed"] = sub["IDTeam2"].map(seeds).fillna(0)
    sub["SeedDiff"] = sub["Team1Seed"] - sub["Team2Seed"]
    sub = sub.fillna(-1)

    # Merge aggregated stats with games and submission data
    games = pd.merge(games, gb, how="left", left_on="IDTeams", right_on="IDTeams_c_score")
    sub = pd.merge(sub, gb, how="left", left_on="IDTeams", right_on="IDTeams_c_score")

    # Define feature columns
    exclude_cols = [
        "ID", "DayNum", "ST", "Team1", "Team2", "IDTeams", "IDTeam1", "IDTeam2",
        "WTeamID", "WScore", "LTeamID", "LScore", "NumOT", "Pred", "ScoreDiff",
        "ScoreDiffNorm", "WLoc"
    ] + c_score_col
    col = [c for c in games.columns if c not in exclude_cols]

    print("Data preprocessing completed.")
    return games, sub, col, seeds, gb

games, sub, col, seeds, gb = preprocess_data(data)

def train_model(games, col):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X = games[col].fillna(-1)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    y = games["Pred"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=42
    )

    model.fit(X_scaled, y)
    pred = model.predict_proba(X_scaled)[:, 1].clip(0.001, 0.999)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(pred, y)
    pred_cal = ir.transform(pred)

    print(f"Log Loss: {log_loss(y, pred_cal):.8f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, pred_cal):.8f}")
    print(f"Brier Score: {brier_score_loss(y, pred_cal):.8f}")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_squared_error")
    print(f"Cross-validated MSE: {-cv_scores.mean():.8f}")

    return model, imputer, scaler, ir

model, imputer, scaler, ir = train_model(games, col)

def predict_submission(sub, col, model, imputer, scaler, ir, output_file="submission.csv"):
    sub_X = sub[col].fillna(-1)
    X_imputed = imputer.transform(sub_X)
    X_scaled = scaler.transform(X_imputed)
    preds = model.predict_proba(X_scaled)[:, 1].clip(0.01, 0.99)
    preds_cal = ir.transform(preds)
    sub["Pred"] = preds_cal
    sub[["ID", "Pred"]].to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

predict_submission(sub, col, model, imputer, scaler, ir)

def save_model(model, imputer, scaler, seeds, col, gb, ir, filename):
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "seeds": seeds,
        "col": col,
        "gb": gb,
        "ir": ir
    }, filename)
    print(f"Model saved to {filename}")

save_model(model, imputer, scaler, seeds, col, gb, ir, "tournament_model.pkl")

def load_model(filename):
    data = joblib.load(filename)
    return data["model"], data["scaler"], data["imputer"], data["seeds"], data["col"], data["gb"], data["ir"]

model, scaler, imputer, seeds, col, gb, ir = load_model("tournament_model.pkl")

if __name__ == "__main__":
    data = load_data(DATA_PATH)
    games, sub, col, seeds, gb = preprocess_data(data)
    model, imputer, scaler, ir = train_model(games, col)
    predict_submission(sub, col, model, imputer, scaler, ir)
    save_model(model, imputer, scaler, seeds, col, gb, ir, "tournament_model.pkl")



