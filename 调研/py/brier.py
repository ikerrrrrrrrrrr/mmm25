# %% [markdown]
# <div style="display: flex; justify-content: space-between; align-items: flex-start;">
#     <div style="text-align: left;">
#         <p style="color:#FFD700; font-size: 15px; font-weight: bold; margin-bottom: 1px; text-align: left;">Published on  February 13, 2025</p>
#         <h4 style="color:#4B0082; font-weight: bold; text-align: left; margin-top: 6px;">Author: Jocelyn C. Dumlao</h4>
#         <p style="font-size: 17px; line-height: 1.7; color: #333; text-align: center; margin-top: 20px;"></p>
#         <a href="https://www.linkedin.com/in/jocelyn-dumlao-168921a8/" target="_blank" style="display: inline-block; background-color: #003f88; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">LinkedIn</a>
#         <a href="https://github.com/jcdumlao14" target="_blank" style="display: inline-block; background-color: transparent; color: #059c99; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px; border: 2px solid #007bff;">GitHub</a>
#         <a href="https://www.youtube.com/@CogniCraftedMinds" target="_blank" style="display: inline-block; background-color: #ff0054; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">YouTube</a>
#         <a href="https://www.kaggle.com/jocelyndumlao" target="_blank" style="display: inline-block; background-color: #3a86ff; color: #fff; text-decoration: none; padding: 5px 10px; border-radius: 10px; margin: 15px;">Kaggle</a>
#     </div>
# </div>

# %% [markdown]
# ## Welcome!
# 
# This notebook explores historical NCAA Division I men's and women's basketball data to predict the outcomes of March Madness tournament games. We'll leverage provided datasets containing team information, game results, and tournament seeds to build a predictive model. Data files are prefixed with 'M' for men's, 'W' for women's,and some span both.
# 
# ## Goal: 
# Minimize the Brier score, the evaluation metric for this competition.
# 
# ## Approach:
#  1. Start with a basic model using seed differences.
#  2. Gradually incorporate more sophisticated features and techniques.
# 

# %% [markdown]
# # <p style="padding:10px;background-color:#00ffc3;margin:0;color:#102d02;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:15px 50px;overflow:hidden;font-weight:500;border: 3px solid #00a5a0;">Brier Score</p>
# 
# 
# The **Brier score** is a strictly proper scoring rule that measures the accuracy of probabilistic predictions. For unidimensional predictions, it is strictly equivalent to the mean squared error as applied to predicted probabilities.
# 
# The Brier score is applicable to tasks in which predictions must assign probabilities to a set of mutually exclusive discrete outcomes or classes. The set of possible outcomes can be either binary or categorical in nature, and the probabilities assigned to this set of outcomes must sum to one (where each individual probability is in the range of 0 to 1). It was proposed by Glenn W. Brier in 1950.[1]
# 
# The Brier score can be thought of as a cost function. More precisely, across all items 
# i
# ∈
# 1...
# N
# in a set of N predictions, the Brier score measures the mean squared difference between:
# 
# - The predicted probability assigned to the possible outcomes for item i
# - The actual outcome 
# o
# i
# 
# Therefore, the lower the Brier score is for a set of predictions, the better the predictions are calibrated. Note that the Brier score, in its most common formulation, takes on a value between zero and one, since this is the square of the largest possible difference between a predicted probability (which must be between zero and one) and the actual outcome (which can take on values of only 0 or 1). In the original (1950) formulation of the Brier score, the range is double, from zero to two.
# 
# The Brier score is appropriate for binary and categorical outcomes that can be structured as true or false, but it is inappropriate for ordinal variables which can take on three or more values.
# 
# Links: https://en.wikipedia.org/wiki/Brier_score

# %% [markdown]
# # <p style="padding:10px;background-color:#00ffc3;margin:0;color:#102d02;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:15px 50px;overflow:hidden;font-weight:500;border: 3px solid #00a5a0;">Import Libraries</p>

# %%
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mean_squared_error, roc_curve, auc
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # <p style="padding:10px;background-color:#00ffc3;margin:0;color:#102d02;font-family:newtimeroman;font-size:100%;text-align:center;border-radius:15px 50px;overflow:hidden;font-weight:500;border: 3px solid #00a5a0;">Load the Data</p>

# %%
data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'

# %%
class TournamentPredictor:
    def __init__(self, data_dir):
        self.data_path = data_dir
        self.data = None
        self.teams = None
        self.seeds = None
        self.games = None
        self.sub = None
        self.gb = None
        self.col = None
        self.model = None # declare model here.
        self.calibration_model = None # declare calibration model here.
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def load_data(self):
        files = glob.glob(self.data_path)
        self.data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p, encoding='latin-1') for p in files}

        teams = pd.concat([self.data['MTeams'], self.data['WTeams']])
        teams_spelling = pd.concat([self.data['MTeamSpellings'], self.data['WTeamSpellings']])
        teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
        teams_spelling.columns = ['TeamID', 'TeamNameCount']
        self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])

        season_cresults = pd.concat([self.data['MRegularSeasonCompactResults'], self.data['WRegularSeasonCompactResults']])
        season_dresults = pd.concat([self.data['MRegularSeasonDetailedResults'], self.data['WRegularSeasonDetailedResults']])
        tourney_cresults = pd.concat([self.data['MNCAATourneyCompactResults'], self.data['WNCAATourneyCompactResults']])
        tourney_dresults = pd.concat([self.data['MNCAATourneyDetailedResults'], self.data['WNCAATourneyDetailedResults']])

        seeds_df = pd.concat([self.data['MNCAATourneySeeds'], self.data['WNCAATourneySeeds']])
        self.seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values}

        self.sub = self.data['SampleSubmissionStage1']

        season_cresults['ST'] = 'S'
        season_dresults['ST'] = 'S'
        tourney_cresults['ST'] = 'T'
        tourney_dresults['ST'] = 'T'

        self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

        self.games['ID'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['IDTeams'] = self.games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1)
        self.games['Team1'] = self.games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
        self.games['Team2'] = self.games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
        self.games['IDTeam1'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.games['IDTeam2'] = self.games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
        self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)
        self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
        self.games['Pred'] = self.games.apply(lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1)
        self.games['ScoreDiffNorm'] = self.games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1)
        self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']
        self.games = self.games.fillna(-1)

        c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        self.gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
        self.gb.columns = [''.join(c) + '_c_score' for c in self.gb.columns]

        self.games = self.games[self.games['ST'] == 'T']

        self.sub['WLoc'] = 3
        self.sub['Season'] = self.sub['ID'].map(lambda x: x.split('_')[0]).astype(int)
        self.sub['Team1'] = self.sub['ID'].map(lambda x: x.split('_')[1])
        self.sub['Team2'] = self.sub['ID'].map(lambda x: x.split('_')[2])
        self.sub['IDTeams'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
        self.sub['IDTeam1'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
        self.sub['IDTeam2'] = self.sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
        self.sub['Team1Seed'] = self.sub['IDTeam1'].map(self.seeds).fillna(0)
        self.sub['Team2Seed'] = self.sub['IDTeam2'].map(self.seeds).fillna(0)
        self.sub['SeedDiff'] = self.sub['Team1Seed'] - self.sub['Team2Seed']
        self.sub = self.sub.fillna(-1)

        self.games = pd.merge(self.games, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        self.sub = pd.merge(self.sub, self.gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col
        self.col = [c for c in self.games.columns if c not in exclude_cols]
        print("Data loading and preprocessing completed.")

    def create_models(self):
      # Create the models here with the same parameters.
      self.model = RandomForestRegressor(
          n_estimators=235,
          random_state=42,
          max_depth=15,
          min_samples_split=2,
          max_features='sqrt',
          n_jobs=-1
      )
      self.calibration_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)



    def train_model(self):
        X = self.games[self.col].fillna(-1)
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        y = self.games['Pred']

        X_train, X_cal, y_train, y_cal = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        train_preds = self.model.predict(X_train).clip(0.001, 0.999)

        cal_preds = self.model.predict(X_cal).clip(0.001, 0.999)
        self.calibration_model.fit(cal_preds.reshape(-1, 1), y_cal)

        train_preds_calibrated = self.calibration_model.predict(train_preds.reshape(-1, 1)).clip(0.001, 0.999)

        print(f'Log Loss (Train): {log_loss(y_train, train_preds_calibrated):.4f}')
        print(f'Brier Score (Train): {brier_score_loss(y_train, train_preds_calibrated):.4f}')
        print(f'MSE (Train): {mean_squared_error(y_train, train_preds_calibrated):.4f}')

        # Plot ROC Curve for the calibration set.
        self.plot_roc_curve(y_cal, cal_preds, "Calibration Set ROC Curve")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mse_scores = []
        cv_logloss_scores = []
        for train_index, val_index in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = y[train_index], y[val_index]

            self.model.fit(X_train, y_train)
            val_preds = self.model.predict(X_val).clip(0.001, 0.999)

            self.calibration_model.fit(val_preds.reshape(-1, 1), y_val)
            val_preds_calibrated = self.calibration_model.predict(val_preds.reshape(-1, 1)).clip(0.001, 0.999)

            mse = mean_squared_error(y_val, val_preds_calibrated)
            logloss = log_loss(y_val, val_preds_calibrated)

            cv_mse_scores.append(mse)
            cv_logloss_scores.append(logloss)

        print(f'Cross-validated MSE: {np.mean(cv_mse_scores):.4f}')
        print(f'Cross-validated LogLoss: {np.mean(cv_logloss_scores):.4f}')

        feature_importances = self.model.feature_importances_
        feature_names = self.col
        self.plot_feature_importance(feature_importances, feature_names)

        self.plot_calibration_curve(y_cal, cal_preds)

        # Plot the distribution of calibrated predictions.
        self.plot_prediction_distribution(train_preds_calibrated, "Distribution of Calibrated Training Predictions")

    def predict_submission(self, output_file='submission.csv'):
        sub_X = self.sub[self.col].fillna(-1)
        sub_X_imputed = self.imputer.transform(sub_X)
        sub_X_scaled = self.scaler.transform(sub_X_imputed)

        preds = self.model.predict(sub_X_scaled).clip(0.001, 0.999)
        preds_calibrated = self.calibration_model.predict(preds.reshape(-1, 1)).clip(0.001, 0.999)

        self.sub['Pred'] = preds_calibrated
        self.sub[['ID', 'Pred']].to_csv(output_file, index=False)
        print(f"Submission file saved to {output_file}")

    def plot_feature_importance(self, importances, feature_names, top_n=20):
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title('Top {} Feature Importances'.format(top_n))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_calibration_curve(self, y_true, y_proba, n_bins=10):

        combined = np.stack([y_proba, y_true], axis=-1)
        combined = combined[np.argsort(combined[:, 0])]
        sorted_probas = combined[:, 0]
        sorted_true = combined[:, 1]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_midpoints = bins[:-1] + (bins[1] - bins[0]) / 2
        bin_assignments = np.digitize(sorted_probas, bins) - 1

        bin_sums = np.bincount(bin_assignments, weights=sorted_probas, minlength=n_bins)
        bin_true = np.bincount(bin_assignments, weights=sorted_true, minlength=n_bins)
        bin_total = np.bincount(bin_assignments, minlength=n_bins)

        fraction_of_positives = bin_true / bin_total
        fraction_of_positives[np.isnan(fraction_of_positives)] = 0

        plt.figure(figsize=(8, 6))
        plt.plot(bin_midpoints, fraction_of_positives, marker='o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

        plt.xlabel('Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction_distribution(self, predictions, title="Distribution of Predictions"):
        """Plots the distribution of model predictions."""
        plt.figure(figsize=(8, 6))
        sns.histplot(predictions, kde=True, color='skyblue')
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
      """Plots the Receiver Operating Characteristic (ROC) curve."""
      fpr, tpr, thresholds = roc_curve(y_true, y_proba)
      roc_auc = auc(fpr, tpr)

      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
      plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(title)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.legend(loc="lower right")
      plt.tight_layout()
      plt.show()


    def run_all(self):
        self.load_data()
        self.create_models()
        self.train_model()
        self.predict_submission()


# %%
# Example usage:
if __name__ == "__main__":
    data_dir = '/kaggle/input/march-machine-learning-mania-2025/**'  # Or a local dir
    predictor = TournamentPredictor(data_dir)
    predictor.run_all()

# %%



