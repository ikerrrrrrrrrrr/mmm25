# %% [markdown]
# # Merging some good baselines

# %%
import numpy as np
import pandas as pd 
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import glob


from sklearn import *
#import redisAI
import glob
import optuna
from sklearn import ensemble
from sklearn.metrics import *
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torch import nn
import torch.optim as optim



# %% [markdown]
# # EDA for all

# %%
regular_m = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams_m = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
# Load and Process Data Women's Tourney
regular_w = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
teams_w = pd.read_csv('./kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
#print(teams_w.columns, regular_w.columns, tourney_m.columns)
# print(len(regular_m), len(tourney_m))


# %% [markdown]
# ## EDA for XGBoost and the MLP method

# %%
# Getting all files
path = "./kaggle/input/march-machine-learning-mania-2025/**"
data = {p.split('/')[-1].split('.')[0].split('\\')[1] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}
df = data["SampleSubmissionStage2"]
# Creating year, left team, and right team columns
"""
df['Year'] = [int(yr[0:4]) for yr in df['ID']]
df['LTeam'] = [int(L[5:9]) for L in df['ID']]
df['RTeam'] = [int(R[10:14]) for R in df['ID']]
"""
df['RTeam'] = [int(R[10:14]) for R in df['ID']]
df['LTeam'] = [int(L[5:9]) for L in df['ID']]

df['LTeam']
df['ID'] # 从示例提交文件格式中获取左侧和右侧队伍的id
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
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)# 只有2003年开始才有detailed results，这里舍弃了compact results
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)# 比赛id：年，1队，2队
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)# 1队，2队
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)# 1队
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)# 年 1队
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

games['ScoreDiff'] = games['WScore'] - games['LScore']
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1) # 1队赢了没
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
games = games.fillna(-1)

c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF'] # 选择和比赛得分相关的列
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
# groupby 分组，同一支队的比赛会被分到一起
# agg 对于 c_score_col 中的每一列，分别计算每个分组（即每一对队伍）的这些聚合统计值。
# 聚合操作后，结果是一个 MultiIndex DataFrame。调用 reset_index() 是为了将其转换为普通的 DataFrame，方便后续处理。
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
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] # 提取各种信息和添加种子特征
sub = sub.fillna(-1)

games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
# 将比赛数据（games）与之前生成的统计特征数据（gb）通过IDTeams进行左连接。
# 将提交数据（sub）与统计特征数据（gb）通过IDTeams进行左连接。

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm',
                                             'WLoc'] + c_score_col]


# %% [markdown]
# # XGB

# %% [markdown]
# ## Params of XGB

# %%
# XGB parameters
param_grid = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'max_depth': 6
}

# %% [markdown]
# ## Predictions of XGBoost

# %%
X = games[col].fillna(-1)
sub_X = sub[col].fillna(-1)

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(**param_grid, device="gpu", random_state=42))
])

# Fitting pipeline
#pipeline.fit(X, games['Pred'])

# Predicting games and submissions
#pred = pipeline.predict(X).clip(0.001, 0.999)
#sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

# Cross validation (for the MSE)
#cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")
# 5 times of cross validation, 1. 1234, 5; 2: 1235,4, ...

#sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

# submission_df = pd.DataFrame({
#     'ID': df['ID'],
#     'Pred': sub_pred
# })

# pred_of_XGBoost = submission_df

# %% [markdown]
# ## Optuna adjustments for XGB

# %%
def objective_xgb(trial):
    # 定义需要优化的参数
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),  # 迭代次数
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log = True),  # 学习率
        'max_depth': trial.suggest_int('max_depth', 3, 10)  # 树的最大深度
    }

    # 定义 Pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(**param_grid, device="gpu", random_state=42))
    ])

    # 使用交叉验证计算均方误差
    cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")
    mse = -cv_scores.mean()  # 转换为 MSE
    return mse

# 创建 Optuna 研究对象
study_xgb = optuna.create_study(direction='minimize')  # 最小化 MSE

# 运行优化
study_xgb.optimize(objective_xgb, n_trials=4, show_progress_bar=True)  # 运行 50 次试验

# 获取最佳超参数
best_params_xgb = study_xgb.best_params
best_score_xgb = study_xgb.best_value

print("Best Hyperparameters:", best_params_xgb)
print("Best MSE:", best_score_xgb)



# %%
optuna.visualization.plot_optimization_history(study_xgb).show()
optuna.visualization.plot_param_importances(study_xgb).show()
optuna.visualization.plot_contour(study_xgb).show()
# 绘制平行坐标图
optuna.visualization.plot_parallel_coordinate(study_xgb, params=['n_estimators', 'learning_rate', 'max_depth']).show()

optuna.visualization.plot_slice(study_xgb, params=['n_estimators', 'learning_rate', 'max_depth']).show()


# %% [markdown]
# # MLP

# %% [markdown]
# ## Params of MLP

# %%
batch_size = 32  # Set batch size
lr = 0.01
n_layers = 2
num_epochs = 5  # Adjust the number of epochs based on performance
dropout_p = 0.3
hidden_layers_count = 100


# %% [markdown]
# ## Funcs of MLP

# %%
class NeuralNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers = n_layers, dropout_p = dropout_p):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden),
            nn.ReLU()]
        for layer in range(n_layers):
            layers += [nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden),nn.ReLU(), nn.Dropout(p=dropout_p)]
        layers += [nn.Linear(d_hidden, d_out)]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
# Training function
def train(model, train_loader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move data to GPU if available

        optimizer.zero_grad()  # Reset gradients
        outputs = model(batch_X).squeeze()  # Forward pass (ensure output shape matches labels)
        loss = loss_fn(outputs, batch_y)  # Compute MSE loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        total_loss += loss.item()  # Accumulate loss
    
    return total_loss / len(train_loader)  # Return average loss per batch

# Evaluation function
def evaluate(model, val_loader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()  # Ensure output shape matches labels
            
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss  # Return MSE loss (lower is better)

# %% [markdown]
# ## Predictions of the MLP method

# %%
X = games[col].fillna(-1)
y = games['Pred']

imputer = SimpleImputer(strategy='mean')  
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)


# turn numpy arrays into tensors
X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Convert features to float
y_train_tensor = torch.tensor(y, dtype=torch.float32)  # Convert labels to long (for classification)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Train, valid split
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



model = NeuralNetwork(X_scaled.shape[1], 1, hidden_layers_count)
#print(model)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate as needed



# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    
    # print(f"Epoch {epoch+1}/{num_epochs}:")
    # print(f"  Train Loss (MSE): {train_loss:.4f}")
    # print(f"  Validation Loss (MSE): {val_loss:.4f}")

X_submit = sub[col].fillna(-1)
X_submit_imputed = imputer.transform(X_submit)
X_submit_scaled = scaler.transform(X_submit_imputed)


X_submit_tensor = torch.tensor(X_submit_scaled, dtype=torch.float32)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_submit_tensor = X_submit_tensor.to(device)

# Set model to evaluation mode
model.eval()

# Make predictions (disable gradients for efficiency)
with torch.no_grad():
    y_preds = model(X_submit_tensor).cpu().numpy()  # Move back to CPU for saving

sub['Pred'] = y_preds
# sub[['ID', 'Pred']].to_csv('submission.csv', index=False)

pred_of_MLP = sub[['ID', 'Pred']]


# %% [markdown]
# ## Optuna for MLP

# %%
def objective_mlp(trial):
    # 定义需要优化的参数
    params = {
    'batch_size': trial.suggest_int('batch_size', 16, 64),  # 批量大小
    'lr': trial.suggest_float('lr', 0.001, 0.1, log=True),  # 学习率
    'n_layers': trial.suggest_int('n_layers', 2, 5),  # 网络层数
    'num_epochs': trial.suggest_int('num_epochs', 5, 30),  # 训练轮数
    'dropout_p': trial.suggest_float('dropout_p', 0.1, 0.5),  # Dropout 概率
    'hidden_layers_count': trial.suggest_int('hidden_layers_count', 50, 200)  # 隐藏层神经元数量
    }

    batch_size, lr, n_layers, num_epochs, dropout_p, hidden_layers_count = params.values()

    imputer = SimpleImputer(strategy='mean')  
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)


    # turn numpy arrays into tensors
    X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Convert features to float
    y_train_tensor = torch.tensor(y, dtype=torch.float32)  # Convert labels to long (for classification)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Train, valid split
    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    model = NeuralNetwork(X_scaled.shape[1], 1, hidden_layers_count, n_layers=n_layers, dropout_p = dropout_p)
    #print(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate as needed



    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
        # print(f"Epoch {epoch+1}/{num_epochs}:")
        # print(f"  Train Loss (MSE): {train_loss:.4f}")
        # print(f"  Validation Loss (MSE): {val_loss:.4f}")

    # X_submit = sub[col].fillna(-1)
    # X_submit_imputed = imputer.transform(X_submit)
    # X_submit_scaled = scaler.transform(X_submit_imputed)


    # X_submit_tensor = torch.tensor(X_submit_scaled, dtype=torch.float32)

    # # Move to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # X_submit_tensor = X_submit_tensor.to(device)

    # # Set model to evaluation mode
    # model.eval()

    # # Make predictions (disable gradients for efficiency)
    # with torch.no_grad():
    #     y_preds = model(X_submit_tensor).cpu().numpy()  # Move back to CPU for saving

    # sub['Pred'] = y_preds
    # # sub[['ID', 'Pred']].to_csv('submission.csv', index=False)

    # pred_of_MLP = sub[['ID', 'Pred']]
    
    return val_loss

# 创建 Optuna 研究对象
study_MLP = optuna.create_study(direction='minimize')  # 最小化 MSE

# 运行优化
study_MLP.optimize(objective_mlp, n_trials=5, show_progress_bar=True)  # 运行 50 次试验

# 获取最佳超参数
best_params_mlp = study_MLP.best_params
best_score_mlp = study_MLP.best_value

print("Best Hyperparameters:", best_params_mlp)
print("Best MSE:", best_score_mlp)


# %% [markdown]
# # Catboost

# %% [markdown]
# ## EDA for catboost

# %%
# men_results = pd.concat([regular_m, tourney_m])[['Season', 'WTeamID', 'LTeamID']].copy()
men_results = tourney_m[['Season', 'WTeamID', 'LTeamID']].copy()
men_results['Result'] = 1 
m_inv = men_results.copy()
m_inv[['WTeamID', 'LTeamID']] = men_results[['LTeamID', 'WTeamID']].values
m_inv['Result'] = 0  # Loss label
men_results_final = pd.concat([men_results, m_inv], ignore_index=True)
# women_results = pd.concat([regular_w, tourney_w])[['Season', 'WTeamID', 'LTeamID']].copy()
women_results = tourney_w[['Season', 'WTeamID', 'LTeamID']].copy()
women_results['Result'] = 1 
w_inv = women_results.copy()
w_inv[['WTeamID', 'LTeamID']] = women_results[['LTeamID', 'WTeamID']].values
w_inv['Result'] = 0  # Loss label
women_results_final = pd.concat([women_results, w_inv], ignore_index=True)
# 居然只用季后赛作为训练集吗，常规赛也值得尝试
# 难道说模型的处理数据能力不足

# %% [markdown]
# ## Params of Catboost

# %%
catboost_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function':
    'Logloss',
    'verbose': 200
}

# %% [markdown]
# ## Predictions of Catboost

# %%
# Train model
all_results = pd.concat([men_results_final, women_results_final], ignore_index=True)

X = all_results[['Season', 'WTeamID', 'LTeamID']]
y = all_results['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(**catboost_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Evaluate model
preds = model.predict_proba(X_test)[:, 1]
print(f"Brier Score: {brier_score_loss(y_test, preds)}")

# Generate submission
input_folder = r"./kaggle/input/march-machine-learning-mania-2025"
required_ids_df = pd.read_csv(os.path.join(input_folder, "SampleSubmissionStage2.csv"))
team_ids = []
for game_id in required_ids_df['ID']:
    # 分割ID字符串，获取队伍ID
    team_ids.extend(game_id.split('_')[1:])  # 去掉年份和下划线

# 使用集合去重，然后转换为numpy数组并排序
all_teams = np.sort(np.unique(team_ids))
pairings = list(combinations(all_teams, 2))

# Load required matchup IDs
required_ids_df = pd.read_csv(os.path.join(input_folder, "SampleSubmissionStage2.csv"))
required_ids = set(required_ids_df['ID'])

def create_submission(pairings, season=2025, max_rows=131407):
    submission = []
    for (team1, team2) in pairings:
        matchup_id = f"{season}_{min(team1, team2)}_{max(team1, team2)}"
        if matchup_id in required_ids:
            input_data = pd.DataFrame({'Season': [season], 'WTeamID': [min(team1, team2)], 'LTeamID': [max(team1, team2)]})
            pred = model.predict_proba(input_data)[0, 1] if len(input_data) > 0 else 0.5
            submission.append([matchup_id, pred])
    submission_df = pd.DataFrame(submission, columns=["ID", "Pred"])
    print(f"Submission file has {submission_df.shape[0]} rows.")
    return submission_df

submission_df = create_submission(pairings)

pred_of_Catboost = submission_df

# 居然用更多的数据效果会更差吗

# %% [markdown]
# ## Optuna for Catboost

# %%
def objective_catboost(trial):
    # 定义需要优化的参
    catboost_params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),  # 迭代次数
        'depth': trial.suggest_int('depth', 3, 10),  # 树的深度
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),  # 学习率
        'loss_function': trial.suggest_categorical('loss_function', ['Logloss']),  # 损失函数
        'verbose': 200  # 打印日志的频率（固定值）
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluate model
    preds = model.predict_proba(X_test)[:, 1]
    #print(f"Brier Score: {brier_score_loss(y_test, preds)}")


    return brier_score_loss(y_test, preds)

# 创建 Optuna 研究对象
study_catboost = optuna.create_study(direction='minimize')  # 最小化 MSE

# 运行优化
study_catboost.optimize(objective_catboost, n_trials=4, show_progress_bar=True)  # 运行 50 次试验

# 获取最佳超参数
best_params_catboost = study_catboost.best_params
best_score_catboost = study_catboost.best_value

print("Best Hyperparameters:", best_params_catboost)
print("Best MSE:", best_score_catboost)


# %% [markdown]
# # ELO

# %% [markdown]
# ## Params of ELO

# %%
init_rating_m = 1200
init_rating_w = 1250
k_m = 125
k_w = 190
alpha_m = None
alpha_w = None
weights_regular_m = 1
weights_tournament_m = 0.7
weights_regular_w = 0.95
weights_tournament_w = 1


# %% [markdown]
# ## Functions of ELO

# %%
def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None, weights=False, nan_score=1):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
        
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    loss = []
    margin_of_victory = 1
    weight = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls, w  in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
                margin_of_victory = (ws - ls)/alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        # Update ratings for winning and losing teams
        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below 1
        if team_dict[lteam] < 1:
            team_dict[lteam] = 1
            
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1-rateW)**2)
        
    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, nan_score=1):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - weights (array-like): Containing weights for each match.

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
    
    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, nan_score)
    # Calculate loss only on tourney results
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"Loss: {loss}")
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

    # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results

def generate_match_predictions(elo_df, teams, season=2025):
    predictions = []
    
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1_id = teams[i]
            team2_id = teams[j]
            
            # 获取两支队伍的 Elo 评分
            rating_team1 = elo_df[(elo_df['TeamID'] == team1_id) & (elo_df['Season'] == season)]['Rating_Last'].values[0]
            rating_team2 = elo_df[(elo_df['TeamID'] == team2_id) & (elo_df['Season'] == season)]['Rating_Last'].values[0]
            
            # 计算胜率
            expected_score_team1 = 1 / (1 + 10 ** ((rating_team2 - rating_team1) / 400))
            
            # 生成 ID
            match_id = f"{season}_{team1_id}_{team2_id}"
            
            # 添加到预测列表
            predictions.append([match_id, expected_score_team1])
    
    # 创建 DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['ID', 'Pred'])
    return predictions_df

# %% [markdown]
# ## EDA for ELO

# %%
regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = weights_regular_m
tourney_m['weight'] = weights_tournament_m

data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = weights_regular_w
tourney_w['weight'] = weights_tournament_w

data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

# %% [markdown]
# ## Predictions of ELO

# %%
elo_df_men = create_elo_data(teams_m.TeamID, data_m, initial_rating=init_rating_m, k=k_m, alpha=alpha_m, weights=data_m['weight'])
elo_df_women = create_elo_data(teams_w.TeamID, data_w, initial_rating=init_rating_w, k=k_w, alpha=alpha_w, weights=data_w['weight'])


# 生成男子组比赛预测
men_teams_2025 = teams_m[teams_m['TeamID'].isin(elo_df_men[elo_df_men['Season'] == 2025]['TeamID'])]['TeamID'].values
men_predictions = generate_match_predictions(elo_df_men, men_teams_2025)

# 生成女子组比赛预测
women_teams_2025 = teams_w[teams_w['TeamID'].isin(elo_df_women[elo_df_women['Season'] == 2025]['TeamID'])]['TeamID'].values
women_predictions = generate_match_predictions(elo_df_women, women_teams_2025)

# 合并所有预测
all_predictions = pd.concat([men_predictions, women_predictions])

pred_of_ELO = all_predictions


# %%
elo_df_men

# %% [markdown]
# ## Optuna for ELO

# %%
def objective_elo(trial):
    # 定义需要优化的参
    params_elo_m = {
        # 男性相关参数
        'init_rating_m': trial.suggest_int('init_rating_m', 1000, 1500),  # 初始评分（男性）
        'k_m': trial.suggest_int('k_m', 100, 250),  # K 值（男性）
        'weights_regular_m': trial.suggest_float('weights_regular_m', 0.5, 1.5),  # 普通比赛权重（男性）
        'weights_tournament_m': trial.suggest_float('weights_tournament_m', 0.5, 1.5),  # 锦标赛权重（男性）
        'alpha_m': None,  # 男性相关固定参数

    }

    regular_m['tourney'] = 0
    tourney_m['tourney'] = 1
    regular_m['weight'] = weights_regular_m
    tourney_m['weight'] = weights_tournament_m

    data_m = pd.concat([regular_m, tourney_m])
    data_m.sort_values(['Season', 'DayNum'], inplace=True)
    data_m.reset_index(inplace=True, drop=True)

    elo_df_men = create_elo_data(teams_m.TeamID, data_m, initial_rating=init_rating_m, k=k_m, alpha=alpha_m, weights=data_m['weight'])


    params_elo_w = {
        # 女性相关参数
        'init_rating_w': trial.suggest_int('init_rating_w', 1000, 1500),  # 初始评分（女性）
        'k_w': trial.suggest_int('k_w', 100, 250),  # K 值（女性）
        'weights_regular_w': trial.suggest_float('weights_regular_w', 0.5, 1.5),  # 普通比赛权重（女性）
        'weights_tournament_w': trial.suggest_float('weights_tournament_w', 0.5, 1.5),  # 锦标赛权重（女性）
        'alpha_w': None,  # 女性相关固定参数
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluate model
    preds = model.predict_proba(X_test)[:, 1]
    #print(f"Brier Score: {brier_score_loss(y_test, preds)}")


    return brier_score_loss(y_test, preds)

# 创建 Optuna 研究对象
study_elo = optuna.create_study(direction='minimize')  # 最小化 MSE

# 运行优化
study_elo.optimize(objective_elo, n_trials=4, show_progress_bar=True)  # 运行 50 次试验

# 获取最佳超参数
best_params_elo = study_elo.best_params
best_score_elo = study_elo.best_value

print("Best Hyperparameters:", best_params_elo)
print("Best MSE:", best_score_elo)


# %%
# print('Catboost', pred_of_Catboost.head(), len(pred_of_Catboost))
# print('XGBoost', pred_of_XGBoost.head(), len(pred_of_XGBoost))
# print('MLP', pred_of_MLP.head(), len(pred_of_MLP))
# print('ELO', pred_of_ELO.head(), len(pred_of_ELO))

# 汇总了和分开跑不一样，TMD



