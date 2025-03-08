import numpy as np
import pandas as pd
from sklearn import *
#import redisAI
import glob

path = '/kaggle/input/march-machine-learning-mania-2025/**'
data = {p.split('/')[-1].split('.')[0] : pd.read_csv(p, encoding='latin-1') for p in glob.glob(path)}

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
#games = pd.concat((season_cresults, tourney_cresults), axis=0, ignore_index=True)
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

col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm', 'WLoc'] + c_score_col]

col

import optuna
from sklearn import ensemble
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  
scaler = StandardScaler()

X = games[col].fillna(-1)
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

y = games['Pred']

print(X_scaled.shape, y.shape)
print(X_scaled.dtype, y.dtype)
y

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

# turn numpy arrays into tensors
X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Convert features to float
y_train_tensor = torch.tensor(y, dtype=torch.float32)  # Convert labels to long (for classification)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Train, valid split
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


batch_size = 32  # Set batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers = 2):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden),
            nn.ReLU()]
        for layer in range(n_layers):
            layers += [nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden),nn.ReLU(), nn.Dropout(p=0.3)]
        layers += [nn.Linear(d_hidden, d_out)]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


import torch.optim as optim

# Define mode, loss function and optimizer

model = NeuralNetwork(X_scaled.shape[1], 1, 100)
print(model)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adjust learning rate as needed

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

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 5  # Adjust the number of epochs based on performance

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss (MSE): {train_loss:.4f}")
    print(f"  Validation Loss (MSE): {val_loss:.4f}")



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
sub[['ID', 'Pred']].to_csv('submission.csv', index=False)





