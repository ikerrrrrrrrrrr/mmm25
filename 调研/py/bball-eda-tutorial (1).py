# Essential Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import ipywidgets as widgets
from IPython.display import display
import warnings

# Niche Imports
import chardet

# Settings
sns.set_style("whitegrid")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")

# Tools
def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    
    # print(result)  # Check detected encoding
    df = pd.read_csv(csv_file, encoding=result["encoding"])
    return df

# View files

def get_comp_files_and_dirs(input_dir):
    file_list = []
    dir_list = []
    try:
        for comp_dir in os.listdir(input_dir):
            comp_path = '/'.join([input_dir, comp_dir])
            print(f"Competition Directory: {comp_path}")
            print("Contains:")
            with os.scandir(comp_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        print(f"- (File) {entry.name}, Size: {entry.stat().st_size} bytes")
                        file_list.append(os.path.join(input_dir, comp_dir, entry))
                    elif entry.is_dir():
                        print(f"- (Folder) {entry.name}")
                        dir_list.append(os.path.join(input_dir, comp_dir, entry))

    except FileNotFoundError:
        print(f"The specified directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission error accessing directory '{directory}'.")
    return file_list, dir_list
    
input_dir = '/kaggle/input'
file_list, dir_list = get_comp_files_and_dirs(input_dir)

comp_dir = '/kaggle/input/march-machine-learning-mania-2025'
data_section_1_mens_list = [
    'MTeams.csv',
    'MSeasons.csv',
    'MNCAATourneySeeds.csv',
    'MRegularSeasonCompactResults.csv',
    'MNCAATourneyCompactResults.csv'
]
data_section_1_womens_list = [
    'WTeams.csv',
    'WSeasons.csv',
    'WNCAATourneySeeds.csv',
    'WRegularSeasonCompactResults.csv',
    'WNCAATourneyCompactResults.csv',
]
sample_submission = 'SampleSubmissionStage1.csv'
data_section_2_mens_list = [
    'MRegularSeasonDetailedResults.csv',
    'MNCAATourneyDetailedResults.csv',
]
data_section_2_womens_list = [
    'WRegularSeasonDetailedResults.csv',
    'WNCAATourneyDetailedResults.csv',
]
data_section_3_mens_list = [
    'Cities.csv',
    'MGameCities.csv',
]
data_section_3_womens_list = [
    'Cities.csv',
    'WGameCities.csv',
]
data_section_4_list = [
    'MMasseyOrdinals.csv'
]
data_section_5_list = [
    'MTeamCoaches.csv'
]

mens_df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WTeams.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

# Find unique values in Team Names
series1 = mens_df['TeamName']
series2 = womens_df['TeamName']
unique_in_series1 = series1[~series1.isin(series2)]

# Values in series2 but not in series1
unique_in_series2 = series2[~series2.isin(series1)]

# Combine results
unique_values = pd.concat([unique_in_series1, unique_in_series2])

print(unique_values)

# Plot the durations of each team
for gender in ['Mens']:
    print(f"Investigating {gender} Team")
    df = load_csv(os.path.join(comp_dir, 'MTeams.csv'))

    # Calculate the width of the bar
    df['Widths'] = df['LastD1Season'] - df['FirstD1Season']
    

    # # Plot bars
    fig, ax = plt.subplots(figsize=(15,60))
    # ax.barh(df['TeamName'], df['Widths'], left=df['FirstD1Season'], color='blue', edgecolor='black')
    # ax.invert_yaxis()
    # # Labels and grid
    # ax.set_xlabel("Value")
    # ax.set_ylabel("Bars")
    # ax.set_title("Horizontal Bars from Start to End Values")
    # ax.grid(axis='x', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    
    # plt.show()

    sns.barplot(
        data=df,
        y="TeamName",        # Y-axis (categorical variable)
        x="Widths",          # X-axis (bar length)
        hue=None,            # No grouping
        orient="h",          # Horizontal bars
        color="blue",        # Bar color
        edgecolor="black",   # Border color
        ax=ax                # Use the existing axis
    )
    # Offset each bar by the start
    for i, (start, width) in enumerate(zip(df["FirstD1Season"], df["Widths"])):
        ax.patches[i].set_x(start)  # Shift bar to start position
    ax.set_xlim(df["FirstD1Season"].min(), df["LastD1Season"].max())  # Fit all bars correctly
    # Show x-axis at the top as well
    ax.xaxis.set_ticks_position("both")  # Show ticks on both top and bottom
    ax.xaxis.set_label_position("top")   # Move x-axis label to the top
    ax.tick_params(axis="x", which="both", labeltop=True, labelbottom=True)  # Show tick labels at the top
    ax.spines["top"].set_visible(True)   # Show the top spine (border)
    ax.set_xlabel("")
    plt.title('Durations of Mens NCAA Teams')
    plt.show()


mens_df = load_csv(os.path.join(comp_dir, 'MSeasons.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WSeasons.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

mens_df = load_csv(os.path.join(comp_dir, 'MNCAATourneySeeds.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WNCAATourneySeeds.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

mens_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonCompactResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonCompactResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

# Look at distributions of winning and losing scores across seasons
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    df['Score Difference']  = df['WScore'] - df['LScore']

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='WScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Winning Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='LScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Losing Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='Score Difference', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Score Difference Distributions')
    plt.tight_layout()
    plt.show()

mens_df = load_csv(os.path.join(comp_dir, 'MNCAATourneyCompactResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WNCAATourneyCompactResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

# Look at distributions of winning and losing scores across seasons
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    df['Score Difference']  = df['WScore'] - df['LScore']

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='WScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Winning Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='LScore', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Losing Score Distributions')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.boxplot(x='Season', y='Score Difference', data=df)
    plt.xticks(rotation=90)
    plt.title(f'{gender} Score Difference Distributions')
    plt.tight_layout()
    plt.show()

ss_df = load_csv(os.path.join(comp_dir, 'SampleSubmissionStage1.csv'))
ss_df.head()

mens_df = load_csv(os.path.join(comp_dir, 'MRegularSeasonDetailedResults.csv'))
womens_df = load_csv(os.path.join(comp_dir, 'WRegularSeasonDetailedResults.csv'))
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    display(df.describe())
    display(df.head())

mens_df.columns

prefix_dict = {
    'W': 'Winning Team',
    'L': 'Losing Team',
}
suffix_dict = {
    'FGM': 'Field goals made', 
    'FGA': 'Field goals attempted', 
    'FGM3': 'Three pointers made', 
    'FGA3': 'Three pointers attempted', 
    'FTM': 'Free throws made', 
    'FTA': 'Free throws attempted', 
    'OR': 'Offensive rebounds', 
    'DR': 'Defensive rebounds', 
    'Ast': 'Assists', 
    'TO': 'Turnovers committed', 
    'Stl': 'Steals', 
    'Blk': 'Blocks', 
    'PF': 'Personal fouls committed',
}
for gender in ['Mens', 'Womens']:
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    for key, value in suffix_dict.items():
        df[f'D{key}'] = df[f'W{key}'] - df[f'L{key}']
        
        plt.figure(figsize=(10,6))
        sns.histplot(df[f'D{key}'], bins=15, kde=True)
        plt.xticks(rotation=90)
        plt.title(f'{gender} {value} Difference')
        plt.tight_layout()
        plt.show()
    

fig, axes = plt.subplots(1, 2, figsize=(40, 20))
for i, gender in enumerate(['Mens', 'Womens']):
    print(f"Investigating {gender} Team")
    if gender == 'Mens':
        df = mens_df
    else:
        df = womens_df
    df.fillna(0, inplace=True)
    df.dropna(inplace=True)
    if len(df) < 1:
        print("Empty df")
    w_columns = [f'W{suffix}' for suffix in suffix_dict.keys()]
    l_columns = [f'L{suffix}' for suffix in suffix_dict.keys()]

    # Group by 'TeamID' and compute the mean for selected columns
    w_heatmap_data_sum = df.groupby('WTeamID')[w_columns].sum()
    l_heatmap_data_sum = df.groupby('LTeamID')[l_columns].sum()
    w_heatmap_data_count = df.groupby('WTeamID')[w_columns].count()
    l_heatmap_data_count = df.groupby('LTeamID')[l_columns].count()

    w_heatmap_data_sum.columns = list(suffix_dict.keys())
    l_heatmap_data_sum.columns = list(suffix_dict.keys())
    w_heatmap_data_count.columns = list(suffix_dict.keys())
    l_heatmap_data_count.columns = list(suffix_dict.keys())

    if w_heatmap_data_sum.empty or l_heatmap_data_sum.empty:
        print("Warning: One of the grouped dataframes is empty!")
    
    heatmap_data = (w_heatmap_data_sum + l_heatmap_data_sum)/(w_heatmap_data_count + l_heatmap_data_count)
    heatmap_data = heatmap_data - np.mean(heatmap_data, axis=0)
    # Plot heatmap
    sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", fmt=".1f", cbar_kws={'label': 'Normalised Value'}, center=0, ax=axes[i])
    axes[i].set_ylabel("Team ID")
    axes[i].set_xlabel("Metric")
    axes[i].set_title(f"Average {gender} metrics per team over ALL games relative to mean")
plt.show()




