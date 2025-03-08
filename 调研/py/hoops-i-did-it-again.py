import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


w_teams = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WTeams.csv").collect()
m_teams = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MTeams.csv").collect()
m_compact_results = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv").collect()
w_compact_results = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv").collect()
m_rankings = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv").collect()

w_seeds = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv").collect()
m_seeds = pl.scan_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv").collect()

submission_stage1 = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv").collect()
submission_stage2 = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv").collect()

m_truth = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv").collect()
w_truth = pl.scan_csv ("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv").collect()

def extract_seeds (seeds : pl.DataFrame) -> pl.DataFrame :
    
    result = seeds.with_columns (pl.col("Seed").str.head(1).alias("Conference"), 
                               pl.col("Seed").str.slice(1,2).cast(pl.UInt8).alias("C_Seed"))
    return result.drop("Seed")

w_seeds_Season = extract_seeds (w_seeds) 
m_seeds_Season = extract_seeds (m_seeds) 

print (m_rankings.head())
def extract_rankings (rankings : pl.DataFrame) -> pl.DataFrame :
    rankings  = rankings.with_columns ((pl.col("Season").cast (pl.String) + "_" + pl.col("TeamID").cast(pl.String)).alias ("Season_Team"))

    last_ranking_day  = rankings.group_by(["Season", "SystemName"]).agg(pl.col("RankingDayNum").max()).sort("Season")

    rankings_last = rankings.join (last_ranking_day, how = "inner", on = ["Season", "SystemName", "RankingDayNum"])


    team_rankings = rankings_last.pivot("SystemName", index="Season_Team", values="OrdinalRank")

    team_rankings = team_rankings.with_columns (pl.col("Season_Team").str.split("_").list.first().cast (pl.Int64).alias ("Season"),
                                               pl.col("Season_Team").str.split("_").list.last().cast (pl.Int64).alias ("TeamID"))

    return team_rankings.drop("Season_Team")

m_team_rankings = extract_rankings (m_rankings)
print (m_team_rankings.head())

w_teams.head()

display (submission_stage1.sort ("ID"))
display (submission_stage2.sort ("ID"))

w_compact_results.get_column("Season").unique()

m_compact_results.filter (pl.col("Season") == 2021).describe()

w_truth

def win_loss_record (teams : pl.DataFrame) -> pl.DataFrame :
    
    result = teams.select ("TeamID")
    base_file =  w_compact_results if (teams.get_column("TeamID").min() > 2000) else m_compact_results
    # season_results = base_file.filter (pl.col("Season") == season)
    
        
    wins = base_file.group_by (["Season","WTeamID"]).len()
    losses = base_file.group_by (["Season","LTeamID"]).len()
    wscores =  base_file.group_by (["Season","WTeamID"]).agg (pl.col("WScore").sum())
    lscores = base_file.group_by (["Season","LTeamID"]).agg (pl.col("LScore").sum())
    opp_wscores = base_file.group_by (["Season","WTeamID"]).agg (pl.col("LScore").sum())
    opp_lscores = base_file.group_by (["Season","LTeamID"]).agg (pl.col("WScore").sum())
    
    result = result.join (wins, how = "left", left_on = "TeamID", right_on = "WTeamID")
    result = result.rename ({"len" : "games won"})
    result = result.join (losses, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"len" : "games lost"})
    result = result.join (wscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"WScore" : "points in win"})
    result = result.join (lscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"LScore" : "points in loss"})
    result = result.join (opp_wscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["WTeamID", "Season"])
    result = result.rename ({"LScore" : "dev points in win"})
    result = result.join (opp_lscores, how = "left", left_on = ["TeamID", "Season"], right_on = ["LTeamID", "Season"])
    result = result.rename ({"WScore" : "dev points in loss"})
    result = result.with_columns ((pl.col("games won") + pl.col("games lost")).alias ("total games"), 
                                  (pl.col("points in win") + pl.col("points in loss")).alias ("total points"), 
                                  (pl.col("dev points in win") + pl.col("dev points in loss")).alias ("total dev points"))
    
    return result.fill_null(0)

male_team_results = win_loss_record (m_teams)

display (male_team_results)

female_team_results = win_loss_record (w_teams)

display (female_team_results)

def create_train (df,truth :pl.DataFrame) -> pl.DataFrame :
    
    truth_is_female = (truth.get_column ("WTeamID").min() > 2000)
    
    result = df.with_columns (pl.col("ID").str.split ("_").list.get(0).cast(pl.Int64).alias ("Season"),
                             pl.col("ID").str.split ("_").list.get(1).cast(pl.Int64).alias ("Team1"),
                             pl.col("ID").str.split ("_").list.get(2).cast(pl.Int64).alias ("Team2"))
   
    if truth_is_female :
        result = result.filter (pl.col('Team1') > 2000)
    else :
        result = result.filter (pl.col('Team1') < 2000)
    
    result = result.join (truth, how = "left", left_on = ["Season", "Team1", "Team2"], right_on = ["Season", "WTeamID", "LTeamID"])
    result = result.join (truth, how = "left", left_on = ["Season", "Team1", "Team2"], right_on = ["Season", "LTeamID", "WTeamID"])
   
   
    result = result.with_columns (pl.when (pl.col("WScore") > 0 ).then (pl.lit(1)).otherwise (
                                  pl.when (pl.col("WScore_right") > 0 ).then (pl.lit(0)).otherwise (
                                           pl.lit(0.5))).alias ("truth")
                                  
    )
    

    return result.drop ('Pred') 
    

male_training = create_train (submission_stage1, m_truth).filter ((pl.col("truth") ==1) | (pl.col("truth") ==0))
female_training = create_train (submission_stage1, w_truth).filter ((pl.col("truth") ==1) | (pl.col("truth") ==0))

male_submission = create_train (submission_stage2, m_truth)
female_submission = create_train (submission_stage2, w_truth)

print (f" total size {male_training.shape  }   ")
print (f" total size {male_submission.shape }   ")

male_training = male_training.join (male_team_results, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
male_training = male_training.join (male_team_results, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])

male_training = male_training.join (m_seeds_Season, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
male_training = male_training.join (m_seeds_Season, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])
#male_training = male_training.join (m_team_rankings, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
#male_training = male_training.join (m_team_rankings, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])

display (male_training)

print (male_training.schema)

female_training = female_training.join (female_team_results, how = "left", left_on = ["Season", "Team1"], right_on = ["Season","TeamID"])
female_training = female_training.join (female_team_results, how = "left", left_on = ["Season", "Team2"], right_on = ["Season","TeamID"])

female_training = female_training.join (w_seeds_Season, how = "left", left_on = ["Season", "Team1"], right_on = ["Season","TeamID"])
female_training = female_training.join (w_seeds_Season, how = "left", left_on = ["Season", "Team2"], right_on = ["Season","TeamID"])

display (female_training)
# X_train = pl.concat ([male_training, female_training], how = "vertical_relaxed").sample (fraction = 1, shuffle = True)

# X_train


male_submission = male_submission.join (male_team_results, how = "left", left_on = ["Season", "Team1"], right_on = ["Season","TeamID"]) 
male_submission = male_submission.join (male_team_results, how = "left", left_on = ["Season", "Team2"], right_on = ["Season","TeamID"]) 
male_submission = male_submission.join (m_seeds_Season, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
male_submission = male_submission.join (m_seeds_Season, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])
#male_submission = male_submission.join (m_team_rankings, how = "left", left_on = ["Season","Team1"], right_on = ["Season","TeamID"])
#male_submission = male_submission.join (m_team_rankings, how = "left", left_on = ["Season","Team2"], right_on = ["Season","TeamID"])

print (male_submission.columns)

female_submission = female_submission.join (female_team_results, how = "left", left_on = ["Season", "Team1"], right_on = ["Season","TeamID"]) 
female_submission = female_submission.join (female_team_results, how = "left", left_on = ["Season", "Team2"], right_on = ["Season","TeamID"]) 
female_submission = female_submission.join (w_seeds_Season, how = "left", left_on = ["Season", "Team1"], right_on = ["Season","TeamID"])
female_submission = female_submission.join (w_seeds_Season, how = "left", left_on = ["Season", "Team2"], right_on = ["Season","TeamID"])


print (f" total size {male_training.shape  }   ")
print (f" total size {male_submission.shape }   ")


['ID', 'Season', 'Team1', 'Team2', 'DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT', 'DayNum_right', 'WScore_right', 'LScore_right', 'WLoc_right', 'NumOT_right', 'truth', 'games won', 'games lost', 'points in win', 'points in loss', 'dev points in win', 'dev points in loss', 'total games', 'total points', 'total dev points', 'games won_right', 'games lost_right', 'points in win_right', 'points in loss_right', 'dev points in win_right', 'dev points in loss_right', 'total games_right', 'total points_right', 'total dev points_right', 'Conference', 'C_Seed', 'Conference_right', 'C_Seed_right']

print (male_training.shape) 

print (male_submission.shape)


!pip install ray==2.10.0
!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor
m_predictor = TabularPredictor(path = '/kaggle/working/march_madness/male',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               # sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID']})
                                 
m_predictor.fit(train_data= male_training.to_pandas(), 
                        presets= 'best_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 16000,
                        num_gpus=0,
                        raise_on_no_models_fitted = True,
                        #dynamic_stacking=False, 
                        #num_stack_levels=0,
                        #hyperparameters=hyper_search,
#                         hyperparameters = my_search_hyperparameters  ,
                        #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )                              
                                

m_predictor.leaderboard()

 

w_predictor = TabularPredictor(path = '/kaggle/working/march_madness/female',
                                       label='truth', 
                               problem_type = 'binary', 
                               eval_metric =  'accuracy',  
                               # sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'ID']})
                                 
w_predictor.fit(train_data= female_training.to_pandas(), 
                        presets= 'best_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 16000,
                        num_gpus=0,
                        raise_on_no_models_fitted = True,
                        #dynamic_stacking=False, 
                        #num_stack_levels=0,
                        #hyperparameters=hyper_search,
#                         hyperparameters = my_search_hyperparameters  ,
                        #hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )     

w_predictor.leaderboard()

schema1 = male_submission.schema
schema2 = male_training.schema    

if schema1 ==schema2 :
    print ("no change detected")
else :
    print ("change detected")

male_submission = male_submission.fill_null(999.0)
male_submission = male_submission.fill_null("W")

male_march_madness_prediction =  m_predictor.predict_proba(male_submission.to_pandas()) 


female_submission = female_submission.fill_null(999.0)
female_submission = female_submission.fill_null("W")
female_march_madness_prediction =  w_predictor.predict_proba(female_submission.to_pandas())


march_madness_prediction = pl.concat ([pl.DataFrame(male_march_madness_prediction), pl.DataFrame(female_march_madness_prediction)], how = "vertical")

combined_id = pl.concat ([male_submission.get_column("ID"), female_submission.get_column("ID")], how = "vertical")

print (march_madness_prediction.head(3))

probabilty_first_team = march_madness_prediction.get_column ("0")
my_submit = pl.DataFrame ([combined_id, probabilty_first_team]) 
    
my_submit = my_submit.rename ({"0" : "Pred"})

my_submit

my_submit.write_csv("submission.csv")

