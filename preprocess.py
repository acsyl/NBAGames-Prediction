import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



# Read file and parse date
data_filename="data/leagues.csv"
dataset = pd.read_csv(data_filename,parse_dates=[0])

# rename columns
dataset.columns = ["Date", "Score Type", "Visitor Team","VisitorPts", "Home Team", "HomePts"]

# create new coloums
dataset["OT?"] = np.nan
dataset["Notes"] = np.nan  
dataset["HomeLastWin"] = False
dataset["VisitorLastWin"] = False


# create && compute HomeWin
dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]

# compute home and visitor team win the last game
won_last = defaultdict(int)
for index, row in dataset.iterrows():
	# update lastwin
	home_team = row["Home Team"]
	visitor_team = row["Visitor Team"]
	row["HomeLastWin"] = won_last[home_team]
	row["VisitorLastWin"] = won_last[visitor_team]
	dataset.ix[index] = row
	# set the most recently win 
	won_last[home_team] = row["HomeWin"]
	won_last[visitor_team] = not row["HomeWin"]

# use the last seanson rank
standings = pd.read_csv("data/leagues_NBA_2013_standings_expanded-standings.csv")
dataset["HomeTeamRanksHigher"] = 0
for index, row in dataset.iterrows():
	home_team = row["Home Team"]
	visitor_team = row["Visitor Team"]
	if home_team == "New Orleans Pelicans":
		home_team = "New Orleans Hornets"
	elif visitor_team == "New Orleans Pelicans":
		visitor_team = "New Orleans Hornets"

	home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
	visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
	row["HomeTeamRanksHigher"] = int(home_rank>visitor_rank)
	dataset.ix[index] = row
last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0
for index, row in dataset.iterrows():
	home_team = row["Home Team"]
	visitor_team = row["Visitor Team"]
	teams = tuple(sorted([home_team, visitor_team]))
	row["HomeTeamWonLast"] = 1  if row["Home Team"] == last_match_winner[teams] else 0
	dataset.ix[index] = row

	# set the last wining
	winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
	last_match_winner[teams] = winner


encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_teams,visitor_teams]).T
onehot = OneHotEncoder()
X_teams_expanded = onehot.fit_transform(X_teams).todense()
x = dataset[["HomeTeamRanksHigher","HomeTeamWonLast","HomeLastWin","VisitorLastWin"]].values
final_x = np.hstack((X_teams_expanded,x))
y_true = dataset["HomeWin"].values
y_true = y_true*1

np.save("x_input2.npy",X_teams_expanded)
np.save("y.npy",y_true)
print(y_true)
print(final_x.shape)







