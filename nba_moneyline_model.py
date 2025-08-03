from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

ADVANCED_STATS = 'nba_24-25_advanced_team_stats.csv'

#map team abbreviations to mascot
team_dict = {'GSW': 'Warriors', 'CHI': 'Bulls', 'CLE': 'Cavaliers', 
            'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets', 
            'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 
            'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 
            'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat', 'MIL': 'Bucks', 
            'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
            'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
            'POR': 'Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 
            'TOR': 'Raptors','UTA': 'Jazz', 'WAS': 'Wizards'}

def getTeam():
    while True:
        team = str(input('Enter betting team abbreviation: '))
        opponent = str(input('Enter opposing team abbreviation: '))
        venue = str(input('Enter betting team home or away: '))
        if isValid(team) and isValid(opponent) and (venue.lower() == 'home' or 
                                                    venue.lower() == 'away'):
            return team, opponent, venue
        else:
            print('Error, try again')

def getOdds():
    while True:
        odds = str(input('Enter moneyline odds: '))
        if isValidOdds(odds):
            return odds
        else:
            print('Error, try again')

def isValidOdds(odds):
    if '+' not in odds and '-' not in odds:
        return False
    elif '+' in odds and '-' in odds:
        return False
    for c in odds[1:]:
        if c.isdigit() == False:
            return False
    return True

def isValid(team):
    if team in team_dict.keys():
        return True
    for abb in team_dict.keys():
        if team.lower() == abb.lower():
            return True
    return False

def getTeamID(team): #helper for getGameLog
    all_teams = teams.get_teams()
    team_x = [t for t in all_teams if t['abbreviation'] == team.upper()][0]
    team_id = team_x['id']
    return team_id

# print(getTeamID('NYK'))

def getOpponent(str): #helper for getGameLog
    if 'vs.' in str:
        opp_index = str.find('.')
        opponent = str[opp_index+2:]
        return opponent, 1
    elif '@' in str:
        opp_index = str.find('@')
        opponent = str[opp_index+2:]
        return opponent, 0

# print(getOpponent('MIL vs. CHA'))

def getGameLog(team): #last 200 games
    team_id = getTeamID(team)
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=str(team_id),
                                        season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]
    opponents = []
    win_loss = []
    home_away = []
    for index, row in games.head(200).iterrows():
        matchup = games.loc[index, 'MATCHUP']
        opponent, homeAway = getOpponent(matchup)
        mascot = team_dict[opponent]
        opponents.append(mascot)
        home_away.append(homeAway)
        outcome = games.loc[index, 'WL']
        if outcome == 'W':
            win_loss.append(1)
        else:
            win_loss.append(0)
    game_log = pd.DataFrame(
        {
            'Opponent': opponents[::-1],
            'H1/A0': home_away[::-1],
            'W/L': win_loss[::-1]
        }
    )
    return game_log

# print(getGameLog('MIL'))

def addStats(): #add feauture variables to gameLog
    team, opponent, venue = getTeam()
    opponent = team_dict[opponent.upper()]
    if venue.lower() == 'home':
        venue = 1
    else:
        venue = 0
    game_log = getGameLog(team)

    df = pd.read_csv(ADVANCED_STATS)
    team_stats = df[['Team', 'SRS','NRtg', 'OeFG%','OTOV%','OORB%', 'DRB%']].copy()
    team_stats['Team'] = team_stats['Team'].apply(lambda x: x.replace('*', ''))
    for team in team_stats['Team']:
        fullName = team.split()
        mascot = fullName[-1]
        team_stats.loc[team_stats['Team'] == team, 'Team'] = mascot
    nrtg = []
    srs = []
    eFG = []
    off_reb_pct = []
    turnover_pct = []
    def_reb_pct = []
    for opposition in game_log['Opponent']:
        temp = team_stats.loc[team_stats['Team'] == opposition]
        row = temp.iloc[0].to_dict()
        nrtg.append(row['NRtg'])
        srs.append(row['SRS'])
        eFG.append(row['OeFG%'])
        off_reb_pct.append(row['OORB%'])
        turnover_pct.append(row['OTOV%'])
        def_reb_pct.append(row['DRB%'])
    features = {'NRtg': nrtg, 'SRS': srs, 'eFG%': eFG, 'ORB%': off_reb_pct,
                'TOV%': turnover_pct, 'DRB%': def_reb_pct}
    for key in features:
        game_log[key] = features[key]

    #now create df for upcoming game against opponent 
    opp_temp = team_stats.loc[team_stats['Team'] == opponent]
    opp_row = opp_temp.iloc[0].to_dict()
    predict_data = pd.DataFrame(
        {
            'H1/A0': [venue],
            'NRtg': [opp_row['NRtg']],
            'SRS': [opp_row['SRS']],
            'eFG%': [opp_row['OeFG%']],
            'ORB%': [opp_row['OORB%']],
            'TOV%': [opp_row['OTOV%']],
            'DRB%': [opp_row['DRB%']]
        }
    )
    return game_log, predict_data

def predictOutcome(data, predict_data):
    vegas_odds = getOdds()
    vegas_probability = getProbability(vegas_odds)
    feature_cols = ['H1/A0', 'NRtg', 'SRS', 'eFG%', 'ORB%', 'TOV%', 'DRB%']
    X = data[feature_cols] #features
    y = data['W/L'] #target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            shuffle=False)
    log_reg = LogisticRegression(C=0.5, solver='saga', max_iter=10**5)
    log_reg.fit(X_train, y_train)
    train_accuracy = log_reg.score(X_train, y_train)
    test_accuracy = log_reg.score(X_test, y_test)
    print(f"Training Accuracy: {round(train_accuracy, 3)}")
    print(f"Test Accuracy: {round(test_accuracy, 3)}")
    result = log_reg.predict_proba(predict_data)
    win_prob = round(result[0, 1], 3)
    return win_prob, vegas_probability

def getProbability(odds):
    if '+' in odds:
        implied_probability = 100 / (int(odds) + 100)
        return round(implied_probability, 3)
    elif '-' in odds:
        implied_probability = abs(int(odds)) / (abs(int(odds)) + 100)
        return round(implied_probability, 3)

def main():
    data, predict_data = addStats()
    pred, vegas = predictOutcome(data, predict_data)
    print(f'Calculated win probability: {pred}')
    print(f'Book probability: {vegas}')
    print(f'Value over book: {round(pred-vegas, 3)}')
main()

