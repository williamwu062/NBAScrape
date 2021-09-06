import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plot
import seaborn as sns
from xgboost import XGBRegressor


def web_scrape():
    url_1 = 'https://web.archive.org/web/20170129032938/https://hoopshype.com/salaries/players/'
    response_1 = requests.get(url_1, timeout=10)
    soup_1 = BeautifulSoup(response_1.text, 'html.parser')

    names = soup_1.find_all('td', attrs={'class': 'name'})[1:]
    names = [name.find('a') for name in names]
    names = [string.text.strip() for string in names]

    salaries = soup_1.find_all('td', attrs={'class': 'hh-salaries-sorted'})[1:]
    salaries = [amount['data-value'] for amount in salaries]

    url_2 = 'https://www.basketball-reference.com/leagues/NBA_2016_per_game.html'
    response_2 = requests.get(url_2, timeout=10)
    soup_2 = BeautifulSoup(response_2.text, 'html.parser')

    stat_names_1 = soup_2.find_all('tr', limit=1)[0].find_all('th')
    stat_names_1 = [string.text for string in stat_names_1]
    stat_names_1 = stat_names_1[2:]

    url_3 = 'https://www.basketball-reference.com/leagues/NBA_2016_advanced.html'
    response_3 = requests.get(url_3, timeout=10)
    soup_3 = BeautifulSoup(response_3.text, 'html.parser')

    stat_names_2 = soup_3.find_all('tr', limit=1)[0].find_all('th')
    stat_names_2 = [string.text for string in stat_names_2]
    stat_names_2 = stat_names_2[7:]

    names = [[string] for string in names]
    for index, name in enumerate(names):
        try:
            stats_1 = soup_2.find('td', string=name).find_parent('tr')
            stats_1 = stats_1.find_all('td')
            stats_1 = [string.text for string in stats_1]
            stats_1 = stats_1[1:]
            stats_2 = soup_3.find('td', string=name).find_parent('tr')
            stats_2 = stats_2.find_all('td')
            stats_2 = [string.text for string in stats_2]
            stats_2 = stats_2[6:]
            names[index].extend(stats_1)
            names[index].extend(stats_2)
        except AttributeError:
            no_info = [''] * 40
            names[index].extend(no_info)

    id_row = ['Players']
    id_row += stat_names_1 + stat_names_2
    data = pd.DataFrame(names, columns=id_row)
    data['Salaries'] = salaries
    data.to_csv('resources/nba_salaries.csv')

    return data;

'''
    rows = zip(names, salaries)
    with open('resources/nba_salaries.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
'''

#web_scrape()
data = pd.read_csv('resources/nba_salaries.csv')

#drop bad data
data.dropna(axis=1, how='all', inplace=True)
data.drop(data.columns[0], axis=1, inplace=True)
bad_players = data.loc[data['MP'].isna()]
bad_players = bad_players.index
data.drop(index=bad_players, axis=0, inplace=True)

data.fillna(0, inplace=True)

#Change Pos values into integers
data.loc[data['Pos'] == 'PG', 'Pos'] = 1
data.loc[data['Pos'] == 'SG', 'Pos'] = 2
data.loc[data['Pos'] == 'SF', 'Pos'] = 3
data.loc[data['Pos'] == 'PF', 'Pos'] = 4
data.loc[data['Pos'] == 'C', 'Pos'] = 5
data.loc[data['Pos'] == 'PF-C', 'Pos'] = 5

plot.figure(figsize=(13,13))
sns.set(font_scale=0.4, font='Helvetica')
res = sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.1f', square=True, linewidths=1)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize=4)
res.set_yticklabels(res.get_ymajorticklabels(), fontsize=4)
plot.show()

X = data.drop(['Salaries', 'Players', 'Tm'], axis=1)
y = data['Salaries']

total = 0
for _ in range(30):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)
    acc = forest.score(X_test, y_test)
    total += acc
print(total/30)