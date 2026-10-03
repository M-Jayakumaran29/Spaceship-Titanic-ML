#EDA-Exploratory Data Analysis

import pandas as pd
from clean import clean,get_fill_values,fill_generic

DATA_DIR = r'D:\New folder\Spaceship Titanic\spaceship-titanic'

train = pd.read_csv(DATA_DIR + r'\train.csv')
test = pd.read_csv(DATA_DIR + r'\test.csv')

print(train.shape, test.shape)
print(train.head())
train.info()
print(train['Cabin'].nunique())
print(train.isnull().any(axis=1).mean())

# 1. Split Cabin into three new columns
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)

# 2. Transported rate for each Deck and each Side
print(train.groupby('Deck')['Transported'].mean())
print(train.groupby('Side')['Transported'].mean())

# 3. Num is text after splitting, so convert it to numbers first
train['Num'] = pd.to_numeric(train['Num'])
print(train['Num'].min(), train['Num'].max())

print(train.groupby('Deck')['Transported'].agg(['mean', 'count']))
print(train.groupby('CryoSleep')['Transported'].agg(['mean', 'count']))
print(train.groupby(['Deck', 'CryoSleep'])['Transported'].agg(['mean', 'count']))

spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
sleepers = train[train['CryoSleep'] == True]
print(sleepers[spend_cols].max())
train['TotalSpend'] = train[spend_cols].sum(axis=1)
zero_spenders = train[train['TotalSpend'] == 0]
print(zero_spenders['CryoSleep'].value_counts(normalize=True))
awake_zero = train[(train['CryoSleep'] == False) & (train['TotalSpend'] == 0)]
print(awake_zero['Age'].describe())

print(train[train['Age'] <= 17]['TotalSpend'].max())
print(train[train['Age'] <= 20].groupby('Age')['TotalSpend'].max())

for col in ['HomePlanet', 'Destination', 'VIP']:
    print(train.groupby(col)['Transported'].agg(['mean', 'count']))


print(pd.crosstab(train['HomePlanet'], train['Deck']))

print(train.groupby(['HomePlanet', 'CryoSleep'])['Transported'].agg(['mean', 'count']))

earth_sleepers = train[(train['HomePlanet'] == 'Earth') & (train['CryoSleep'] == True)]
print(earth_sleepers.groupby('Deck')['Transported'].agg(['mean', 'count']))

e_sleepers = train[(train['Deck'] == 'E') & (train['CryoSleep'] == True)]
print(e_sleepers.groupby('HomePlanet')['Transported'].agg(['mean', 'count']))

train['Group'] = train['PassengerId'].str.split('_').str[0]
print(train.groupby('Group')['HomePlanet'].nunique().max())
print(train.isnull().sum())
print(test.isnull().sum())
kids = train[train['Age'] <= 12]
print(kids['CryoSleep'].value_counts(normalize=True))

adult_zero = train[(train['Age'] > 12) & (train['TotalSpend'] == 0)]
print(adult_zero['CryoSleep'].value_counts(normalize=True))


before = train['CryoSleep'].isnull()
cleaned = clean(train)
print(cleaned.loc[before, 'CryoSleep'].value_counts(dropna=False))
print(train.isnull().sum())
print(cleaned.isnull().sum())
print(cleaned.columns)
print(train['Age'].mean(), train['Age'].median())
print(train['RoomService'].mean(), train['RoomService'].median())
awake_adults = train[(train['CryoSleep'] == False) & (train['Age'] > 12)]
print(awake_adults['RoomService'].median())

print(train['Num'].diff().abs().describe())



above = train['Cabin'].shift(1)
below = train['Cabin'].shift(-1)
same_neighbors = (above == below)

known = train['Cabin'].notnull()
print((train['Cabin'] == above)[same_neighbors & known].mean())
print((train['Cabin'].isnull() & same_neighbors).sum())

group_cabin = train.groupby('Group')['Cabin'].transform('first')
print((train['Cabin'].isnull() & group_cabin.notnull()).sum())

sizes = train.groupby('Group')['PassengerId'].transform('count')
multi = train[sizes > 1]
print((multi.groupby('Group')['Cabin'].nunique() == 1).mean())

print((multi.groupby('Group')['Deck'].nunique() == 1).mean())
print((multi.groupby('Group')['Side'].nunique() == 1).mean())


cleaned = clean(train)
comparison = pd.DataFrame({
    'before': train.isnull().sum(),
    'after': cleaned.isnull().sum()
})
print(comparison)

train_c = clean(train)
fv = get_fill_values(train_c)
train_c = fill_generic(train_c, fv)
print(train_c.isnull().sum())
test_c = fill_generic(clean(test), fv)
print(test_c.isnull().sum())