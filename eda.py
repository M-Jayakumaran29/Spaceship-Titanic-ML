# eda.py - Exploratory Data Analysis for Spaceship Titanic
# Each check is followed by its result and the decision it led to.
# Rule numbers refer to the fill rules in clean.py.

import pandas as pd
from clean import clean, get_fill_values, fill_generic, encode, spend_cols

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------------------------------------------------------
section("1. Overview")
# ---------------------------------------------------------------
print("train / test shape:", train.shape, test.shape)
print(train.head())
train.info()

print("Fraction of rows with any missing value:", train.isnull().any(axis=1).mean())
# -> 24% of passengers have at least one gap. Dropping rows is not an option
#    (and test rows can't be dropped at all).

print("Overall transported rate:", train['Transported'].mean())
# -> 0.504: balanced target. Baseline "always predict True" = 50.4%.


# ---------------------------------------------------------------
# Helper columns for exploration only.
# (clean.py builds its own versions; these are just for EDA.)
# ---------------------------------------------------------------
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
train['Num'] = pd.to_numeric(train['Num'])
train['TotalSpend'] = train[spend_cols].sum(axis=1)   # note: sum() treats missing as 0
train['Group'] = train['PassengerId'].str.split('_').str[0]


# ---------------------------------------------------------------
section("2. Cabin -> Deck / Num / Side")
# ---------------------------------------------------------------
print("Unique cabins:", train['Cabin'].nunique())
# -> 6560 cabins for ~8500 people: too specific to generalize (overfitting risk).
#    Use the parts (Deck, Num, Side) instead.

print("Num range:", train['Num'].min(), train['Num'].max())

print("Transported by Deck:\n", train.groupby('Deck')['Transported'].agg(['mean', 'count']))
# -> B 73%, C 68%, E 36%. Deck T has only 5 people: not trusted.

print("Transported by Side:\n", train.groupby('Side')['Transported'].mean())
# -> P 45% vs S 56%. A real signal: ~4000 people per side, so a 10-point gap isn't chance.


# ---------------------------------------------------------------
section("3. CryoSleep and its interactions")
# ---------------------------------------------------------------
print("Transported by CryoSleep:\n", train.groupby('CryoSleep')['Transported'].agg(['mean', 'count']))
# -> 82% asleep vs 33% awake. Strongest single signal.

print("Deck x CryoSleep:\n", train.groupby(['Deck', 'CryoSleep'])['Transported'].agg(['mean', 'count']))
# -> Sleepers 94-99% on most decks, ~65% on E and G. Awake 28-46%.
#    Deck matters much more for sleepers (interaction).

print("HomePlanet x Deck (counts):\n", pd.crosstab(train['HomePlanet'], train['Deck']))
# -> Decks A, B, C, T are 100% Europa. Deck G is 100% Earth. -> Rule 5

print("HomePlanet x CryoSleep:\n", train.groupby(['HomePlanet', 'CryoSleep'])['Transported'].agg(['mean', 'count']))
# -> Sleepers: Earth 66%, Europa 99%, Mars 91%.

earth_sleepers = train[(train['HomePlanet'] == 'Earth') & (train['CryoSleep'] == True)]
print("Earth sleepers by Deck:\n", earth_sleepers.groupby('Deck')['Transported'].agg(['mean', 'count']))
# -> 1322 of ~1346 Earth sleepers are on deck G.
#    Deck G and Earth effects can't be separated.

e_sleepers = train[(train['Deck'] == 'E') & (train['CryoSleep'] == True)]
print("Deck E sleepers by HomePlanet:\n", e_sleepers.groupby('HomePlanet')['Transported'].agg(['mean', 'count']))
# -> Mars 46% (n=99) vs 91% for Mars sleepers overall. Europa unaffected (98%).
#    Deck E matters specifically for Martian sleepers (3-way interaction).
#    Note: AI's earlier estimate said Mars AND Europa were lower; this check corrected it.


# ---------------------------------------------------------------
section("4. Spending, CryoSleep and Age rules")
# ---------------------------------------------------------------
sleepers = train[train['CryoSleep'] == True]
print("Max spending among sleepers:\n", sleepers[spend_cols].max())
# -> All 0. -> Rule 1 (asleep -> spending 0) and Rule 2 (spent anything -> awake)

zero_spenders = train[train['TotalSpend'] == 0]
print("CryoSleep among zero-spenders:\n", zero_spenders['CryoSleep'].value_counts(normalize=True))
# -> 85% asleep. (My guess was 95%.)

awake_zero = train[(train['CryoSleep'] == False) & (train['TotalSpend'] == 0)]
print("Age of awake zero-spenders:\n", awake_zero['Age'].describe())
# -> 511 people, median age 7, 75% are 12 or younger: mostly kids.

# Rejected: my first cutoff guess of 17
print("Max spend, age <= 17:", train[train['Age'] <= 17]['TotalSpend'].max())
# -> 24217: teens do spend, so 17 is wrong.

print("Max spend by age (0-20):\n", train[train['Age'] <= 20].groupby('Age')['TotalSpend'].max())
# -> Ages 0-12 always 0, jumps at 13. -> Rule 4 (age <= 12 -> spending 0)

kids = train[train['Age'] <= 12]
print("CryoSleep among kids:\n", kids['CryoSleep'].value_counts(normalize=True))
# -> ~52/48: a kid's zero spending says nothing about sleep.

adult_zero = train[(train['Age'] > 12) & (train['TotalSpend'] == 0)]
print("CryoSleep among adult zero-spenders:\n", adult_zero['CryoSleep'].value_counts(normalize=True))
# -> 96% asleep. -> Rule 3 (adult zero-spender -> asleep). Up from 85% by excluding kids.


# ---------------------------------------------------------------
section("5. HomePlanet, Destination, VIP")
# ---------------------------------------------------------------
for col in ['HomePlanet', 'Destination', 'VIP']:
    print(f"Transported by {col}:\n", train.groupby(col)['Transported'].agg(['mean', 'count']))
# -> HomePlanet: Earth 42%, Europa 66%, Mars 52% (biggest gap).
#    Destination: 47-61%.
#    VIP: 51% vs 38%, but only 199 VIPs (2%) -> VIP dropped.


# ---------------------------------------------------------------
section("6. Groups")
# ---------------------------------------------------------------
print("Max HomePlanets in one group:", train.groupby('Group')['HomePlanet'].nunique().max())
# -> 1: groups never mix planets. -> Rule 6

train['GroupSize'] = train.groupby('Group')['PassengerId'].transform('count')
multi = train[train['GroupSize'] > 1]
print("Multi-person groups sharing one Cabin:", (multi.groupby('Group')['Cabin'].nunique() == 1).mean())
print("... sharing one Deck:", (multi.groupby('Group')['Deck'].nunique() == 1).mean())
print("... sharing one Side:", (multi.groupby('Group')['Side'].nunique() == 1).mean())
# -> Cabin 70%, Deck 70% (identical: split groups always change deck), Side 100%.

group_cabin = train.groupby('Group')['Cabin'].transform('first')
print("Missing cabins fillable from a groupmate:", (train['Cabin'].isnull() & group_cabin.notnull()).sum())
# -> 100 rows. -> Rule 7 (fill Cabin from a groupmate)

print("Transported by GroupSize:\n", train.groupby('GroupSize')['Transported'].agg(['mean', 'count']))
# -> Solo 45%, groups 54-64%. Size 8 is only 13 groups, so its 39% is noisy.
#    -> GroupSize kept as a feature (raw Group number dropped: see section 9).


# ---------------------------------------------------------------
section("7. Rejected Cabin-filling ideas")
# ---------------------------------------------------------------
# Idea: fill Num as the average of the rows above and below
print("Num difference between neighboring rows:\n", train['Num'].diff().abs().describe())
# -> Median 91.5: neighbors are usually far apart. Rejected.

# Idea: fill Cabin when the rows above and below share the same cabin
above = train['Cabin'].shift(1)
below = train['Cabin'].shift(-1)
same_neighbors = (above == below)
known = train['Cabin'].notnull()
print("Neighbor-rule accuracy:", (train['Cabin'] == above)[same_neighbors & known].mean())
print("Neighbor-rule rows filled:", (train['Cabin'].isnull() & same_neighbors).sum())
# -> 90.6% accurate but fills only 29 rows. Replaced by the groupmate rule (Rule 7).


# ---------------------------------------------------------------
section("8. Choosing generic fill values")
# ---------------------------------------------------------------
print("Age mean / median:", train['Age'].mean(), train['Age'].median())
# -> 28.8 vs 27: close, either works. Chose mean.

print("RoomService mean / median:", train['RoomService'].mean(), train['RoomService'].median())
# -> 225 vs 0: heavily skewed, so use the median.

awake_adults = train[(train['CryoSleep'] == False) & (train['Age'] > 12)]
print("RoomService median, awake adults:", awake_adults['RoomService'].median())
# -> 9, not 0. Fill spending from the group that actually has the gaps.


# ---------------------------------------------------------------
section("9. Pipeline checks (on fresh, unmodified data)")
# ---------------------------------------------------------------
# Reload so clean() is tested on raw data, not on the EDA-modified 'train'.
raw_train = pd.read_csv('train.csv')
raw_test = pd.read_csv('test.csv')

train_c = clean(raw_train)
before = raw_train['CryoSleep'].isnull()
print("Originally-missing CryoSleep after rules:\n", train_c.loc[before, 'CryoSleep'].value_counts(dropna=False))
# -> 119 False, 70 True (37%, matches ~35% base rate), 28 left for generic fill.

print("Missing before vs after rules:\n", pd.DataFrame({
    'before': raw_train.isnull().sum(),
    'after': train_c.isnull().sum(),
}))

fv = get_fill_values(train_c)                      # fill values come from train only
train_c = fill_generic(train_c, fv)
test_c = fill_generic(clean(raw_test), fv)
print("Missing after all fills (train):\n", train_c.isnull().sum())
print("Missing after all fills (test):\n", test_c.isnull().sum())
# -> 0 everywhere except raw Cabin (dropped in encode).

print("Groups shared between train and test:", len(set(train_c['Group']) & set(test_c['Group'])))
# -> 0: raw Group number can't generalize. Dropped; GroupSize kept instead.

train_f = encode(train_c)
print("Column types after encoding:\n", train_f.dtypes)
print("Missing after encoding:\n", train_f.isnull().sum())
# -> All numeric/bool except PassengerId. No gaps (a .map() mismatch would show up here).