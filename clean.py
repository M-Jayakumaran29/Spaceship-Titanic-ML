import pandas as pd

spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def clean(df):
    df = df.copy()   # work on a copy so the original stays untouched
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')

    #rule-7
    group_cabin = df.groupby('Group')['Cabin'].transform('first')
    df['Cabin'] = df['Cabin'].fillna(group_cabin)
 
 
    # Helper columns
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'])
    
    

    # Rule 5: Deck A/B/C/T -> Europa, Deck G -> Earth
    m=(df["Deck"].isin(['A','B','C','T'])) & df['HomePlanet'].isnull()
    df.loc[m,'HomePlanet']="Europa"
    f=(df['Deck']=="G") & df['HomePlanet'].isnull()
    df.loc[f,'HomePlanet']='Earth'


    # Rule 6: fill HomePlanet from a groupmate
    group_planet = df.groupby('Group')['HomePlanet'].transform('first')
    df['HomePlanet'] = df['HomePlanet'].fillna(group_planet)

    # Rule 4: Age <= 12 -> spending = 0
    mask=(df['Age']<=12) 
    df.loc[mask, spend_cols]=df.loc[mask,spend_cols].fillna(0)

    # Rule 2: spent anything -> CryoSleep = False
    df['TS'] = df[spend_cols].sum(axis=1) 
    mask= (df['TS']>0) & df['CryoSleep'].isnull()
    df.loc[mask,'CryoSleep']=False

    # Rule 3 (strict): Age > 12 and all five spending known and zero -> CryoSleep = True
    all_zero = (df[spend_cols] == 0).all(axis=1)
    mask = df['CryoSleep'].isnull() & (df['Age'] > 12) & all_zero
    df.loc[mask, 'CryoSleep'] = True

    # Rule 1: CryoSleep = True -> spending = 0
    mask = df['CryoSleep'] == True
    df.loc[mask, spend_cols] = df.loc[mask, spend_cols].fillna(0)

    df['TotalSpend'] = df[spend_cols].sum(axis=1)   # recompute after filling
    return df


def get_fill_values(train_df):
    awake_adults = train_df[(train_df['CryoSleep'] == False) & (train_df['Age'] > 12)]
    return {
        'Age': train_df['Age'].mean(),
        'Destination': train_df['Destination'].mode()[0],
        'spend': awake_adults[spend_cols].median(),
    }

def fill_generic(df,fv):
    df=df.copy()

    df['HomePlanet']=df['HomePlanet'].fillna('Unknown')
    df['CryoSleep']=df['CryoSleep'].astype('boolean').fillna(False).astype(bool)

    df['Deck']=df['Deck'].fillna('Unknown')
    df['Num']=df['Num'].fillna(-1)
    df['Side']=df['Side'].fillna(-1)

    df['Age']=df['Age'].fillna(fv['Age'])

    df['Destination']=df['Destination'].fillna('TRAPPIST-1e')
    df[spend_cols]=df[spend_cols].fillna(fv['spend'])
    df=df.drop(columns=['VIP','Name','TS'])
    return df

def encode(df):
    df=df.copy()
    planet_map = {'Earth': 0, 'Europa': 1, 'Mars': 2, 'Unknown': 3}
    df['HomePlanet'] = df['HomePlanet'].map(planet_map)

    destination={'TRAPPIST-1e':0,'PSO J318.5-22':1,'55 Cancri e':2}
    df['Destination']=df['Destination'].map(destination)

    deck={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7,'Unknown':8}
    df['Deck']=df['Deck'].map(deck)

    side={'P':0,'S':1,-1:2}
    df['Side']=df['Side'].map(side)

    df=df.drop(columns=['Group','Cabin'])


    return df
