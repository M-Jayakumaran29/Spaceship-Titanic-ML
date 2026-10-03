import pandas as pd
from clean import clean, get_fill_values, fill_generic, encode
from sklearn.model_selection import GroupShuffleSplit


train = pd.read_csv('train.csv')
train_c = clean(train)
fv = get_fill_values(train_c)
train_f = encode(fill_generic(train_c, fv))

X = train_f.drop(columns=['PassengerId', 'Transported'])
y = train_f['Transported']
print(X.shape, y.shape)

groups = train_c['Group']
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
print(X_train.shape, X_val.shape)
print(len(set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])))
print(y_val.mean())
