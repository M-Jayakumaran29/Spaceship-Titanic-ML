import pandas as pd
import numpy as np
from clean import clean, get_fill_values, fill_generic, encode
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=15))
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
print("Train accuracy:", train_acc)
print("Validation accuracy:", val_acc)
print("Iterations actually used:", model[-1].n_iter_)

weights = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
print(weights)
print("Intercept:", model[-1].intercept_[0])

x = model[0].transform(X_val.iloc[[0]])[0]                    # Step 0: scaled feature vector
z = np.dot(model[-1].coef_[0], x) + model[-1].intercept_[0]   # Step 2: score
p = 1 / (1 + np.exp(-z))                                      # Step 3: sigmoid
print("manual:", p)
print("sklearn:", model.predict_proba(X_val.iloc[[0]])[0, 1])

probs = model.predict_proba(X_train)[:, 1]
print("Average predicted p:", probs.mean())
print("Actual transported rate:", y_train.mean())

results = []
all_weights = {}
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
    m.fit(X_train, y_train)
    results.append({
        'C': C,
        'train_acc': accuracy_score(y_train, m.predict(X_train)),
        'val_acc': accuracy_score(y_val, m.predict(X_val)),
    })
    all_weights[C] = m[-1].coef_[0]

print(pd.DataFrame(results))
print(pd.DataFrame(all_weights, index=X.columns).round(3))
for seed in range(5):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(X, y, groups))
    row = {'seed': seed}
    for C in [0.1, 1]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        m.fit(X.iloc[tr], y.iloc[tr])
        row[f'C={C}'] = round(accuracy_score(y.iloc[va], m.predict(X.iloc[va])), 4)
    print(row)

awake = train_c[train_c['CryoSleep'] == False]
print(awake.groupby(awake['FoodCourt'] > 0)['Transported'].agg(['mean', 'count']))
print(awake.groupby(awake['Spa'] > 0)['Transported'].agg(['mean', 'count']))


