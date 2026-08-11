####Load whatver you need


conda activate abr_genomes
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import os

###I usually verify the directory, but it isnt strictly necessary ! Change as needed

current_directory = os.getcwd()
print(current_directory)
os.chdir('OneDrive - University of Illinois Chicago/Desktop/lab_materials/WSL')

###upload data and verify
###optionally, you can change these to represent single antibiotic tables, up to u
###You will nead to read a feather or database if you used those instead of csvs
###This step will take a really long time if you're using like, 10mers or above

data = pd.read_csv('kmc_outputs/km_final.csv')
print(data.head())
data = data.drop('genome', axis = 1)

###split data for training
X = df.iloc[:, :-1]
X = X.drop('label', axis = 1)
y = data['label']
###I like to make sure it looks okay - make sure "label" column is not in X, that's probably the most likely error
X.head()
y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

parameters = {
    'objective': 'reg:squarederror',  
    'booster': 'gbtree',              
    'learning_rate': 0.0625,          
    'subsample': 0.75,                
    'colsample_bytree': 0.75,         
    'max_depth': 4,                   
    'eval_metric': 'rmse'             
}


evals = [(dtrain, 'train'), (dtest, 'test')]
kf = KFold(n_splits=10, shuffle=True)


#####################w1 scores

antibiotic_cols = ["Ceftazidime_Avibactam", "Meropenem_Vaborbactam", "Imipenem_Relebactam", "Meropenem"]
w1_scores = []
w1_per_abx = {col: [] for col in antibiotic_cols}

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=25,
        verbose_eval=False
    )

    best_iter = model.best_iteration if model.best_iteration is not None else 999
    preds = model.predict(dtest, iteration_range=(0, best_iter + 1))

    y_test_a = y_test.values
    eps = 1e-8
    
    within_factor = (
        (abs(preds) >= (abs(y_test_a) + eps)/2) &
        (abs(preds) <= (abs(y_test_a) + eps)*2)
    )
    percentage_ea = np.mean(within_factor) * 100
    w1_scores.append(percentage_ea)

    
    for col in antibiotic_cols:
        mask = X_test[col].values == 1

        if np.sum(mask) == 0:
            continue  # skip if no samples in this fold

        preds_sub = preds[mask]
        y_sub = y_test_a[mask]

        within_factor_sub = (
            (abs(preds_sub) >= (abs(y_sub) + eps)/2) &
            (abs(preds_sub) <= (abs(y_sub) + eps)*2)
        w1_per_abx[col].append(w1_abx)


print(f"Overall W1: {np.mean(w1_scores):.2f}% ± {np.std(w1_scores):.2f}")

for col in antibiotic_cols
    scores = w1_per_abx[col]
    if len(scores) > 0:
        print(f"{col}: {np.mean(scores):.2f}% ± {np.std(scores):.2f}")
        )

        w1_abx = np.mean(within_factor_sub) * 100
