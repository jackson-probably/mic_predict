###Hi my name is jackson and I wrote this code to model MICs using genomic features (kmers)
###If you find it useful please let me know! It took me a long time lmao
###Much of this code was adapted from our collaborator Marcus Nguyen https://github.com/Tinyman392/GenomicModelCreator



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
import shap

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
###IMPORTANT: I label my y variable as "label" but if you use a different name be sure to change that here!!!

X = data.drop(columns=["label"])
y = data['label']

###I like to make sure it looks okay - make sure "label" column is not in X, that's probably the most likely error
X.head()
y.head()
print("\nNumber of samples:", len(X))
print("Number of features:", X.shape[1])


###This is the part where we actually split the data and build the model
###MAKE SURE YOU CHANGE NAMES TO FIT YOUR DATA
###If you use 12mers like I did, this part til the end will take a really long time, even with an HPC. Like overnight probably.


parameters = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'learning_rate': 0.0625,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'max_depth': 4,
    'eval_metric': 'rmse'
}

###THis will actually split the data, you can use however many folds you prefer
###I set random state to my birthday, but literally do whatever you want!
kf = KFold(n_splits=10, shuffle=True, random_state=101294)

###And then this is where we actually establish our empty list variables. I would keep all these, they are all important model metrics.
###That said you can call them whatever you want !!

w1_scores = []
rmse_scores = []
mae_scores = []
r2_scores = []
best_iterations = []
fold_results = []
all_shap_values = []
all_X_test = []

###And now we are off. This for loop will roll through all 10 folds, again pay attention to naming.

for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    print(f"\n{'=' * 60}")
    print(f"Fold {fold}/10")
    print(f"{'=' * 60}")

    X_train_full = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()

    y_train_full = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()

    print("Training samples:", len(X_train_full))
    print("Test samples:", len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        shuffle=True,
        random_state=101294 + fold
    )



    dtrain = xgb.DMatrix(X_train,label=y_train)
    dval = xgb.DMatrix(X_val,label=y_val)
    dtest = xgb.DMatrix(X_test,label=y_test)

    model = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[
            (dtrain, "train"),
            (dtest, "validation")
        ],
        early_stopping_rounds=25,
        verbose_eval=False
    )

    best_iter = model.best_iteration
    best_iterations.append(best_iter)
    print("Best iteration:", best_iter)


    preds = model.predict(dtest,iteration_range=(0, best_iter + 1))
    y_test_array = y_test.to_numpy()
    eps = 1e-8
    within_2fold = np.abs(preds - y_test_array) <= 1
    percentage_2fold = np.mean(within_2fold) * 100

    rmse = np.sqrt(mean_squared_error(y_test_array,preds))
    mae = mean_absolute_error(y_test_array,preds)
    r2 = r2_score(y_test_array,preds)
    w1_scores.append(percentage_2fold)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    fold_results.append({
        "fold": fold,
        "W1_percent": percentage_2fold,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "best_iteration": best_iter
    })
    print(f"W1 (±1): {percentage_2fold:.2f}%")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    explainer = shap.TreeExplainer(model)

    shap_values_fold = explainer.shap_values(X_test)

    all_shap_values.append(shap_values_fold)

    all_X_test.append(X_test)

print("\n")
print("=" * 60)
print("OVERALL 10-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

print(f"W1 (±1): "f"{np.mean(w1_scores):.2f}% "f"± {np.std(w1_scores):.2f}%")

print(f"RMSE: "f"{np.mean(rmse_scores):.4f} "f"± {np.std(rmse_scores):.4f}")

print(f"MAE: "f"{np.mean(mae_scores):.4f} "f"± {np.std(mae_scores):.4f}")

print(f"R²: "f"{np.mean(r2_scores):.4f} "f"± {np.std(r2_scores):.4f}")

print(f"Best iteration: "f"{np.mean(best_iterations):.1f} "f"± {np.std(best_iterations):.1f}")



os.makedirs("shap_results",exist_ok=True)


fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv("shap_results/fold_metrics.csv",index=False)



all_shap_values = np.vstack(all_shap_values)
all_X_test = pd.concat(all_X_test,axis=0)
shap_df = pd.DataFrame(all_shap_values,columns=X.columns,index=all_X_test.index)
mean_shap = (shap_df.mean().sort_values(ascending=False))
mean_abs_shap = (shap_df.abs().mean().sort_values(ascending=False))
top_positive = mean_shap.head(20)
top_negative = mean_shap.tail(20)
top_features = mean_abs_shap.head(20)

###IMPORTANT: If you are using multiple dataframe inputs, be sure to adjust the output folders here!!!!! So that it doesn't overwrite your old data
shap_df.to_csv("shap_results/shap_values.csv",index=True)


all_X_test.to_csv(
    "shap_results/X_test_for_shap.csv",
    index=True
)

shap_importance_df = pd.DataFrame({
    "feature": mean_abs_shap.index,
    "mean_abs_shap": mean_abs_shap.values,
    "mean_shap": mean_shap.loc[
        mean_abs_shap.index
    ].values
})

shap_importance_df.to_csv(
    "shap_results/shap_feature_importance.csv",
    index=False
)



print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print("shap_results/fold_metrics.csv")

print("shap_results/shap_values.csv")

print("shap_results/X_test_for_shap.csv")

print("shap_results/shap_feature_importance.csv")

print("shap_results/shap_summary_plot.png")


print(f"\nOverall W1: "f"{np.mean(w1_scores):.2f}% ± {np.std(w1_scores):.2f}%")
