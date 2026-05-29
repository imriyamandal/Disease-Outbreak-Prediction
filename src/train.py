import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

from preprocessing import preprocess_data
from features import create_features


features = pd.read_csv(
    "data/dengue_features_train.csv"
)

labels = pd.read_csv(
    "data/dengue_labels_train.csv"
)


df = pd.merge(
    features,
    labels,
    on=['city', 'year', 'weekofyear']
)


df = preprocess_data(df)

df = create_features(df)


threshold = df['total_cases'].quantile(0.75)

df['outbreak'] = (
    df['total_cases'] > threshold
).astype(int)


df.to_csv(
    "data/cleaned_dataset.csv",
    index=False
)


X = df.drop(
    ['total_cases', 'outbreak'],
    axis=1
)

X = X.select_dtypes(include=np.number)

y = df['outbreak']


tscv = TimeSeriesSplit(n_splits=5)


for train_index, test_index in tscv.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


scale = (
    len(y_train[y_train == 0]) /
    len(y_train[y_train == 1])
)


model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss'
)


model.fit(
    X_train,
    y_train
)


pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    pred
)

precision = precision_score(
    y_test,
    pred
)

recall = recall_score(
    y_test,
    pred
)

f1 = f1_score(
    y_test,
    pred
)


print("\nModel Performance\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        pred
    )
)


importance = model.feature_importances_


feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})


feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)


print("\nTop Important Features\n")

print(
    feature_df.head(15)
)


joblib.dump(
    model,
    "models/xgboost_model.pkl"
)


print("\nModel Trained Successfully")

