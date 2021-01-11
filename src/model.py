import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from eda import paid_vs_unpaid
from eda import *
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import RandomizedSearchCV
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50


def split(df):
    """
      Performs a 80/20 train,test split on the DataFrame.

      Input: DataFrame
      Output: X/Y train and test splits. Also returns X and y values.
    """
    train_df, test_df = train_test_split(df, test_size=0.20)
    train_df = under_sampling(train_df)
    X_train = train_df.drop('unpaid', axis=1)
    y_train = train_df['unpaid']
    X_test = test_df.drop('unpaid', axis=1)
    y_test = test_df['unpaid']

    return X_train, X_test, y_train, y_test


def under_sampling(df):
    # under sample X_train and make sure that classes are balanced.
    paid, un_paid = paid_vs_unpaid(df)
    paid_sample = paid.sample(n=len(un_paid['unpaid']))
    return pd.concat([paid_sample, un_paid])


def random_forest(df):
    """
    Applies a random forest model to the DataFrame

    Input: DataFrame
    Output: Accuracy Score, Confusion Matrix
    """
    X_train, X_test, y_train, y_test = split(df)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

    # feature importance

    # Come back to this and figure out the descrepency between notebook and here*********************************************************
    df = clean_cols(df)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    X = df.drop(['Unnamed: 0', 'unpaid'], axis=1)
    X = df.drop('Unnamed: 0', axis=1)
    cols = X.columns.to_numpy()
    col_sort = cols[indices]
    importance_sort = importances[indices]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('Feature Importance', size=20)
    plt.bar(col_sort[:10], importance_sort[:10], edgecolor='black', lw=1.5)
    plt.xticks(rotation=40, size=18)
    plt.yticks(size=18)
    plt.xlabel('Feature', size=19)
    plt.ylabel('Feature Importance', size=19)
    plt.show()
    plt.savefig('importance_hist_pycharm.png')


def grid_search():
    est = RandomForestClassifier(n_jobs=-1)
    grid = {'max_depth': [3, 5, None],
            'max_features': [1, 2, 3, 4, None],
            'min_samples_split': [2, 4, 8],
            'min_samples_leaf': [1, 5, 10, 20],
            'bootstrap': [True, False],
            'n_estimators': [50, 100, 200],
            'random_state': [1]}
    gridsearch = RandomizedSearchCV(est, grid, scoring='precision', n_iter=100, cv=5, verbose=True)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
    print(f'Best Params: {gridsearch.best_params_}')
    print(f'Best F1 Score: {gridsearch.best_score_:.3f}')

    return model


def gradient_boost(df):
    """
    Applies a gradient_boost model to the DataFrame

    Input: DataFrame
    Output: Accuracy Score, Confusion Matrix
    """
    X_train, X_test, y_train, y_test = split(df)
    gb_model = GradientBoostingClassifier(learning_rate=0.2, n_estimators=300, random_state=42,
                                          min_samples_leaf=200, max_depth=3, max_features=3)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    print("Gradient Boost Accuracy:", metrics.accuracy_score(y_test, y_pred_gb))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred_gb))
    print("GB Precision:", precision_score(y_test, y_pred_gb))
    print("GB Recall:", recall_score(y_test, y_pred_gb))

    df = clean_cols(df)
    importances = gb_model.feature_importances_
    df = df.rename(columns={'dti': 'debt to income'})
    indices = np.argsort(importances)[::-1]
    X = df.drop(['Unnamed: 0', 'unpaid'], axis=1)
    X = df.drop('Unnamed: 0', axis=1)
    cols = X.columns.to_numpy()
    col_sort = cols[indices]
    importance_sort = importances[indices]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('Feature Importance', size=20)
    plt.bar(col_sort[:5], importance_sort[:5], edgecolor='black', lw=1.5)
    plt.xticks(rotation=40, size=18)
    plt.yticks(size=18)
    #plt.xlabel('Feature', size=19)
    plt.ylabel('Feature Importance', size=19)
    plt.show()
    plt.savefig('importance_hist_pycharm_gb.png')


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    random_forest(df)
    gradient_boost(df)
    # grid_search(model)
