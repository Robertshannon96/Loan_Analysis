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


def profit_curve():
    """
    Function that creates profit curve for both models used.

    (Important to note that this function was copied over from a messy juypter notebook
    and does not currently run on this file.)
    """
    cost_TP = 8000; cost_FP = -2000; cost_FN = 0; cost_TN = 0
    cb_matrix = np.array([[cost_TN, cost_FP], [cost_FN, cost_TP]])

    profits_gb= []
    profits_rf= []
    thresholds = []
    cfs = []
    cfs2 = []
    count = 0
    models = [model, gb_model]
    for threshold in np.arange(0.0,1.0,0.01):
        thresholds.append(threshold)
        for model_type in models:
            count += 1
            predicted_proba = model_type.predict_proba(X_test2)
            predicted = (predicted_proba [:,1] >= threshold).astype('int')
            accuracy = metrics.accuracy_score(y_test2, predicted)
            #recall_score_1 = recall_score(y_test, y_pred)
            if count % 2 ==1 :
                cf1 = confusion_matrix(y_test, predicted)
                cf12 = (cf1[0,1] + cf1[1,1]) / cf1.sum()
                cfs2.append(cf12)
                full_matrix_rf = (confusion_matrix(y_test, predicted) * cb_matrix)
                profits_rf.append(np.sum(full_matrix_rf))
                #print('Cost matrix sum for RF:', np.sum(full_matrix_rf), 'Threshold:', threshold)
                #print(f'Accuracy score for RF: {accuracy}')
                #print("Recall:", recall_score1)
                
            else:
                cf = confusion_matrix(y_test2, predicted)
                cf2 = (cf[0,1] + cf[1,1]) / cf.sum()
                cfs.append(cf2)
                full_matrix_gb = confusion_matrix(y_test2, predicted) * cb_matrix
                profits_gb.append(np.sum(full_matrix_gb))
                #print(f'Accuracy score for GB: {accuracy}')
                #print('Cost matrix sum for GB:', np.sum(full_matrix_gb),'Threshold:', threshold)

    _, ax = plt.subplots(figsize=(12,8))
    ax.plot(cfs2, profits_rf, label='Random Forest', c='r')
    ax.plot(cfs, profits_gb, label='Gradient Boost', c='b')
    plt.title('Profit Curve', size=26)
    #ax.yaxis.zoom(3) 
    plt.xlabel('Loan Acceptance Rate', size=26)
    plt.ylabel('Profit', size=26)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.legend()
    plt.savefig('../images/profit_curve_for_git.png')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    random_forest(df)
    gradient_boost(df)
    # grid_search(model)
    # profit_curve
