import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from eda import paid_vs_unpaid


class PrepModel:
    def __init__(self, df):
        self.df = df

    def under_sampling(self):
        # under sample X_train and make sure that classes are balanced.
        paid, un_paid = paid_vs_unpaid(self.df)
        paid_sample = paid.sample(n=len(un_paid['unpaid']))
        return pd.concat([paid_sample, un_paid])

    def split(self):
        """
          Performs a 80/20 train,test split on the DataFrame.

          Input: DataFrame
          Output: X/Y train and test splits. Also returns X and y values.
        """
        train_df, test_df = train_test_split(self.df, test_size=0.20)
        train_df = under_sampling(train_df)
        X_train = train_df.drop('unpaid', axis=1)
        y_train = train_df['unpaid']
        X_test = test_df.drop('unpaid', axis=1)
        y_test = test_df['unpaid']

        return X_train, X_test, y_train, y_test


class RandomForest:
    def __init__(self, df):
        self.df = df
        pass

    def random_forest(self):
        """
        Applies a random forest model to the DataFrame

        Input: DataFrame
        Output: Accuracy Score, Confusion Matrix
        """
        X_train, X_test, y_train, y_test = split(self.df)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("Confusion Matrix:", confusion_matrix(y_test, y_pred))


class GradientBoost:
    def __init__(self):

    def gradient_boost(self):
        """
        Applies a gradient_boost model to the DataFrame

        Input: DataFrame
        Output: Accuracy Score, Confusion Matrix
        """
        pass


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    random_forest(df)
