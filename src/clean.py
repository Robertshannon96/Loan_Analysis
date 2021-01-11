import pandas as pd
from sklearn.model_selection import train_test_split


class CleanPipeline:
    def __init__(self, df):
        self.df = df

    def clean_cols(self):
        self.df = self.df.rename(columns={'not.fully.paid': 'unpaid'})
        cols = self.df.columns.tolist()
        cols = [col.replace('.', '_') for col in cols]
        self.df.columns = cols

    def one_hot_encode(self):
        variable = ['purpose']
        self.df = pd.get_dummies(self.df, columns=variable, drop_first=True)

    def create_holdout(self):
        train, holdout = train_test_split(self.df, test_size=.2, random_state=1)
        train.to_csv('../data/data.csv')
        holdout.to_csv('../data/holdout.csv')


def main():
    df = pd.read_csv('../data/loan_data.csv')
    df_pipe = CleanPipeline(df)
    df_pipe.clean_cols()
    df_pipe.one_hot_encode()
    df_pipe.create_holdout()
    df = df_pipe.df
    print(df)
    print(df.columns)


if __name__ == '__main__':
    main()
