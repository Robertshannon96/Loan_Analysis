import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def clean_cols(df):
    """
    This is specifically for graphing purposes only. 
    This function cleans the underscore out of the columns
    to make graphs cleaner. 
    """
    cols = df.columns.tolist()
    cols = [col.replace('_', ' ') for col in cols]
    df.columns = cols
    
    return df


def paid_vs_unpaid(df):
    paid_customers = df[df['unpaid'] == 0]
    un_paid_customers = df[df['unpaid'] == 1]

    return paid_customers, un_paid_customers


def make_histogram(paid_customers, un_paid_customers, title, variable):
    fig, ax = plt.subplots(2, figsize=(12, 8))
    fig.suptitle(title)
    ax[0].hist(paid_customers[variable], edgecolor='black', lw=1.5)
    ax[0].set_title('Paid Customers')
    ax[1].hist(un_paid_customers[variable], edgecolor='black', lw=1.5)
    ax[1].set_title('Un Paid Customers')
    plt.savefig(f'../images/{title}.png')


def count_plot(df):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='purpose', hue='unpaid', data=df, edgecolor='black', lw=1.5)
    plt.title('What account is most likely to not pay back the loan?')
    plt.savefig(f'../images/count_plot.png')


def layered_histogram(un_paid_customers):
    plt.figure(figsize=(10, 6))
    un_paid_customers[un_paid_customers['credit_policy'] ==
                      1]['fico'].hist(alpha=0.5, bins=25, label='Credit Policy= 1')
    un_paid_customers[un_paid_customers['credit_policy'] ==
                      0]['fico'].hist(alpha=0.5, bins=25, label='Credit Policy= 0')
    plt.legend()
    plt.title('Fico scores in different Credit Policies')
    plt.xlabel('Fico score')
    plt.ylabel('Count')
    plt.savefig('../images/layered_histogram.png')


if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    clean_cols(df)
    paid_customers, un_paid_customers = paid_vs_unpaid(df)
    make_histogram(paid_customers=paid_customers, un_paid_customers=un_paid_customers,
                   title='Interest rates in paid vs unpaid loan customers',
                   variable='int_rate')
    make_histogram(paid_customers=paid_customers, un_paid_customers=un_paid_customers,
                   title='Fico scores in paid vs unpaid loan customers',
                   variable='fico')
    # count_plot(df)
    layered_histogram(un_paid_customers)

