#  Included: HK Housing price, HK GDP, HK Inflation, HSI, HK Unemployment,
#  Included: Chinese housing price, Chinese GDP, Chinese Inflation, CSI300,
#  Included: sp500, US interest, USD:RMB
#  Not include: chi-hk, Gold, Chinese Deposit, wage, Chinese interest
import quandl
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import bs4 as bs
from sklearn import svm, preprocessing, cross_validation
from sklearn.linear_model import SGDClassifier

# HSI Volume, HSI
# Sourced from: Yahoo
def hsi_data():
    df = quandl.get("YAHOO/INDEX_HSI", authtoken="zpFWg7jpwtBPmzA8sT2Z")
    df.rename(columns={'Adjusted Close':'HSI', 'Volume':'HSI Volume'}, inplace=True)
    df.drop(['Open', 'High', 'Low', 'Close'], 1, inplace=True)
    return df
# print hsi_data()

def custom_stock(stock):
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock, "yahoo", start, end)
    df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], 1, inplace=True)
    df.rename(columns={'Adj Close':stock}, inplace=True)
    return df
# print custom_stock("8153.HK").head()

# Format all data
def format_data(stock):
    p1 = hsi_data()
    p2 = custom_stock(stock)
    df = p1.join([p2])
    return df
# print format_data("1217.HK")

def create_labels(cur, fut):
    profit_counter=1
    if fut > 0.03:  # if rise 3%
        profit_counter = profit_counter * (fut)
        return 1
    elif fut < -0.03:
        return -1
    else:
        return 0
    print profit_counter


def process(stock):
    df = format_data(stock)
    df[['HSI Volume', 'HSI', stock]] = df[['HSI Volume', 'HSI', stock]].pct_change()

    # shift future value to current date
    df[stock+'_future'] = df[stock].shift(-1)
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['label'] = list(map(create_labels, df[stock], df[stock+'_future']))
    X = np.array(df.drop(['label', stock+'_future'], 1)) # 1 = column
    X = preprocessing.scale(X)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    # print "Number of Data: ", len(df)
    return clf.score(X_test, y_test)

# print process("1217.HK")

def final(stock):
    accuracies = []
    for j in range(10):
        number = process(stock)
        accuracies.append(number)
    print 'Mean Accuracy for 10 tests for ', stock, ' : ', sum(accuracies) / len(accuracies)

final("1217.HK")
