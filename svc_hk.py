import quandl
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import bs4 as bs
from sklearn import svm, preprocessing, cross_validation
from sklearn.linear_model import SGDClassifier
from sklearn import kernel_approximation
from sklearn.naive_bayes import GaussianNB
import requests
import pickle
import os

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
    #elif fut < -0.03:
    #    return -1
    else:
        return 0
    print profit_counter

def predict(stock):
    action_dict = {1:"Buy", -1:"Sell", 0:"Hold"}
    df = format_data(stock)
    df[stock+'_90ma'] = df[stock].rolling(window=90).mean()
    df['HSI_90ma'] = df['HSI'].rolling(window=90).mean()
    df = df.pct_change()

    # shift future value to current date
    df[stock+'_future'] = df[stock].shift(-1)

    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df['label'] = list(map(create_labels, df[stock], df[stock+'_future']))

    X = np.array(df.drop(['label', stock+'_future'], 1)) # 1 = column
    X = preprocessing.scale(X)
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    if os.path.exists("my_dumped_classifier.pkl"):
        with open('my_dumped_classifier.pkl', 'rb') as fid:
            clf = pickle.load(fid)
    else:
        clf = svm.LinearSVC()

    clf.fit(X_train, y_train)

    with open('my_dumped_classifier.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    #current_hsi = hsi_data().tail(1)
    #current_stock = custom_stock(stock).tail(1)
    #df2 = current_hsi.join(current_stock)
    accuracy = clf.score(X_test, y_test)
    #prediction = clf.predict(df2)[0]
    return accuracy #, prediction
    #return "Accuracy for {}: {}%, Recommend action for {}: {}".format(stock, int(accuracy*100), stock, action_dict[prediction])

# print predict("1217.HK")
# print predict("8153.HK")

def save_HSI_tickers():
    resp = requests.get('http://www.aastocks.com/tc/stocks/market/index/hk-index-con.aspx')
    soup = bs.BeautifulSoup(resp.text, "html5lib")
    table = soup.find('table', {'class':'tblM s2'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('a')[0].text
        tickers.append(str(ticker))

    #with open("sp500tickers.pickle", "wb") as f:
    #    pickle.dump(tickers, f)

    # print tickers
    return tickers

# save_HSI_tickers()

def test_all():
    tickers = save_HSI_tickers()
    accuracies = []
    for count, ticker in enumerate(tickers):
        accuracy = predict(ticker[1:])
        print "Trained {} stock with accuracy {}".format(count+1, accuracy)
        print "\n"
    #print "Total accuracy: {}".format(sum(accuracies)/len(accuracies))

test_all()
