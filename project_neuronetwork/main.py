from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow import keras

import numpy as np
import os
import pandas as pd
import math
import pickle
from portfolio_functions import *

# каталог, где лежат датафреймы с котировками
dataPath = "quotes_data"

# получаем имена файлов с данными о котировках активов
dataFileNames = os.listdir(dataPath)

# загружаем данные в словарь датафреймов
dataDict = {}
for filename in dataFileNames:
    with open(fr"{dataPath}\{filename}", mode="rb") as f:
        dataDict[filename.replace(".bin", "")] = pickle.load(f)

# список тикеров активов
tickers = list(dataDict.keys())

# формируем датафрейм нужной структуры
assets_data_df = dataDict[tickers[0]][["<CLOSE>"]]
assets_data_df.index = dataDict[tickers[0]]["<DATE>"]
assets_data_df.columns = [tickers[0]]
for ticker in tickers[1:]:
    newDF = dataDict[ticker][["<CLOSE>"]]
    newDF.index = dataDict[ticker]["<DATE>"]
    newDF.columns = [ticker]
    assets_data_df = pd.concat([assets_data_df, newDF], axis=1)

# вычисляем месячные доходности активов
assets_returns_df = assets_data_df.copy()
assets_returns_df.iloc[0, :] = math.nan
for i in range(len(tickers)):
    assets_returns_df.iloc[1:, i] = getYields(y=assets_data_df[tickers[i]].tolist())

train_len_m = 120  # длина обучающей выборки в месяцах
test_len_m = 12  # длина тестовой выборки в месяцах
period_len = 12  # длина рассматриваемого периода в месяцах (в данном случае год)

# вычисляем годовые доходности активов по месячным
asset_period_returns_df = DataFrame(columns=assets_returns_df.columns)
for i in range(len(tickers)):
    period_returns = periodReturnsFromMonth(assets_returns_df[tickers[i]].tolist(), period_len)
    period_indices_numbers = list(range(-1, -len(assets_returns_df), -period_len)[::-1])
    period_indices = assets_returns_df.index[period_indices_numbers[-len(period_returns):]]
    for j in range(len(period_indices)):
        asset_period_returns_df.loc[period_indices[j], tickers[i]] = period_returns[j]

# индексы train и test
testInd = assets_returns_df.index[-test_len_m:]
# для каждого актива обучающая выборка имеет разную длину! Поэтому словарь
trainIndDict = dict()
for ticker in tickers:
    trainIndDict[ticker] = assets_returns_df[ticker][[pd.notna(x) for x in assets_returns_df[ticker]]].index
trainInd = assets_returns_df.index[len(assets_returns_df) - train_len_m - test_len_m:-test_len_m]

# годовые индексы train и test
testIndPeriod = asset_period_returns_df.index[-(test_len_m // period_len):]
trainIndPeriodDict = dict()
for ticker in tickers:
    trainIndPeriodDict[ticker] = asset_period_returns_df.index[len(asset_period_returns_df)\
        - (len(trainIndDict[ticker]) - test_len_m) // period_len - test_len_m // period_len:-test_len_m\
        // period_len]
trainIndPeriod = asset_period_returns_df.index[len(asset_period_returns_df) - train_len_m // period_len
                                               - test_len_m // period_len:-test_len_m // period_len]

# датафрейм с прогнозными данными
df_pred_allcounts = DataFrame(columns=["return", "risk", "sharp", "w1", "w2", "w3", "w4", "maxRisk"])

# теперь будем прогнозировать
# тут будут сидеть предсказанные месячные значения
assets_returns_pred_df = DataFrame(columns=tickers)

# прогнозируем доходности каждого актива
for ticker in tickers:
    # обучающая выборка (месячные данные)
    train = list(assets_returns_df.loc[trainIndDict[ticker], ticker])
    # тестовая выборка (месячные данные)
    test = list(assets_returns_df.loc[testInd, ticker])

    # ************ НЕЙРОСЕТЬ ************
    # оставляем доходность и доходность через test_len_m месяцев
    data_set = pd.concat([pd.Series(train[:-test_len_m]), pd.Series(train[test_len_m:])], axis=1).values
    X = []
    y = []
    history_points = 12  # кол-во прошлых месяцев, по которым осуществляется прогноз

    for i in range(history_points, data_set.shape[0]):
        X.append(data_set[i - history_points:i, 0])
        y.append(data_set[i, 1])
    X = np.array(X)
    y = np.array(y)
    # print(X.shape)
    # print(y.shape)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    # print(X.shape)

    X_train = X[:-test_len_m]
    y_train = y[test_len_m:]
    X_test = X[-test_len_m:]

    try:
        regressor = Sequential()
        regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.3))
        regressor.add(LSTM(units=100, return_sequences=True))
        regressor.add(Dropout(0.3))
        regressor.add(LSTM(units=100, return_sequences=True))
        regressor.add(Dropout(0.3))
        regressor.add(LSTM(units=100))
        regressor.add(Dropout(0.3))
        regressor.add(Dense(units=1))
        adam = keras.optimizers.Adam(learning_rate=0.0001)
        regressor.compile(optimizer=adam, loss='mean_squared_error')
        regressor.fit(X_train, y_train, epochs=100, batch_size=32)

        predicted = regressor.predict(X_test)
    except:
        raise Exception("Таки при обучении нейросети и прогнозировании что-то произошло нехорошее")
    # ************ НЕЙРОСЕТЬ. КОНЕЦ ************

    # засовываем прогноз по активу в датафрейм
    assets_returns_pred_df[ticker] = predicted.flatten()
# end for

assets_returns_pred_df.index = testInd  # прописываем корректные индексы в датафрейме прогнозов

# получим предсказанные годовые значения
asset_period_returns_pred_df = DataFrame(columns=assets_returns_pred_df.columns)
for i in range(len(tickers)):
    period_returns = periodReturnsFromMonth(assets_returns_pred_df[tickers[i]].tolist(), period_len)
    period_indices_numbers = list(range(-1, -len(assets_returns_pred_df), -period_len)[::-1])
    period_indices = assets_returns_pred_df.index[period_indices_numbers[-len(period_returns):]]
    for j in range(len(period_indices)):
        asset_period_returns_pred_df.loc[period_indices[j], tickers[i]] = period_returns[j]

portfolio_size = len(tickers)
best_tickers = tickers[:portfolio_size]
# trainIndPeriod определяем по самому короткому из n активов
# (хотя в нашем случае они все должны быть одной длины)
trainIndPeriodTrue = trainIndPeriod
for ti in tickers[:portfolio_size]:
    if len(trainIndPeriodDict[ti][:test_len_m]) < len(trainIndPeriodTrue):
        trainIndPeriodTrue = trainIndPeriodDict[ti][:test_len_m]

# переходим к формированию портфелей
# максимальный риск определяем как риск портфеля из активов в равных долях
maxRisk = portfolioStanDev(weights=[1 / portfolio_size] * portfolio_size,
                           df=asset_period_returns_df.loc[trainIndPeriodTrue, best_tickers])
# оптимизация портфеля по доходности с ограничением по риску
weights = portfolioOptimize(dfYield=asset_period_returns_pred_df.loc[testIndPeriod, best_tickers],
                            targFun=portfolioYield, bounds=[(0.00, 1)] * portfolio_size, invert=True,
                            maxRisk=maxRisk,
                            dfR=pd.concat([asset_period_returns_df.loc[trainIndPeriodTrue, best_tickers],
                                        asset_period_returns_pred_df.loc[
                                            testIndPeriod, best_tickers]], axis=0))
# вычисляем параметры портфеля при полученных весах активов
return1, risk, sharp = portfolioParameters(df=asset_period_returns_df.loc[:, best_tickers],
                                           train_ind=trainIndPeriodTrue,
                                           test_ind=testIndPeriod, w=weights)
# засовываем результаты в датафрейм
df_pred_allcounts.loc["-".join(tickers)] = [return1, risk, sharp, weights[0], weights[1],
                                            weights[2] if len(weights) >= 3 else None,
                                            weights[3] if len(weights) >= 4 else None,
                                            maxRisk]
# выводим на экран
print(df_pred_allcounts.to_string())
