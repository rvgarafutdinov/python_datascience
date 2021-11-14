from scipy.optimize import minimize
from statistics import mean, stdev
from scipy.stats import pearsonr
from math import sqrt
from pandas import DataFrame


# функция округления по российским правилам (честно позаимствовано из Сети)
def roundRus(x, y=0):
    neg = 1 if x >= 0 else -1
    x = abs(x)
    ''' A classical mathematical rounding by Voznica '''
    m = int('1'+'0'*y)  # multiplier - how many positions to the right
    q = x*m # shift to the right by multiplier
    c = int(q) # new number
    i = int( (q-c)*10 ) # indicator number on the right
    if i >= 5:
        c += 1
    return neg * c/m
# конец функции roundRus()

# вычисление стандартного отклонения портфеля
def portfolioStanDev(weights, df, pInvert=1):
    # df - датафрейм с доходностями активов
    # weights - список с весами активов
    from itertools import combinations
    res = 0
    for i in range(len(weights)):
        res += weights[i] ** 2 * stdev(list(df.iloc[:, i].values)) ** 2
    for i, j in list(combinations(range(len(weights)), r=2)):
        res += 2 * weights[i] * weights[j] * pearsonr(list(df.iloc[:, i].values), list(df.iloc[:, j].values))[0] * \
               stdev(list(df.iloc[:, i].values)) * stdev(list(df.iloc[:, j].values))
    return pInvert * sqrt(res)
# конец функции portfolioStanDev()


# вычисление доходности портфеля
def portfolioYield(weights, df, pInvert=1):
    # df - датафрейм с доходностями активов
    # weights - список с весами активов
    res = 0
    for i in range(len(weights)):
        res += mean(list(df.iloc[:, i].values)) * weights[i]
    return pInvert * res
# конец функции portfolioYield()

# вычисление коэффициента Шарпа (доходность/риск) портфеля
# доходность и риск портфеля оцениваются по разным датафреймам!
def portfolioSharpRatio(weights, dfYield, dfRisk, pInvert=1):
    # df - датафрейм с доходностями активов
    # weights - список с весами активов
    return pInvert * portfolioYield(weights=weights, df=dfYield) / portfolioStanDev(weights=weights, df=dfRisk)
# конец функции portfolioSharpRatio()


# вычислим доходности инструмента по месяцам за заданный период
def periodReturnsFromMonth(month_returns, period_len):
    yearys = []
    for i in range(-period_len, -len(month_returns) - 1, -period_len):
        cur_month_returns = month_returns[i:i + period_len] if i + period_len != 0 else month_returns[i:]
        yeary = 1
        for cur_return in cur_month_returns:
            yeary *= (1 + cur_return)
        yeary -= 1
        yearys = [yeary] + yearys
    return yearys
# конец функции yearReturnsFromMonth()

# оптимизация структуры портфеля
# targFun - целевая функция
# bounds - список кортежей с ограничениями на доли активов вида [(0.1, 1), (0.1, 1), ...]
# dfYield, dfRisk - дело в том, что доходность и риск оцениваются по разным датафреймам
# Это становится важно при максимизации коэф. Шарпа как целевой функции. Поэтому эти оба датафрейма могут быть переданы
# minReturn - минимально допустимая доходность
# maxRisk - максимально допустимый риск
def portfolioOptimize(targFun, bounds: list, dfYield=None, dfRisk=None, invert=True, maxRisk=None, dfR=None):
    if dfYield is not None and dfRisk is None:
        args = dfYield, -1 if invert else 1
        if dfR is None:
            dfR = dfYield
        assets_count = len(dfYield.columns)  # число активов в портфеле
    elif dfYield is None and dfRisk is not None:
        args = dfRisk, -1 if invert else 1
        if dfR is None:
            dfR = dfRisk
        assets_count = len(dfRisk.columns)  # число активов в портфеле
    elif dfYield is not None and dfRisk is not None:
        args = dfYield, dfRisk, -1 if invert else 1
        if dfR is None:
            dfR = dfRisk
        assets_count = len(dfYield.columns)  # число активов в портфеле
    weights = [0] * assets_count
    if maxRisk is not None:  # если задано ограничение по риску
        result = minimize(fun=targFun,
                          x0=weights,
                          args=args,
                          constraints=[{"type": "eq", "fun": lambda x: sum(x) - 1},  # функция равна 0
                                       {"type": "ineq", "fun": lambda x: -portfolioStanDev(df=dfR, weights=x) + maxRisk}],  # функция >= 0
                          bounds=bounds,
                          options={'eps': 1e-10})
    else:
        result = minimize(fun=targFun,
                          x0=weights,
                          args=args,
                          constraints=[{"type": "eq", "fun": lambda x: sum(x) - 1}],
                          bounds=bounds,
                          options={'eps': 1e-10})
    return result.x  # список долей активов
# конец функции portfolioOptimize()

# вычисление характеристик портфеля: доходность, риск, Шарп
# df - датафрейм с входными данными
# train_ind - индексы обучающей выборки
# test_ind - индексы тестовой выборки
# w - список долей активов
def portfolioParameters(df: DataFrame, train_ind, test_ind, w, verbose=False):
    stdevOpt = portfolioStanDev(df=df.loc[train_ind.union(test_ind), ], weights=w)
    yieldOpt = portfolioYield(df=df.loc[test_ind, ], weights=w)
    if verbose:
        print(f"Доходность, риск, Шарп: {roundRus(yieldOpt, 3)}\t{roundRus(stdevOpt, 3)}\t{roundRus(yieldOpt / stdevOpt, 3)}"
              f"\t{df.columns.tolist()} {[roundRus(x, 2) for x in w]}")
    return yieldOpt, stdevOpt, yieldOpt / stdevOpt
# конец функции portfolioParameters()


def getYields(y, price0=0, backToPrices=False):
    """
    Функция, вычисляющая доходности и выполняющая обратное преобразование доходностей в цены

    :param y: исходный ряд, который надо преобразовать
    :param price0: значение цены, предшествующее ряду доходностей (при обратном преобразовании)
    :param backToPrices: флаг обратного преобразования
    :return: результат (доходности или цены)
    """
    # доходности
    if not backToPrices:
        y_price = y.copy()
        y_yield = list(map(lambda x, y: x / y - 1, y_price[1:], y_price[:-1]))
        return y_yield
    # цены (обратное преобразование)
    else:
        y_yield = y.copy()
        y_price = [price0] + y_yield
        for i in range(1, len(y_price)):
            y_price[i] = (y_price[i] + 1) * y_price[i - 1]
        return y_price
# конец функции getYields()