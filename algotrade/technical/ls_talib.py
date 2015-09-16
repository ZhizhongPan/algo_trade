# coding:utf-8
from __future__ import division
# from talib.abstract import MOM
import numpy as np
import pandas as pd
import talib as ta
from talib.abstract import Function
import scipy.ndimage.interpolation as inp


def is_close(a, b, tol=1e-8):
    return np.abs(a - b) < tol


def ACC(prices, timeperiod=12):
    """
    :param prices: stock price DataFrame
    :param timeperiod: lookback periods
    :return: acc
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    MOM = ta.abstract.MOM
    mom = MOM(df_price, timeperiod)
    ret = mom - mom.shift(timeperiod)

    return ret


# @jit
def ACD(prices, timeperiod=14):
    """
    DIF = CLOSE-IF(CLOSE>CLOSE[1],MIN(LOW,CLOSE[1]),MAX(HIGH,CLOSE[1]))
    ACD(N) = SUM(IF(CLOSE==CLOSE[1],0,DIF),N)   //N为参数

    :param prices: price data containing low,high,open,close,vol
    :param timeperiod: N
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort(ascending=True)

    high, low, close = df_price[['high', 'low', 'close']].T.values
    close1 = inp.shift(close, 1, order=0, cval=np.nan)

    min_low_close1 = np.min(np.column_stack((low, close1)), axis=1)
    max_high_close1 = np.max(np.column_stack((high, close1)), axis=1)

    minuend = max_high_close1

    cond1 = close > close1

    minuend[cond1] = min_low_close1[cond1]

    dif = close - minuend

    dif_or_0 = dif

    cond2 = np.isclose(close, close1)  # close == close1

    dif_or_0[cond2] = 0

    SUM = ta.SUM
    acd = SUM(dif_or_0, timeperiod)

    return pd.Series(acd, index=df_price.index)

    # close1 = df_price['close'].shift(1).values
    # close = df_price['close'].values
    # low = df_price['low'].values
    # high = df_price['high'].values

    # dif = np.zeros_like(close1, dtype=float)
    #
    # for idx in xrange(len(dif)):
    #     dif[idx] = close[idx] - (min(low[idx], close1[idx]) if close[idx] > close1[idx] else max(high[idx], close1[idx]))

    # def _dif(row):
    #     ret1 = min(row['low'], row['close1']) if row['close'] > row[
    #         'close1'] else max(row['high'], row['close1'])
    #     return row['close'] - ret1
    #
    # df_price['DIF'] = df_price.apply(_dif, axis=1)

    # def func(row):
    #     return 0 if np.abs(row['close'] - row['close1']) < 1e-6 else row['DIF']
    #
    # # 实数判断相等
    # df_price['DIF2'] = df_price.apply(
    #     func, axis=1)

    # dif2 = np.zeros_like(close, dtype=float)
    #
    # for idx in xrange(len(dif2)):
    #     dif2[idx] = 0 if np.isclose(close[idx], close1[idx]) else dif[idx]
    #
    # acd = ta.SUM(dif2, timeperiod)
    # return pd.Series(acd, index=df_price.index)


# @jit
def ADTM(prices, timeperiod=14):
    """
    说明：ADTM是用开盘价的向上波动幅度和向下波动幅度的距离差值来描述人气高低的指标。

    计算方法：
    DTM = IF(OPEN<=OPEN[1],0,MAX(HIGH-OPEN,OPEN-OPEN[1]))
    DBM = IF(OPEN>=OPEN[1],0,MAX(OPEN-LOW,OPEN-OPEN[1]))
    STM(N) = SUM(DTM,N)
    SBM(N) = SUM(DBM,N)
    ADTM = IF(STM>SBM,(STM-SBM)/STM, IF(STM<SBM, (STM-SBM)/SBM, 0))

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    open, high, low = df_price[['open', 'high', 'low']].T.values

    open1 = inp.shift(open, 1, order=0, cval=np.nan)

    max1 = np.amax(np.column_stack((high - open, open - open1)), axis=1)
    cond1 = open <= open1
    dtm = max1.copy()
    dtm[cond1] = 0.0

    max2 = np.amax(np.column_stack((open - low, open - open1)), axis=1)
    cond2 = open >= open1
    dbm = max2.copy()
    dbm[cond2] = 0.0

    SUM = ta.SUM
    stm_n = SUM(dtm, timeperiod)
    sbm_n = SUM(dbm, timeperiod)

    # adtm
    cond3 = stm_n > sbm_n
    cond4 = stm_n < sbm_n

    adtm = np.empty(shape=open1.shape)
    adtm[cond3] = (stm_n[cond3] - sbm_n[cond3]) / stm_n[cond3]
    adtm[cond4] = ((stm_n - sbm_n) / sbm_n)[cond4]
    adtm[np.isclose(stm_n, sbm_n)] = 0.0
    ## IMPORTANT!!
    adtm[np.isnan(stm_n) | np.isnan(sbm_n)] = np.nan


    # open = df_price['open'].values
    # open1 = df_price['open'].shift(1).values
    # high = df_price['high'].values
    # low = df_price['low'].values
    #
    # dtm = np.zeros_like(open1, dtype=float)
    # for idx in xrange(len(dtm)):
    #     dtm[idx] = 0 if open[idx] <= open1[idx] else np.max((high[idx] - open[idx], open[idx] - open1[idx]))
    #
    # # df_price['DTM'] = df_price.apply(
    # #     lambda row: 0 if row['open'] <= row['open1'] else max(
    # #         row['high'] - row['open'], row['open'] - row['open1']), axis=1)
    #
    # dbm = np.zeros_like(open1, dtype=float)
    # for idx in xrange(len(dbm)):
    #     dbm[idx] = 0 if open[idx] >= open1[idx] else np.max((open[idx] - low[idx], open[idx] - open1[idx]))
    #
    # # df_price['DBM'] = df_price.apply(
    # #     lambda row: 0 if row['open'] >= row['open1'] else max(
    # #         row['open'] - row['low'], row['open'] - row['open1']), axis=1)
    #
    # stm = ta.SUM(dtm, timeperiod)
    # sbm = ta.SUM(dbm, timeperiod)
    # # df_price['STM'] = pd.rolling_sum(df_price['DTM'], timeperiod)
    # # df_price['SBM'] = pd.rolling_sum(df_price['DBM'], timeperiod)
    #
    # # def _adtm(row):
    # #     ret = None
    # #     if np.isnan(row['STM']) or np.isnan(row['SBM']):
    # #         ret = np.nan
    # #     elif row['STM'] > row['SBM']:
    # #         ret = (row['STM'] - row['SBM']) / row['STM']
    # #     elif row['STM'] < row['SBM']:
    # #         ret = (row['STM'] - row['SBM']) / row['SBM']
    # #     else:
    # #         ret = 0
    # #
    # #     return ret
    #
    # adtm = np.zeros_like(open1, dtype=float)
    # for idx in xrange(len(adtm)):
    #     if np.isnan(stm[idx]) or np.isnan(sbm[idx]):
    #         adtm[idx] = np.nan
    #     elif stm[idx] > sbm[idx]:
    #         adtm[idx] = (stm[idx] - sbm[idx]) / stm[idx]
    #     elif stm[idx] < sbm[idx]:
    #         adtm[idx] = (stm[idx] - sbm[idx]) / sbm[idx]
    #     else:
    #         adtm[idx] = 0

    adtm = pd.Series(adtm, index=df_price.index)
    return adtm


def AR(prices, timeperiod=14):
    """
    说明：AR指标是反映市场当前情况下多空双方力量发展对比的结果。
    它是以当日的开盘价为基点。与当日最高价相比较，依固定公式计算出来的强弱指标。
        BR指标也是反映当前情况下多空双方力量争斗的结果。
        不同的是它是以前一日的收盘价为基础，与当日的最高价、最低价相比较，
        依固定公式计算出来的强弱指标。

        计算方法：
        AR(N) = SUM(HIGH-OPEN,N)/SUM(OPEN-LOW,N)*100

    :param prices:
    :param timeperiod: N
    :return:AR(N)
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    ar = pd.rolling_sum(df_price['high'] - df_price['open'], timeperiod) / \
         pd.rolling_sum(df_price['open'] - df_price['low'], timeperiod) * 100
    return ar


# @jit
def BR(prices, timeperiod=14):
    """
    BR指标也是反映当前情况下多空双方力量争斗的结果。
    不同的是它是以前一日的收盘价为基础，与当日的最高价、最低价相比较，依固定公式计算出来的强弱指标。
    计算方法：
    BR(N) = SUM(MAX(0,HIGH-CLOSE[1]),N)/SUM(MAX(0,CLOSE[1]-LOW),N)*100

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    high, low, close = df_price[['high', 'low', 'close']].T.values
    # low = df_price['low'].values
    # close = df_price['close'].values

    close1 = inp.shift(close, 1, order=0, cval=np.nan)

    # 分子
    max1 = high - close1
    max1[max1 < 0] = 0.0

    # 分母
    max2 = close1 - low
    max2[max2 < 0] = 0.0

    # 最后
    SUM = ta.SUM
    br = SUM(max1, timeperiod) / SUM(max2, timeperiod) * 100



    # high_close1 = np.zeros_like(high, dtype=float)
    # for idx in xrange(len(high_close1)):
    #     high_close1[idx] = np.max((0, high[idx] - close1[idx]))
    #
    # close1_low = np.zeros_like(close1, dtype=float)
    # for idx in xrange(len(close1_low)):
    #     close1_low[idx] = np.max([0, close1[idx] - low[idx]])

    # br = np.zeros_like(close1, dtype=float)
    #
    # for idx in xrange(len(br)):
    # br = ta.SUM(high_close1, timeperiod) / ta.SUM(close1_low, timeperiod) * 100

    # df_price['high-close1'] = df_price.apply(
    #     lambda row: max(0, row['high'] - row['close1']), axis=1)
    # df_price['close1-low'] = df_price.apply(
    #     lambda row: max(0, row['close1'] - row['low']), axis=1)
    # SUM = pd.rolling_sum
    # br = SUM(high_close1, timeperiod) / SUM(close1_low, timeperiod) * 100
    return pd.Series(br, index=df_price.index)


def ARC(prices, timeperiod=14):
    """
    ARC变化率指数均值
    说明：ARC指标是股票的价格变化率RC指标的均值，用以判断前一段交易周期内股票的平均价格变化率。
    计算方法：
    RC(N) = CLOSE/CLOSE[N]
    ARC(N) = MA(RC[1],N)
    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    df_price['rcn'] = df_price['close'] / df_price['close'].shift(timeperiod)
    arc = pd.rolling_mean(df_price['rcn'].shift(1), timeperiod)
    return arc


# @jit
def ASI(prices, timeperiod=14):
    """
    7.  ASI累计振动升降指标
    说明：累计振动升降指标(ASI) 以开盘、最高、最低、收盘价与前一交易日的各种价格相比较作为计算因子，研判市场的方向性。

    计算方法：
    A = ABS(HIGH-CLOSE[1])
    B = ABS(LOW-CLOSE[1])
    C = ABS(HIGH-LOW[1])
    D = ABS(CLOSE[1]-OPEN[1])
    E = CLOSE-CLOSE[1]
    F = CLOSE-OPEN
    G = CLOSE[1]-OPEN[1]
    X = E+0.5F+G
    K = MAX(A,B)
    R = IF(A>B AND A>C, A+0.5B+0.25D, IF(B>A AND B>C, B+0.5A+0.25D, C+0.25D))
    SI = 16*X/R*K
    ASI(N) = SUM(SI,N)

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    open, high, low, close = df_price[['open', 'high', 'low', 'close']].T.values

    close1 = inp.shift(close, 1, order=0, cval=np.nan)
    low1 = inp.shift(low, 1, order=0, cval=np.nan)
    open1 = inp.shift(open, 1, order=0, cval=np.nan)
    # high = df_price['high'].values
    # low = df_price['low'].values
    # close = df_price['close'].values
    # open = df_price['open'].values

    A = np.abs(high - close1)
    B = np.abs(low - close1)
    C = np.abs(high - low1)
    D = np.abs(close1 - open1)

    E = close - close1
    F = close - open
    G = close1 - open1

    X = E + 0.5 * F + G

    K = np.amax(np.column_stack((A, B)), axis=1)

    # K = np.zeros_like(open1)
    # for idx in xrange(len(K)):
    #     K[idx] = np.max((A[idx], B[idx]))
    # K = df_price.apply(
    #     lambda row: max(row['A'], row['B']), axis=1)

    R = np.empty(shape=open1.shape)

    cond1 = (A > B) & (A > C)
    cond2 = (B > A) & (B > C)

    R[cond1] = (A + 0.5 * B + 0.25 * D)[cond1]
    R[~cond1 & cond2] = (B + 0.5 * A + 0.25 * D)[~cond1 & cond2]

    R[~(cond1 | cond2)] = (C + 0.25 * D)[~(cond1 | cond2)]


    # for idx in xrange(len(R)):
    #     if A[idx] > B[idx] and A[idx] > C[idx]:
    #         R[idx] = A[idx] + 0.5 * B[idx] + 0.25 * D[idx]
    #     elif B[idx] > A[idx] and B[idx] > C[idx]:
    #         R[idx] = B[idx] + 0.5 * A[idx] + 0.25 * D[idx]
    #     else:
    #         R[idx] = C[idx] + 0.25 * D[idx]



    # def _R(row):
    #     if row['A'] > row['B'] and row['A'] > row['C']:
    #         return row['A'] + 0.5 * row['B'] + 0.25 * row['D']
    #     if row['B'] > row['A'] and row['B'] > row['C']:
    #         return row['B'] + 0.5 * row['A'] + 0.25 * row['D']
    #     else:
    #         return row['C'] + 0.25 * row['D']
    #
    # df_price['R'] = df_price.apply(_R, axis=1)

    si = 16 * X / R * K

    asi = ta.SUM(si, timeperiod)
    # df_price['SI'] = 16 * df_price['X'] / df_price['R'] * df_price['K']

    # asi = pd.rolling_sum(df_price['SI'], timeperiod)
    return pd.Series(asi, index=df_price.index)


def BBI(prices, timeperiod1=3, timeperiod2=6, timeperiod3=12, timeperiod4=24):
    """
    BBI多空指标
    说明：BBI多空指标，是一种将不同日数移动平均线加权平均之后的综合指标，属于均线型指标，
    一般选用3日、 6日、 12日、 24日等4条平均线。在BBI指标中，近期数据较多，远期数据利用次数较少，
    因而是一种变相的加权计算。 BBI指标既有短期移动平均线的灵敏，又有明显的中期趋势特征。

    计算方法：
    BBI(N1,N2,N3,N4) = (MA(CLOSE,N1)+MA(CLOSE,N2)+MA(CLOSE,N3)+MA(CLOSE,N4))/4

    :param prices:
    :param timeperiod1:
    :param timeperiod2:
    :param timeperiod3:
    :param timeperiod4:
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2, timeperiod3, timeperiod4)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    assert isinstance(timeperiod3, int)
    assert isinstance(timeperiod4, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    close = df_price['close']
    MA = pd.rolling_mean
    bbi = (MA(close, timeperiod1) + MA(close, timeperiod2) +
           MA(close, timeperiod3) + MA(close, timeperiod4)) / 4
    return bbi


def BIAS(prices, timeperiod=14):
    """
    说明：乖离率（BIAS）简称Y值也叫偏离率，是反映一定时期内股价与其移动平均数偏离程度的指标。
    移动平均数一般可视为某一时期内买卖双方都能接受的均衡价格。因此，股价距离移动平均线太远时会重新向平均线靠拢。
    乖离率指标就是通过测算股价在波动过程中与移动平均线出现的偏离程度，从而得出股价在剧烈波动时因偏离移动平均趋势可能形成的回档或反弹。

    计算方法：
    BIAS(N) = (CLOSE-MA(CLOSE,N))/MA(CLOSE,N)*100

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    close = df_price['close']
    MA = pd.rolling_mean
    bias = (close - MA(close, timeperiod)) / MA(close, timeperiod) * 100
    return bias


# @jit
def CMF(prices, timeperiod=20):
    """
    12. CMF蔡金货币流量指标（Chaikin Money Flow，CMF）
    说明：佳庆指标是基于AD曲线的指数移动均线而计算得到的。

    计算方法：默认N=20
    CLV = ((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-CLOSE)*VOL
    CMF(N) = SUM(CLV,N)/SUM(VOL,N)

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    close = df_price['close'].values
    high = df_price['high'].values
    low = df_price['low'].values
    volume = df_price['volume'].values

    # clv = np.zeros_like(low, dtype=float)
    # for idx in xrange(len(clv)):
    #     if is_close(high[idx] - close[idx], 0):
    #         clv[idx] = volume[idx]
    #     else:
    #         clv[idx] = ((close[idx] - low[idx]) - (high[idx] - close[idx])) / (high[idx] - close[idx]) * volume[idx]

    clv = ((close - low) - (high - close)) / (high - close) * volume
    inf_idx = np.isinf(clv)
    clv[inf_idx] = volume[inf_idx]

    # df_price['high_close'] = df_price['high'] - df_price['close']

    # df_price['CLV'] = df_price.apply(
    #     lambda row: ((row['close'] - row['low']) - row['high_close']) / row['high_close'] * row['volume'] if not np.is_close(row['high_close'], 0) else row[
    #         'volume'], axis=1
    # )
    SUM = pd.rolling_sum
    cmf = SUM(clv, timeperiod) / SUM(volume, timeperiod)
    # cmf = SUM(df_price['CLV'], timeperiod) / SUM(df_price['volume'], timeperiod)
    return pd.Series(cmf, index=df_price.index)


# @jit
def CVI(prices, timeperiod=14):
    """
    说明：蔡金波动性指标-- 计算最高价和最低价之间的价差。
    以在最大和最小之间的振幅为基础蔡金波动指标来断定波动价值。
    与真实范围平均数不同, 蔡金波动制表在账户中没有间隔。
    根据Chaikin的诠释，指标价值的增长直接关系到短的时间空隙，
    就是说价格接近他们的最小值(像当惊慌卖出)，在长时间里指标波动减缓，
    表明价格处于繁忙状态（例如，条件成熟牛市的状态）。

    计算方法：
    CVl(N) = (EMA(HIGH-LOW,N)-EMA(HIGH-LOW,N)[N])/ EMA(HIGH-LOW,N)*100

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    EMA = ta.EMA

    high_low = (df_price['high'] - df_price['low']).values.astype(float)
    ema_high_low_N = EMA(high_low, timeperiod=timeperiod)

    ema_high_low_NN = inp.shift(ema_high_low_N, shift=timeperiod, order=0, cval=np.nan)
    # df_price['ema_high_low'] = ema_high_low
    # df_price['ema_high_low_N'] = df_price['ema_high_low'].shift(timeperiod)

    # cvi = np.zeros_like(high_low, dtype=float)
    # for idx in xrange(len(cvi)):
    #     cvi[idx] = (ema_high_low_N[idx] - ema_high_low_NN[idx]) / ema_high_low_N[idx] * 100 if not is_close(
    #         ema_high_low_N[idx], 0) else 0

    cvi = (ema_high_low_N - ema_high_low_NN) / ema_high_low_N * 100.0
    inf_idx = np.isinf(cvi)
    cvi[inf_idx] = 0
    # def func(row):
    #     return (row['ema_high_low'] - row['ema_high_low_N']) / row['ema_high_low'] * 100 if not np.is_close(row['ema_high_low'], 0) else 0
    #
    # df_price['CVI'] = df_price.apply(
    #     func, axis=1
    # )

    return pd.Series(cvi, index=df_price.index)


# @autojit 不要用jit
def CR(prices, timeperiod=14):
    """
    说明：CR指标以上一个计算周期（如N日）的中间价比较当前周期（如日）的最高价、最低价，
    计算出一段时期内股价的“强弱”，从而在分析一些股价的异常波动行情时，有其独到的功能。
    另外， CR指标不但能够测量人气的热度、价格动量的潜能，而且能够显示出股价的压力带和支撑带，
    为分析预测股价未来的变化趋势，判断买卖股票的时机提供重要的参考。

    计算方法：
    MID = (HIGH+LOW+CLOSE)/3
    CR(N) = SUM(MAX(0,HIGH-MID[1],N)/SUM(MAX(0,MID[1]-LOW,N)*100

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    high, low, close = df_price[['high', 'low', 'close']].T.values

    mid = (high + low + close) / 3.0
    # df_price['mid'] = (df_price['high'] + df_price['low'] + df_price['close']) / 3



    mid1 = inp.shift(mid, 1, order=0, cval=np.nan)
    # df_price['mid1'] = df_price['mid'].shift(1)

    # df_price['max_high_mid1'] = df_price.apply(
    #     lambda row: max(0, row['high'] - row['mid1']), axis=1)

    high_mid1 = high - mid1

    max_high_mid1 = np.max(np.column_stack((high_mid1, np.zeros(shape=high_mid1.shape))), axis=1)


    # df_price['max_high_mid1'] = df_price[['']]

    # df_price['max_mid1_low'] = df_price.apply(
    #     lambda row: max(0, row['mid1'] - row['low']), axis=1)

    # mid1_low = (df_price['mid1'] - df_price['low']).values
    mid1_low = mid1 - low
    max_mid1_low = np.max(np.column_stack((mid1_low, np.zeros(shape=mid1_low.shape))), axis=1)

    # SUM = pd.rolling_sum
    SUM = ta.SUM
    # cr = SUM(df_price['max_high_mid1'], timeperiod) / \
    #      SUM(df_price['max_mid1_low'], timeperiod) * 100
    cr = SUM(max_high_mid1, timeperiod) / SUM(max_mid1_low, timeperiod) * 100
    return pd.Series(cr, index=df_price.index)


def DBCD(prices, timeperiod1=14, timeperiod2=14, timeperiod3=14):
    """
    DBCD异同离差乖离率(DBCD)
    说明：DBCD异同离差乖离率先计算乖离率BIAS，然后计算不同日的乖离率之间的离差，
    最后对离差进行指数移动平滑处理。优点是能够保持指标的紧密同步，并且线条光滑，
    信号明确，能够有效的过滤掉伪信号。

    计算方法：
    BIAS(N) = (CLOSE-SMA(CLOSE,N))/SMA(CLOSE,N)
    DIF(N1) = BIAS-BIAS[N1]
    DBCD(N2) = SMA(DIF(N1),N2)

    :param prices:
    :param timeperiod1:N
    :param timeperiod2:N1
    :param timeperiod3:N2
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2, timeperiod3)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    assert isinstance(timeperiod3, int), 'period must be positive integer'
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    df_price['BIAS'] = BIAS(prices, timeperiod1)
    df_price['DIF'] = df_price['BIAS'] - df_price['BIAS'].shift(timeperiod2)
    dbcn = pd.rolling_sum(df_price['DIF'], timeperiod3)
    return dbcn


def DDI(prices, timeperiod=20):
    """
    19. DDI方向标准离差指标(Directional Divergence Index，DDI)
    说明：

    计算方法：默认N=20
    DMZ = IF(HIGH+LOW<=HIGH[1]+LOW[1],0,MAX(ABS(HIGH-HIGH[1]),ABS(LOW-LOW[1])))
    DMF = IF(HIGH+LOW>HIGH[1]+LOW[1],0, MAX(ABS(HIGH-HIGH[1]),ABS(LOW-LOW[1])))
    DIZ = SUM(DMZ,N)/(SUM(DMZ,N)+SUM(DMF,N))
    DIF = SUM(DMF,N)/(SUM(DMZ,N)+SUM(DMF,N))
    DDI=DIZ-DIF

    :param prices:
    :param timeperiod:
    :return:
    """

    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()

    df_price = prices.sort_index(ascending=True)
    high, low = df_price[['high', 'low']].T.values
    high1 = inp.shift(high, 1, cval=np.nan)
    low1 = inp.shift(low, 1, cval=np.nan)

    abs_high_high1 = np.abs(high - high1)
    abs_low_low1 = np.abs(low - low1)
    val_max = np.max(np.column_stack((abs_high_high1, abs_low_low1)), axis=1)

    # DMZ
    cond1 = high + low <= high1 + low1
    dmz = val_max.copy()
    dmz[cond1] = 0

    # DMF
    dmf = val_max
    dmf[~cond1] = 0

    SUM = ta.SUM
    sum_dmz = SUM(dmz, timeperiod)
    sum_dmf = SUM(dmf, timeperiod)
    sum_dmz_dmf = sum_dmz + sum_dmf
    diz = sum_dmz / sum_dmz_dmf
    dif = sum_dmf / sum_dmz_dmf

    ddi = pd.Series((diz - dif) * 100, index=df_price.index)

    # df_price['high1'] = df_price['high'].shift(1)
    # df_price['low1'] = df_price['low'].shift(1)
    #
    # df_price['DMZ'] = df_price.apply(
    #     lambda row: 0 if row['high'] + row['low'] <= row['high1'] + row['low1'] else max(abs(row['high'] - row['high1']), abs(row['low'] - row['low1'])),
    # axis=1
    # )
    # df_price['DMF'] = df_price.apply(
    #     lambda row: 0 if row['high'] + row['low'] > row['high1'] + row['low1'] else max(abs(row['high'] - row['high1']), abs(row['low'] - row['low1'])),
    # axis=1
    # )
    # SUM = pd.rolling_sum
    # df_price['SUM_DMZ'] = SUM(df_price['DMZ'], timeperiod)
    # df_price['SUM_DMF'] = SUM(df_price['DMF'], timeperiod)
    # df_price['SUM_DMZ_DMF'] = df_price['SUM_DMZ'] + df_price['SUM_DMF']
    # df_price['DIZ'] = df_price.apply(
    #     lambda row: row['SUM_DMZ'] / row['SUM_DMZ_DMF'] if not np.isclose(row['SUM_DMZ_DMF'], 0) else 0, axis=1
    # )
    # df_price['DIF'] = df_price.apply(
    #     lambda row: row['SUM_DMF'] / row['SUM_DMZ_DMF'] if not np.isclose(row['SUM_DMZ_DMF'], 0) else 0, axis=1
    # )

    # ddi = (df_price['DIZ'] - df_price['DIF']) * 100
    return ddi


def DPO(prices, timeperiod=14):
    """
    20. DPO去趋势价格摆动指标(Detrended Price Oscillator)
    说明：去趋势价格摆动指标试图消除价格的趋势，让你更容易辨别出周期和超买超卖水平。

    计算方法:
    K = N/2+1
    DPO(N)[K] = CLOSE[-K]-SMA(CLOSE,N)[-K]

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    K = int(timeperiod / 2) + 1
    close = df_price['close']
    SMA = pd.rolling_mean
    dpo = close.shift(-K) - SMA(close, timeperiod).shift(-K)
    return dpo


def DMI(prices, timeperiod1=5, timeperiod2=10):
    """
    22. DMI动态动量指标（Dynamic Momentum Index, DMI）
    说明：除了时间区间的长度是变动而不是固定的，动态动量指标与相对强弱指标是一致的。
    价格的波动性越高，DMI对价格的变化就越敏感。

    计算方法：
    DMI(N1,N2) = 14/(STD(CLOSE,N1)/SMA(STD(CLOSE,N1),N2))
    默认取值N1=5, N2=10

    :param prices:
    :param timeperiod1: N1
    :param timeperiod2: N2
    :return:
    """
    assert prices is not None
    timeperiod = timeperiod1 + timeperiod2
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    stdn1 = pd.rolling_std(df_price['close'], timeperiod1)
    dmi = 14 * pd.rolling_mean(stdn1, timeperiod2) / stdn1
    return dmi


def EMV(prices):
    """
    23. EMV简易波动指标(Ease of Movement)
    说明：简易波动指标（Ease of Movement Value）又称EMV指标，
    它是由Richard W．Arm Jr．根据等量图和压缩图的原理设计而成,
    目的是将价格与成交量的变化结合成一个波动指标来反映股价或指数的变动状况。
    由于股价的变化和成交量的变化都可以引发该指标数值的变动,因此,EMV实际上也是一个量价合成指标。

    计算方法:
    MM = ((HIGH-LOW)-(HIGH[1]-LOW[1]))/2
    BR = VOL/(HIGH-LOW)/10000
    EMV = MM/BR

    :param prices:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), 1)

    high = prices['high']
    low = prices['low']
    vol = prices['volume']

    mm = ((high - low) - (high.shift(1) - low.shift(1))) / 2
    br = vol / (high - low) / 10000
    emv = mm / br
    return emv


# @jit
def IMI(prices, timeperiod=14):
    """
    说明：
    日内动量指标（IMI）是由相对强弱指标和蜡烛图分析嫁接而成，
    IMI超过70表示潜在的超买，低于30表示潜在的超卖

    计算方法：
    USUM(N) = SUM(IF(CLOSE>OPEN, CLOSE-OPEN,0),N)
    DSUM(N) = SUM(IF(CLOSE<=OPEN, OPEN-CLOSE,0),N)
    IMI(N) = USUM/(USUM(N)+DSUM(N))*100

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    open, close = df_price[['open', 'close']].T.values

    cond = close > open

    u = np.zeros(shape=open.shape)
    d = u.copy()
    u[cond] = (close - open)[cond]
    d[~cond] = (open - close)[~cond]

    SUM = ta.SUM
    usum = SUM(u, timeperiod)
    dsum = SUM(d, timeperiod)

    rmi = usum / (usum + dsum) * 100

    return pd.Series(rmi, index=df_price.index)


    # df_price['U'] = df_price.apply(
    #     lambda row: row['close'] - row['open'] if row['close'] > row['open'] else 0, axis=1
    # )
    # df_price['D'] = df_price.apply(
    #     lambda row: row['open'] - row['close'] if row['close'] <= row['open'] else 0, axis=1
    # )
    # df_price['USUM'] = pd.rolling_sum(df_price['U'], timeperiod)
    # df_price['DSUM'] = pd.rolling_sum(df_price['D'], timeperiod)
    # df_price['IMI'] = df_price.apply(
    #     lambda row: row['USUM'] / (row['USUM'] + row['DSUM']) * 100 if not np.isclose(row['USUM'] + row['DSUM'], 0) else 100, axis=1
    # )
    # # imi = df_price['USUM'] / (df_price['USUM'] + df_price['DSUM']) * 100
    # return df_price['IMI']


def KVO(prices, timeperiod1=34, timeperiod2=55, timeperiod3=13):
    """
    TP_TODAY = (HTODAY + LTODAY + CTODAY) / 3
    TP_YESTERDAY = (H_YESTERDAY + L_YESTERDAY + C_YESTERDAY) / 3

    where

    H = High, L = Low, and C = Close, respectively

    Based on the TP values, assign a signed value (SV) to today’s volume (V):

    If TP_TODAY > TP_YESTERDAY, then SV = +V
    If TP_TODAY < TP_YESTERDAY, then SV = -V

    In our research, we found no statement as to how to treat the case

    TP_TODAY = TP_YESTERDAY

    So for equal TP values, SV=+V. That is, we treat this case as positive.

    Klinger refers to the signed volume value SV as the “volume force.”
    A positive volume force indicates accumulation,
    while a negative volume force indicates distribution.

    Next, calculate two EMAs of the signed volume value and their difference:

    KVO = EMA(34)(SV) - EMA(55)(SV)

    Finally, calculate a 13-period EMA of the KVO,
    used as a trigger line for the KVO:

    EMA(13)(KVO)

    :param prices:
    :param timeperiod1:
    :param timeperiod2:
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)



    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    high, low, close = df_price[['high', 'low', 'close']].T.values
    typical_price = (high + low + close) / 3.0
    typical_price1 = inp.shift(typical_price, 1, order=0, cval=np.nan)
    cond = typical_price > typical_price1
    sv = df_price['volume'].values.astype(float)
    sv[~cond] = - df_price['volume'].values[~cond]

    EMA = ta.EMA
    kvo = EMA(sv, timeperiod1) - EMA(sv, timeperiod2)

    trig = EMA(kvo, timeperiod3)

    return pd.Series(trig, index=df_price.index)

    # df_price['typical_price'] = (
    #                                 df_price['high'] + df_price['low'] + df_price['close']) / 3
    # df_price['typical_price1'] = df_price['typical_price'].shift(1)
    #
    # df_price['SV'] = df_price.apply(
    #     lambda row: row['volume'] if row['typical_price'] >= row['typical_price1'] else -row['volume'], axis=1
    # )
    # EWA = ta.EMA
    # df_price['kvo'] = EWA(
    #     df_price['SV'].values.astype(float), timeperiod1) - EWA(df_price['SV'].values.astype(float), timeperiod2)
    # df_price['timeperiod3'] = EWA(df_price['kvo'].values.astype(float), timeperiod3)
    # return df_price['timeperiod3']


def MI(prices, timeperiod=9):
    """
    28. MI质量指标(Mass Index, MI)
    说明：质量指标通过度量最高价与最低价之间区间的变宽和变窄来辨别趋势反转。

    计算方法:
    MI(N) = SUM(EMA(HIGH-LOW,9)/EMA(EMA(HIGH-LOW,9),9),N)

    :param prices:
    :param timeperiod:N
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    EMA = ta.EMA
    df_price['ema_high_low'] = EMA(
        (df_price['high'] - df_price['low']).values.astype(float), timeperiod=timeperiod)
    df_price['ema_ema_high_low'] = EMA(
        df_price['ema_high_low'].values.astype(float), timeperiod=timeperiod)
    df_price['MI'] = pd.rolling_sum(
        (df_price['ema_high_low'] / df_price['ema_ema_high_low']).values.astype(float), timeperiod)
    return df_price['MI']


def MTM(prices, timeperiod=20):
    """
    26. MTM动量指标(Momentum，MTM)
    说明：动量指标度量在给定时间价格的变动

    计算方法:默认N=20
    MTM(N) = CLOSE-CLOSE[N]

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    mtm = df_price['close'] - df_price['close'].shift(timeperiod)
    return mtm


def NVI(prices):
    """

    :param prices:
    :return:
    """
    # TODO:性能稍差与其他程序，很难向量化，jit优化更耗时
    assert prices is not None
    _assert_greater_or_equal(len(prices), 1)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    close, volume = df_price[['close', 'volume']].T.values

    close1 = inp.shift(close, 1, order=0, cval=np.nan)
    volume1 = inp.shift(volume, 1, order=0, cval=np.nan)

    N = len(df_price)
    # nvi = np.empty(shape=close1.shape)

    # @jit
    def __nvi(N, close, close1, volume, volume1):
        nvi = np.empty(shape=close1.shape)
        nvi[0] = 1000
        for ind in range(1, N):
            nvi[ind] = nvi[ind - 1] - (close - close1)[ind] / close1[
                ind] * nvi[ind - 1] if volume[ind] < volume1[ind] else nvi[ind - 1]
        return nvi

    nvi = __nvi(N, close, close1, volume, volume1)
    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['volume1'] = df_price['volume'].shift(1)
    # N = len(prices)
    # nvi = np.zeros(shape=(N,), dtype=np.float64)
    # nvi[0] = 1000
    # for ind in range(1, N):
    #     nvi[ind] = nvi[ind - 1] - (df_price['close'] - df_price['close1'])[ind] / df_price['close1'][
    #         ind] * nvi[ind - 1] if df_price['volume'][ind] < df_price['volume1'][ind] else nvi[ind - 1]

    nvi = pd.Series(nvi, index=df_price.index)
    return nvi


def PFE(prices, timeperiod1=10, timeperiod2=5):
    """
    32. PFE极化分形效率指标(Polarized Fractal Efficiency, PFE)
    说明：PFE利用数学方法来确定价格在两个点之间变动的效率，价格变动越呈现线性，效率越高，
    价格在两个点之间运动的距离就越短

    计算方法：N,N1=10,5
    P1 = SQRT((CLOSE-CLOSE[N])^2+N^2)
    P2 = SUM(SQRT((CLOSE-CLOSE[1])^2+1)，N)
    FET = SIGN(CLOSE-CLOSE[N])*P1/P2*100
    PFE = EMA(FET,N1)

    :param prices:
    :param timeperiod1: N
    :param timeperiod2: N1
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    df_price['closen'] = df_price['close'].shift(timeperiod1)
    df_price['close1'] = df_price['close'].shift(1)

    p1 = np.sqrt(
        (df_price['close'] - df_price['closen']) ** 2 + timeperiod1 ** 2)
    SUM = pd.rolling_sum
    p2 = SUM(
        np.sqrt((df_price['close'] - df_price['close1']) ** 2 + 1), timeperiod1)
    fet = np.sign(df_price['close'] - df_price['closen']) * p1 / p2 * 100
    EMA = ta.EMA
    df_price['PFE'] = EMA(fet.values.astype(float), timeperiod2)
    return df_price['PFE']


def PVI(prices):
    """
    33. PVI正量指标(Positive Volume Index, PVI)
    说明：正量指标集中关注那些成交量比前期增加了的交易日，显示了并不那么聪明的资金正在干什么。

    计算方法:
    PVI = IF(VOL>VOL[1],PVI[1]+(CLOSE-CLOSE[1])/CLOSE[1]*PVI[1],PVI[1])
    PVI初始值设为1000

    :param prices:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), 1)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    close, volume = df_price[['close', 'volume']].T.values

    close1 = inp.shift(close, 1, order=0, cval=np.nan)
    volume1 = inp.shift(volume, 1, order=0, cval=np.nan)
    N = len(prices)
    pvi = np.zeros(shape=(N,), dtype=np.float64)
    pvi[0] = 1000
    for ind in range(1, N):
        pvi[ind] = pvi[ind - 1] + (close - close1)[ind] / close1[
            ind] * pvi[ind - 1] if volume[ind] > volume1[ind] else pvi[ind - 1]

    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['volume1'] = df_price['volume'].shift(1)
    # N = len(prices)
    # pvi = np.zeros(shape=(N,), dtype=np.float64)
    # pvi[0] = 1000
    # for ind in range(1, N):
    #     pvi[ind] = pvi[ind - 1] + (df_price['close'] - df_price['close1'])[ind] / df_price['close1'][
    #         ind] * pvi[ind - 1] if df_price['volume'][ind] > df_price['volume1'][ind] else pvi[ind - 1]

    pvi = pd.Series(pvi, index=df_price.index)
    return pvi


def PVT(prices):
    """
    34. PVT价量趋势指标(Price and Volume Trend, PVT)
    说明：价量趋势指标类似于能量潮（OBV），也是随收盘价变化而调整的累积成交量。
    然而，OBV是加上当期收盘价高于前期收盘价的交易日的全部成交量和减去当期收盘价低于前期收盘价的交易日的全部成交量，
    而PVT仅仅加上或减去日成交量的一部分。

    计算方法：
    PVT = (CLOSE-CLOSE[1])/CLOSE[1]*VOL+PVT[1]
    PVT初始值= (CLOSE-CLOSE[1])/CLOSE[1]*VOL

    :param prices:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), 1)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    df_price['close1'] = df_price['close'].shift(1)
    N = len(prices)
    pvt = np.zeros(shape=(N,), dtype=np.float64)

    pvt[0] = np.nan
    pvt[1] = (df_price['close'] - df_price['close1'])[1] / \
             df_price['close1'][1] * df_price['volume'][1]
    for ind in range(2, N):
        pvt[ind] = pvt[ind - 1] + (df_price['close'] - df_price['close1'])[ind] / \
                                  df_price['close1'][ind] * df_price['volume'][ind]

    pvt = pd.Series(pvt, index=df_price.index)
    return pvt


def QST(prices, timeperiod=14):
    """
    说明：过去一段时间收盘价与开盘价之差的平均值

    计算方法：
    QST(N) = SUM(CLOSE-OPEN,N)/N

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    SUM = pd.rolling_sum
    qst = SUM(df_price['close'] - df_price['open'], timeperiod) / timeperiod
    return qst


# @jit
def RI(prices, timeperiod1=20, timeperiod2=5):
    """
    36. RI区域指标(Range Indicator, RI)
    说明：区域指标显示了期间内最高价到最低价的区域超过了期间之间收盘价到收盘价的区域的时间，
    这种方法在辨别趋势的开始与结束方面被证明是有用的。

    计算方法：默认N1=20，N2=5
    TR = MAX(HIGH-LOW,ABS(CLOSE[1]-HIGH),ABS(CLOSE[1]-LOW))
    W = IF(CLOSE>CLOSE[1],TR/(CLOSE-CLOSE[1]),TR)
    SR(N1)=IF(MAX(W,N1)-MIN(W,N1)>0,(W-MIN(W,N1))/(MAX(W,N1)-MIN(W,N1))*100, (W-MIN(W,N1))*100)
    RI(N1,N2) = EMA(SR(N1),N2)

    :param prices:
    :param timeperiod1: N1
    :param timeperiod2: N2
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    high, low, close = df_price[['high', 'low', 'close']].T.values

    close1 = inp.shift(close, 1, cval=np.nan)

    tr = TR(df_price).values

    w = tr.copy()
    cond1 = close > close1
    w[cond1] = tr[cond1] / (close[cond1] - close1[cond1])

    MIN = ta.MIN
    MAX = ta.MAX

    min_w_n1 = MIN(w, timeperiod1)
    max_w_n1 = MAX(w, timeperiod1)

    # sr
    cond2 = max_w_n1 > min_w_n1

    sr = (w - min_w_n1) * 100.0
    sr[cond2] = (w - min_w_n1)[cond2] / (max_w_n1[cond2] - min_w_n1[cond2]) * 100

    # ri
    EMA = ta.EMA
    ri = EMA(sr, timeperiod2)

    return pd.Series(ri, index=df_price.index)



    # df_price['close1'] = df_price['close'].shift(1)
    #
    # def func1(row):
    #     return max(row['high'] - row['low'], abs(row['close1'] - row['high']), abs(row['close1'] - row['low']))
    #
    # df_price['TR'] = df_price.apply(
    #     func1, axis=1
    # )
    #
    # def func2(row):
    #     return row['TR'] / (row['close'] - row['close1']) if row['close'] > row['close1'] else row['TR']
    #
    # df_price['W'] = df_price.apply(
    #     func2, axis=1
    # )

    # MAX = pd.rolling_max
    # df_price['MAX_W_N1'] = MAX(df_price['W'], timeperiod1)
    # MIN = pd.rolling_min
    # df_price['MIN_W_N1'] = MIN(df_price['W'], timeperiod1)
    #
    # def func3(row):
    #     return (row['W'] - row['MIN_W_N1']) / (row['MAX_W_N1'] - row['MIN_W_N1']) * 100 if row['MAX_W_N1'] > row['MIN_W_N1'] else (row['W'] - row[
    #         'MIN_W_N1']) * 100
    #
    # df_price['SR_N1'] = df_price.apply(
    #     func3, axis=1
    # )
    # EMA = ta.EMA
    # df_price['RI'] = EMA(df_price['SR_N1'].values.astype(float), timeperiod2)
    # return df_price['RI']


# @jit
def RMI(prices, timeperiod1=5, timeperiod2=14):
    """
    37. RMI相对动量指标(Relative Momentum Indicator, RMI)
    说明：相对动量指标是相对强弱指标加上动量成分后的变形，相对动量指标从收盘价相对于n期前收盘价
    计算上涨交易日和下跌交易日。

    计算方法: 默认值N1=5, N2=14
    UM(N1) = IF(CLOSE-CLOSE[N1]>0,CLOSE-CLOSE[N1],0)
    DM(N1) = IF(CLOSE-CLOSE[N1]<0,CLOSE[N1]-CLOSE,0)
    UA(N2) = (UA[1]*(N2-1)+UM)/N2
    DA(N2) = (DA[1]*(N2-1)+DM)/N2
    RMI = 100*(UA/(UA+DA))

    UA初始值 = SMA(UM,N)
    DA初始值 = SMA(DM,N)

    :param prices:
    :param N1: N1
    :param N2: N2
    :return:
    """
    assert prices is not None
    timeperiod = timeperiod1 + timeperiod2
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    #     df_price = prices.copy()
    df_price = prices.sort(ascending=True).astype(float)

    N1 = timeperiod1
    N2 = timeperiod2

    close = df_price['close'].values.astype(float)
    close_n1 = inp.shift(close, N1, order=0, cval=np.nan)

    # um, dm
    cond1 = close > close_n1
    um_n1 = np.zeros(shape=close.shape)
    um_n1[cond1] = close[cond1] - close_n1[cond1]
    um_n1[np.isnan(close_n1)] = np.nan

    cond2 = close < close_n1
    dm_n1 = np.zeros(shape=close_n1.shape)
    dm_n1[cond2] = close_n1[cond2] - close[cond2]
    dm_n1[np.isnan(close_n1)] = np.nan


    # close = df_price['close'].values
    # close_n1 = df_price['close'].shift(N1).values
    #
    # um_n1 = np.zeros_like(close, dtype=float)
    # for idx in xrange(len(um_n1)):
    #     if close[idx] > close_n1[idx]:
    #         um_n1[idx] = close[idx] - close_n1[idx]
    #     else:
    #         um_n1[idx] = 0.0
    #
    # dm_n1 = np.zeros_like(close)
    # for idx in xrange(len(dm_n1)):
    #     dm_n1[idx] = close_n1[idx] - close[idx] if close[idx] < close_n1[idx] else 0.0

    ua = np.zeros_like(close, dtype=float)
    da = np.zeros_like(close, dtype=float)

    SMA = ta.SMA

    ua[:(N1 + N2)] = np.nan
    ua[(N1 + N2)] = SMA(um_n1, N2)[(N1 + N2)]

    da[:(N1 + N2)] = np.nan
    da[(N1 + N2)] = SMA(dm_n1, N2)[(N1 + N2)]

    for ind in xrange((N1 + N2 + 1), len(close)):
        ua[ind] = (ua[ind - 1] * (N2 - 1) + um_n1[ind]) / N2
        da[ind] = (da[ind - 1] * (N2 - 1) + dm_n1[ind]) / N2

    # rmi = np.zeros_like(close, dtype=float)
    #
    # for idx in xrange(len(rmi)):
    #     rmi[idx] = ua[idx] / (ua[idx] + da[idx]) * 100 if not np.isclose(ua[idx] + da[idx], 0) else 50

    rmi = ua / (ua + da) * 100
    inf_idx = np.isinf(rmi)
    rmi[inf_idx] = 50

    rmi = pd.Series(rmi, index=df_price.index)
    return rmi


def RVI(prices, timeperiod1=10, timeperiod2=14):
    """
    39. RVI相对波动率指标(Relative Volatility Index, RVI)
    说明：相对波动率指标用来度量价格波动的方向，它的计算与相对强弱指标一样，只是RVI测量的最高价或最低价的10日标准差。

    计算方法：默认值N1=10,N2=14
    UM(PRICE,N1) = IF(PRICE>PRICE[1],STD(PRICE,N1)，0)
    DM(PRICE,N1) = IF(PRICE<PRICE[1],STD(PRICE,N1)，0)
    UA(N2) = (UA[1]*(N-1)+UM)/N2
    DA(N2) = (DA[1]*(N-1)+DM)/N2
    RS(PRICE) = 100*UA/(UA+DA)
    RVI = (RS(HIGH)+RS(LOW))/2

    UA初始值 = SMA(UM,N2)
    DA初始值 = SMA(DM,N2)

    :param prices:
    :param timeperiod1:
    :param timeperiod2:
    :return:
    """
    assert prices is not None
    timeperiod = timeperiod1 + timeperiod2
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    N1 = timeperiod1
    N2 = timeperiod2

    STD = ta.STDDEV

    def _UM(price, N1):
        price1 = inp.shift(price, 1, order=0, cval=np.nan)
        cond1 = price > price1
        um = np.zeros(shape=price1.shape)
        um[cond1] = STD(price, N1)[cond1]
        # return std
        # df_price[price + str(1)] = df_price[price].shift(1)
        # STD = pd.rolling_std
        # std_price_n1 = 'std_' + price + '_' + str(N1)
        # df_price[std_price_n1] = STD(df_price[price], N1)

        # def func(row):
        #     return row[std_price_n1] if row[price] > row[price + str(1)] else 0
        #
        # um = df_price.apply(
        #     func, axis=1
        # )
        um[:N1] = np.nan
        # print('um={0}'.format(um))
        return um

    def _DM(price, N1):
        # df_price[price + str(1)] = df_price[price].shift(1)
        # STD = pd.rolling_std
        # std_price_n1 = 'std_' + price + '_' + str(N1)
        # df_price[std_price_n1] = STD(df_price[price], N1)
        #
        # def func(row):
        #     return row[std_price_n1] if row[price] < row[price + str(1)] else 0

        price1 = inp.shift(price, 1, order=0, cval=np.nan)
        dm = np.zeros(shape=price1.shape)
        cond = price < price1
        dm[cond] = STD(price, N1)[cond]
        # dm = df_price.apply(
        #     func, axis=1
        # )
        dm[:N1] = np.nan
        # print('dm={0}'.format(dm))
        return dm

    def _UA(price, N1, N2):
        ua = np.zeros_like(price)
        um = _UM(price, N1)
        SMA = ta.SMA

        ua[:(N1 + N2)] = np.nan

        sma_um_n2 = SMA(um, N2)
        ua[(N1 + N2)] = sma_um_n2[(N1 + N2)]

        for ind in range((N1 + N2 + 1), len(price)):
            ua[ind] = (ua[ind - 1] * (N2 - 1) + um[ind]) / N2
        # ua = pd.Series(ua, index=um.index)
        # print('ua={0}'.format(ua))
        return ua

    def _DA(price, N1, N2):

        SMA = ta.SMA

        dm_n1 = _DM(price, N1)
        da = np.zeros_like(price)
        da[:(N1 + N2)] = np.nan
        sma_dm_n2 = SMA(dm_n1, N2)
        da[(N1 + N2)] = sma_dm_n2[(N1 + N2)]
        for ind in range((N1 + N2 + 1), len(price)):
            da[ind] = (da[ind - 1] * (N2 - 1) + dm_n1[ind]) / N2
        # print('da={0}'.format(da))
        # da = pd.Series(da, index=dm_n1.index)
        return da

    def _RS(price):
        ua = _UA(price, N1, N2)
        da = _DA(price, N1, N2)
        # rs = ua / (ua + da) * 100
        # rs = np.zeros_like(ua)
        rs = ua / (ua + da) * 100
        inf_idx = np.isinf(rs)
        rs[inf_idx] = 0

        # for ind in range(len(ua)):
        #     rs[ind] = ua[ind] / (ua[ind] + da[ind]) * 100 if not np.isclose((ua[ind] + da[ind]), 0) else 0
        # rs = pd.Series(rs, index=ua.index)
        return rs

    high, low = df_price[['high', 'low']].T.values

    rvi = (_RS(high) + _RS(low)) / 2
    # df_price['RVI'] = (_RS('high') + _RS('low')) / 2

    return pd.Series(rvi, index=df_price.index)


def SMI(prices, timeperiod1=10, timeperiod2=3, timeperiod3=3):
    """
    说明：SMI显示收盘价相对于近期最高价/最低价区间的中点的位置。

    计算方法：N,N1,N2=10,3,3
    C(N) = (MAX(HIGH,N)+MAX(LOW,N))/2
    H = CLOSE-C(N)
    SH1 = EMA(H,N1)
    SH2 = EMA(SH1,N2)
    R = MAX(HIGH,N)-MAX(LOW,N)
    SR1 = EMA(R,N1)
    SR2 = EMA(SR1,N2)/2
    SMI = (SH2/SR2)*100

    :param prices:
    :param timeperiod1: N
    :param timeperiod2: N1
    :param timeperiod3: N2
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2, timeperiod3)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    assert isinstance(timeperiod3, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    MAX = pd.rolling_max
    cn = (MAX(df_price['high'], timeperiod1) +
          MAX(df_price['low'], timeperiod1)) / 2
    h = df_price['close'] - cn
    EMA = ta.EMA
    df_price['SH1'] = EMA(h.values.astype(float), timeperiod2)
    df_price['SH2'] = EMA(df_price['SH1'].values.astype(float), timeperiod3)
    r = MAX(df_price['high'], timeperiod1) - MAX(df_price['low'], timeperiod1)
    df_price['SR1'] = EMA(r.values.astype(float), timeperiod2)
    df_price['SR2'] = EMA(df_price['SR1'].values.astype(float), timeperiod3) / 2
    smi = df_price['SH2'] / df_price['SR2'] * 100
    return smi


def SRSI(prices, timeperiod1=14, timeperiod2=14):
    """
    42. SRSI随机相对强弱指标(Stochastic Relative Strength Index，SRSI)
    说明：随机摆动指标中的价格数据换成RSI

    计算方法：默认N，N1=14
    SRSI = (RSI(N)-MAX(RSI(N),N1))/(MAX(RSI(N),N1)-MIN(RSI(N),N1)) * 100

    :param prices:
    :param timeperiod1: N
    :param timeperiod2: N1
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    RSI = Function("RSI")
    MAX = ta.MAX
    MIN = ta.MIN

    rsi_n = RSI(df_price, timeperiod1).values

    srsi = (rsi_n - MAX(rsi_n, timeperiod2)) / (MAX(rsi_n, timeperiod2) - MIN(rsi_n, timeperiod2)) * 100

    inf_index = np.isinf(srsi)
    srsi[inf_index] = 0

    return pd.Series(srsi, index=df_price.index)
    # RSI = Function("RSI")
    # MAX = pd.rolling_max
    # MIN = pd.rolling_min
    # df_price['RSI_N'] = RSI(df_price, timeperiod1)
    # df_price['MAX_RSI_N_N1'] = MAX(df_price['RSI_N'], timeperiod2)
    # df_price['MIN_RSI_N_N1'] = MIN(df_price['RSI_N'], timeperiod2)
    # df_price['MAX_MINUS_MIN_RSI'] = df_price['MAX_RSI_N_N1'] - df_price['MIN_RSI_N_N1']
    #
    # def func(row):
    #     return (row['RSI_N'] - row['MAX_RSI_N_N1']) / row['MAX_MINUS_MIN_RSI'] * 100 if not np.isclose(row['MAX_MINUS_MIN_RSI'], 0) else 0
    #
    # df_price['SRSI'] = df_price.apply(
    #     func, axis=1)
    # return df_price['SRSI']


def TS(prices, timeperiod=20):
    """
    说明：一个用来统计近期涨跌情况的简单指标，当价格上涨分数为正，价格下跌分数为负。

    计算方法：默认N=20
    TS = SUM(IF(CLOSE>=CLOSE[1],1,-1),N)

    :param prices:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    close = df_price['close'].values.astype(float)
    close1 = inp.shift(close, 1, order=0, cval=np.nan)
    SUM = ta.SUM
    ts = np.ones(shape=close1.shape, dtype=float)
    cond = close >= close1
    ts[~cond] = -1.0
    return pd.Series(SUM(ts, timeperiod), index=df_price.index)

    # SUM = pd.rolling_sum
    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['close_close1'] = df_price.apply(
    #     lambda row: 1 if row['close'] >= row['close1'] else -1, axis=1
    # )
    # ts = SUM(df_price['close_close1'], timeperiod)
    # return ts


def TMA(prices, timeperiod=10, price='close'):
    """
    46. TMA三角移动平均线(Triangular Moving Average，TMA)
    说明：
    三角移动平均线(TMA)是对简单移动平均线进行再平均，并使其更加平滑，即TMA 是平均价格的平均值。
    与其他移动平均线一样，TMA 的主要缺点是反应迟钝，是一个滞后的指标，总是落后于价格。
    然而，相比于其他移动平均线，TMA 有时能够更快地对价格变化作出反应。
    这是因为 TMA 产生的线条比简单移动平均线更平滑、更像波浪。

    计算方法：默认N=10
    TMA = IF(N%2=0,SMA(SMA(REAL,N/2+1),N/2),SMA(SMA(REAL,(N+1)/2),(N+1)/2))

    :param prices:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod + 1)
    assert isinstance(timeperiod, int)
    assert price in ('open', 'close', 'high', 'low')
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    SMA = ta.SMA

    if timeperiod % 2 == 0:
        df_price['TMA'] = SMA(SMA(df_price[price].values.astype(float), timeperiod // 2 + 1),
                              timeperiod // 2)
    else:
        df_price['TMA'] = SMA(SMA(df_price[price].values.astype(float), (timeperiod + 1) // 2),
                              (timeperiod + 1) // 2)
    return df_price['TMA']


def TR(prices):
    """
    47. TR真实波动范围(True Range,TR)
    说明：真实波动范围

    计算方法：
    TR = MAX(HIGH-LOW,ABS(HIGH-CLOSE[1]),ABS(LOW-CLOSE[1]))

    :param prices:
    :return:
    """
    assert prices is not None

    _assert_greater_or_equal(len(prices), 1)
    # assert isinstance(staticmethod, int)
    # assert price in ('open', 'close', 'high', 'low')
    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    high, low, close = df_price[['high', 'low', 'close']].T.values
    close1 = inp.shift(close, 1, cval=np.nan)

    tr = np.max(np.column_stack((high - low, np.abs(high - close1), np.abs(low - close1))), axis=1)

    return pd.Series(tr, index=df_price.index)

    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['TR'] = df_price.apply(
    #     lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close1']), abs(row['low'] - row['close1'])), axis=1
    # )
    # return df_price['TR']


def VIDYA(prices, timeperiod=20):
    """
    54. VIDYA钱得动量摆动平均指数（Chande’s Variable Index Dynamic Average，VIDYA）
    说明：
    VIDYA 由Tushar Chande 创立，在第一个版本中，标准差被用作波动率指数。
    1995 年 10 月，Chande 对 VIDYA 进行了修改，使用新的钱德动量摆动指标 (CMO)作为波动率指数。
    VIDYA钱得动量摆动平均指数是一个统计指标，使用收盘价进行计算，
    它是平滑参数由9个时间段钱德动量摆动指标CMO决定的指数移动平均.当市价突破 VIDYA 上轨时，
    可能表明上升趋势的开始；当市价突破 VIDYA 下轨时，可能表明下降趋势的开始。

    计算方法:默认N=20
    SC=2/(N+1)
    VI=CMO(CLOSE)
    VMA=SC*VI*CLOSE+(1-SC*VI)*CLOSE[1]

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    sc = 2.0 / (timeperiod + 1)
    CMO = ta.CMO

    vi = CMO(df_price['close'].values.astype(float), timeperiod)
    vma = sc * vi * df_price['close'] + \
          (1 - sc * vi) * df_price['close'].shift(1)
    return vma


def TSI(prices, timeperiod1=25, timeperiod2=13):
    """
    32. TSI真实强度指数（True Strength Index，TSI）
    说明：
    真实强弱指数（True Strength Index，TSI）是相对强弱指数 (RSI) 的变体。TSI 使用价格动量的双重平滑指数移动平均线，
    剔除价格的震荡变化并发现趋势的变化。TSI 可帮助判断市场趋势。TSI 线上扬表示上升趋势。反之，TSI 线下挫表示下跌趋势。

    计算方法:
    MOM=CLOSE-CLOSE[1]
    TSI=EMA(EMA(MOM,N),M)/EMA(EMA(ABS(MOM),N),M)*100
    N为平滑价格动量的周期数，通常取25，M为平滑已平滑动量的周期数，通常取13.

    :param prices:
    :param timeperiod1: N
    :param timeperiod2: M
    :return:
    """
    assert prices is not None
    timeperiod = timeperiod1 + timeperiod2
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)

    df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    close = df_price['close'].values.astype(float)

    mom = close = inp.shift(close, 1, cval=np.nan)
    EMA = ta.EMA

    tsi = EMA(EMA(mom, timeperiod1), timeperiod2) / EMA(EMA(np.abs(mom), timeperiod1), timeperiod2) * 100
    inf_idx = np.isinf(tsi)
    tsi[inf_idx] = 0.0
    return pd.Series(tsi, index=df_price.index)


    # df_price['mom'] = df_price['close'] - df_price['close'].shift(1)
    #
    # EMA = ta.EMA
    # df_price['EMA_EMA_MOM'] = EMA(EMA(df_price['mom'].values.astype(float), timeperiod1), timeperiod2)
    # df_price['EMA_EMA_ABS_MOM'] = EMA(EMA(df_price['mom'].abs().values.astype(float), timeperiod1), timeperiod2)
    # df_price['TSI'] = df_price.apply(
    #     lambda row: row['EMA_EMA_MOM'] / row['EMA_EMA_ABS_MOM'] * 100 if not np.isclose(row['EMA_EMA_ABS_MOM'], 0) else 0, axis=1
    # )
    # # tsi = EMA(EMA(df_price['mom'].values.astype(float), timeperiod1), timeperiod2) / \
    # #     EMA(EMA(df_price['mom'].abs().values.astype(float), timeperiod1), timeperiod2) * 100
    # # df_price['tsi'] = tsi
    # return df_price['TSI']


def UI(prices, timeperiod=14):
    """
    说明：
    Ulcer Index是由Peter Martin 1987年提出的测度证券市场风险的波动率指标，
    、先求出过去n日每日收盘价相对于n日内最高价的变动率R, R平方的平均值开二次方后得到Ulcer指标。

    计算方法:
    R=(CLOSE-MAX(CLOSE,N))/MAX(CLOSE,N)*100
    UI=SQRT(SUMSQ(R,N)/N)

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)
    # assert isinstance(timeperiod2, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    max_close_n = pd.rolling_max(df_price['close'], timeperiod)
    r = (df_price['close'] - max_close_n) / max_close_n * 100
    ui = pd.rolling_sum(r ** 2, timeperiod) / timeperiod
    return ui


def VAMA(prices, timeperiod=20, price='close'):
    """
    37. VAMA交易量调整的移动平均
    说明：

    计算方法:
    VAMA=SUM(CLOSE*VOL,N)/SUM(VOL,N)

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)
    assert price in ('high', 'low', 'open', 'close')
    # assert isinstance(timeperiod2, int)

    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)
    SMA = ta.SMA
    df_price['VAMA'] = \
        SMA((df_price[price] * df_price['volume']).values.astype(float), timeperiod) / \
        SMA(df_price['volume'].values.astype(float), timeperiod)
    return df_price['VAMA']


def VHF(prices, timeperiod=20):
    """
    说明：
    垂直水平过滤指标（VHF）确定价格是处于趋势阶段还是整理阶段。
    该指标首次由亚当.怀特（Adam White）1991年提出。MACD和移动平均线在趋势性市场中表现佳，
    摆动指标如RSI、随机摆动指标在波动时表现佳，
    VHF指标试图确定价格运动的“趋势性”以帮助你决定采用哪种指标。
    （1） VHF指标值越高，价格的趋势性程度越高，采用趋势跟踪指标的效果会更好；
    （2） 上升的VHF表示正在发展的趋势，下跌的VHF表示价格可能进入整理阶段；
    （3） 预期高VHF指标值后将是整理阶段，预期低VHF指标值后将是趋势阶段。

    计算方法:
    HCP=MAX(HIGH,N)
    LCP=MIN(LOW,N)
    A=ABS(HCP-LCP)
    B=SUM(ABS(CLOSE-CLOSE[1]),N)
    VHF=A/B
    N默认为5

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)


    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    high, low, close = df_price[['high', 'low', 'close']].T.values

    MAX = ta.MAX
    MIN = ta.MIN
    SUM = ta.SUM

    hcp = MAX(high, timeperiod)
    lcp = MIN(low, timeperiod)
    a = np.abs(hcp - lcp)
    b = SUM(np.abs(close - inp.shift(close, 1, cval=np.nan)), timeperiod)

    vhf = a / b
    vhf[np.isinf(vhf)] = 0.0
    return pd.Series(vhf, index=df_price.index)

    # hcp = pd.rolling_max(df_price['high'], timeperiod)
    # lcp = pd.rolling_min(df_price['low'], timeperiod)
    # df_price['A'] = (hcp - lcp).abs()
    # abs_colse_close1 = (df_price['close'] - df_price['close'].shift(1)).abs()
    # df_price['B'] = pd.rolling_sum(abs_colse_close1, timeperiod)
    # df_price['VHF'] = df_price.apply(
    #     lambda row: row['A'] / row['B'] if not np.isclose(row['B'], 0) else 0, axis=1
    # )
    # return df_price['VHF']


def VMACD(prices, timeperiod1=12, timeperiod2=26, timeperiod3=9):
    """
    39. VMACD量指数平滑异同移动平均线
    （Vol Moving Average Convergence and Divergence，VMACD）
    说明：
    量平滑异同移动平均线（VMACD）用于衡量量能的发展趋势，属于量能引趋向指标。
    MACD称为指数平滑异同平均线。分析的数学公式都是一样的，只是分析的物理量不同。
    VMACD对成交量VOL进行分析计算，而MACD对收盘价CLOSE进行分析计算。
    计算方法:
    SHORT=EMA(VOL,N1)
    LONG=EMA(VOL,N2)
    DIFF=SHORT-LONG
    DEA=EMA(DIFF,M)
    VMACD=DIFF-DEA
    通常N1=12，N2=26，M=9

    :param prices:
    :param timeperiod1: N1
    :param timeperiod2: N2
    :param timeperiod3: N3
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2, timeperiod3)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    assert isinstance(timeperiod3, int)
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    EMA = ta.EMA
    short = EMA(df_price['volume'].values.astype(float), timeperiod1)
    long = EMA(df_price['volume'].values.astype(float), timeperiod2)
    diff = short - long
    dea = EMA(diff, timeperiod3)
    vmacd = diff - dea
    df_price['VMACD'] = vmacd
    return df_price['VMACD']


def VO(prices, timeperiod1=2, timeperiod2=5):
    """
    40. VO成交量摆动指标（Volume Oscillator，VO）
    说明：
    成交量摆动指标显示证券成交量的两个移动均值之差，以点或百分比形式表示。
    摆动指标升至零以上，是短期成交量移动均线升至长期成交量移动均线之上的信号，
    因此短期成交量趋势高于长期成交量趋势。上涨的价格伴随着放大的成交量是上涨预期（更多的买方）
    增加的信号，这将引导市场继续上升；相反，下跌的价格伴随增加的成交量（更多的卖方）
    是上涨预期减少的信号。
    计算方法:
    (SMA(VOL,2)-SMA(VOL,5))/SMA(VOL,5)*100

    :param prices:
    :param timeperiod1: N1
    :param timeperiod2: N2
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    SMA = ta.SMA
    sma_t1 = SMA(df_price['volume'].values.astype(float), timeperiod1)
    sma_t2 = SMA(df_price['volume'].values.astype(float), timeperiod2)
    df_price['VO'] = (sma_t1 - sma_t2) / sma_t2 * 100
    return df_price['VO']


def VOSC(prices, timeperiod1=12, timeperiod2=26):
    """
    说明：
    VOSC指标又名移动平均成交量指标，但是，它并非仅仅计算成交量的移动平均线，
    而是通过对成交量的长期移动平均线和短期移动平均线之间的比较，
    分析成交量的运行趋势和及时研判趋势转变方向。先分别计算短期移动平均线(SHORT)和
    长期移动平均线(LONG)，然后算两者的差值，再求差值与短期移动平均线(SHORT)的比，
    最后将比值放大100倍，得到VOSC值。
    计算方法:
    SHORT=SUM(VOL,N)/N
    LONG=SUM(VOL,M)/M
    VOSC=(SHORT-LONG)/SHORT*100
    N一般取12天，    M取26天

    :param prices:
    :param timeperiod1: N
    :param timeperiod2: M
    :return:
    """
    assert prices is not None
    timeperiod = max(timeperiod1, timeperiod2)
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod1, int)
    assert isinstance(timeperiod2, int)
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    SUM = pd.rolling_sum
    short = SUM(df_price['volume'], timeperiod1) / timeperiod1
    long = SUM(df_price['volume'], timeperiod2) / timeperiod2
    vosc = (short - long) / short * 100
    return vosc


def VR(prices, timeperiod=26):
    """
    42. VR成交量比率指标（Volume Ratio，VR）
    说明：
    成交量比率（Volume Ratio，VR），是通过分析股价上升日成交量与股价下降日成交量比值，
    从而掌握市场买卖气势的中期技术指标。主要用于个股分析，其理论基础是“量价同步”及“量须先予价”，
    以成交量的变化确认低价和高价，从而确定买卖时法。

    计算方法:
    A=IF(CLOSE>CLOSE[1],VOL,0)
    B=IF(CLOSE<CLOSE[1],VOL,0)
    VR=SUM(A,N)/SUM(B,N)*100
    N多为26日。

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)
    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)

    close, volume = df_price[['close', 'volume']].T.values

    close1 = inp.shift(close, 1, cval=np.nan)

    cond1 = close > close1
    cond2 = close < close1

    a = np.zeros(shape=close1.shape, dtype=float)
    b = a.copy()
    a[cond1] = volume[cond1]
    b[cond2] = volume[cond2]

    SUM = ta.SUM
    vr = SUM(a, timeperiod) / SUM(b, timeperiod) * 100
    vr[np.isinf(vr)] = 0.0
    return pd.Series(vr, index=df_price.index)


    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['A'] = df_price.apply(
    #     lambda row: row['volume'] if row['close'] > row['close1'] else 0, axis=1
    # )
    # df_price['B'] = df_price.apply(
    #     lambda row: row['volume'] if row['close'] < row['close1'] else 0, axis=1
    # )
    #
    # SUM = pd.rolling_sum
    # df_price['SUM_A'] = SUM(df_price['A'], timeperiod)
    # df_price['SUM_B'] = SUM(df_price['B'], timeperiod)
    # df_price['VR'] = df_price.apply(
    #     lambda row: row['SUM_A'] / row['SUM_B'] * 100 if not np.isclose(row['SUM_B'], 0) else 0, axis=1
    # )
    #
    # return df_price['VR']


def VROC(prices, timeperiod=14):
    """
    43. VROC量变动速率指标（Volume Rate-of-Change，ROC）
    说明：
    量变动速率指标的计算是将最近n期内成交量的变化量除以n期前成交量。如果今日成交量大于n期前成交量，那么VROC将为正，
    如果今日成交量小于n期前成交量，那么VROC为负。几乎每一个显著的形态（如顶、底、突破）都伴随着成交量的急剧增加，
    成交量VROC显示了成交量变动的速度。

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    volume_n = df_price['volume'].shift(timeperiod)
    vroc = (df_price['volume'] - volume_n) / volume_n * 100
    return vroc


def VRSI(prices, timeperiod=14):
    """
    44. VRSI量相对强弱指标（Volume Relative Strength Index，VRSI）
    说明：
    VRSI是从RSI（强弱指数）演变出的一种指标，即成交量的相对强弱指数。
    它计算一段时间内价格上升日与下跌日成交量的比值，来反应成交量与价格升跌的关系。
    其原理与RSI和VR相类似。此指标的计算方式同RSI，只是将收盘价改成成交手数，应用方式请参照RSI。

    计算方法:
    U=IF(CLOSE>CLOSE[1],VOL,IF(CLOSE=CLOSE[1],VOL/2，0))
    D=IF(CLOSE<CLOSE[1],VOL, IF(CLOSE=CLOSE[1],VOL/2，0))
    UU=((N-1)U[1]+U)/N
    DD=((N-1)D[1]+D)/N
    VRSI=100*UU/(UU+DD)

    :param prices:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    _assert_greater_or_equal(len(prices), timeperiod)
    assert isinstance(timeperiod, int)
    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    close, volume = df_price[['close', 'volume']].T.values
    close1 = inp.shift(close, 1, cval=np.nan)

    cond1 = close > close1
    cond_e = np.isclose(close, close1)
    cond2 = close < close1

    u = np.empty(shape=close1.shape)
    d = u.copy()

    u[cond1] = volume[cond1]
    u[cond_e] = 0.5 * volume[cond_e]
    u[~(cond1 | cond_e)] = 0.0

    d[cond2] = volume[cond2]
    d[cond_e] = 0.5 * volume[cond_e]
    d[~(cond2 | cond_e)] = 0.0

    uu = ((timeperiod - 1) * inp.shift(u, 1, order=0, cval=np.nan) + u) / timeperiod
    dd = ((timeperiod - 1) * inp.shift(d, 1, order=0, cval=np.nan) + d) / timeperiod

    vrsi = 100 * uu / (uu + dd)

    return pd.Series(vrsi, index=df_price.index)

    # df_price['close1'] = df_price['close'].shift(1)
    #
    # def _U(row):
    #     ret = None
    #     if row['close'] > row['close1']:
    #         ret = row['volume']
    #     elif np.isclose(row['close'], row['close1']):
    #         ret = row['volume'] / 2
    #     else:
    #         ret = 0
    #     return ret
    #
    # def _D(row):
    #     ret = None
    #     if row['close'] < row['close1']:
    #         ret = row['volume']
    #     elif np.isclose(row['close'], row['close1']):
    #         ret = row['volume'] / 2
    #     else:
    #         ret = 0
    #     return ret
    #
    # df_price['U'] = df_price.apply(_U, axis=1)
    # df_price['D'] = df_price.apply(_D, axis=1)
    #
    # UU = ((timeperiod - 1) * df_price['U'].shift(1) + df_price['U']) / timeperiod
    # DD = ((timeperiod - 1) * df_price['D'].shift(1) + df_price['D']) / timeperiod
    # vrsi = UU / (UU + DD) * 100
    # return vrsi


def WC(prices):
    """
    说明：
    加权收盘价的计算是将收盘价乘以2，再加上最高价和最低价，最后除以4，
    结果就是赋予收盘价额外权重的加权收盘价.只显示收盘价的线条图会误导投资者，
    因为其忽视最高价和最低价，通过描绘每天的包含最高价、最低价和收盘价的单个点图，
    加权收盘价图结合了线条图的简单性和条形图的视野。

    计算方法:
    WC=(CLOSE*2+HIGH-LOW)/4

    :param prices:
    :return:
    """
    _assert_not_none(prices)

    _assert_greater_or_equal(len(prices), 0)
    df_price = prices.copy()
    df_price = df_price.sort_index(ascending=True)

    wc = (df_price['close'] * 2 + df_price['high'] - df_price['low']) / 4
    return wc


def WAD(prices):
    """
    48. WAD威廉姆斯累积/派发指标（Williams’s Accumulation/Distribution，WAD)
    说明：
    累积是用于描述被买方控制的市场术语，派发则是描述被卖方控制的市场。
    该指标是拉里.威廉姆斯（Larry Williams）发明，他建议基于背离来运用该指标进行交易。
    当市场创出新高而累积/派发指标未能创出新高时表示证券被派发，卖出；
    当市场创出新低而累积/派发指标未能创出新低时表示证券被积累，买入。

    计算方法:
    TRH=MAX(CLOSE[1],HIGH)
    TRL=MIN(CLOSE[1],LOW)
    A/D=IF(CLOSE>CLOSE[1],CLOSE-TRL,IF(CLOSE<CLOSE[1],CLOSE-TRH,0))
    WAD=CUM(A/D)

    :param price:
    :param timeperiod:
    :return:
    """
    assert prices is not None
    # _assert_greater_or_equal(len(prices), timeperiod)
    # assert isinstance(timeperiod, int)
    # df_price = prices.copy()
    df_price = prices.sort_index(ascending=True)
    high, low, close = df_price[['high', 'low', 'close']].T.values

    close1 = inp.shift(close, 1, cval=np.nan)

    # CUM = np.cumsum

    trh = np.max(np.column_stack((close1, high)), axis=1)

    trl = np.min(np.column_stack((close1, low)), axis=1)

    cond1 = close > close1
    cond2 = close < close1

    a_over_d = np.empty(shape=close1.shape, dtype=float)
    a_over_d[cond1] = (close - trl)[cond1]
    a_over_d[~cond1 & cond2] = (close - trh)[~cond1 & cond2]
    a_over_d[~(cond1 | cond2)] = 0.0

    wad = np.cumsum(a_over_d)

    return pd.Series(wad, index=df_price.index)

    # df_price['close1'] = df_price['close'].shift(1)
    # df_price['TRH'] = df_price.apply(
    #     lambda row: max(row['close1'], row['high']), axis=1)
    # df_price['TRL'] = df_price.apply(
    #     lambda row: min(row['close1'], row['low']), axis=1)
    #
    # def _a_over_d(row):
    #     if row['close'] > row['close1']:
    #         ret = row['close'] - row['TRL']
    #     elif row['close'] < row['close1']:
    #         ret = row['close'] - row['TRH']
    #     else:
    #         ret = 0
    #
    #     return ret
    #
    # df_price['A_OVER_D'] = df_price.apply(_a_over_d, axis=1)
    #
    # wad = df_price['A_OVER_D'].cumsum()
    # return wad


def _assert_not_none(a):
    assert a is not None, str(a) + 'is None'


def _assert_greater_or_equal(a, b):
    assert a >= b, str(a) + ' must be greater than ' + str(b)


__all__ = {
    'ACC',
    'ACD',
    'ADTM',
    'AR',
    'ARC',
    'ASI',
    'BBI',
    'BIAS',
    'BR',
    'CMF',
    'CR',
    'CVI',
    'DBCD',
    'DDI',
    'DMI',
    'DPO',
    'EMV',
    'IMI',
    'KVO',
    'MI',
    'MTM',
    'NVI',
    'PFE',
    'PVI',
    'PVT',
    'QST',
    'RI',
    'RMI',
    'RVI',
    'SMI',
    'SRSI',
    'TMA',
    'TR',
    'TS',
    'TSI',
    'UI',
    'VAMA',
    'VHF',
    'VIDYA',
    'VMACD',
    'VO',
    'VOSC',
    'VR',
    'VROC',
    'VRSI',
    'WAD',
    'WC'
}
if __name__ == '__main__':
    p = pd.read_csv('../orcl-2000.csv', index_col=0, parse_dates=True)
    p.columns = [str.lower(col) for col in p.columns]


    # p = pd.concat([p for _ in xrange(100)], ignore_index=True)


    # test('BR')
    # main()

    def main():
        # p = pd.read_csv('../orcl-2000.csv', index_col=0, parse_dates=True)
        # p.columns = [str.lower(col) for col in p.columns]
        ret = CVI(p)
        print(ret)


        # main()
        # import inspect
        #
        # decorated_func = inspect.getmembers(RMI)
        # print(decorated_func)


    def test(func_name):
        import ls_talib_benchmark as benchmark
        import ls_talib
        import talib
        from time import time
        # import pandas as pd



        df = p
        N = 10
        func1 = benchmark.__getattribute__(func_name)
        t0 = time()
        for idx in range(N):
            ret1 = func1(df)
        t1 = time() - t0
        print('Benchmarks nan is {0}, total time: {1}'.format(ret1.isnull().sum(), t1))

        func2 = ls_talib.__getattribute__(func_name)
        t0 = time()
        for idx in xrange(N):
            ret2 = func2(df)
        t2 = time() - t0
        print('{0} nan is {1}, total time: {2}'.format(func2.__name__, ret2.isnull().sum(), t2))

        assert np.allclose(ret1.dropna(), ret2.dropna())

        t0 = time()
        for idx in xrange(N):
            talib.abstract.CCI(df)
        t3 = time() - t0
        print('talib function total time: {0}'.format(t3))

        print("your func is {0} times faster than Benchmark, is {1} times compared with talib ".format(t1 / t2, t3 / t2))


    def count_function_number():
        import inspect
        import ls_talib
        ret = inspect.getmembers(ls_talib, predicate=inspect.isfunction)
        # print(ret)
        return filter(lambda x: x.isupper(), dict(ret).keys())


    test('RVI')

    # print count_function_number()
    # print len(count_function_number())
