本包把自己编写的技术指标和talib的集成到了一起。提供和talib一致的调用方法。

函数的输入：
param: dataframe,包含open,high,low,close,volume字段，注意字段名称是小写
timeperiod:
timeperiod1, timeperiod2,... : 各函数均不同，有些函数只有一个dataframe参数，有些包含
                               timeperiod参数，有些函数的参数包含timeperiod1,timeperiod2
                               
returns: DataSeries,talib中的某些函数返回一个dataframe

本包的优点在于可以很方便的获得lookback

使用方法：
import technical as ta

1.
    acd = ta.ACD # 自己编写的函数

    acd.lookback # 使用默认的参数得到的lookback
    # 14, 默认timeperiod = 14

    acd.parameters = {'timeperiod':50}

    acd.lookack
    # 50


