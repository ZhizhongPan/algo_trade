�������Լ���д�ļ���ָ���talib�ļ��ɵ���һ���ṩ��talibһ�µĵ��÷�����

���������룺
param: dataframe,����open,high,low,close,volume�ֶΣ�ע���ֶ�������Сд
timeperiod:
timeperiod1, timeperiod2,... : ����������ͬ����Щ����ֻ��һ��dataframe��������Щ����
                               timeperiod��������Щ�����Ĳ�������timeperiod1,timeperiod2
                               
returns: DataSeries,talib�е�ĳЩ��������һ��dataframe

�������ŵ����ڿ��Ժܷ���Ļ��lookback

ʹ�÷�����
import technical as ta

1.
    acd = ta.ACD # �Լ���д�ĺ���

    acd.lookback # ʹ��Ĭ�ϵĲ����õ���lookback
    # 14, Ĭ��timeperiod = 14

    acd.parameters = {'timeperiod':50}

    acd.lookack
    # 50


