#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

import inspect

import talib.abstract as ab
import numpy as np

import ls_talib

_LS_FUNCTION_NAMES = set(ls_talib.__all__)


# TODO: 遇到问题：如果用jit修饰后，无法用inspect获得默认参数

class Function(ab.Function):
    def __init__(self, func_name, *args, **kwargs):
        """

        :type kwargs: object
        """
        self.__name = func_name.upper()
        self.__parameters = {}
        # self.__opt_inputs = OrderedDict()
        # self.__info = None

        if self.__name in _LS_FUNCTION_NAMES:
            pass
            # self.parameters = {}
        else:
            super(Function, self).__init__(func_name, *args, **kwargs)

        if kwargs:
            self.parameters = kwargs
            # raise Exception('%s not supported by LS_TA-LIB.' % self.__name)
            # self.set_function_args(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if not self.parameters:
            self.parameters.update(**kwargs)

        if self.__name in _LS_FUNCTION_NAMES:
            func = ls_talib.__getattribute__(self.__name)
            return func(*args, **kwargs)
        else:
            return super(Function, self).__call__(*args, **kwargs)

    @property
    def parameters(self):
        if self.__name in _LS_FUNCTION_NAMES:
            return self.__parameters
        else:
            return super(Function, self).parameters

    @parameters.setter
    def parameters(self, parameters):
        if self.__name in _LS_FUNCTION_NAMES:
            self.__parameters.update(parameters)
        else:
            super(Function, self).set_parameters(parameters)

    @property
    def lookback(self):
        if self.__name in _LS_FUNCTION_NAMES:
            kwargs = self.parameters if self.parameters else self.__get_default_args(self.__name)
            return self.__lookback(self.__name, **kwargs)
        else:
            return super(Function, self).lookback

    @staticmethod
    def __lookback(func_name, timeperiod=np.nan, timeperiod1=np.nan, timeperiod2=np.nan, timeperiod3=np.nan, timeperiod4=np.nan):
        tables = {
            # =====================================0 个周期
            'WC'   : 0,
            'EMV'  : 1,
            'PVT'  : 1,
            'TR'   : 1,
            'PVI'  : 0,
            'NVI'  : 0,
            # =====================================1 个周期
            'ACC'  : timeperiod * 2,
            'ACD'  : timeperiod,
            'ADTM' : timeperiod,
            'AR'   : timeperiod - 1,
            'ARC'  : timeperiod * 2,
            'ASI'  : timeperiod,
            'BIAS' : timeperiod - 1,
            'BR'   : timeperiod,
            'CMF'  : timeperiod - 1,
            'CR'   : timeperiod,
            'CVI'  : 2 * timeperiod - 1,
            'DDI'  : timeperiod,
            'DPO'  : timeperiod - 1,
            'IMI'  : timeperiod - 1,
            'MI'   : (timeperiod - 1) * 3,
            'MTM'  : timeperiod,
            'QST'  : timeperiod - 1,
            'TMA'  : timeperiod - 1,
            'TS'   : timeperiod - 1,
            'UI'   : timeperiod * 2 - 2,
            # 'UPN'  : None,
            'VAMA' : timeperiod - 1,
            'VHF'  : timeperiod,
            'VIDYA': timeperiod,
            'VR'   : timeperiod - 1,
            'VROC' : timeperiod,
            'VRSI' : 1,
            'WAD'  : 0,
            # =============================================== 4 个周期
            'BBI'  : max(timeperiod1, timeperiod2, timeperiod3, timeperiod4) - 1,
            # =============================================== 3 个周期
            'SMI'  : timeperiod1 + timeperiod2 + timeperiod3 - 3,
            'VMACD': max(timeperiod1, timeperiod2) + timeperiod3 - 2,
            'DBCD' : timeperiod1 + timeperiod2 + timeperiod3 - 2,
            'KVO'  : max(timeperiod1, timeperiod2) + timeperiod3 - 2,
            # ============================================== 2 个周期
            'VOSC' : max(timeperiod1, timeperiod2) - 1,
            'RVI'  : timeperiod1 + timeperiod2,
            'TSI'  : timeperiod1 + timeperiod2 - 1,
            'SRSI' : timeperiod1 + timeperiod2 - 1,
            'DMI'  : timeperiod1 + timeperiod2 - 2,
            'RI'   : timeperiod1 + timeperiod2 - 1,
            'VO'   : max(timeperiod1, timeperiod2) - 1,
            'RMI'  : timeperiod1 + timeperiod2,
            'PFE'  : timeperiod1 + timeperiod2 - 1
        }

        # print len(tables)
        return tables[func_name]

    @property
    def default_args(self):
        return self.__get_default_args(self.__name)

    def __get_default_args(self, func_name):
        """
        returns a dictionary of arg_name:default_values for the input function
        """
        func = ls_talib.__getattribute__(func_name)
        args, varargs, keywords, defaults = inspect.getargspec(func)
        # print(func.__name__)
        # print('args={0},varargs={1},keywords={2},defaults={3}'.format(
        #     args, varargs, keywords, defaults))
        if defaults:
            ret = dict(zip(reversed(args), reversed(defaults)))
            # 本来函数的default应该是周期，是整型.
            # 比如ret={'timeperiod1': 14, timeperiod2: 20}
            # 但是有一些函数的缺省值是字符串。这些函数
            # 是为了方便，可以使用不同的price来计算.
            # 比如TMA(prices, timeperiod=14, price='high')
            # 我们要去掉这些字符型的字典项
            numeric_value_dict = {
                key: val for key, val in ret.iteritems() if isinstance(val, int)}
            return numeric_value_dict
        else:
            # print func_name
            return {}


def test_ls_talib():
    for func_name in _LS_FUNCTION_NAMES:
        dict_param = dict(
            timeperiod=np.random.randint(10, 100, 1)[0],
            timeperiod1=np.random.randint(10, 100, 1)[0],
            timeperiod2=np.random.randint(10, 100, 1)[0],
            timeperiod3=np.random.randint(10, 100, 1)[0],
            timeperiod4=np.random.randint(10, 100, 1)[0]
            # timeperiod1 = np.random.randint(10,100,1),
        )
        func = Function(func_name, **dict_param)
        lookback = func.lookback
        default_args = func.default_args
        real_args = default_args.copy()
        for key, val in real_args.iteritems():
            real_args[key] = dict_param[key]

        NAs = func(p, **real_args).isnull().sum()
        print(func_name)
        print(dict_param)
        print('lookback={0}'.format(lookback))
        print('number of NA={0}'.format(NAs))
        assert lookback == NAs


def test_talib():
    func_names = filter(str.isupper, dir(ab))
    func_names = filter(lambda x: not x.startswith('_'), func_names)
    print func_names
    ad = Function('ADOSC')
    param = {'fastperiod': 20}
    ad.parameters = param
    ret = ad(p)
    print(dir(ad))
    print(ret)
    print ad.lookback
    # print(len(func_names))


for name in _LS_FUNCTION_NAMES:
    exec "%s = Function('%s')" % (name, name)

__all__ = ['Function'] + list(_LS_FUNCTION_NAMES)

if __name__ == '__main__':

    aroon = Function('AROON')

    import pandas as pd

    p = pd.read_csv('../orcl-2000.csv', index_col=0, parse_dates=True)
    p.columns = [str.lower(col) for col in p.columns]
    p = p.astype(float)

    # print(rmi(p))
    test_ls_talib()
    # test_talib()
