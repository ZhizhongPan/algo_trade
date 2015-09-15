#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

import pandas as pd
from numpy.testing.utils import (assert_array_almost_equal,assert_allclose)
from algotrade.technical.abstract import Function
import algotrade.technical.ls_talib_benchmark as benchmark
import talib

def test_Function():
    p = pd.read_csv('../../orcl-2000.csv', index_col=[0], parse_dates=True)
    p.columns = [str.lower(col) for col in p.columns]
    func = Function('AROON')
    ret = func(p)
    print('length of ret is {0}, NA number is {1}'.format(len(ret), ret.isnull().sum()))

    # benchmark_ret = benchmark.ACD(p)
    benchmark_ret = talib.abstract.AROON(p)
    print('length of ret is {0}, NA number is {1}'.format(len(benchmark_ret), benchmark_ret.isnull().sum()))

    assert_array_almost_equal(ret.dropna(), benchmark_ret.dropna())
