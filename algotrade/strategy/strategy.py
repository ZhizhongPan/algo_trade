#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect

import pandas as pd
import concurrent.futures
import numpy as np
from numpy.testing.utils import assert_almost_equal

import algotrade.technical as lsta
from algotrade.const import EventType
from algotrade.event_engine import Event
from algotrade.technical.utils import (get_ta_functions, get_default_args, num_bars_to_accumulate)
from algotrade.barfeed import CSVBarFeed
from algotrade.broker import BackTestingBroker
from algotrade import const

__author__ = 'phil.zhang'


def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''

    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class BaseStrategy:
    def __init__(self, barfeed, broker, ta_factors=None):
        self.barfeed = barfeed
        self.broker = broker
        self.__ta_factors = ta_factors
        self.dataframe = pd.DataFrame()
        self.event_engine = barfeed.event_engine
        self.event_engine.register(EventType.EVENT_BAR_ARRIVE, self.on_bar)
        #: TODO: 添加talib里面的函数
        self.func_lib = merge_dicts(inspect.getmembers(lsta, inspect.isfunction), get_ta_functions())

    def run(self):
        self.event_engine.start()
        gen = self.__barfeed.next_bar()
        try:
            while True:
                self.event_engine.put(
                    Event(EventType.EVENT_BAR_ARRIVE, gen.next()))
                self.event_engine.run()
        except StopIteration:
            self.on_finish()

    def on_bar(self, event):
        event_type, msg = event.type_, event.dict_
        self.__append_data_frame(msg.values())
        self.__calc_signals()
        # self.on_finish()

    def __calc_signals(self):
        for func_name, param_dict in self.__ta_factors:

            try:
                func = self.func_lib[func_name]
            except KeyError as e:
                raise e

            if not param_dict and func_name in dict(inspect.getmembers(lsta, inspect.isfunction)):
                param_dict = get_default_args(func)

            if func_name in dict(inspect.getmembers(lsta, inspect.isfunction)):
                max_period = num_bars_to_accumulate(func_name=func_name, **param_dict)
            else:
                max_period = func.lookback + 1
            if len(self.dataframe) < max_period:
                continue
            else:
                try:
                    # TODO: 这里需要修改，合成一句
                    ret = func(self.dataframe.iloc[-max_period:], **param_dict)
                    if len(ret.shape) > 1:
                        ret = ret.ix[:, 0]
                    ind = self.dataframe.index[-1]
                    print len(self.dataframe), func_name, ind, ret[-1]
                    self.dataframe.ix[ind, func_name] = ret[-1]
                except IndexError as e:
                    print('__name={0},ind={1}'.format(func_name, ind))
                    raise e(func_name + ' ' + ind)

    def __calc_signals_concurrency(self):
        """
        deprecated: 速度慢
        :return:
        """
        futures = set()
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for func_name, param_dict in self.__ta_factors:
                try:
                    func = self.func_lib[func_name]
                except KeyError as e:
                    raise e

                if not param_dict and func_name in inspect.getmembers(lsta, inspect.isfunction):
                    param_dict = get_default_args(func)

                if func_name in inspect.getmembers(lsta, inspect.isfunction):
                    max_period = num_bars_to_accumulate(func_name=func_name, **param_dict)
                else:
                    max_period = func.lookback + 1
                if len(self.dataframe) < max_period:
                    continue
                else:
                    future = executor.submit(func, self.dataframe.tail(max_period), **param_dict)
                    futures.add(future)

        for future in concurrent.futures.as_completed(futures):
            ret = future.result()
            if len(ret.shape) > 1:
                ret = ret.ix[:, 0]
            ind = self.dataframe.index[-1]
            print len(self.dataframe), func_name, ind, ret[-1]
            self.dataframe.ix[ind, func_name] = ret[-1]

    def on_finish(self):
        print(self.dataframe)

    def __append_data_frame(self, bar):
        # bar_ = {key: [val] for key,val in bar.iteritems()}
        bar_frame = pd.DataFrame([bar], index=[bar['date_time']])
        self.dataframe = self.dataframe.append(bar_frame)

    def __update_data_frame(self, minibar):
        self.dataframe[self.dataframe.index[-1]] = minibar

    def limit_order(self, instrument, limit_price, quantity, good_till_canceled=False, all_or_none=False):
        """Submits a limit order.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param limit_price: Limit price.
        :type limit_price: float.
        :param quantity: The amount of shares. Positive means buy, negative means sell.
        :type quantity: int/float.
        :param good_till_canceled: True if the order is good till canceled. If False then the order gets automatically canceled when the session closes.
        :type good_till_canceled: boolean.
        :param all_or_none: True if the order should be completely filled or not at all.
        :type all_or_none: boolean.
        :rtype: The :class:`chinascope_algotrade.broker.LimitOrder` submitted.
        """

        ret = None
        if quantity > 0:
            ret = self.broker.create_limit_order(const.OrderAction.BUY, instrument, limit_price, quantity)
        elif quantity < 0:
            ret = self.broker.create_limit_order(const.OrderAction.SELL, instrument, limit_price, quantity * -1)
        if ret:
            ret.setGoodTillCanceled(good_till_canceled)
            ret.setAllOrNone(all_or_none)
            self.broker.submit_order(ret)
        return ret

    def stop_order(self, instrument, stop_price, quantity, good_till_canceled=False, all_or_none=False):
        """Submits a stop order.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param stop_price: Stop price.
        :type stop_price: float.
        :param quantity: The amount of shares. Positive means buy, negative means sell.
        :type quantity: int/float.
        :param good_till_canceled: True if the order is good till canceled. If False then the order gets automatically canceled when the session closes.
        :type good_till_canceled: boolean.
        :param all_or_none: True if the order should be completely filled or not at all.
        :type all_or_none: boolean.
        :rtype: The :class:`chinascope_algotrade.broker.StopOrder` submitted.
        """

        ret = None
        if quantity > 0:
            ret = self.broker.create_stop_order(const.OrderAction.BUY, instrument, stop_price, quantity)
        elif quantity < 0:
            ret = self.broker.create_stop_order(const.OrderAction.SELL, instrument, stop_price, quantity * -1)
        if ret:
            ret.setGoodTillCanceled(good_till_canceled)
            ret.setAllOrNone(all_or_none)
            self.broker.submit_order(ret)
        return ret

    def stop_limit_order(self, instrument, stop_price, limit_price, quantity, good_till_canceled=False, all_or_none=False):
        """Submits a stop limit order.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param stop_price: Stop price.
        :type stop_price: float.
        :param limit_price: Limit price.
        :type limit_price: float.
        :param quantity: The amount of shares. Positive means buy, negative means sell.
        :type quantity: int/float.
        :param good_till_canceled: True if the order is good till canceled. If False then the order gets automatically canceled when the session closes.
        :type good_till_canceled: boolean.
        :param all_or_none: True if the order should be completely filled or not at all.
        :type all_or_none: boolean.
        :rtype: The :class:`chinascope_algotrade.broker.StopLimitOrder` submitted.
        """

        ret = None
        if quantity > 0:
            ret = self.broker.create_stop_limit_order(const.OrderAction.BUY, instrument, stop_price, limit_price, quantity)
        elif quantity < 0:
            ret = self.broker.create_stop_limit_order(const.OrderAction.SELL, instrument, stop_price, limit_price, quantity * -1)
        if ret:
            ret.setGoodTillCanceled(good_till_canceled)
            ret.setAllOrNone(all_or_none)
            self.broker.submit_order(ret)
        return ret

    def market_order(self, instrument, quantity, on_close=False, good_till_canceled=False, all_or_none=False):
        """Submits a market order.

        :param instrument: Instrument identifier.
        :type instrument: string.
        :param quantity: The amount of shares. Positive means buy, negative means sell.
        :type quantity: int/float.
        :param on_close: True if the order should be filled as close to the closing price as possible (Market-On-Close order). Default is False.
        :type on_close: boolean.
        :param good_till_canceled: True if the order is good till canceled. If False then the order gets automatically canceled when the session closes.
        :type good_till_canceled: boolean.
        :param all_or_none: True if the order should be completely filled or not at all.
        :type all_or_none: boolean.
        :rtype: The :class:`chinascope_algotrade.broker.MarketOrder` submitted.
        """

        ret = None
        if quantity > 0:
            ret = self.broker.create_market_order(const.OrderAction.BUY, instrument, quantity, on_close)
        elif quantity < 0:
            ret = self.broker.create_market_order(const.OrderAction.SELL, instrument, quantity * -1, on_close)
        if ret:
            ret.setGoodTillCanceled(good_till_canceled)
            ret.setAllOrNone(all_or_none)
            self.broker.submit_order(ret)
        return ret

    def enter_long(self):
        pass

    def enter_short(self):
        pass


class FixedPeriodStrategy(BaseStrategy):
    def __init__(self, barfeed=None, broker=None, initial_code_list=None):
        super(FixedPeriodStrategy).__init__(barfeed, broker, tafactors=None)
        self.previous_code_list = set(initial_code_list.keys())
        self.current_code_list = set(initial_code_list.keys())
        self.to_sell = set()
        self.to_buy = set()
        # self.event_engine.register

    def on_code_list_arrive(self, code_list_dict):
        """

        :param code_list: a dict,containing codes to buy,and its weights
        :return:
        """
        code_list = set(code_list_dict.keys())

        assert_almost_equal(1, np.sum(code_list_dict.values()))

        self.previous_code_list = self.current_code_list
        self.current_code_list = code_list
        codes_to_sell = self.previous_code_list - self.current_code_list
        codes_to_buy = self.current_code_list - self.previous_code_list
        codes_not_changed = self.previous_code_list & self.current_code_list

        total = self.broker.total

        for code in codes_not_changed:
            share = self.broker.shares[code]

            # share 比例过大，sell
            if share / total > code_list_dict[code]:
                self.to_sell.add(
                    self.broker.create_market_order(action=const.OrderAction.SELL, instrument=code, quantity=share - code_list_dict[code] * total))
            # 买进
            elif share / total < code_list_dict[code]:
                self.to_buy.add(
                    self.broker.create_market_order(action=const.OrderAction.BUY, instrument=code, quantity=code_list_dict[code] * total - share))

        for code in codes_to_sell:
            self.to_sell.add(
                self.broker.create_market_order(action=const.OrderAction.SELL, instrument=code, quantity=self.broker.shares[code]))

    def on_bar(self, event):
        type_, msg = event
        print type_, event


def main():
    ta_fac1 = [
        ('ACD', {'timeperiod': 14}),
        ('ACC', {}),
        ('CVI', {'timeperiod': 30}),
        ('BBI', {}),
        ('DBCD', {}),
        ('SMI', {}),
        ('RI', {})
    ]
    ta_fac2 = [
        ('AD', {}),
        ('ADOSC', {}),
        ('ADX', {}),
        ('AROON', {}),
        ('AROONOSC', {}),
        ('ULTOSC', {}),
        ('OBV', {})
    ]
    barfeed = CSVBarFeed(['orcl_2000', 'orcl_2000_copy'])
    barfeed.load_data_from_csv('../')
    broker = BackTestingBroker(cash=10000, barfeed=barfeed)
    my_strategy = FixedPeriodStrategy(barfeed=barfeed, broker=broker)
    import time
    t = time.time()
    my_strategy.run()
    print('total time {}'.format(time.time() - t))


if __name__ == '__main__':
    # ret = inspect.getmembers(lsta, inspect.isfunction)
    # print(ret)
    main()
