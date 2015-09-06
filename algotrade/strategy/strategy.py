#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'
import pandas as pd

from algotrade.const import EVENT_TYPE
from algotrade.event_engine import Event


class BaseStrategy:
    def __init__(self, barfeed, broker, ta_factors=None):
        self.__barfeed = barfeed
        self.__broker = broker
        self.__ta_factors = ta_factors
        self.__dataframe = pd.DataFrame()
        self.event_engine = barfeed.event_engine
        self.event_engine.register(EVENT_TYPE.EVENT_BAR_ARRIVE, self.on_bar)

    def run(self):
        self.event_engine.start()
        gen = self.__barfeed.next_bar()
        try:
            while True:
                self.event_engine.put(Event(EVENT_TYPE.EVENT_BAR_ARRIVE, gen.next()))
                self.event_engine.run()
        except StopIteration:
            self.on_finish()

    def on_bar(self, event):
        event_type, msg = event.type_, event.dict_
        self.__append_data_frame(msg.values())
        self.__calc_signals()

    def __calc_signals(self):
        for func_name, param_dict in self.__ta_factors:

    def on_finish(self):
        print(self.__dataframe)

    def __append_data_frame(self, bar):
        # bar_ = {key: [val] for key,val in bar.iteritems()}
        bar_frame = pd.DataFrame([bar], index=[bar['date_time']])
        self.__dataframe = self.__dataframe.append(bar_frame)

    def __update_data_frame(self, minibar):
        self.__dataframe[self.__dataframe.index[-1]] = minibar


if __name__ == '__main__':
    from algotrade.barfeed import CSVBarFeed
    from algotrade.broker import BaseBroker

    barfeed = CSVBarFeed()
    barfeed.load_data_from_csv('../orcl-2000.csv')
    broker = BaseBroker()
    my_strategy = BaseStrategy(barfeed, broker)
    my_strategy.run()
