#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from abc import abstractmethod, ABCMeta
import os

import six
import pandas as pd

from algotrade.bar import Bar
from algotrade.event_engine import Event
from algotrade.event_engine import EventEngineMixin
from algotrade.const import EventType, FREQUENCY


class BaseBarFeed(six.with_metaclass(ABCMeta), EventEngineMixin):
    def __init__(self):
        super(BaseBarFeed, self).__init__()
        self.event_engine = EventEngineMixin.event_engine

    @abstractmethod
    def next_bar(self):
        raise NotImplementedError


class CSVBarFeed(BaseBarFeed):
    def __init__(self, instruments):
        """

        :param instruments: a list of strings, instrument code
        :return:
        """
        super(CSVBarFeed, self).__init__()
        self.__instruments = instruments
        self.__dict_dataframe = dict.fromkeys(instruments)

    def next_bar(self):

        generators = dict([(instrument, self.__dict_dataframe[instrument].iterrows()) for instrument in self.__instruments])
        while True:
            for instrument, gen in generators.items():
                idx, row = gen.next()
                bar_dict = dict(row)
                # bar_dict.update(frequency=FREQUENCY.DAY, date_time=row.name)
                bar_ = Bar(date_time=row.name,
                           open_=bar_dict['open'],
                           high=bar_dict['high'],
                           low=bar_dict['low'],
                           close=bar_dict['close'],
                           adj_close=bar_dict['adj_close'],
                           volume=bar_dict['volume'],
                           frequency=FREQUENCY.DAY,
                           instrument=instrument
                           )
                yield bar_

    def __next_bar(self):
        # try:
        #     bar = self._next_bar()
        # except StopIteration:
        #     pass
        # else:
        #     self.event_engine.put(Event(EventType.EVENT_BAR_ARRIVE, bar))
        #     return bar
        row = self.next_bar().next()
        self.event_engine.put(Event(type_=EventType.EVENT_BAR_ARRIVE, dict_=row))
        # return row

    def __load_data_from_csv(self, path, instrument):
        filename = instrument + '.csv'
        p = os.path.join(os.path.abspath(path), filename)
        print(p)
        self.__dict_dataframe[instrument] = pd.read_csv(p, index_col=0, parse_dates=True).sort()
        self.__dict_dataframe[instrument].columns = self.__dict_dataframe[instrument].columns.str.lower().str.replace(' ', '_')
        print self.__dict_dataframe[instrument].columns

    def load_data_from_csv(self, path):
        [self.__load_data_from_csv(path, instrument) for instrument in self.__instruments]


if __name__ == '__main__':

    csvfeed = CSVBarFeed(['orcl_2000', 'orcl_2000_copy'])
    csvfeed.load_data_from_csv('..\\')
    for item in csvfeed.next_bar():
        print(item)
