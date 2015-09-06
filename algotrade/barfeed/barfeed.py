#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from abc import abstractmethod, ABCMeta

import six
import pandas as pd

from algotrade.bar import Bar
from algotrade.event_engine import Event
from algotrade.event_engine import EventEngineMixin
from algotrade.const import EVENT_TYPE, FREQUENCY


class BaseBarFeed(six.with_metaclass(ABCMeta), EventEngineMixin):
    def __init__(self):
        super(BaseBarFeed, self).__init__()
        self.event_engine = EventEngineMixin.event_engine

    @abstractmethod
    def next_bar(self):
        raise NotImplementedError


class CSVBarFeed(BaseBarFeed):
    def __init__(self):
        super(CSVBarFeed, self).__init__()
        self.__dataframe = None

    def next_bar(self):
        for idx, row in self.__dataframe.iterrows():
            bar_dict = dict(row)
            bar_dict.update(frequency=FREQUENCY.DAY, date_time=row.name)
            bar_ = Bar(date_time=bar_dict['date_time'],
                       open_=bar_dict['open'],
                       high=bar_dict['high'],
                       low=bar_dict['low'],
                       close=bar_dict['close'],
                       adj_close=bar_dict['adj close'],
                       volume=bar_dict['volume'],
                       frequency=bar_dict['frequency']
                       )
            yield bar_

    def __next_bar(self):
        # try:
        #     bar = self._next_bar()
        # except StopIteration:
        #     pass
        # else:
        #     self.event_engine.put(Event(EVENT_TYPE.EVENT_BAR_ARRIVE, bar))
        #     return bar
        row = self.next_bar().next()
        self.event_engine.put(Event(type_=EVENT_TYPE.EVENT_BAR_ARRIVE, dict_=row))
        # return row

    def load_data_from_csv(self, path):
        self.__dataframe = pd.read_csv(path, index_col=0, parse_dates=True).sort()
        self.__dataframe.columns = [str.lower(col_names) for col_names in self.__dataframe.columns]


if __name__ == '__main__':

    csvfeed = CSVBarFeed()
    csvfeed.load_data_from_csv('../orcl-2000.csv')
    for item in csvfeed.next_bar():
        print(item)
