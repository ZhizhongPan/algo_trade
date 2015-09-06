#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'
from algotrade.const import EVENT_TYPE
from algotrade.event_engine import Event


class BaseStrategy:
    def __init__(self, barfeed, broker):
        self.__barfeed = barfeed
        self.__broker = broker
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
            pass

    def on_bar(self, event):
        event_type, msg = event.type_, event.dict_
        print(msg)


if __name__ == '__main__':
    from algotrade.barfeed import CSVBarFeed
    from algotrade.broker import BaseBroker

    barfeed = CSVBarFeed()
    barfeed.load_data_from_csv('../orcl-2000.csv')
    broker = BaseBroker()
    my_strategy = BaseStrategy(barfeed, broker)
    my_strategy.run()
