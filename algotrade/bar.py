#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'


class Bar(object):
    # Optimization to reduce memory footprint.
    __slots__ = (
        '__date_time',
        '__open',
        '__close',
        '__high',
        '__low',
        '__volume',
        '__adj_close',
        '__frequency',
        '__use_adjusted_value'
    )

    def __init__(self, date_time=None, open_=None, high=None, low=None, close=None, volume=None, adj_close=None, frequency=None):
        if high < low:
            raise Exception("high < low on %s" % date_time)
        elif high < open_:
            raise Exception("high < open on %s" % date_time)
        elif high < close:
            raise Exception("high < close on %s" % date_time)
        elif low > open_:
            raise Exception("low > open on %s" % date_time)
        elif low > close:
            raise Exception("low > close on %s" % date_time)

        self.__date_time = date_time
        self.__open = open_
        self.__close = close
        self.__high = high
        self.__low = low
        self.__volume = volume
        self.__adj_close = adj_close
        self.__frequency = frequency
        self.__use_adjusted_value = False

    def __repr__(self):
        return 'Bar({date_time},{open},{high},{low},{close},{volume},{adj_close},{frequency})'.format(
            date_time=self.__date_time,
            open=self.__open,
            high=self.__high,
            low=self.__low,
            close=self.__close,
            volume=self.__volume,
            adj_close=self.__adj_close,
            frequency=self.__frequency)

    def __str__(self):
        return '{date_time},{open},{high},{low},{close},{volume},{adj_close},{frequency}'.format(
            date_time=self.__date_time,
            open=self.__open,
            high=self.__high,
            low=self.__low,
            close=self.__close,
            volume=self.__volume,
            adj_close=self.__adj_close,
            frequency=self.__frequency)

    def __setstate__(self, state):
        (self.__date_time,
         self.__open,
         self.__close,
         self.__high,
         self.__low,
         self.__volume,
         self.__adj_close,
         self.__frequency,
         self.__use_adjusted_value) = state

    def __getstate__(self):
        return (
            self.__date_time,
            self.__open,
            self.__close,
            self.__high,
            self.__low,
            self.__volume,
            self.__adj_close,
            self.__frequency,
            self.__use_adjusted_value
        )

    @property
    def date_time(self):
        return self.__date_time

    @property
    def open(self):
        return self.__open

    @property
    def high(self):
        return self.__high

    @property
    def low(self):
        return self.__low

    @property
    def close(self):
        return self.__close

    @property
    def adj_close(self):
        return self.__adj_close

    @property
    def volume(self):
        return self.__volume

    @property
    def frequency(self):
        return self.__frequency

    def values(self):
        return {
            'date_time': self.date_time,
            'open'     : self.open,
            'high'     : self.high,
            'low'      : self.low,
            'close'    : self.close,
            'adj_close': self.adj_close,
            'volume'   : self.volume,
            'frequency': self.frequency
        }
