#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from abc import ABCMeta

import six

from algotrade.event_engine import EventEngineMixin


class BaseBroker(six.with_metaclass(ABCMeta), EventEngineMixin):
    def __init__(self):
        self.event_engine = EventEngineMixin.event_engine
