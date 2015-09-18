#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from enum import Enum


# EventType = Enum(
#     'EventType',
#
#     [  # 系统相关
#        'EVENT_TIMER',  # 计时器事件，每隔1秒发送一次
#        'EVENT_LOG',  # 日志事件，全局通用
#
#        # Gateway相关
#        'EVENT_TICK',                 # TICK行情事件，可后接具体的vtSymbol
#        'EVENT_TRADE',                # 成交回报事件
#        'EVENT_ORDER',                # 报单回报事件
#        'EVENT_POSITION',             # 持仓回报事件
#        'EVENT_ACCOUNT',              # 账户回报事件
#        'EVENT_ERROR'                 # 错误回报事件
#        'EVENT_MARKET_DATA',          # 常规行情事件
#        'EVENT_BAR_ARRIVE',           # Phil定义
#        'EVENT_MARKET_DATA_CONTRACT',  # 特定合约行情事件
#        'EVENT_INVESTOR',             # 投资者查询回报
#        'EVENT_INSTRUMENT',           # 合约查询回报
#        'EVENT_ORDER_ORDER_REF',      # 特定合约行情事件
#        'EVENT_TD_LOGIN',
#     ]
# )

class EventType(Enum):
    # 系统相关
    EVENT_TIMER = 1  # 计时器事件，每隔1秒发送一次
    EVENT_LOG = 2  # 日志事件，全局通用
    # Gateway相关
    EVENT_TICK = 3  # TICK行情事件，可后接具体的vtSymbol
    EVENT_TRADE = 4  # 成交回报事件
    EVENT_ORDER = 5  # 报单回报事件
    EVENT_POSITION = 6  # 持仓回报事件
    EVENT_ACCOUNT = 7  # 账户回报事件
    EVENT_ERROR = 8  # 错误回报事件
    EVENT_MARKET_DATA = 9  # 常规行情事件
    EVENT_BAR_ARRIVE = 10  # Phil定义
    EVENT_MARKET_DATA_CONTRACT = 11  # 特定合约行情事件
    EVENT_INVESTOR = 12  # 投资者查询回报
    EVENT_INSTRUMENT = 13  # 合约查询回报
    EVENT_ORDER_ORDER_REF = 14  # 特定合约行情事件
    EVENT_TD_LOGIN = 15


class FREQUENCY(Enum):
    SECOND = 1,
    MINUTE = 60,
    HOUR = 60 * 60,
    DAY = 24 * 60 * 60,
    WEEK = 24 * 60 * 60 * 7,
    MONTH = 24 * 60 * 60 * 31


# class OrderStatus(Enum):
#     ACCEPTED = 1  # BaseOrder has been acknowledged by the broker.
#     CANCELED = 2  # BaseOrder has been canceled.
#     PARTIALLY_FILLED = 3  # BaseOrder has been partially filled.
#     FILLED = 4  # BaseOrder has been completely filled.


class OrderAction(Enum):
    BUY = 1
    BUY_TO_COVER = 2
    SELL = 3
    SELL_SHORT = 4


class OrderStatus(Enum):
    INITIAL = 1  # Initial state.
    SUBMITTED = 2  # BaseOrder has been submitted.
    ACCEPTED = 3  # BaseOrder has been acknowledged by the broker.
    CANCELED = 4  # BaseOrder has been canceled.
    PARTIALLY_FILLED = 5  # BaseOrder has been partially filled.
    FILLED = 6  # BaseOrder has been completely filled.


class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4
