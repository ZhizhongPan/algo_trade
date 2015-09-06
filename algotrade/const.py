#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from enum import Enum, enum

EVENT_TYPE = Enum(
    'EVENT_TYPE',

    [  # 系统相关
       'EVENT_TIMER',  # 计时器事件，每隔1秒发送一次
       'EVENT_LOG',  # 日志事件，全局通用

       # Gateway相关
       'EVENT_TICK',                 # TICK行情事件，可后接具体的vtSymbol
       'EVENT_TRADE',                # 成交回报事件
       'EVENT_ORDER',                # 报单回报事件
       'EVENT_POSITION',             # 持仓回报事件
       'EVENT_ACCOUNT',              # 账户回报事件
       'EVENT_ERROR'                 # 错误回报事件
       'EVENT_MARKET_DATA',          # 常规行情事件
       'EVENT_BAR_ARRIVE',           # Phil定义
       'EVENT_MARKET_DATA_CONTRACT',  # 特定合约行情事件
       'EVENT_INVESTOR',             # 投资者查询回报
       'EVENT_INSTRUMENT',           # 合约查询回报
       'EVENT_ORDER_ORDER_REF',      # 特定合约行情事件
       'EVENT_TD_LOGIN',
    ]
)


class FREQUENCY(Enum):
    SECOND = 1,
    MINUTE = 60,
    HOUR = 60 * 60,
    DAY = 24 * 60 * 60,
    WEEK = 24 * 60 * 60 * 7,
    MONTH = 24 * 60 * 60 * 31
