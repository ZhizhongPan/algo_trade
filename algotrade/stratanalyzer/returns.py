# chinascope_algotrade
#
# Copyright 2011-2015 Gabriel Martin Becedillas Ruiz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
.. moduleauthor:: Gabriel Martin Becedillas Ruiz <gabriel.becedillas@gmail.com>
"""

import math

from algotrade import stratanalyzer
# from algotrade import observer
# from algotrade import dataseries


# Helper class to calculate time-weighted returns in a portfolio.
# Check http://www.wikinvest.com/wiki/Time-weighted_return
class TimeWeightedReturns(object):
    def __init__(self, initialValue):
        self.__lastValue = initialValue
        self.__flows = 0.0
        self.__lastPeriodRet = 0.0
        self.__cumRet = 0.0

    def deposit(self, amount):
        self.__flows += amount

    def withdraw(self, amount):
        self.__flows -= amount
        
    @property
    def current_value(self):
        return self.__lastValue

    # Update the value of the portfolio.
    def update(self, current_value):
        if self.__lastValue:
            ret_sub_period = (current_value - self.__lastValue - self.__flows) / float(self.__lastValue)
        else:
            ret_sub_period = 0.0

        self.__cumRet = (1 + self.__cumRet) * (1 + ret_sub_period) - 1
        self.__lastPeriodRet = ret_sub_period
        self.__lastValue = current_value
        self.__flows = 0.0

    @property
    def last_period_returns(self):
        return self.__lastPeriodRet

    # Note that this value is not annualized.
    @property
    def cumulative_returns(self):
        return self.__cumRet


# Helper class to calculate returns and profit over a single instrument and a single position (not
# the whole portfolio).
class PositionTracker(object):
    def __init__(self, instrument_traits):
        self.__instrument_traits = instrument_traits
        self.reset()

    def reset(self):
        self.__cash = 0.0
        self.__shares = 0
        self.__commissions = 0.0
        self.__costPerShare = 0.0  # Volume weighted average price per share.
        self.__costBasis = 0.0

    def __update(self, quantity, price, commission):
        assert (quantity != 0)

        if self.__shares == 0:
            # Opening a new position
            totalShares = quantity
            self.__costPerShare = price
            self.__costBasis = abs(quantity) * price
        else:
            totalShares = self.__instrument_traits.round_quantity(self.__shares + quantity)
            if totalShares != 0:
                prevDirection = math.copysign(1, self.__shares)
                txnDirection = math.copysign(1, quantity)

                if prevDirection != txnDirection:
                    if abs(quantity) > abs(self.__shares):
                        # Going from long to short or the other way around.
                        # Update costs as a new position being opened.
                        self.__costPerShare = price
                        diff = self.__instrument_traits.round_quantity(self.__shares + quantity)
                        self.__costBasis = abs(diff) * price
                    else:
                        # Reducing the position.
                        pass
                else:
                    # Increasing the current position.
                    # Calculate a volume weighted average price per share.
                    prevCost = self.__costPerShare * self.__shares
                    txnCost = quantity * price
                    self.__costPerShare = (prevCost + txnCost) / totalShares
                    self.__costBasis += abs(quantity) * price
            else:
                # Closing the position.
                self.__costPerShare = 0.0

        self.__cash += price * quantity * -1
        self.__commissions += commission
        self.__shares = totalShares

    def cash(self):
        return self.__cash - self.__commissions

    def shares(self):
        return self.__shares

    def getCostPerShare(self):
        # Returns the weighted average cost per share for the open position.
        return self.__costPerShare

    def commissions(self):
        return self.__commissions

    def getCostBasis(self):
        return self.__costBasis

    def getNetProfit(self, price=None, includeCommissions=True):
        ret = self.__cash
        if price is None:
            price = self.__costPerShare
        ret += price * self.__shares
        if includeCommissions:
            ret -= self.__commissions
        return ret

    def getReturn(self, price=None, includeCommissions=True):
        ret = 0
        netProfit = self.getNetProfit(price, includeCommissions)
        if self.__costBasis != 0:
            ret = netProfit / float(self.__costBasis)
        return ret

    def buy(self, quantity, price, commission=0):
        assert (quantity > 0)
        self.__update(quantity, price, commission)

    def sell(self, quantity, price, commission=0):
        assert (quantity > 0)
        self.__update(quantity * -1, price, commission)


class ReturnsAnalyzerBase(stratanalyzer.StrategyAnalyzer):
    def __init__(self):
        self.__event = observer.Event()
        self.__portfolioReturns = None

    @classmethod
    def getOrCreateShared(cls, strat):
        name = cls.__name__
        # Get or create the shared ReturnsAnalyzerBase.
        ret = strat.get_named_analyzer(name)
        if ret is None:
            ret = ReturnsAnalyzerBase()
            strat.attach_analyzer_ex(ret, name)
        return ret

    def attached(self, strat):
        self.__portfolioReturns = TimeWeightedReturns(strat.broker.equity)

    # An event will be notified when return are calculated at each bar. The hander should receive 1 parameter:
    # 1: The current datetime.
    # 2: This analyzer's instance
    def getEvent(self):
        return self.__event

    def getNetReturn(self):
        return self.__portfolioReturns.last_period_returns

    def getCumulativeReturn(self):
        return self.__portfolioReturns.cumulative_returns

    def beforeOnBars(self, strat, bars):
        self.__portfolioReturns.update(strat.broker.equity)

        # Notify that new returns are available.
        self.__event.emit(bars.date_time, self)


class Returns(stratanalyzer.StrategyAnalyzer):
    """A :class:`chinascope_algotrade.stratanalyzer.StrategyAnalyzer` that calculates
    time-weighted returns for the whole portfolio."""

    def __init__(self):
        self.__netReturns = dataseries.SequenceDataSeries()
        self.__cumReturns = dataseries.SequenceDataSeries()

    def beforeAttach(self, strat):
        # Get or create a shared ReturnsAnalyzerBase
        analyzer = ReturnsAnalyzerBase.getOrCreateShared(strat)
        analyzer.getEvent().subscribe(self.__onReturns)

    def __onReturns(self, date_time, returnsAnalyzerBase):
        self.__netReturns.appendWithDateTime(date_time, returnsAnalyzerBase.getNetReturn())
        self.__cumReturns.appendWithDateTime(date_time, returnsAnalyzerBase.getCumulativeReturn())

    def getReturns(self):
        """Returns a :class:`chinascope_algotrade.dataseries.DataSeries` with the returns for each bar."""
        return self.__netReturns

    def cumulative_returns(self):
        """Returns a :class:`chinascope_algotrade.dataseries.DataSeries` with the cumulative returns for each bar."""
        return self.__cumReturns
