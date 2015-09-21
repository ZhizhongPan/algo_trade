# chinascope_algotrade
#
# Copyright 2011-2015 Gabriel Martin Becedillas Ruiz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
.. moduleauthor:: Gabriel Martin Becedillas Ruiz <gabriel.becedillas@gmail.com>
"""

from abc import abstractmethod, ABCMeta
from algotrade import const
import six

from algotrade.broker import broker
# from algotrade.broker import fillstrategy
# from algotrade import warninghelpers
# from chinascope_algotrade import logger
# import chinascope_algotrade.bar


######################################################################
# Commission models

class Commission(six.with_metaclass(ABCMeta)):
    """Base class for implementing different commission schemes.

    .. note::
        This is a base class and should not be used directly.
    """

    # __metaclass__ = abc.ABCMeta

    @abstractmethod
    def calculate(self, order, price, quantity):
        """Calculates the commission for an order execution.

        :param order: The order being executed.
        :type order: :class:`chinascope_algotrade.broker.BaseOrder`.
        :param price: The price for each share.
        :type price: float.
        :param quantity: The order size.
        :type quantity: float.
        :rtype: float.
        """
        raise NotImplementedError()


class NoCommission(Commission):
    """A :class:`Commission` class that always returns 0."""

    def calculate(self, order, price, quantity):
        return 0


class FixedPerTrade(Commission):
    """A :class:`Commission` class that charges a fixed amount for the whole trade.

    :param amount: The commission for an order.
    :type amount: float.
    """

    def __init__(self, amount):
        self.__amount = amount

    def calculate(self, order, price, quantity):
        ret = 0
        # Only charge the first fill.
        if order.execution_info is None:
            ret = self.__amount
        return ret


class TradePercentage(Commission):
    """A :class:`Commission` class that charges a percentage of the whole trade.

    :param percentage: The percentage to charge. 0.01 means 1%, and so on. It must be smaller than 1.
    :type percentage: float.
    """

    def __init__(self, percentage):
        assert (percentage < 1)
        self.__percentage = percentage

    def calculate(self, order, price, quantity):
        return price * quantity * self.__percentage


######################################################################
# Orders

class BacktestingOrder(object):
    def __init__(self):
        self.__accepted = None

    @property
    def accepted_date_time(self):
        return self.__accepted

    @accepted_date_time.setter
    def accepted_date_time(self, date_time):
        self.__accepted = date_time

    # Override to call the fill strategy using the concrete order type.
    # return FillInfo or None if the order should not be filled.
    def process(self, broker_, bar_):
        raise NotImplementedError()


class MarketOrder(broker.MarketOrder, BacktestingOrder):
    def __init__(self, action, instrument, quantity, on_close, instrument_traits):
        broker.MarketOrder.__init__(self, action, instrument, quantity, on_close, instrument_traits)
        BacktestingOrder.__init__(self)

    def process(self, broker_, bar_):
        return broker_.fill_strategy.fillMarketOrder(broker_, self, bar_)


class LimitOrder(broker.LimitOrder, BacktestingOrder):
    def __init__(self, action, instrument, limit_price, quantity, instrument_traits):
        broker.LimitOrder.__init__(self, action, instrument, limit_price, quantity, instrument_traits)
        BacktestingOrder.__init__(self)

    def process(self, broker_, bar_):
        return broker_.fill_strategy.fillLimitOrder(broker_, self, bar_)


class StopOrder(broker.StopOrder, BacktestingOrder):
    def __init__(self, action, instrument, stop_price, quantity, instrument_traits):
        broker.StopOrder.__init__(self, action, instrument, stop_price, quantity, instrument_traits)
        BacktestingOrder.__init__(self)
        self.__stop_hit = False

    def process(self, broker_, bar_):
        return broker_.fill_strategy.fillStopOrder(broker_, self, bar_)

    @property
    def stop_hit(self):
        return self.__stop_hit

    @stop_hit.setter
    def stop_hit(self, stop_hit):
        self.__stop_hit = stop_hit


# http://www.sec.gov/answers/stoplim.htm
# http://www.interactivebrokers.com/en/trading/orders/stopLimit.php
class StopLimitOrder(broker.StopLimitOrder, BacktestingOrder):
    def __init__(self, action, instrument, stop_price, limit_price, quantity, instrument_traits):
        broker.StopLimitOrder.__init__(self, action, instrument, stop_price, limit_price, quantity, instrument_traits)
        BacktestingOrder.__init__(self)
        self.__stop_hit = False  # Set to true when the limit order is activated (stop price is hit)

    @property
    def stop_hit(self):
        return self.__stop_hit

    @stop_hit.setter
    def stop_hit(self, stop_hit):
        self.__stop_hit = stop_hit

    def is_limit_order_active(self):
        # TODO: Deprecated since v0.15. Use stop_hit instead.
        return self.__stop_hit

    def process(self, broker_, bar_):
        return broker_.fill_strategy.fillStopLimitOrder(broker_, self, bar_)


######################################################################
# BackTestingBroker

class BackTestingBroker(broker.BaseBroker):
    """Backtesting broker.

    :param cash: The initial amount of cash.
    :type cash: int/float.
    :param barfeed: The bar feed that will provide the bars.
    :type barfeed: :class:`chinascope_algotrade.barfeed.CSVBarFeed`
    :param commission: An object responsible for calculating order commissions.
    :type commission: :class:`Commission`
    """

    LOGGER_NAME = "broker.backtesting"

    def __init__(self, cash, barfeed, commission=None):
        broker.BaseBroker.__init__(self)

        assert (cash >= 0)
        self.__cash = cash
        if commission is None:
            self.__commission = NoCommission()
        else:
            self.__commission = commission
        self.__shares = {}
        self.__active_orders = {}
        self.__use_adjustedValues = False
        # self.__fill_strategy = fillstrategy.DefaultStrategy()
        # self.__logger = logger.logger(BackTestingBroker.LOGGER_NAME)

        # It is VERY important that the broker subscribes to barfeed events before the strategy.
        barfeed.get_new_values_event().subscribe(self.on_bars)
        self.__barfeed = barfeed
        self.__allow_negative_cash = False
        self.__nextOrderId = 1

    def _get_next_order_id(self):
        ret = self.__nextOrderId
        self.__nextOrderId += 1
        return ret

    def _get_bar(self, bars, instrument):
        ret = bars.bar(instrument)
        if ret is None:
            ret = self.__barfeed.get_last_bar(instrument)
        return ret

    def _register_order(self, order):
        assert (order.id not in self.__active_orders)
        assert (order.id is not None)
        self.__active_orders[order.id] = order

    def _unregister_order(self, order):
        assert (order.id in self.__active_orders)
        assert (order.id is not None)
        del self.__active_orders[order.id]

    @property
    def logger(self):
        return self.__logger

    def set_allow_negative_cash(self, allow_negative_cash):
        self.__allow_negative_cash = allow_negative_cash

    def get_cash(self, include_short=True):
        ret = self.__cash
        if not include_short and self.__barfeed.current_bars is not None:
            bars = self.__barfeed.current_bars
            for instrument, shares in self.__shares.iteritems():
                if shares < 0:
                    instrument_price = self._get_bar(bars, instrument).close(self.use_adjusted_values)
                    ret += instrument_price * shares
        return ret

    def set_cash(self, num_cash):
        self.__cash = num_cash

    @property
    def commission(self):
        """Returns the strategy used to calculate order commissions.

        :rtype: :class:`Commission`.
        """
        return self.__commission

    @commission.setter
    def commission(self, commission):
        """Sets the strategy to use to calculate order commissions.

        :param commission: An object responsible for calculating order commissions.
        :type commission: :class:`Commission`.
        """

        self.__commission = commission

    @property
    def fill_strategy(self):
        """Returns the :class:`chinascope_algotrade.broker.fillstrategy.FillStrategy` currently set."""
        return self.__fill_strategy

    @fill_strategy.setter
    def fill_strategy(self, strategy):
        """Sets the :class:`chinascope_algotrade.broker.fillstrategy.FillStrategy` to use."""
        self.__fill_strategy = strategy

    @property
    def use_adjusted_values(self):
        return self.__use_adjustedValues

    def set_use_adjusted_values(self, use_adjusted, deprecationCheck=None):
        # Deprecated since v0.15
        if not self.__barfeed.bars_have_adj_close():
            raise Exception("The barfeed doesn't support adjusted close values")
        if deprecationCheck is None:
            pass
            # warninghelpers.deprecation_warning(
            #     "set_use_adjusted_values will be deprecated in the next version. Please use set_use_adjusted_values on the strategy instead.",
            #     stacklevel=2
            # )
        self.__use_adjustedValues = use_adjusted

    def active_orders(self, instrument=None):
        if instrument is None:
            ret = self.__active_orders.values()
        else:
            ret = [order for order in self.__active_orders.values() if order.instrument == instrument]
        return ret

    def get_pending_orders(self):
        # warninghelpers.deprecation_warning(
        #     "get_pending_orders will be deprecated in the next version. Please use active_orders instead.",
        #     stacklevel=2
        # )
        return self.active_orders()

    @property
    def _current_datetime(self):
        return self.__barfeed.current_datetime

    def instrument_traits(self, instrument):
        return broker.IntegerTraits()

    def shares(self, instrument):
        return self.__shares.get(instrument, 0)

    def positions(self):
        return self.__shares

    @property
    def active_instruments(self):
        return [instrument for instrument, shares in self.__shares.iteritems() if shares != 0]

    def __get_equity_with_bars(self, bars):
        ret = self.cash
        if bars is not None:
            for instrument, shares in self.__shares.iteritems():
                instrument_price = self._get_bar(bars, instrument).close(self.use_adjusted_values)
                ret += instrument_price * shares
        return ret

    @property
    def equity(self):
        """Returns the portfolio value (cash + shares)."""
        return self.__get_equity_with_bars(self.__barfeed.current_bars)

    # Tries to commit an order execution.
    def commit_order_execution(self, order, date_time, fill_info):
        price = fill_info.price
        quantity = fill_info.quantity

        if order.is_a_buy_order:
            cost = price * quantity * -1
            assert (cost < 0)
            shares_delta = quantity
        elif order.is_a_sell_order:
            cost = price * quantity
            assert (cost > 0)
            shares_delta = quantity * -1
        else:  # Unknown action
            assert False

        commission = self.commission.calculate(order, price, quantity)
        cost -= commission
        resulting_cash = self.cash + cost

        # Check that we're ok on cash after the commission.
        if resulting_cash >= 0 or self.__allow_negative_cash:

            # Update the order before updating internal state since add_execution_info may raise.
            # add_execution_info should switch the order state.
            order_execution_info = broker.OrderExecutionInfo(price, quantity, commission, date_time)
            order.add_execution_info(order_execution_info)

            # Commit the order execution.
            self.__cash = resulting_cash
            updated_shares = order.instrument_traits.round_quantity(
                self.shares(order.instrument) + shares_delta
            )
            if updated_shares == 0:
                del self.__shares[order.instrument]
            else:
                self.__shares[order.instrument] = updated_shares

            # Let the strategy know that the order was filled.
            self.__fill_strategy.onOrderFilled(self, order)

            # Notify the order update
            if order.is_filled:
                self._unregister_order(order)
                self.notify_order_event(broker.OrderEvent(order, broker.OrderEvent.Type.FILLED, order_execution_info))
            elif order.is_partially_filled:
                self.notify_order_event(
                    broker.OrderEvent(order, broker.OrderEvent.Type.PARTIALLY_FILLED, order_execution_info)
                )
            else:
                assert False
        else:
            self.__logger.debug("Not enough cash to fill %s order [%s] for %d share/s" % (
                order.instrument,
                order.id,
                order.remaining
            ))

    def submit_order(self, order):
        if order.is_initial:
            order.submit_date_time(self._get_next_order_id(), self._current_datetime)
            self._register_order(order)
            # Switch from INITIAL -> SUBMITTED
            # IMPORTANT: Do not emit an event for this switch because when using the position interface
            # the order is not yet mapped to the position and Position.on_order_updated will get called.
            order.switch_state(broker.BaseOrder.State.SUBMITTED)
        else:
            raise Exception("The order was already processed")

    # Return True if further processing is needed.
    def __pre_process_order(self, order, bar_):
        ret = True

        # For non-GTC orders we need to check if the order has expired.
        if not order.good_till_canceled:
            expired = bar_.date_time.date() > order.accepted_date_time.date()

            # Cancel the order if it is expired.
            if expired:
                ret = False
                self._unregister_order(order)
                order.switch_state(broker.BaseOrder.State.CANCELED)
                self.notify_order_event(broker.OrderEvent(order, broker.OrderEvent.Type.CANCELED, "Expired"))

        return ret

    def __post_process_order(self, order, bar_):
        # For non-GTC orders and daily (or greater) bars we need to check if orders should expire right now
        # before waiting for the next bar.
        if not order.good_till_canceled:
            expired = False
            if self.__barfeed.frequency >= const.Frequency.DAY:
                expired = bar_.date_time.date() >= order.accepted_date_time.date()

            # Cancel the order if it will expire in the next bar.
            if expired:
                self._unregister_order(order)
                order.switch_state(broker.BaseOrder.State.CANCELED)
                self.notify_order_event(broker.OrderEvent(order, broker.OrderEvent.Type.CANCELED, "Expired"))

    def __process_order(self, order, bar_):
        if not self.__pre_process_order(order, bar_):
            return

        # Double dispatch to the fill strategy using the concrete order type.
        fill_info = order.process(self, bar_)
        if fill_info is not None:
            self.commit_order_execution(order, bar_.date_time, fill_info)

        if order.is_active:
            self.__post_process_order(order, bar_)

    def __on_bars_Impl(self, order, bars):
        # IF WE'RE DEALING WITH MULTIPLE INSTRUMENTS WE SKIP ORDER PROCESSING IF THERE IS NO BAR FOR THE ORDER'S
        # INSTRUMENT TO GET THE SAME BEHAVIOUR AS IF WERE BE PROCESSING ONLY ONE INSTRUMENT.
        bar_ = bars.bar(order.instrument)
        if bar_ is not None:
            # Switch from SUBMITTED -> ACCEPTED
            if order.is_submitted:
                order.setAcceptedDateTime(bar_.date_time)
                order.switch_state(broker.BaseOrder.State.ACCEPTED)
                self.notify_order_event(broker.OrderEvent(order, broker.OrderEvent.Type.ACCEPTED, None))

            if order.is_active:
                # This may trigger orders to be added/removed from __active_orders.
                self.__process_order(order, bar_)
            else:
                # If an order is not active it should be because it was canceled in this same loop and it should
                # have been removed.
                assert order.is_canceled
                assert (order not in self.__active_orders)

    def on_bars(self, date_time, bars):
        # Let the fill strategy know that new bars are being processed.
        self.__fill_strategy.on_bars(self, bars)

        # This is to froze the orders that will be processed in this event, to avoid new getting orders introduced
        # and processed on this very same event.
        orders_to_process = self.__active_orders.values()

        for order in orders_to_process:
            # This may trigger orders to be added/removed from __active_orders.
            self.__on_bars_Impl(order, bars)

    # def start(self):
    #     pass
    #
    # def stop(self):
    #     pass
    #
    # def join(self):
    #     pass
    #
    # def eof(self):
    #     # If there are no more events in the barfeed, then there is nothing left for us to do since all processing took
    #     # place while processing barfeed events.
    #     return self.__barfeed.eof()
    #
    # def dispatch(self):
    #     # All events were already emitted while handling barfeed events.
    #     pass
    #
    # def peek_datetime(self):
    #     return None

    def create_market_order(self, action, instrument, quantity, on_close=False):
        # In order to properly support market-on-close with intraday feeds I'd need to know about different
        # exchange/market trading hours and support specifying routing an order to a specific exchange/market.
        # Even if I had all this in place it would be a problem while paper-trading with a live feed since
        # I can't tell if the next bar will be the last bar of the market session or not.
        if on_close is True and self.__barfeed.is_intraday():
            raise Exception("Market-on-close not supported with intraday feeds")

        return MarketOrder(action, instrument, quantity, on_close, self.instrument_traits(instrument))

    def create_limit_order(self, action, instrument, limit_price, quantity):
        return LimitOrder(action, instrument, limit_price, quantity, self.instrument_traits(instrument))

    def create_stop_order(self, action, instrument, stop_price, quantity):
        return StopOrder(action, instrument, stop_price, quantity, self.instrument_traits(instrument))

    def create_stop_limit_order(self, action, instrument, stop_price, limit_price, quantity):
        return StopLimitOrder(action, instrument, stop_price, limit_price, quantity, self.instrument_traits(instrument))

    def cancel_order(self, order):
        active_order = self.__active_orders.get(order.id)
        if active_order is None:
            raise Exception("The order is not active anymore")
        if active_order.is_filled:
            raise Exception("Can't cancel order that has already been filled")

        self._unregister_order(active_order)
        active_order.switch_state(broker.BaseOrder.State.CANCELED)
        self.notify_order_event(
            broker.OrderEvent(active_order, broker.OrderEvent.Type.CANCELED, "User requested cancellation")
        )
