#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'phil.zhang'

from abc import ABCMeta, abstractmethod, abstractproperty

import six

from algotrade.event_engine import EventEngineMixin


# class BaseBroker(six.with_metaclass(ABCMeta), EventEngineMixin):
#     def __init__(self):
#         self.event_engine = EventEngineMixin.event_engine


######################################################################
# Base broker class
class BaseBroker(six.with_metaclass(ABCMeta), EventEngineMixin):
    """Base class for brokers.

    .. note::

        This is a base class and should not be used directly.
    """

    # __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.event_engine = EventEngineMixin.event_engine

    # def notify_order_event(self, order_event):
    #     self.__order_event.emit(self, order_event)

    # Handlers should expect 2 parameters:
    # 1: broker instance
    # 2: OrderEvent instance
    @property
    def order_updated_event(self):
        return self.__order_event

    @abstractmethod
    def instrument_traits(self, instrument):
        raise NotImplementedError()

    @abstractmethod
    def get_cash(self, include_short=True):
        """
        Returns the available cash.

        :param include_short: Include cash from short positions.
        :type include_short: boolean.
        """
        raise NotImplementedError()

    @abstractmethod
    def shares(self, instrument):
        """Returns the number of shares for an instrument."""
        raise NotImplementedError()

    @abstractproperty
    def positions(self):
        """Returns a dictionary that maps instruments to shares."""
        raise NotImplementedError()

    @abstractmethod
    def active_orders(self, instrument=None):
        """Returns a sequence with the orders that are still active.

        :param instrument: An optional instrument identifier to return only the active orders for the given instrument.
        :type instrument: string.
        """
        raise NotImplementedError()

    @abstractmethod
    def submit_order(self, order):
        """Submits an order.

        :param order: The order to submit.
        :type order: :class:`BaseOrder`.

        .. note::
            * After this call the order is in SUBMITTED state and an event is not triggered for this transition.
            * Calling this twice on the same order will raise an exception.
        """
        raise NotImplementedError()

    # def place_order(self, order):
    #     # Deprecated since v0.16
    #     warninghelpers.deprecation_warning("place_order will be deprecated in the next version. Please use submit_order instead.", stacklevel=2)
    #     return self.submit_order(order)

    @abstractmethod
    def create_market_order(self, action, instrument, quantity, on_close=False):
        """Creates a Market order.
        A market order is an order to buy or sell a stock at the best available price.
        Generally, this type of order will be executed immediately. However, the price at which a market order will be executed
        is not guaranteed.

        :param action: The order action.
        :type action: BaseOrder.Action.BUY, or BaseOrder.Action.BUY_TO_COVER, or BaseOrder.Action.SELL or BaseOrder.Action.SELL_SHORT.
        :param instrument: Instrument identifier.
        :type instrument: string.
        :param quantity: BaseOrder quantity.
        :type quantity: int/float.
        :param on_close: True if the order should be filled as close to the closing price as possible (Market-On-Close order). Default is False.
        :type on_close: boolean.
        :rtype: A :class:`MarketOrder` subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_limit_order(self, action, instrument, limit_price, quantity):
        """Creates a Limit order.
        A limit order is an order to buy or sell a stock at a specific price or better.
        A buy limit order can only be executed at the limit price or lower, and a sell limit order can only be executed at the
        limit price or higher.

        :param action: The order action.
        :type action: BaseOrder.Action.BUY, or BaseOrder.Action.BUY_TO_COVER, or BaseOrder.Action.SELL or BaseOrder.Action.SELL_SHORT.
        :param instrument: Instrument identifier.
        :type instrument: string.
        :param limit_price: The order price.
        :type limit_price: float
        :param quantity: BaseOrder quantity.
        :type quantity: int/float.
        :rtype: A :class:`LimitOrder` subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_stop_order(self, action, instrument, stop_price, quantity):
        """Creates a Stop order.
        A stop order, also referred to as a stop-loss order, is an order to buy or sell a stock once the price of the stock
        reaches a specified price, known as the stop price.
        When the stop price is reached, a stop order becomes a market order.
        A buy stop order is entered at a stop price above the current market price. Investors generally use a buy stop order
        to limit a loss or to protect a profit on a stock that they have sold short.
        A sell stop order is entered at a stop price below the current market price. Investors generally use a sell stop order
        to limit a loss or to protect a profit on a stock that they own.

        :param action: The order action.
        :type action: BaseOrder.Action.BUY, or BaseOrder.Action.BUY_TO_COVER, or BaseOrder.Action.SELL or BaseOrder.Action.SELL_SHORT.
        :param instrument: Instrument identifier.
        :type instrument: string.
        :param stop_price: The trigger price.
        :type stop_price: float
        :param quantity: BaseOrder quantity.
        :type quantity: int/float.
        :rtype: A :class:`StopOrder` subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_stop_limit_order(self, action, instrument, stop_price, limit_price, quantity):
        """Creates a Stop-Limit order.
        A stop-limit order is an order to buy or sell a stock that combines the features of a stop order and a limit order.
        Once the stop price is reached, a stop-limit order becomes a limit order that will be executed at a specified price
        (or better). The benefit of a stop-limit order is that the investor can control the price at which the order can be executed.

        :param action: The order action.
        :type action: BaseOrder.Action.BUY, or BaseOrder.Action.BUY_TO_COVER, or BaseOrder.Action.SELL or BaseOrder.Action.SELL_SHORT.
        :param instrument: Instrument identifier.
        :type instrument: string.
        :param stop_price: The trigger price.
        :type stop_price: float
        :param limit_price: The price for the limit order.
        :type limit_price: float
        :param quantity: BaseOrder quantity.
        :type quantity: int/float.
        :rtype: A :class:`StopLimitOrder` subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def cancel_order(self, order):
        """Requests an order to be canceled. If the order is filled an Exception is raised.

        :param order: The order to cancel.
        :type order: :class:`BaseOrder`.
        """
        raise NotImplementedError()
