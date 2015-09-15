#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import (ABCMeta)

import six

from algotrade.const import (OrderStatus, OrderAction, OrderType)

__author__ = 'phil.zhang'


# from algotrade import const


class BaseOrder(six.with_metaclass(ABCMeta)):
    """Base class for orders.

    :param type_: The order type
    :type type_: :class:`OrderType`
    :param action: The order action.
    :type action: :class:`OrderAction`
    :param instrument: Instrument identifier.
    :type instrument: string.
    :param quantity: BaseOrder quantity.
    :type quantity: int/float.

    .. note::
        This is a base class and should not be used directly.

        Valid **type** parameter values are:

         * OrderType.MARKET
         * OrderType.LIMIT
         * OrderType.STOP
         * OrderType.STOP_LIMIT

        Valid **action** parameter values are:

         * OrderAction.BUY
         * OrderAction.BUY_TO_COVER
         * OrderAction.SELL
         * OrderAction.SELL_SHORT
    """
    # OrderStatus = const.OrderStatus
    # OrderAction = const.OrderAction

    # class Action(object):
    #     def __init__(self):
    #         pass
    #
    #     BUY = 1
    #     BUY_TO_COVER = 2
    #     SELL = 3
    #     SELL_SHORT = 4
    #
    # class State(object):
    #     def __init__(self):
    #         pass
    #
    #     INITIAL = 1  # Initial state.
    #     SUBMITTED = 2  # BaseOrder has been submitted.
    #     ACCEPTED = 3  # BaseOrder has been acknowledged by the broker.
    #     CANCELED = 4  # BaseOrder has been canceled.
    #     PARTIALLY_FILLED = 5  # BaseOrder has been partially filled.
    #     FILLED = 6  # BaseOrder has been completely filled.

    # @classmethod
    def to_string(state):
        if state == OrderStatus.INITIAL:
            return "INITIAL"
        elif state == OrderStatus.SUBMITTED:
            return "SUBMITTED"
        elif state == OrderStatus.ACCEPTED:
            return "ACCEPTED"
        elif state == OrderStatus.CANCELED:
            return "CANCELED"
        elif state == OrderStatus.PARTIALLY_FILLED:
            return "PARTIALLY_FILLED"
        elif state == OrderStatus.FILLED:
            return "FILLED"
        else:
            raise Exception("Invalid state")


            # class OrderType(object):
            #     def __init__(self):
            #         pass
            #
            #     MARKET = 1
            #     LIMIT = 2
            #     STOP = 3
            #     STOP_LIMIT = 4

    # Valid state transitions.
    VALID_TRANSITIONS = {
        OrderStatus.INITIAL         : [OrderStatus.SUBMITTED, OrderStatus.CANCELED],
        OrderStatus.SUBMITTED       : [OrderStatus.ACCEPTED, OrderStatus.CANCELED],
        OrderStatus.ACCEPTED        : [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED],
        OrderStatus.PARTIALLY_FILLED: [OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED],
    }

    def __init__(self, type_, action, instrument, quantity, instrument_traits):
        if quantity <= 0:
            raise Exception("Invalid quantity")
        self.__id = None
        self.__order_type = type_
        self.__action = action
        self.__instrument = instrument
        self.__quantity = quantity
        self.__instrument_traits = instrument_traits
        self.__filled = 0
        self.__avgFillPrice = None
        self.__executionInfo = None
        self.__good_till_canceled = False
        self.__commissions = 0
        self.__all_or_none = False
        self.__state = OrderStatus.INITIAL
        self.__submitDateTime = None

        # This is to check that orders are not compared directly. order ids should be compared.

    #    def __eq__(self, other):
    #        if other is None:
    #            return False
    #        assert(False)

    # This is to check that orders are not compared directly. order ids should be compared.
    #    def __ne__(self, other):
    #        if other is None:
    #            return True
    #        assert(False)

    @property
    def instrument_traits(self):
        return self.__instrument_traits

    @property
    def id(self):
        """
        Returns the order id.

        .. note::

            This will be None if the order was not submitted.
        """
        return self.__id

    @property
    def order_type(self):
        """Returns the order type. Valid order types are:

         * OrderType.MARKET
         * OrderType.LIMIT
         * OrderType.STOP
         * OrderType.STOP_LIMIT
        """
        return self.__order_type

    @property
    def submit_date_time(self):
        """Returns the datetime when the order was submitted."""
        return self.__submitDateTime

    # @submit_date_time.setter
    def submit_date_time(self, order_id, date_time):
        assert (self.__id is None or order_id == self.__id)
        self.__id = order_id
        self.__submitDateTime = date_time

    @property
    def action(self):
        """Returns the order action. Valid order actions are:

         * OrderAction.BUY
         * OrderAction.BUY_TO_COVER
         * OrderAction.SELL
         * OrderAction.SELL_SHORT
        """
        return self.__action

    @property
    def state(self):
        """Returns the order state. Valid order states are:

         * OrderStatus.INITIAL (the initial state).
         * OrderStatus.SUBMITTED
         * OrderStatus.ACCEPTED
         * OrderStatus.CANCELED
         * OrderStatus.PARTIALLY_FILLED
         * OrderStatus.FILLED
        """
        return self.__state

    @property
    def is_active(self):
        """Returns True if the order is active."""
        return self.__state not in [OrderStatus.CANCELED, OrderStatus.FILLED]

    @property
    def is_initial(self):
        """Returns True if the order state is OrderStatus.INITIAL."""
        return self.__state == OrderStatus.INITIAL

    @property
    def is_submitted(self):
        """Returns True if the order state is OrderStatus.SUBMITTED."""
        return self.__state == OrderStatus.SUBMITTED

    @property
    def is_accepted(self):
        """Returns True if the order state is OrderStatus.ACCEPTED."""
        return self.__state == OrderStatus.ACCEPTED

    @property
    def is_canceled(self):
        """Returns True if the order state is OrderStatus.CANCELED."""
        return self.__state == OrderStatus.CANCELED

    @property
    def is_partially_filled(self):
        """Returns True if the order state is OrderStatus.PARTIALLY_FILLED."""
        return self.__state == OrderStatus.PARTIALLY_FILLED

    @property
    def is_filled(self):
        """Returns True if the order state is OrderStatus.FILLED."""
        return self.__state == OrderStatus.FILLED

    @property
    def instrument(self):
        """Returns the instrument identifier."""
        return self.__instrument

    @property
    def quantity(self):
        """Returns the quantity."""
        return self.__quantity

    def number_filled(self):
        """Returns the number of shares that have been executed."""
        return self.__filled

    @property
    def remaining(self):
        """Returns the number of shares still outstanding."""
        return self.__instrument_traits.round_quantity(self.__quantity - self.__filled)

    @property
    def average_fill_rice(self):
        """Returns the average price of the shares that have been executed, or None if nothing has been filled."""
        return self.__avgFillPrice

    @property
    def commissions(self):
        return self.__commissions

    @property
    def good_till_canceled(self):
        """Returns True if the order is good till canceled."""
        return self.__good_till_canceled

    @good_till_canceled.setter
    def good_till_canceled(self, good_till_canceled):
        """Sets if the order should be good till canceled.
        Orders that are not filled by the time the session closes will be will be automatically canceled
        if they were not set as good till canceled

        :param good_till_canceled: True if the order should be good till canceled.
        :type good_till_canceled: boolean.

        .. note:: This can't be changed once the order is submitted.
        """
        if self.__state != OrderStatus.INITIAL:
            raise Exception("The order has already been submitted")
        self.__good_till_canceled = good_till_canceled

    @property
    def all_or_none(self):
        """Returns True if the order should be completely filled or else canceled."""
        return self.__all_or_none

    @all_or_none.setter
    def all_or_none(self, all_or_none):
        """Sets the All-Or-None property for this order.

        :param all_or_none: True if the order should be completely filled.
        :type all_or_none: boolean.

        .. note:: This can't be changed once the order is submitted.
        """
        if self.__state != OrderStatus.INITIAL:
            raise Exception("The order has already been submitted")
        self.__all_or_none = all_or_none

    def add_execution_info(self, order_execution_info):
        if order_execution_info.quantity > self.remaining:
            raise Exception("Invalid fill size. %s remaining and %s filled" % (self.remaining, order_execution_info.quantity))

        if self.__avgFillPrice is None:
            self.__avgFillPrice = order_execution_info.price
        else:
            self.__avgFillPrice = (self.__avgFillPrice * self.__filled + order_execution_info.price() * order_execution_info.quantity) / float(
                self.__filled + order_execution_info.quantity)

        self.__executionInfo = order_execution_info
        self.__filled = self.instrument_traits.round_quantity(self.__filled + order_execution_info.quantity)
        self.__commissions += order_execution_info.commission

        if self.remaining == 0:
            self.switch_state(OrderStatus.FILLED)
        else:
            assert (not self.__all_or_none)
            self.switch_state(OrderStatus.PARTIALLY_FILLED)

    def switch_state(self, new_state):
        valid_transitions = BaseOrder.VALID_TRANSITIONS.get(self.__state, [])
        if new_state not in valid_transitions:
            raise Exception("Invalid order state transition from %s to %s" % (OrderStatus.to_string(self.__state), OrderStatus.to_string(new_state)))
        else:
            self.__state = new_state

    def set_state(self, new_state):
        self.__state = new_state

    @property
    def execution_info(self):
        """Returns the last execution information for this order, or None if nothing has been filled so far.
        This will be different every time an order, or part of it, gets filled.

        :rtype: :class:`OrderExecutionInfo`.
        """
        return self.__executionInfo

    # Returns True if this is a BUY or BUY_TO_COVER order.
    @property
    def is_a_buy_order(self):
        return self.__action in [OrderAction.BUY, OrderAction.BUY_TO_COVER]

    # Returns True if this is a SELL or SELL_SHORT order.
    @property
    def is_a_sell_order(self):
        return self.__action in [OrderAction.SELL, OrderAction.SELL_SHORT]


class MarketOrder(BaseOrder):
    """Base class for market orders.

    .. note::

        This is a base class and should not be used directly.
    """

    def __init__(self, action, instrument, quantity, on_close, instrument_traits):
        super(MarketOrder).__init__(self, OrderType.MARKET, action, instrument, quantity, instrument_traits)
        self.__on_close = on_close

    @property
    def is_fill_on_close(self):
        """Returns True if the order should be filled as close to the closing price as possible (Market-On-Close order)."""
        return self.__on_close


class LimitOrder(BaseOrder):
    """Base class for limit orders.

    .. note::

        This is a base class and should not be used directly.
    """

    def __init__(self, action, instrument, limit_price, quantity, instrument_traits):
        super(LimitOrder).__init__(self, OrderType.LIMIT, action, instrument, quantity, instrument_traits)
        self.__limit_price = limit_price

    @property
    def limit_price(self):
        """Returns the limit price."""
        return self.__limit_price


class StopOrder(BaseOrder):
    """Base class for stop orders.

    .. note::

        This is a base class and should not be used directly.
    """

    def __init__(self, action, instrument, stop_price, quantity, instrument_traits):
        super(StopOrder).__init__(self, OrderType.STOP, action, instrument, quantity, instrument_traits)
        self.__stop_price = stop_price

    @property
    def stop_price(self):
        """Returns the stop price."""
        return self.__stop_price


class StopLimitOrder(BaseOrder):
    """Base class for stop limit orders.

    .. note::

        This is a base class and should not be used directly.
    """

    def __init__(self, action, instrument, stop_price, limit_price, quantity, instrument_traits):
        super(StopLimitOrder).__init__(self, OrderType.STOP_LIMIT, action, instrument, quantity, instrument_traits)
        self.__stop_price = stop_price
        self.__limit_price = limit_price

    @property
    def stop_price(self):
        """Returns the stop price."""
        return self.__stop_price

    @property
    def limit_price(self):
        """Returns the limit price."""
        return self.__limit_price


class OrderExecutionInfo(object):
    """Execution information for an order."""

    def __init__(self, price, quantity, commission, date_time):
        self.__price = price
        self.__quantity = quantity
        self.__commission = commission
        self.__date_time = date_time

    def __str__(self):
        return "%s - Price: %s - Amount: %s - Fee: %s" % (self.__date_time, self.__price, self.__quantity, self.__commission)

    @property
    def price(self):
        """Returns the fill price."""
        return self.__price

    @property
    def quantity(self):
        """Returns the quantity."""
        return self.__quantity

    @property
    def commission(self):
        """Returns the commission applied."""
        return self.__commission

    @property
    def date_time(self):
        """Returns the :class:`datatime.datetime` when the order was executed."""
        return self.__date_time


class OrderEvent(object):
    class Type:
        def __init__(self):
            pass

        ACCEPTED = 1  # BaseOrder has been acknowledged by the broker.
        CANCELED = 2  # BaseOrder has been canceled.
        PARTIALLY_FILLED = 3  # BaseOrder has been partially filled.
        FILLED = 4  # BaseOrder has been completely filled.

    def __init__(self, order, event_type, event_info):
        self.__order = order
        self.__event_type = event_type
        self.__event_info = event_info

    @property
    def order(self):
        return self.__order

    @property
    def event_type(self):
        return self.__event_type

    # This depends on the event type:
    # ACCEPTED: None
    # CANCELED: A string with the reason why it was canceled.
    # PARTIALLY_FILLED: An OrderExecutionInfo instance.
    # FILLED: An OrderExecutionInfo instance.
    @property
    def event_info(self):
        return self.__event_info
