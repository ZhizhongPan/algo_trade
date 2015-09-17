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

import datetime

from algotrade.stratanalyzer import returns
from algotrade import warninghelpers
from algotrade import const


class PositionState(object):
    def on_enter(self, position):
        pass

    # Raise an exception if an order can't be submitted in the current state.
    def can_submit_order(self, position, order):
        raise NotImplementedError()

    def on_order_event(self, position, order_event):
        raise NotImplementedError()

    def is_open(self, position):
        raise NotImplementedError()

    def exit(self, position, stop_price=None, limit_price=None, good_till_canceled=None):
        raise NotImplementedError()


class WaitingEntryState(PositionState):
    def can_submit_order(self, position, order):
        if position.entry_active:
            raise Exception("The entry order is still active")

    def on_order_event(self, position, order_event):
        # Only entry order events are valid in this state.
        assert (position.entry_order.id == order_event.order.id)

        if order_event.event_type in (const.OrderStatus.FILLED, const.OrderStatus.PARTIALLY_FILLED):
            position.switch_state(OpenState())
            position.strategy.on_enter_ok(position)
        elif order_event.event_type == const.OrderStatus.CANCELED:
            assert (position.entry_order.number_filled == 0)
            position.switch_state(ClosedState())
            position.strategy.onEnterCanceled(position)

    def is_open(self, position):
        return True

    def exit(self, position, stop_price=None, limit_price=None, good_till_canceled=None):
        assert position.shares() == 0
        assert position.entry_order.is_active
        position.strategy.broker.cancel_order(position.entry_order)


class OpenState(PositionState):
    def on_enter(self, position):
        entry_datetime = position.entry_order.execution_info.date_time
        position.set_entry_datetime(entry_datetime)

    def can_submit_order(self, position, order):
        # Only exit orders should be submitted in this state.
        pass

    def on_order_event(self, position, order_event):
        if position.exit_order and position.exit_order.id == order_event.order.id:
            if order_event.event_type == const.OrderStatus.FILLED:
                if position.shares() == 0:
                    position.switch_state(ClosedState())
                    position.strategy.on_exit_ok(position)
            elif order_event.event_type == const.OrderStatus.CANCELED:
                assert (position.shares() != 0)
                position.strategy.on_exit_canceled(position)
        elif position.entry_order.id == order_event.order.id:
            # Nothing to do since the entry order may be completely filled or canceled after a partial fill.
            assert (position.shares() != 0)
        else:
            raise Exception("Invalid order event '%s' in OpenState" % order_event.event_type)

    def is_open(self, position):
        return True

    def exit(self, position, stop_price=None, limit_price=None, good_till_canceled=None):
        assert (position.shares() != 0)

        # Fail if a previous exit order is active.
        if position.exit_active:
            raise Exception("Exit order is active and it should be canceled first")

        # If the entry order is active, request cancellation.
        if position.entry_active:
            position.strategy.broker.cancel_order(position.entry_order)

        position._submit_exit_order(stop_price, limit_price, good_till_canceled)


class ClosedState(PositionState):
    def on_enter(self, position):
        # Set the exit datetime if the exit order was filled.
        if position.exit_filled:
            exit_datetime = position.exit_order.execution_info.date_time
            position.set_exit_datetime(exit_datetime)

        assert (position.shares() == 0)
        position.strategy.unregister_position(position)

    def can_submit_order(self, position, order):
        raise Exception("The position is closed")

    def on_order_event(self, position, order_event):
        raise Exception("Invalid order event '%s' in ClosedState" % order_event.event_type)

    def is_open(self, position):
        return False

    def exit(self, position, stop_price=None, limit_price=None, good_till_canceled=None):
        pass


class Position(object):
    """Base class for positions.

    Positions are higher level abstractions for placing orders.
    They are essentially a pair of entry-exit orders and allow
    to track returns and PnL easier that placing orders manually.

    :param strategy: The strategy that this position belongs to.
    :type strategy: :class:`chinascope_algotrade.strategy.BaseStrategy`.
    :param entry_order: The order used to enter the position.
    :type entry_order: :class:`chinascope_algotrade.broker.BaseOrder`
    :param good_till_canceled: True if the entry order should be set as good till canceled.
    :type good_till_canceled: boolean.
    :param all_or_none: True if the orders should be completely filled or not at all.
    :type all_or_none: boolean.

    .. note::
        This is a base class and should not be used directly.
    """

    def __init__(self, strategy, entry_order, good_till_canceled, all_or_none):
        # The order must be created but not submitted.
        assert entry_order.is_initial

        self.__state = None
        self.__active_orders = {}
        self.__shares = 0
        self.__strategy = strategy
        self.__entry_order = None
        self.__entry_datetime = None
        self.__exit_order = None
        self.__exit_datetime = None
        self.__pos_tracker = returns.PositionTracker(entry_order.instrument_traits)
        self.__all_or_none = all_or_none

        self.switch_state(WaitingEntryState())

        entry_order.good_till_canceled = good_till_canceled
        entry_order.all_or_none = all_or_none
        self.__submit_and_register_order(entry_order)
        self.__entry_order = entry_order

    def __submit_and_register_order(self, order):
        assert order.is_initial

        # Check if an order can be submitted in the current state.
        self.__state.can_submit_order(self, order)

        # This may raise an exception, so we wan't to submit the order before moving forward and registering
        # the order in the strategy.
        self.strategy.broker.submit_order(order)

        self.__active_orders[order.id] = order
        self.strategy.register_position_order(self, order)

    def set_entry_datetime(self, date_time):
        self.__entry_datetime = date_time

    def set_exit_datetime(self, date_time):
        self.__exit_datetime = date_time

    def switch_state(self, new_state):
        self.__state = new_state
        self.__state.on_enter(self)

    @property
    def strategy(self):
        return self.__strategy

    def get_last_price(self):
        return self.__strategy.get_last_price(self.instrument)

    def active_orders(self):
        return self.__active_orders.values()

    def shares(self):
        """Returns the number of shares.
        This will be a positive number for a long position, and a negative number for a short position.

        .. note::
            If the entry order was not filled, or if the position is closed, then the number of shares will be 0.
        """
        return self.__shares

    @property
    def entry_active(self):
        """Returns True if the entry order is active."""
        return self.__entry_order is not None and self.__entry_order.is_active

    @property
    def entry_filled(self):
        """Returns True if the entry order was filled."""
        return self.__entry_order is not None and self.__entry_order.is_filled

    @property
    def exit_active(self):
        """Returns True if the exit order is active."""
        return self.__exit_order is not None and self.__exit_order.is_active

    @property
    def exit_filled(self):
        """Returns True if the exit order was filled."""
        return self.__exit_order is not None and self.__exit_order.is_filled

    @property
    def entry_order(self):
        """Returns the :class:`chinascope_algotrade.broker.BaseOrder` used to enter the position."""
        return self.__entry_order

    @property
    def exit_order(self):
        """Returns the :class:`chinascope_algotrade.broker.BaseOrder` used to exit the position. If this position hasn't been closed yet, None is returned."""
        return self.__exit_order

    def instrument(self):
        """Returns the instrument used for this position."""
        return self.__entry_order.instrument

    def get_return(self, including_commissions=True):
        """Calculates cumulative percentage returns up to this point.
        If the position is not closed, these will be unrealized returns.

        :param including_commissions: True to include commissions in the calculation.
        :type including_commissions: boolean.
        """

        ret = 0
        price = self.get_last_price()
        if price is not None:
            ret = self.__pos_tracker.getReturn(price, including_commissions)
        return ret

    def get_unrealized_return(self, price=None):
        # Deprecated in v0.15.
        warninghelpers.deprecation_warning("get_unrealized_return will be deprecated in the next version. Please use get_return instead.", stacklevel=2)
        if price is not None:
            raise Exception("Setting the price to get_unrealized_return is no longer supported")
        return self.get_return(False)

    def get_PnL(self, including_commissions=True):
        """Calculates PnL up to this point.
        If the position is not closed, these will be unrealized PnL.

        :param including_commissions: True to include commissions in the calculation.
        :type including_commissions: boolean.
        """

        ret = 0
        price = self.get_last_price()
        if price is not None:
            ret = self.__pos_tracker.get_net_profit(price, including_commissions)
        return ret

    def get_net_profit(self, including_commissions=True):
        # Deprecated in v0.15.
        warninghelpers.deprecation_warning("get_net_profit will be deprecated in the next version. Please use get_PnL instead.", stacklevel=2)
        return self.get_PnL(including_commissions)

    def get_unrealized_net_profit(self, price=None):
        # Deprecated in v0.15.
        warninghelpers.deprecation_warning("get_unrealized_net_profit will be deprecated in the next version. Please use get_PnL instead.", stacklevel=2)
        if price is not None:
            raise Exception("Setting the price to get_unrealized_net_profit is no longer supported")
        return self.get_PnL(False)

    def quantity(self):
        # Deprecated in v0.15.
        warninghelpers.deprecation_warning("quantity will be deprecated in the next version. Please use abs(self.shares()) instead.", stacklevel=2)
        return abs(self.shares())

    def cancel_entry(self):
        """Cancels the entry order if its active."""
        if self.entry_active:
            self.strategy.broker.cancel_order(self.entry_order)

    def cancel_exit(self):
        """Cancels the exit order if its active."""
        if self.exit_active:
            self.strategy.broker.cancel_order(self.exit_order)

    def exit_market(self, good_till_canceled=None):
        """Submits a market order to close this position.

        :param good_till_canceled: True if the exit order is good till canceled. If False then the order gets automatically canceled when the session closes.
        If None, then it will match the entry order.
        :type good_till_canceled: boolean.

        .. note::
            * If the position is closed (entry canceled or exit filled) this won't have any effect.
            * If the exit order for this position is pending, an exception will be raised. The exit order should be canceled first.
            * If the entry order is active, cancellation will be requested.
        """

        self.__state.exit(self, None, None, good_till_canceled)

    def exit_limit(self, limit_price, good_till_canceled=None):
        """Submits a limit order to close this position.

        :param limit_price: The limit price.
        :type limit_price: float.
        :param good_till_canceled: True if the exit order is good till canceled. If False then the order gets automatically canceled when the session closes.
        If None, then it will match the entry order.
        :type good_till_canceled: boolean.

        .. note::
            * If the position is closed (entry canceled or exit filled) this won't have any effect.
            * If the exit order for this position is pending, an exception will be raised. The exit order should be canceled first.
            * If the entry order is active, cancellation will be requested.
        """

        self.__state.exit(self, None, limit_price, good_till_canceled)

    def exit_stop(self, stop_price, good_till_canceled=None):
        """Submits a stop order to close this position.

        :param stop_price: The stop price.
        :type stop_price: float.
        :param good_till_canceled: True if the exit order is good till canceled. If False then the order gets automatically canceled when the session closes.
        If None, then it will match the entry order.
        :type good_till_canceled: boolean.

        .. note::
            * If the position is closed (entry canceled or exit filled) this won't have any effect.
            * If the exit order for this position is pending, an exception will be raised. The exit order should be canceled first.
            * If the entry order is active, cancellation will be requested.
        """

        self.__state.exit(self, stop_price, None, good_till_canceled)

    def exit_stop_limit(self, stop_price, limit_price, good_till_canceled=None):
        """Submits a stop limit order to close this position.

        :param stop_price: The stop price.
        :type stop_price: float.
        :param limit_price: The limit price.
        :type limit_price: float.
        :param good_till_canceled: True if the exit order is good till canceled. If False then the order gets automatically canceled when the session closes.
        If None, then it will match the entry order.
        :type good_till_canceled: boolean.

        .. note::
            * If the position is closed (entry canceled or exit filled) this won't have any effect.
            * If the exit order for this position is pending, an exception will be raised. The exit order should be canceled first.
            * If the entry order is active, cancellation will be requested.
        """

        self.__state.exit(self, stop_price, limit_price, good_till_canceled)

    def exit(self, stop_price=None, limit_price=None, good_till_canceled=None):
        # Deprecated in v0.15.
        if stop_price is None and limit_price is None:
            warninghelpers.deprecation_warning("exit will be deprecated in the next version. Please use exit_market instead.", stacklevel=2)
        elif stop_price is None and limit_price is not None:
            warninghelpers.deprecation_warning("exit will be deprecated in the next version. Please use exit_limit instead.", stacklevel=2)
        elif stop_price is not None and limit_price is None:
            warninghelpers.deprecation_warning("exit will be deprecated in the next version. Please use exit_stop instead.", stacklevel=2)
        elif stop_price is not None and limit_price is not None:
            warninghelpers.deprecation_warning("exit will be deprecated in the next version. Please use exit_stop_limit instead.", stacklevel=2)

        self.__state.exit(self, stop_price, limit_price, good_till_canceled)

    def _submit_exit_order(self, stop_price, limit_price, good_till_canceled):
        assert (not self.exit_active)

        exit_order = self.build_exit_order(stop_price, limit_price)

        # If good_till_canceled was not set, match the entry order.
        if good_till_canceled is None:
            good_till_canceled = self.__entry_order.good_till_canceled
        exit_order.good_till_canceled = good_till_canceled

        exit_order.all_or_none = self.__all_or_none

        self.__submit_and_register_order(exit_order)
        self.__exit_order = exit_order

    def on_order_event(self, order_event):
        self.__update_pos_tracker(order_event)

        order = order_event.order
        if not order.is_active:
            del self.__active_orders[order.id]

        # Update the number of shares.
        if order_event.event_type in (const.OrderStatus.PARTIALLY_FILLED, const.OrderStatus.FILLED):
            exec_info = order_event.event_info
            # round_quantity is used to prevent bugs like the one triggered in testcases.bitstamp_test:TestCase.testRoundingBug
            if order.is_a_buy_order:
                self.__shares = order.instrument_traits.round_quantity(self.__shares + exec_info.quantity)
            else:
                self.__shares = order.instrument_traits.round_quantity(self.__shares - exec_info.quantity)

        self.__state.on_order_event(self, order_event)

    def __update_pos_tracker(self, order_event):
        if order_event.event_type in (const.OrderStatus.PARTIALLY_FILLED, const.OrderStatus.FILLED):
            order = order_event.order
            exec_info = order_event.event_info
            if order.is_a_buy_order:
                self.__pos_tracker.buy(exec_info.quantity, exec_info.price, exec_info.commission)
            else:
                self.__pos_tracker.sell(exec_info.quantity, exec_info.price, exec_info.commission)

    def build_exit_order(self, stop_price, limit_price):
        raise NotImplementedError()

    def is_open(self):
        """Returns True if the position is open."""
        return self.__state.is_open(self)

    @property
    def age(self):
        """Returns the duration in open state.

        :rtype: datetime.timedelta.

        .. note::
            * If the position is open, then the difference between the entry datetime and the datetime of the last bar is returned.
            * If the position is closed, then the difference between the entry datetime and the exit datetime is returned.
        """
        ret = datetime.timedelta()
        if self.__entry_datetime is not None:
            if self.__exit_datetime is not None:
                last = self.__exit_datetime
            else:
                last = self.__strategy.current_datetime
            ret = last - self.__entry_datetime
        return ret


# This class is responsible for order management in long positions.
class LongPosition(Position):
    def __init__(self, strategy, instrument, stop_price, limit_price, quantity, good_till_canceled, all_or_none):
        if limit_price is None and stop_price is None:
            entry_order = strategy.broker.create_market_order(const.OrderAction.BUY, instrument, quantity, False)
        elif limit_price is not None and stop_price is None:
            entry_order = strategy.broker.create_limit_order(const.OrderAction.BUY, instrument, limit_price, quantity)
        elif limit_price is None and stop_price is not None:
            entry_order = strategy.broker.create_stop_order(const.OrderAction.BUY, instrument, stop_price, quantity)
        elif limit_price is not None and stop_price is not None:
            entry_order = strategy.broker.create_stop_limit_order(const.OrderAction.BUY, instrument, stop_price, limit_price, quantity)
        else:
            assert False

        Position.__init__(self, strategy, entry_order, good_till_canceled, all_or_none)

    def build_exit_order(self, stop_price, limit_price):
        quantity = self.shares()
        assert (quantity > 0)
        if limit_price is None and stop_price is None:
            ret = self.strategy.broker.create_market_order(const.OrderAction.SELL, self.instrument, quantity, False)
        elif limit_price is not None and stop_price is None:
            ret = self.strategy.broker.create_limit_order(const.OrderAction.SELL, self.instrument, limit_price, quantity)
        elif limit_price is None and stop_price is not None:
            ret = self.strategy.broker.create_stop_order(const.OrderAction.SELL, self.instrument, stop_price, quantity)
        elif limit_price is not None and stop_price is not None:
            ret = self.strategy.broker.create_stop_limit_order(const.OrderAction.SELL, self.instrument, stop_price, limit_price, quantity)
        else:
            assert False

        return ret


# This class is responsible for order management in short positions.
class ShortPosition(Position):
    def __init__(self, strategy, instrument, stop_price, limit_price, quantity, good_till_canceled, all_or_none):
        if limit_price is None and stop_price is None:
            entry_order = strategy.broker.create_market_order(const.OrderAction.SELL_SHORT, instrument, quantity, False)
        elif limit_price is not None and stop_price is None:
            entry_order = strategy.broker.create_limit_order(const.OrderAction.SELL_SHORT, instrument, limit_price, quantity)
        elif limit_price is None and stop_price is not None:
            entry_order = strategy.broker.create_stop_order(const.OrderAction.SELL_SHORT, instrument, stop_price, quantity)
        elif limit_price is not None and stop_price is not None:
            entry_order = strategy.broker.create_stop_limit_order(const.OrderAction.SELL_SHORT, instrument, stop_price, limit_price, quantity)
        else:
            assert False

        Position.__init__(self, strategy, entry_order, good_till_canceled, all_or_none)

    def build_exit_order(self, stop_price, limit_price):
        quantity = self.shares() * -1
        assert (quantity > 0)
        if limit_price is None and stop_price is None:
            ret = self.strategy.broker.create_market_order(const.OrderAction.BUY_TO_COVER, self.instrument, quantity, False)
        elif limit_price is not None and stop_price is None:
            ret = self.strategy.broker.create_limit_order(const.OrderAction.BUY_TO_COVER, self.instrument, limit_price, quantity)
        elif limit_price is None and stop_price is not None:
            ret = self.strategy.broker.create_stop_order(const.OrderAction.BUY_TO_COVER, self.instrument, stop_price, quantity)
        elif limit_price is not None and stop_price is not None:
            ret = self.strategy.broker.create_stop_limit_order(const.OrderAction.BUY_TO_COVER, self.instrument, stop_price, limit_price, quantity)
        else:
            assert False

        return ret
