# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import logging

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from scipy import stats


class SupportResistance(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "720": 0,
        # "60": 0.01,
        # "30": 0.02,
        # "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    use_custom_stoploss = True
    stoploss = -1

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    start_bounce_loss = -0.01  # Loss threshold where start to bounce
    max_bounce_loss = -0.05  # Loss for selling bounce
    max_bounce_count = 2  # Number of bounces from support / resistance levels
    min_bounce_diff = 0.005  # Minimum difference between bounces
    min_bounce_interval = 30  # Minimum interval between bounces in minutes

    # Strategy parameters
    # buy_rsi = IntParameter(10, 40, default=30, space="buy")
    # sell_rsi = IntParameter(60, 90, default=70, space="sell")

    plot_config = {
        "main_plot": {
            "resistance_0": {"color": "red"},
            "resistance_1": {"color": "red"},
            "resistance_2": {"color": "red"},
            "support_0": {"color": "green"},
            "support_1": {"color": "green"},
            "support_2": {"color": "green"},
        },
        "subplots": {
        }
    }


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Momentum Indicators
        # ------------------------------------

        def support_agg(x, k=0):
            h, e = np.histogram(x, 50, range=(x.min(), x.iloc[-1]))
            if h.max() < 2:
                return np.NAN
            # h, e = np.histogram(x, 10)
            return e.take(np.argsort(h)[::-1][k])

        def resistance_agg(x, k=0):
            h, e = np.histogram(x, 50, range=(x.iloc[-1], x.max()))
            if h.max() < 2:
                return np.NAN
            # h, e = np.histogram(x, 10)
            return e.take(np.argsort(h)[::-1][k])

        # dataframe["rsi"] = ta.RSI(dataframe)
        # dataframe["tema"] = ta.TEMA(dataframe)

        # dataframe["support_0"] = dataframe["support_1"] = dataframe["support_2"] = dataframe["low"]
        # dataframe["resistance_0"] = dataframe["resistance_1"] = dataframe["resistance_2"] = dataframe["high"]
        # dataframe.update(dataframe.rolling(12).agg({
        #     "support_0": lambda x: support_agg(x),
        #     "resistance_0": lambda x: resistance_agg(x)
        # }))
        # dataframe.update(dataframe.rolling(100).agg({
        #     "support_1": lambda x: support_agg(x),
        #     "resistance_1": lambda x: resistance_agg(x)
        # }))
        # dataframe.update(dataframe.rolling(200).agg({
        #     "support_2": lambda x: support_agg(x),
        #     "resistance_2": lambda x: resistance_agg(x)
        # }))

        dataframe_1h = self.dp.get_pair_dataframe(metadata['pair'], '1h')

        for i, win in enumerate([3, 9, 24]):
            sr_1h = dataframe_1h.rolling(win).agg({
                "low": 'min',
                "high": 'max',
            })
            sr_1h = sr_1h.rename({"low": f"support_{i}", "high": f"resistance_{i}"}, axis=1)
            # We need to shift forward because we need to set up 1H candle close time instead of open time
            sr_1h = pd.concat([dataframe_1h['date'], sr_1h.shift(1)], axis=1)
            dataframe = dataframe.merge(sr_1h, on='date', how='left').ffill()

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]
        """


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # shifted = dataframe.shift(1)
        targets = dataframe["resistance_0"]
        dataframe.loc[
            (
                (dataframe["low"] < dataframe["support_1"]) &
                (targets / dataframe["close"] >= 1.005) &
                (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long"] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'enter_short'] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe["close"], dataframe.shift(1)["resistance_0"])) &  # Signal: RSI crosses above sell_rsi
        #         (dataframe["volume"] > 0)  # Make sure Volume is not 0
        #     ),
        #     "exit_long"] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)

        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'exit_short'] = 1

        return dataframe

    def bot_loop_start(self, **kwargs) -> None:
        for trade in Trade.get_open_trades():
            if trade.get_custom_data('target_rate') is None and trade.is_open:
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                start_candle = dataframe.iloc[-1]
                target = start_candle['resistance_0']
                trade.set_custom_data('target_rate', target)
        return super().bot_loop_start(**kwargs)

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        if current_rate >= last_candle['resistance_0']:
            if current_profit > 0.05:
                return 'immediate_profit'

        if trade.get_custom_data('target_rate'):
            if current_rate >= trade.get_custom_data('target_rate'):
                logging.info(f'Force exit: {current_rate} >= {trade.get_custom_data("target_rate")}')
                return 'target_rate'

        bounce_count = trade.get_custom_data('bounce_count', 0)
        average_profit = (current_rate - trade.open_rate) / trade.open_rate
        if bounce_count > 0 and average_profit >= 0.005:
            return 'bounce_draw'

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> Optional[float]:

        bounce_count = trade.get_custom_data('bounce_count', 0)
        if after_fill and bounce_count >= self.max_bounce_count:
            return self.max_bounce_loss
        return -1

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float, current_entry_rate: float,
                              current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        bounce_count = trade.get_custom_data('bounce_count', 0)
        bounce_rate = trade.get_custom_data('bounce_rate', trade.open_rate)
        if bounce_count >= self.max_bounce_count:
            return None
        if current_entry_rate * (1 + self.min_bounce_diff) >= bounce_rate:
            return None
        if current_profit > self.start_bounce_loss:
            return None
        if current_time - trade.date_last_filled_utc < timedelta(minutes=self.min_bounce_interval):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        if current_rate <= last_candle['support_2']:
            trade.set_custom_data('bounce_count', bounce_count + 1)
            trade.set_custom_data('bounce_rate', current_entry_rate)
            stake = trade.amount * current_entry_rate
            return stake
        return None
