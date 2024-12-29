def GetSinglePnL(
    all_time_hist_data,
    result_hour,
    compound_column_name,
    group_num=20,
    long_factor_combination_list=[1, 2, 3],
    update_position_time=1,
    leverage=1,
    trade_with_rank=0,
):
    logger.info(f"start get SinglePnl: {result_hour}")
    FACTOR_NAME = compound_column_name + f"_{update_position_time}day"

    # future_sharp = future_ret / realized_volatility
    # use it for debug
    fut_sharp = compound_column_name + f"_{update_position_time}day_sharp"
    result_hour = result_hour.with_columns(
        (pl.col(FACTOR_NAME) / pl.col("close_price_volatility")).alias(fut_sharp)
    )

    FUT_N_DAY_RET_COL_NAME = f"open_price_fut_{update_position_time}day_ret"
    future_n_day_open_ret = (
        result_hour[["open_time", "symbol", FUT_N_DAY_RET_COL_NAME]]
        .pivot(index="open_time", columns="symbol", values=FUT_N_DAY_RET_COL_NAME)
        .sort("open_time")
    )


    # 默认情况，做多做空的总金额，是50比50
    # 偶尔会改成 60比 40
    long_scale = (
        result_hour[["close_time", "symbol", f'long_value_scale_{update_position_time}day']]
        .pivot(index="close_time", columns="symbol", values = f'long_value_scale_{update_position_time}day')
        .sort("close_time")
    )

    # 因为此时的因子值是在close_time的时候计算的，所以要用close_time
    factors = (
        result_hour[["close_time", "symbol", FACTOR_NAME]]
        .pivot(index="close_time", columns="symbol", values=FACTOR_NAME)
        .sort("close_time")
    )

    # 每根k线的开盘价
    open_df = (
        result_hour[["open_time", "symbol", "open"]]
        .pivot(index="open_time", columns="symbol", values="open")
        .sort("open_time")
    )

    # all_time_close_return = (
    #     all_time_hist_data[["close_time", "symbol", "close_return"]]
    #     .pivot(index="close_time", columns="symbol", values="close_return")
    #     .sort("close_time")
    # )

    # relalized volatility for OPEN price
    # volatility_open_price = (
    #     result_hour[["open_time", "symbol", "open_price_volatility"]]
    #     .pivot(index="open_time", columns="symbol", values="open_price_volatility")
    #     .sort("open_time")
    # )

    # relalized volatility for CLOSE price
    # volatility_close_price = (
    #     result_hour[["close_time", "symbol", "close_price_volatility"]]
    #     .pivot(index="close_time", columns="symbol", values="close_price_volatility")
    #     .sort("close_time")
    # )

    # # position factor
    # position_factor = (
    #     all_time_hist_data[["close_time", "symbol", "pos_signals"]]
    #     .pivot(index="close_time", columns="symbol", values="pos_signals")
    #     .sort("close_time")
    # )

    bar_close_vol = (
        result_hour[["close_time", "symbol", "volume"]]
        .pivot(index="close_time", columns="symbol", values="volume")
        .sort("close_time")
    )

    columns = future_n_day_open_ret.columns[1:]  # means all symbols
    all_symbols = columns

    # 在每一根k线走完的时候计算因子值，然后调仓
    # 所以要遍历每一个close_time，进行调仓
    factors.sort("close_time")
    time_array = factors["close_time"].to_numpy()

    time_array = time_array[:-1]  # the last line doesn't have next day return

    today_pnl = 1
    pnl = []
    long_stocks = []
    short_stocks = []
    cash = START_CASH
    cur_position = pl.DataFrame(
        {col: [value] for col, value in zip(columns, [0] * len(columns))}
    )
    next_step_position = pl.DataFrame(
        {col: [value] for col, value in zip(columns, [0] * len(columns))}
    )

    # print (f'update_position_time: {update_position_time}')

    # 用日线数据的时候，每一行是一天；小时数据的时候，每24行是一天
    update_row_index = update_position_time
    fut_avg_long_pos_ret, fut_avg_short_pos_ret = 0, 0

    # used to count the PnL for current pos & stop loss early
    cur_pos_open_price: pl.DataFrame = None
    cur_hold_long_symbol_list = []
    cur_hold_short_symbol_list = []

    for i, cur_close_time in enumerate(time_array):

        # 用下一根k线的开盘价作为调仓价格，open_time是1ms后
        time_delta: np.timedelta64 = np.timedelta64(1, "ms")

        next_bar_open_time = cur_close_time + time_delta
        all_open_price_when_open_pos = open_df.filter(
            pl.col("open_time") == next_bar_open_time
        ).drop("open_time")

        # for debug and check pnl
        # 计算 {update_position_time} 天后的平仓时间(也是在开盘价平仓，所以是open_time)
        bar_open_time_to_close_pos = next_bar_open_time + np.timedelta64(
            update_position_time, "D"
        )

        logger.debug(
            f"cur time: {cur_close_time} == next bar open time: {next_bar_open_time} == close pos time: {bar_open_time_to_close_pos}"
        )

        if i % update_row_index == 0:
            # Before update the position, calc the total cash+position value

            assert all_open_price_when_open_pos.columns == cur_position.columns

            # 选取当前时间点的数据
            current_factors = factors.filter(
                pl.col("close_time") == cur_close_time
            ).drop("close_time")

            tradeable_symbol_list = GetTradeableSymbolList(
                current_factors, future_n_day_open_ret, bar_open_time_to_close_pos
            )

            # try to avoid trade the last xx% volume symbol ============
            cur_bar_close_vol = bar_close_vol.filter(
                pl.col("close_time") == cur_close_time
            )

            vol_for_all_symbols = []
            for symbol in tradeable_symbol_list:
                vol_for_all_symbols.append(
                    cur_bar_close_vol[symbol].to_list()[0]
                )
            vol_threshold = np.percentile(vol_for_all_symbols, VOL_FILTER_RATIO)
            logger.debug (f'close time {cur_close_time} == vol_threshold: {vol_threshold}')

            tradeable_symbol_list_with_large_vol = []
            for symbol in tradeable_symbol_list:
                assert len ( cur_bar_close_vol[symbol].to_list()) > 0, f'{ cur_bar_close_vol[symbol].to_list()} == {symbol}'
                if cur_bar_close_vol[symbol].to_list()[0] > vol_threshold:
                    tradeable_symbol_list_with_large_vol.append(symbol)
            logger.debug (f'{cur_close_time} == all trade symbol {len (tradeable_symbol_list)} == remain {len (tradeable_symbol_list_with_large_vol)} after VOL filter')
            tradeable_symbol_list = tradeable_symbol_list_with_large_vol


            if trade_with_rank == 0 and len(tradeable_symbol_list) < group_num:
                # 如果可以交易的symbol数目小于组数，那么无法进行交易
                pnl.append(today_pnl)
                continue

            # 把所有可以交易的symbol的因子值找出来
            long_factors = current_factors.select(tradeable_symbol_list).melt(
                id_vars=[],
                value_vars=tradeable_symbol_list,
                variable_name="symbol",
                value_name="factor_value",
            )

            assert len (tradeable_symbol_list) > 0 and long_factors.shape[0] > 0, f'no tradeable symbol at {cur_close_time}'
            assert (
                long_factors["factor_value"].is_null().sum() == 0
            ), "factor_value column contains null values"
            assert (
                long_factors["factor_value"].is_nan().sum() == 0
            ), "factor_value column contains NaN values"

            # 对全部symbol按照因子值排序
            sorted_long_factors = long_factors.sort("factor_value")

            # 对全部symbol进行分组，计算每个组的symbol数目
            group_size = max(
                int(sorted_long_factors.height / group_num), 1
            )  # 避免除以零

            if trade_with_rank != 0:
                # 基于排序的名次交易（只做多前xx个），而不是基于分组交易
                rank_labels = (pl.arange(0, sorted_long_factors.height)).cast(pl.UInt32)
                sorted_long_factors = sorted_long_factors.with_columns(
                    rank_labels.alias("rank")
                )

                # Create Reverse Rank label
                max_rank = sorted_long_factors.height - 1
                reverse_rank_labels = (max_rank - rank_labels).cast(pl.UInt32)
                sorted_long_factors = sorted_long_factors.with_columns(
                    reverse_rank_labels.alias("reverse_rank")
                )
            else:
                # 创建组标签
                # 使用 pl.arange 生成行索引，然后除以每组大小，并取整获得组号
                group_labels = (
                    pl.arange(0, sorted_long_factors.height) / group_size
                ).cast(pl.UInt32)

                # 将组标签列添加到 DataFrame
                sorted_long_factors = sorted_long_factors.with_columns(
                    group_labels.alias("group")
                )

            logger.debug(f"sorted_long_factors: {sorted_long_factors}")

            # temp for verify the production
            symbol_list = sorted_long_factors['symbol'].to_list()
            # logger.debug (f'long: {symbol_list[-22:]}')
            # logger.debug (f'short: {symbol_list[:12]}')

            if trade_with_rank != 0:
                # 如果 trade_with_rank 不为 0，那么使用 rank 进行交易
                # trade_with_rank 为正数时，做多排名靠前的股票，同时做空排名靠后的股票
                # 为负数时，做空排名靠前的股票, 同时做多排名靠后的股票

                # 有可能 2 * trade_with_rank 的绝对值大于 sorted_long_factors.height(全部symbol数目)
                cur_trade_num = min(
                    abs(trade_with_rank), sorted_long_factors.height / 2
                )

                assert (
                    LONG_TRADE_RANK_RATIO + SHORT_TRADE_RANK_RATIO
                ) * cur_trade_num <= sorted_long_factors.height, (
                    "total of long & short should not exceed total symbol number"
                )

                cur_rank_column = "rank" if trade_with_rank > 0 else "reverse_rank"
                long_stock_set = sorted_long_factors.filter(
                    pl.col(cur_rank_column) < cur_trade_num * LONG_TRADE_RANK_RATIO
                )
                short_stock_set = sorted_long_factors.filter(
                    pl.col(cur_rank_column)
                    > sorted_long_factors.height
                    - 1
                    - cur_trade_num * SHORT_TRADE_RANK_RATIO
                )
            else:
                # 计算做空组的索引
                short_combination_list = [
                    group_num - 1 - x for x in long_factor_combination_list
                ]

                # 提取做多的股票集合
                long_stock_set = sorted_long_factors.filter(
                    pl.col("group").is_in(long_factor_combination_list)
                )
                # 提取做空的股票集合
                short_stock_set = sorted_long_factors.filter(
                    pl.col("group").is_in(short_combination_list)
                )

            long_symbol_list: list = (
                long_stock_set.select("symbol").unique().to_pandas().squeeze().to_list()
            )
            short_symbol_list: list = (
                short_stock_set.select("symbol")
                .unique()
                .to_pandas()
                .squeeze()
                .to_list()
            )

            # 改为每次开仓都使用固定值
            each_side_symbol_total_val = 100000.0  # 单边的总价值
            long_value_ratio = 1.0
            short_value_ratio = 1.0

            if DYNAMIC_POS_SCALE:
                if (
                    fut_avg_long_pos_ret > POS_RET_THRESHOLD
                    and fut_avg_short_pos_ret > POS_RET_THRESHOLD
                ):
                    long_value_ratio, short_value_ratio = 1.2, 0.8
                elif (
                    fut_avg_long_pos_ret < -POS_RET_THRESHOLD
                    and fut_avg_short_pos_ret < -POS_RET_THRESHOLD
                ):
                    long_value_ratio, short_value_ratio = 0.8, 1.2
            
            cur_long_scale = long_scale.filter(pl.col("close_time") == cur_close_time)[long_symbol_list[0]].to_numpy()[0]
            # assert (cur_long_scale == long_value_ratio), f'{cur_close_time}=====cur_long_scale: {cur_long_scale} not match == long_value_ratio: {long_value_ratio}'

            if cur_long_scale != long_value_ratio:
                logger.warning (f'{cur_close_time}=====cur_long_scale: {cur_long_scale} not match == long_value_ratio: {long_value_ratio}')

            assert (
                long_value_ratio + short_value_ratio == 2.0
            ), "sum ratio should be 2.0"

            logger.info (f'cur close time: {cur_close_time} == long_value_ratio: {long_value_ratio} == short_value_ratio: {short_value_ratio} --- {cur_long_scale}')

            total_long_pos_value = each_side_symbol_total_val * long_value_ratio
            total_short_pos_value = -each_side_symbol_total_val * short_value_ratio

            # Initialize long_symbol_pos_ratio_dict
            long_symbol_pos_ratio_dict = {
                symbol: 1.0 / len(long_symbol_list) for symbol in long_symbol_list
            }
            short_symbol_pos_ratio_dict = {
                symbol: 1.0 / len(short_symbol_list) for symbol in short_symbol_list
            }


            # for future calc position ratio
            mean_expected_ret = sorted_long_factors['factor_value'].drop_nans().drop_nulls().mean()
            logger.debug (f'mean expected ret (mean factor): {mean_expected_ret}')

            # 用 next_step_value 计算调仓后，每个symbol的市场价值
            next_step_value = pl.DataFrame({col: [0] for col in all_symbols})
            for symbol in long_symbol_list:
                assert symbol in next_step_value.columns
                next_step_value = next_step_value.with_columns(
                    pl.lit(
                        total_long_pos_value * long_symbol_pos_ratio_dict[symbol]
                    ).alias(symbol)
                )

            # Set short positions
            for symbol in short_symbol_list:
                assert symbol in next_step_value.columns
                next_step_value = next_step_value.with_columns(
                    pl.lit(
                        total_short_pos_value * short_symbol_pos_ratio_dict[symbol]
                    ).alias(symbol)
                )

            # 用调仓后的市场价值, 除以每个symbol的开仓价,得到每个symbol的调仓后的仓位
            next_step_position = next_step_value / all_open_price_when_open_pos

            logger.debug(f"begin to update pos: =========")
            logger.debug(f"next_step_value: {next_step_value}")
            logger.debug(
                f"all_open_price_when_open_pos: {all_open_price_when_open_pos}"
            )

            cur_pos_open_price = all_open_price_when_open_pos.clone()

            # 对于此时没上市的symbol, all_open_price_when_open_pos is None, so the next_step_position will be null
            # then diff_position will be null, which will lead to PnL calc wrong
            # 所以提前把null填充0
            next_step_position = next_step_position.fill_null(0)

            # 需要调仓的仓位变动, 对于每个symbol，相当于是卖出 diff_position个，所以累加到cash里
            diff_position = cur_position - next_step_position
            realized_pnl = diff_position * all_open_price_when_open_pos

            def sum_row(df: pl.DataFrame) -> float:
                return df.select(
                    pl.sum_horizontal(
                        # pl.all().filter(pl.col.is_numeric())
                        pl.all()
                    ).fill_null(0)
                ).item()

            # new pl version
            cash += sum_row(diff_position * all_open_price_when_open_pos)

            abs_diff_position = diff_position.select(
                [pl.col(column).sum().abs() for column in diff_position.columns]
            )
            abs_diff_trading_value = sum_row(
                abs_diff_position * all_open_price_when_open_pos
            )
            # 调仓交易额的手续费
            cash -= abs_diff_trading_value * commission

            cur_position = next_step_position.clone()  # 完成调仓
            cur_hold_long_symbol_list = long_symbol_list
            cur_hold_short_symbol_list = short_symbol_list
            logger.debug(
                f"update position: {cur_close_time} === diff_trading_value: {abs_diff_trading_value} == cash: {cash} "
            )

        latest_pnl = cash + sum_row(all_open_price_when_open_pos * cur_position)

        if len(pnl) > 0:
            logger.info(
                f"pnl: {cur_close_time} {latest_pnl} === {(latest_pnl / pnl[-1] - 1) * 100:.3f}%"
            )
        pnl.append(latest_pnl)
    return pnl, AnalysePnLTrace(pnl), " "