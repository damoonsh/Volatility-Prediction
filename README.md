# Competition Description

High volatility is associated to periods of market turbulence and to large price swings, while low volatility describes more calm and quiet markets. For trading firms like Optiver, accurately predicting volatility is essential for the trading of options, whose price is directly related to the volatility of the underlying product. This [competition](https://www.kaggle.com/c/optiver-realized-volatility-prediction) is about predicting the realized volatility. 

The data provided contains stock market data relevant to the practical execution of trades in the financial markets. In particular, it includes order book snapshots and executed trades. With one second resolution, it provides a uniquely fine grained look at the micro-structure of modern financial markets.


# Data Preprocessing

In order to process data effectively one has to know about the competition's data:

**book_[train/test].parquet** A parquet file partitioned by stock_id. Provides order book data on the most competitive buy and sell orders entered into the market. The top two levels of the book are shared. The first level of the book will be more competitive in price terms, it will then receive execution priority over the second level.

1. ***stock_id***: ID code for the stock. Not all stock IDs exist in every time bucket. Parquet coerces this column to the categorical data type when loaded; you may wish to convert it to int8.
2. ***time_id***: ID code for the time bucket. Time IDs are not necessarily sequential but are consistent across all stocks.
seconds_in_bucket: Number of seconds from the start of the bucket, always starting from 0.
3. ***bid_price[1/2]***: Normalized prices of the most/second most competitive buy level.
4. ***ask_price[1/2]***: Normalized prices of the most/second most competitive sell level.
5. ***bid_size[1/2]***: The number of shares on the most/second most competitive buy level.
6. ***ask_size[1/2]***: The number of shares on the most/second most competitive sell level.

**trade_[train/test].parquet** A parquet file partitioned by stock_id. Contains data on trades that actually executed. Usually, in the market, there are more passive buy/sell intention updates (book updates) than actual trades, therefore one may expect this file to be more sparse than the order book.

1. ***stock_id***: Same as above.
2. ***time_id***: Same as above.
3. ***seconds_in_bucket***: Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.
4. ***price***: The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
5. ***size***: The sum number of shares traded.
6. ***order_count***: The number of unique trade orders taking place.

Given this data the following structure is used to aggregate data:

```python
dic = {
    'row_id': f"{stock_id}-{time_id}",
    # Weighted Average Price metrics
    'wap1_mean': book_slice['wap1'].mean(),
    'wap1_std':book_slice['wap1'].std(),
    'wap1_max':book_slice['wap1'].max(),
    # Weighted Average Price metrics            
    'wap2_mean': book_slice['wap2'].mean(),
    'wap2_std':book_slice['wap2'].std(),
    'wap2_max':book_slice['wap2'].max(),

    'h_spread_l1_mean': book['h_spread_l1'].mean(),
    'h_spread_l1_std': book['h_spread_l1'].std(),
    'h_spread_l1_std': book['h_spread_l1'].max(),
                    
    'h_spread_l2_mean': book['h_spread_l2'].mean(),
    'h_spread_l2_std': book['h_spread_l2'].std(),
    'h_spread_l2_max': book['h_spread_l2'].max(),
                    
    'v_spread_b_mean': book['v_spread_b'].mean(),
    'v_spread_b_std': book['v_spread_b'].std(),
    'v_spread_b_max': book['v_spread_b'].max(),
                    
    'v_spread_a_mean': book['v_spread_a'].mean(),
    'v_spread_a_std': book['v_spread_a'].std(),
    'v_spread_a_max': book['v_spread_a'].max(),
                    
    'log_return1_mean': book_slice['log_return1'].mean(),
    'log_return1_std':book_slice['log_return1'].std(),
    'log_return1_max':book_slice['log_return1'].max(),
                    
    'log_return2_mean': book_slice['log_return2'].mean(),
    'log_return2_std':book_slice['log_return2'].std(),
    'log_return2_max':book_slice['log_return2'].max(),
                    
    'bas_mean': book_slice['bas'].mean(),
    'bas_std': book_slice['bas'].std(),
    'bas_max': book_slice['bas'].max(),
                    
    'ask_size_mean': book_slice['ask_size1'].mean(),
    'ask_size_std': book_slice['ask_size1'].std(),
                    
    'ask_price_mean': book_slice['ask_price1'].mean(),
    'ask_price_std': book_slice['ask_price1'].std(),
                    
    'bid_size_mean': book_slice['bid_size1'].mean(),
    'bid_size_std': book_slice['bid_size1'].std(),
                    
    'bid_price_mean': book_slice['bid_price1'].mean(),
    'bid_price_std': book_slice['bid_price1'].std(),
                    
    'actual_price_mean': trade['price'].mean(),
    'actual_price_std': trade['price'].std(),
    'actual_price_max': trade['price'].max(),
                    
    'size_mean': trade['size'].mean(),
    'size_std': trade['size'].std(),
                    
    'order_count_mean': trade['order_count'].mean(),
    'order_count_std': trade['order_count'].std(),
}
```

WAP stands for Weighted Average Price which is calculated using 

$$
(bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
$$

This formula helps us estimate what the price of a stock should be in the given timeframe.

Most of the columns are focused on the measures of central tendency (mean and standard deviation).
