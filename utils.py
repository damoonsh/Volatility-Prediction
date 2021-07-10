import pandas as pd

def rmspe(y_true, y_pred):
    return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_true))))*100


def validate(model):
    val_data = dval if type(model) == xgb.core.Booster else X_val
    y_pred = model.predict(val_data)
    print(
        f'MAE: {mae(y_pred, y_val)}, \n R2: {r2(y_pred, y_val)} \n RMSPE: {rmspe(y_val, y_pred)}')


def log_return(stock_prices):
    return np.log(stock_prices).diff()


class DataManager:
    """ Reads in parquet files from both book and trade data
    and aggregates the data into one dataset."""

    def __init__(self, train=True):
        self._train = train
        self._book_file_list = TRAIN_BOOK_PATHS if train else TEST_BOOK_PATHS
        self._trade_file_list = TRAIN_TRADE_PATHS if train else TEST_TRADE_PATHS
        self.measures_list = []

    def _traverse_book(self):
        """ Goes through each of the training files. """
        for book_file_path, trade_file_path in zip(self._book_file_list, self._trade_file_list):
            stock_id = book_file_path.split("=")[1]  # Getting the stock_id

            # Reading the book info and preparing it for aggregation
            book = pd.read_parquet(book_file_path)

            book.sort_values(by=['time_id', 'seconds_in_bucket'])
            book['wap1'] = (book['bid_price1'] * book['ask_size1'] + book['ask_price1']
                            * book['bid_size1']) / (book['bid_size1'] + book['ask_size1'])
#             print(book.isna().sum(), book['wap1'].std())
            book['log_return1'] = book.groupby(
                ['time_id'])['wap1'].apply(log_return)
            book = book[~book['log_return1'].isnull()]

            book['wap2'] = (book['bid_price2'] * book['ask_size2'] + book['ask_price2']
                            * book['bid_size2']) / (book['bid_size2'] + book['ask_size2'])
            book['log_return2'] = book.groupby(
                ['time_id'])['wap2'].apply(log_return)
            book = book[~book['log_return2'].isnull()]

            # Different spreads: Get the max of these for each time_id
            book['h_spread_l1'] = book['ask_price1'] - book['bid_price1']
            book['h_spread_l2'] = book['ask_price2'] - book['bid_price2']
            book['v_spread_b'] = book['bid_price1'] - book['bid_price2']
            book['v_spread_a'] = book['ask_price1'] - book['bid_price2']

            book.loc[:, 'bas'] = (book.loc[:, ('ask_price1', 'ask_price2')].min(
                axis=1) / book.loc[:, ('bid_price1', 'bid_price2')].max(axis=1) - 1)

            # Reading the trade info
            trade = pd.read_parquet(trade_file_path)

            # Slicing the train data based on stock_id
            book_stock_slice = train[train['stock_id'] == int(stock_id)]

            for time_id in book['time_id'].unique():
                # Slicing based on time_id
                book_slice = book[book['time_id'] == time_id]
                # Features
                dic = {
                    # Fixing row-id from here
                    'row_id': f"{stock_id}-{time_id}",
                    'wap1_mean': book_slice['wap1'].mean(),
                    'wap1_std': book_slice['wap1'].std(),
                    'wap2_mean': book_slice['wap2'].mean(),
                    'wap2_std': book_slice['wap2'].std(),
                    'h_spread_l1': book['h_spread_l1'].max(),
                    'h_spread_l2': book['h_spread_l2'].max(),
                    'v_spread_b': book['v_spread_b'].max(),
                    'v_spread_a': book['v_spread_a'].max(),
                    'log_return1_mean': book_slice['log_return1'].mean(),
                    'log_return1_std': book_slice['log_return1'].std(),
                    'log_return2_mean': book_slice['log_return2'].mean(),
                    'log_return2_std': book_slice['log_return2'].std(),
                    'bas_mean': book_slice['bas'].mean(),
                    'bas_std': book_slice['bas'].std(),
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
                    'size_mean': trade['size'].mean(),
                    'size_std': trade['size'].std(),
                    'order_count_mean': trade['order_count'].mean(),
                    'order_count_std': trade['order_count'].std(),
                }

                # Note: When getting the test_data ready, there is no target column.
                if self._train:
                    dic['target'] = book_stock_slice[book_stock_slice['time_id']
                                                     == time_id]['target'].values[0]

                self.measures_list.append(dic)

    def get_processed(self):
        """ Returns the processed the data. """
        self._traverse_book()
#         print(self.measures_list)
        return pd.DataFrame(self.measures_list)
