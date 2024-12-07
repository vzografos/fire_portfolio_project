import numpy as np
from scipy import stats
from forex_python.converter import CurrencyRates

import logging


class DataPreprocessor:
    def __init__(self, data, preprocessing_functions=None):
        self.data = data
        self.c = CurrencyRates()
        self.available_functions = {
            "remove_duplicate_indices": self.remove_duplicate_indices,
            "handle_outliers": self.handle_outliers,
            "handle_missing_values": self.handle_missing_values,
            "align_time_series": self.align_time_series,
            "adjust_for_dividends_and_splits": self.adjust_for_dividends_and_splits,
            "convert_currency": self.convert_currency,
        }
        self.preprocessing_functions = preprocessing_functions or list(
            self.available_functions.keys()
        )

    def remove_duplicate_indices(self):
        self.data = self.data[~self.data.index.duplicated(keep="first")]

    def handle_outliers(self, method="iqr", threshold=3):
        if method == "iqr":
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[
                ~((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any(
                    axis=1
                )
            ]
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(self.data))
            self.data = self.data[(z_scores < threshold).all(axis=1)]

    def handle_missing_values(self, method="ffill"):
        if method == "ffill":
            self.data = self.data.ffill()
        elif method == "bfill":
            self.data = self.data.bfill()
        elif method == "interpolate":
            self.data = self.data.interpolate()

    def align_time_series(self):
        self.data = self.data.resample("D").last()  # Resample to daily frequency
        self.data = self.data.fillna(
            method="ffill"
        )  # Forward fill any missing values after resampling

    def adjust_for_dividends_and_splits(self, dividend_data, split_data):
        # NOTE: This is a simplified version. In practice, you'd need detailed dividend and split data for each ETF
        raise NotImplementedError(
            "This function is not yet implemented. Please implement it."
        )
        for etf in self.data.columns:
            if etf in dividend_data:
                self.data[etf] = self.data[etf] - dividend_data[etf].cumsum()
            if etf in split_data:
                self.data[etf] = self.data[etf] / split_data[etf].cumprod()

    def convert_currency(self, target_currency="EUR"):
        raise NotImplementedError(
            "This function is not yet implemented. Please implement it."
        )
        for column in self.data.columns:
            currency = column.split("_")[
                -1
            ]  # NOTE: Assuming currency is the last part of the column name
            if currency != target_currency:
                exchange_rate = self.c.get_rate(currency, target_currency)
                self.data[column] = self.data[column] * exchange_rate

    def preprocess(self, functions=None):
        functions_to_execute = functions or self.preprocessing_functions
        for func_name in functions_to_execute:
            if func_name not in self.available_functions:
                raise ValueError(f"Unknown preprocessing function: {func_name}")
            logging.debug(f"Applying preprocessing function: {func_name}")
            self.available_functions[func_name]()
        return self.data
