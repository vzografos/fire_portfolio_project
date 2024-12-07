"""
Example usage:

    etf_symbols = ["SPY", "QQQ", "IWM", "EFA", "AGG"]

    # Initialize the components
    fetching_engine = DataFetchingEngine()
    data_collector = DataCollector(fetching_engine)
    etf_manager = ETFDataManager(data_collector)
    etf_data = etf_manager.get_etf_data(etf_symbols)
    etf_manager.save_data(etf_data, "etf_data.csv")
"""

import yfinance as yf
from datetime import datetime, timedelta
import logging


from src.data_processing.data_preprocesor import DataPreprocessor


class DataFetchingEngine:
    def __init__(self):
        self.data_sources = {"yfinance": self.fetch_from_yfinance}
        # add other sources here

    def fetch_data(self, source, symbols, start_date, end_date):
        if source not in self.data_sources:
            raise ValueError(f"Unsupported data source: {source}")
        return self.data_sources[source](symbols, start_date, end_date)

    def fetch_from_yfinance(self, symbols, start_date, end_date):
        try:
            data = yf.download(symbols, start=start_date, end=end_date)
            return data["Adj Close"]
        except Exception as e:
            logging.error(f"Error fetching data from yfinance: {str(e)}")
            raise


class DataCollector:
    def __init__(self, fetching_engine, source):
        self.fetching_engine = fetching_engine
        self.source = source

    def collect_data(self, symbols, start_date, end_date):
        try:
            logging.info(
                f"Collecting data for {len(symbols)} symbols from {self.source}"
            )
            data = self.fetching_engine.fetch_data(
                self.source, symbols, start_date, end_date
            )
            logging.info(f"Data collected successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Error in data collection: {str(e)}")
            raise


class ETFDataManager:
    def __init__(self, data_collector):
        self.data_collector = data_collector
        self.raw_data = None
        self.processed_data = None

    def get_etf_data(self, symbols, years=5):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        self.raw_data = self.data_collector.collect_data(symbols, start_date, end_date)
        return self.raw_data

    def preprocess_data(self, preprocessing_functions=None):
        if self.raw_data is None:
            raise ValueError("No raw data available. Please call get_etf_data first.")

        preprocessor = DataPreprocessor(self.raw_data, preprocessing_functions)
        self.processed_data = preprocessor.preprocess()
        return self.processed_data

    def save_data(self, filename, processed=True):
        data_to_save = (
            self.processed_data
            if processed and self.processed_data is not None
            else self.raw_data
        )
        if data_to_save is None:
            raise ValueError("No data available to save.")

        try:
            data_to_save.to_csv(filename)
            logging.info(f"Data saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")
            raise
