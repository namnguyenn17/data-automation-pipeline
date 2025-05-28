"""
Yahoo Finance API Client for the Data Automation Pipeline.

This module handles all interactions with Yahoo Finance using yfinance including:
- Stock price data retrieval
- Company overview information
- Error handling
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import yfinance as yf

from scripts.config import Config


class YahooFinanceClient:
    """Client for interacting with Yahoo Finance API using yfinance."""
    
    def __init__(self):
        self.logger = logging.getLogger('YahooFinanceClient')

    def get_daily_prices(self, symbol: str, period: str = '100d') -> pd.DataFrame:
        """
        Get daily OHLCV prices for a symbol for the last 100 days (default).
        Args:
            symbol: Stock symbol
            period: Period string for yfinance (e.g., '100d', '1y')
        Returns:
            DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d', auto_adjust=True)
            if df.empty:
                self.logger.warning(f"No data returned for {symbol} from Yahoo Finance.")
                return pd.DataFrame()
            df = df.reset_index()
            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            df['symbol'] = symbol
            self.logger.info(f"Retrieved {len(df)} records for {symbol} from Yahoo Finance")
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self.logger.info(f"Retrieved company overview for {symbol}")
            return info
        except Exception as e:
            self.logger.error(f"Failed to get company overview for {symbol}: {e}")
            return {}
    
    def get_multiple_stocks_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get daily stock data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        stock_data = {}
        
        for symbol in symbols:
            try:
                df = self.get_daily_prices(symbol)
                stock_data[symbol] = df
                self.logger.info(f"Successfully retrieved data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to retrieve data for {symbol}: {e}")
                stock_data[symbol] = pd.DataFrame()  # Empty DataFrame for failed requests
        
        return stock_data
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], timestamp: str = None) -> Dict[str, str]:
        """
        Save raw data to files.
        
        Args:
            data: Dictionary of DataFrames by symbol
            timestamp: Timestamp for file naming
            
        Returns:
            Dictionary mapping symbols to file paths
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        file_paths = {}
        
        for symbol, df in data.items():
            if not df.empty:
                filename = f"{symbol}_daily_{timestamp}.csv"
                filepath = f"{Config.DATA_RAW_DIR}/{filename}"
                df.to_csv(filepath, index=False)
                file_paths[symbol] = filepath
                self.logger.info(f"Saved raw data for {symbol} to {filepath}")
        
        return file_paths 