"""
Data Processing module for the Data Automation Pipeline.

This module handles:
- Data cleaning and validation
- Missing data handling
- Data type conversions
- Feature engineering
- Technical indicators calculation
- Data transformation and aggregation
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from scripts.config import Config


class DataProcessor:
    """Handles all data processing and transformation tasks."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data.
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        original_length = len(df)
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle date column - Yahoo Finance returns 'date' column, not index
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
            df_clean.set_index('date', inplace=True)
        elif not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Remove duplicate dates
        df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
        
        # Sort by date
        df_clean.sort_index(inplace=True)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Validate data types
        df_clean = self._validate_data_types(df_clean)
        
        # Remove invalid data (negative prices, zero volume)
        df_clean = self._remove_invalid_data(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        cleaned_length = len(df_clean)
        removed_records = original_length - cleaned_length
        
        if removed_records > 0:
            self.logger.info(f"Cleaned data: removed {removed_records} invalid records")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in stock data."""
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df
        
        self.logger.info(f"Handling {missing_count} missing values")
        
        # For price columns, forward fill then backward fill
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # For volume, fill with median volume
        if 'volume' in df.columns:
            median_volume = df['volume'].median()
            df['volume'] = df['volume'].fillna(median_volume)
        
        # Drop rows that still have missing values
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            df.dropna(inplace=True)
            self.logger.warning(f"Dropped {remaining_missing} rows with persistent missing values")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for stock data."""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _remove_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid stock data records."""
        original_length = len(df)
        
        # Remove rows with negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # Remove rows where high < low (invalid)
        if 'high' in df.columns and 'low' in df.columns:
            df = df[df['high'] >= df['low']]
        
        # Remove rows where close is outside open-high or open-low range
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df = df[
                (df['close'] >= df['low']) & 
                (df['close'] <= df['high'])
            ]
        
        removed = original_length - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} invalid data records")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in stock data using IQR method."""
        # Focus on price changes rather than absolute prices
        if 'close' not in df.columns:
            return df
        
        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # Remove extreme outliers (beyond 3 standard deviations)
        mean_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()
        
        outlier_threshold = 3 * std_return
        outliers = abs(df['daily_return'] - mean_return) > outlier_threshold
        
        if outliers.sum() > 0:
            self.logger.info(f"Identified {outliers.sum()} outlier records")
            # Cap outliers rather than remove them
            df.loc[outliers, 'daily_return'] = np.sign(df.loc[outliers, 'daily_return']) * outlier_threshold
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.
        
        Args:
            df: Stock price DataFrame
            
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df_with_indicators = df.copy()
        
        # Simple Moving Averages
        for period in Config.TECHNICAL_INDICATORS['SMA']:
            df_with_indicators[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in Config.TECHNICAL_INDICATORS['EMA']:
            df_with_indicators[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI (Relative Strength Index)
        df_with_indicators['rsi'] = self._calculate_rsi(df['close'], Config.TECHNICAL_INDICATORS['RSI'])
        
        # MACD
        macd_data = self._calculate_macd(df['close'], *Config.TECHNICAL_INDICATORS['MACD'])
        df_with_indicators = pd.concat([df_with_indicators, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'], *Config.TECHNICAL_INDICATORS['BBANDS'])
        df_with_indicators = pd.concat([df_with_indicators, bb_data], axis=1)
        
        # Additional indicators
        df_with_indicators['daily_return'] = df['close'].pct_change()
        df_with_indicators['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        df_with_indicators['price_change'] = df['close'].diff()
        df_with_indicators['volume_sma_20'] = df['volume'].rolling(20).mean() if 'volume' in df.columns else np.nan
        
        self.logger.info(f"Calculated technical indicators for {len(df_with_indicators)} records")
        return df_with_indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        })
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': upper_band - lower_band,
            'bb_position': (prices - lower_band) / (upper_band - lower_band)
        })
    
    def create_comparison_dataframe(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create a comparison DataFrame for multiple stocks.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            
        Returns:
            Combined DataFrame for comparison analysis
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Filter out empty DataFrames
        valid_data = {symbol: df for symbol, df in data_dict.items() if not df.empty}
        
        if not valid_data:
            self.logger.warning("No valid data available for comparison")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = {}
        
        for symbol, df in valid_data.items():
            # Select key columns for comparison
            if 'close' in df.columns:
                comparison_data[f'{symbol}_close'] = df['close']
                comparison_data[f'{symbol}_volume'] = df.get('volume', np.nan)
                comparison_data[f'{symbol}_daily_return'] = df.get('daily_return', np.nan)
                
                # Technical indicators
                if 'rsi' in df.columns:
                    comparison_data[f'{symbol}_rsi'] = df['rsi']
                if 'sma_20' in df.columns:
                    comparison_data[f'{symbol}_sma_20'] = df['sma_20']
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate relative performance
        symbols = list(valid_data.keys())
        if len(symbols) >= 2:
            close_cols = [f'{sym}_close' for sym in symbols]
            if all(col in comparison_df.columns for col in close_cols):
                # Normalize prices to start from 100
                for col in close_cols:
                    first_valid = comparison_df[col].first_valid_index()
                    if first_valid is not None:
                        comparison_df[f'{col}_normalized'] = (
                            comparison_df[col] / comparison_df.loc[first_valid, col] * 100
                        )
                
                # Calculate spread between the two stocks
                comparison_df['price_spread'] = (
                    comparison_df[close_cols[0]] - comparison_df[close_cols[1]]
                )
                comparison_df['price_ratio'] = (
                    comparison_df[close_cols[0]] / comparison_df[close_cols[1]]
                )
        
        self.logger.info(f"Created comparison DataFrame with {len(comparison_df)} records")
        return comparison_df
    
    def generate_summary_statistics(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, float]:
        """
        Generate summary statistics for stock data.
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary of summary statistics
        """
        if df.empty:
            return {}
        
        stats = {}
        
        # Price statistics
        if 'close' in df.columns:
            stats['current_price'] = df['close'].iloc[-1]
            stats['price_change_1d'] = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) > 1 else 0
            stats['price_change_pct_1d'] = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            stats['max_price'] = df['close'].max()
            stats['min_price'] = df['close'].min()
            stats['avg_price'] = df['close'].mean()
            
            # Performance metrics
            if len(df) >= 30:
                stats['price_change_30d'] = ((df['close'].iloc[-1] / df['close'].iloc[-30]) - 1) * 100
            if len(df) >= 7:
                stats['price_change_7d'] = ((df['close'].iloc[-1] / df['close'].iloc[-7]) - 1) * 100
        
        # Volume statistics
        if 'volume' in df.columns:
            stats['avg_volume'] = df['volume'].mean()
            stats['volume_trend'] = df['volume'].tail(5).mean() / df['volume'].mean() if len(df) > 5 else 1
        
        # Volatility
        if 'daily_return' in df.columns:
            stats['volatility'] = df['daily_return'].std() * np.sqrt(252) * 100  # Annualized
            stats['sharpe_ratio'] = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252) if df['daily_return'].std() > 0 else 0
        
        # Technical indicators
        if 'rsi' in df.columns:
            stats['current_rsi'] = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else None
        
        # Trend analysis
        if 'sma_20' in df.columns and 'close' in df.columns:
            stats['price_vs_sma20'] = ((df['close'].iloc[-1] / df['sma_20'].iloc[-1]) - 1) * 100 if not pd.isna(df['sma_20'].iloc[-1]) else None
        
        if 'sma_50' in df.columns and 'sma_20' in df.columns:
            current_sma20 = df['sma_20'].iloc[-1]
            current_sma50 = df['sma_50'].iloc[-1]
            if not pd.isna(current_sma20) and not pd.isna(current_sma50):
                stats['sma20_vs_sma50'] = ((current_sma20 / current_sma50) - 1) * 100
        
        symbol_info = f" for {symbol}" if symbol else ""
        self.logger.info(f"Generated summary statistics{symbol_info}")
        return stats 