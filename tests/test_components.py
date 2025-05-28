#!/usr/bin/env python3
"""
Test script to debug data acquisition and processing components.
"""

from scripts.api_client import YahooFinanceClient
from scripts.data_processor import DataProcessor
import pandas as pd

def test_components():
    # Test data acquisition
    client = YahooFinanceClient()
    processor = DataProcessor()

    print('Testing data acquisition...')
    df = client.get_daily_prices('AMD', period='5d')
    print(f'Raw data shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print(df.head())

    print('\nTesting data processing...')
    cleaned_df = processor.clean_stock_data(df)
    print(f'Cleaned data shape: {cleaned_df.shape}')
    print(cleaned_df.head())

    print('\nTesting technical indicators...')
    enhanced_df = processor.calculate_technical_indicators(cleaned_df)
    print(f'Enhanced data shape: {enhanced_df.shape}')
    print(f'New columns: {[col for col in enhanced_df.columns if col not in df.columns]}')

if __name__ == "__main__":
    test_components() 