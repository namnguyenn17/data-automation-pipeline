#!/usr/bin/env python3
"""
Test script for the Data Automation Pipeline.

This script demonstrates the pipeline functionality using mock data
when no API key is available.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add scripts to path
sys.path.append('scripts')

from scripts.data_automation_pipeline.config import Config
from scripts.data_automation_pipeline.data_processor import DataProcessor
from scripts.data_automation_pipeline.storage import DatabaseManager
from scripts.data_automation_pipeline.report_generator import ReportGenerator
from scripts.api_client import YahooFinanceClient
from scripts.storage import SupabaseManager


def create_mock_stock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create mock stock data for testing."""
    print(f"Creating mock data for {symbol}...")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove weekends (basic approximation)
    dates = dates[dates.weekday < 5]
    
    # Generate realistic stock price data
    np.random.seed(42 if symbol == 'AMD' else 123)  # Different seeds for different stocks
    
    # Starting price
    if symbol == 'AMD':
        start_price = 120.0
        volatility = 0.03
    else:  # NVDA
        start_price = 800.0
        volatility = 0.035
    
    # Generate price series using random walk
    returns = np.random.normal(0.001, volatility, len(dates))  # Slight upward bias
    prices = [start_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_range = close * np.random.uniform(0.01, 0.05)
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)
        
        # Ensure OHLC relationships are maintained
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = int(np.random.uniform(1000000, 10000000))
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'symbol': symbol
        })
    
    df = pd.DataFrame(data, index=dates)
    print(f"Created {len(df)} records for {symbol}")
    return df


def test_pipeline_components():
    """Test individual pipeline components."""
    print("🧪 Testing Data Automation Pipeline Components")
    print("=" * 60)
    
    # Create mock data
    symbols = ['AMD', 'NVDA']
    mock_data = {}
    
    for symbol in symbols:
        mock_data[symbol] = create_mock_stock_data(symbol)
    
    # Test Data Processor
    print("\n📊 Testing Data Processor...")
    processor = DataProcessor()
    
    processed_data = {}
    for symbol, df in mock_data.items():
        print(f"Processing {symbol}...")
        cleaned_df = processor.clean_stock_data(df)
        enhanced_df = processor.calculate_technical_indicators(cleaned_df)
        processed_data[symbol] = enhanced_df
        
        # Generate statistics
        stats = processor.generate_summary_statistics(enhanced_df, symbol)
        print(f"  Current Price: ${stats.get('current_price', 0):.2f}")
        print(f"  Daily Change: {stats.get('price_change_pct_1d', 0):.2f}%")
        print(f"  Volatility: {stats.get('volatility', 0):.2f}%")
    
    # Create comparison data
    comparison_df = processor.create_comparison_dataframe(processed_data)
    print(f"Created comparison dataset with {len(comparison_df)} records")
    
    # Test Database Manager
    print("\n💾 Testing Database Manager...")
    db_manager = DatabaseManager()
    
    for symbol, df in processed_data.items():
        records_saved = db_manager.save_stock_data(df, symbol)
        print(f"Saved {records_saved} records for {symbol}")
        
        # Save technical indicators
        db_manager.save_technical_indicators(df, symbol)
        
        # Save statistics
        stats = processor.generate_summary_statistics(df, symbol)
        db_manager.save_summary_statistics(stats, symbol)
    
    # Test Report Generator
    print("\n📈 Testing Report Generator...")
    report_generator = ReportGenerator()
    
    # Generate statistics for reports
    stats_dict = {}
    for symbol, df in processed_data.items():
        stats_dict[symbol] = processor.generate_summary_statistics(df, symbol)
    
    # Generate reports
    reports = report_generator.generate_comprehensive_report(
        data_dict=processed_data,
        comparison_df=comparison_df,
        stats_dict=stats_dict,
        timestamp=datetime.now()
    )
    
    print(f"Generated {len(reports)} report files:")
    for report_type, filepath in reports.items():
        print(f"  - {report_type.upper()}: {filepath}")
    
    # Database stats
    print("\n📊 Database Statistics:")
    db_stats = db_manager.get_database_stats()
    for key, value in db_stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Pipeline test completed successfully!")
    print("\n🎯 Next Steps:")
    print("1. Get a free Alpha Vantage API key from: https://www.alphavantage.co/support/#api-key")
    print("2. Add your API key to the .env file")
    print("3. Run the full pipeline with: uv run python scripts/data_automation_pipeline/main.py")
    
    return reports


def test_yahoo_finance_client():
    client = YahooFinanceClient()
    symbols = Config.get_stock_symbols()
    for symbol in symbols:
        df = client.get_daily_prices(symbol, period='5d')
        print(f"{symbol} - {len(df)} rows")
        print(df.head())

def test_supabase_manager():
    # Use a small DataFrame for testing
    data = {
        'date': pd.date_range(end=pd.Timestamp.today(), periods=3),
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [104, 105, 106],
        'volume': [1000, 1100, 1200],
        'symbol': ['TEST']*3
    }
    df = pd.DataFrame(data)
    manager = SupabaseManager()
    n = manager.save_stock_data(df, 'TEST')
    print(f"Inserted {n} records into Supabase.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Yahoo Finance client...")
    test_yahoo_finance_client()
    print("Testing Supabase manager...")
    test_supabase_manager()
    
    # Setup directories
    Config.setup_directories()
    
    # Run tests
    test_pipeline_components() 