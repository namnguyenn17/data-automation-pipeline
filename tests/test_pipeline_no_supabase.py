#!/usr/bin/env python3
"""
Test pipeline without Supabase - just data processing and report generation.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from scripts.config import Config
from scripts.api_client import YahooFinanceClient
from scripts.data_processor import DataProcessor
from scripts.report_generator import ReportGenerator

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('test_pipeline')

def test_pipeline():
    """Test the pipeline without Supabase."""
    logger = setup_logging()
    
    logger.info("Starting test pipeline (no Supabase)")
    
    # Initialize components
    api_client = YahooFinanceClient()
    data_processor = DataProcessor()
    report_generator = ReportGenerator()
    
    # Get stock symbols
    symbols = ['AMD', 'NVDA', 'BYD', 'TSLA']
    logger.info(f"Processing symbols: {', '.join(symbols)}")
    
    # Step 1: Data Acquisition
    logger.info("Step 1: Data Acquisition")
    raw_data = {}
    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}")
            df = api_client.get_daily_prices(symbol, period='100d')
            raw_data[symbol] = df
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            raw_data[symbol] = pd.DataFrame()
    
    # Step 2: Data Processing
    logger.info("Step 2: Data Processing")
    processed_data = {}
    for symbol, df in raw_data.items():
        try:
            if df.empty:
                logger.warning(f"No data for {symbol}")
                processed_data[symbol] = df
                continue
            
            # Clean the data
            cleaned_df = data_processor.clean_stock_data(df)
            # Calculate technical indicators
            enhanced_df = data_processor.calculate_technical_indicators(cleaned_df)
            processed_data[symbol] = enhanced_df
            logger.info(f"Processed {len(enhanced_df)} records for {symbol}")
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            processed_data[symbol] = pd.DataFrame()
    
    # Create comparison data
    try:
        comparison_df = data_processor.create_comparison_dataframe(processed_data)
        logger.info(f"Created comparison data with {len(comparison_df)} records")
    except Exception as e:
        logger.error(f"Failed to create comparison data: {e}")
        comparison_df = pd.DataFrame()
    
    # Step 3: Generate Statistics
    logger.info("Step 3: Generate Statistics")
    stats_dict = {}
    for symbol, df in processed_data.items():
        try:
            if not df.empty:
                stats = data_processor.generate_summary_statistics(df, symbol)
                stats_dict[symbol] = stats
                logger.info(f"Generated stats for {symbol}: {len(stats)} metrics")
        except Exception as e:
            logger.error(f"Failed to generate stats for {symbol}: {e}")
            stats_dict[symbol] = {}
    
    # Step 4: Generate Reports
    logger.info("Step 4: Generate Reports")
    try:
        reports = report_generator.generate_comprehensive_report(
            data_dict=processed_data,
            comparison_df=comparison_df,
            stats_dict=stats_dict,
            timestamp=datetime.now()
        )
        
        if reports:
            logger.info(f"Generated {len(reports)} reports:")
            for report_type, filepath in reports.items():
                logger.info(f"  - {report_type.upper()}: {filepath}")
        else:
            logger.warning("No reports generated")
            
    except Exception as e:
        logger.error(f"Failed to generate reports: {e}")
    
    logger.info("Test pipeline completed")

if __name__ == "__main__":
    test_pipeline() 