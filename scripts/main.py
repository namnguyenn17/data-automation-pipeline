"""
Main Pipeline Orchestrator for the Data Automation Pipeline.

This module coordinates all components:
- Data acquisition from Yahoo Finance API
- Data cleaning and processing
- Technical analysis calculation
- Data storage and persistence (Supabase)
- Report generation in multiple formats
- Logging and error handling
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
import os
import pandas as pd

# Add the scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.config import Config
from scripts.api_client import YahooFinanceClient
from scripts.data_processor import DataProcessor
from scripts.report_generator import ReportGenerator


class DataAutomationPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.logger = self._setup_logging()
        self.start_time = None
        
        # Initialize components
        try:
            self.api_client = YahooFinanceClient()
            self.data_processor = DataProcessor()
            self.report_generator = ReportGenerator()
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create logger
        logger = logging.getLogger('data_automation_pipeline')
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatters
        formatter = logging.Formatter(Config.LOG_FORMAT)
        
        # File handler
        Config.setup_directories()
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_pipeline(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run the complete data automation pipeline.
        
        Args:
            force_refresh: Force refresh of all data regardless of last update
            
        Returns:
            Dictionary with pipeline execution results
        """
        self.start_time = time.time()
        execution_results = {
            'success': False,
            'symbols_processed': [],
            'records_processed': 0,
            'reports_generated': [],
            'execution_time': 0,
            'error_message': None
        }
        
        try:
            self.logger.info("="*60)
            self.logger.info("Starting Data Automation Pipeline")
            self.logger.info("="*60)
            
            # Validate configuration
            self._validate_configuration()
            
            # Get stock symbols to process
            symbols = Config.get_stock_symbols()
            self.logger.info(f"Processing symbols: {', '.join(symbols)}")
            
            # Step 1: Data Acquisition
            self.logger.info("Step 1: Data Acquisition")
            raw_data = self._acquire_data(symbols, force_refresh)
            
            # Step 2: Data Processing
            self.logger.info("Step 2: Data Processing and Cleaning")
            processed_data = self._process_data(raw_data)
            
            # Step 3: Data Storage
            self.logger.info("Step 3: Data Storage (Supabase)")
            storage_results = self._store_data(processed_data)
            
            # Step 4: Analysis and Statistics
            self.logger.info("Step 4: Analysis and Statistics Generation")
            analysis_results = self._perform_analysis(processed_data)
            
            # Step 5: Report Generation
            self.logger.info("Step 5: Report Generation")
            reports = self._generate_reports(processed_data, analysis_results)
            
            # Update execution results
            execution_results.update({
                'success': True,
                'symbols_processed': symbols,
                'records_processed': sum(len(df) for df in processed_data.values()),
                'reports_generated': list(reports.keys()) if reports else [],
                'execution_time': time.time() - self.start_time
            })
            
            self.logger.info("Pipeline completed successfully")
            self._log_execution_summary(execution_results)
            
        except Exception as e:
            execution_results.update({
                'success': False,
                'error_message': str(e),
                'execution_time': time.time() - self.start_time if self.start_time else 0
            })
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
        
        finally:
            # Optionally, record pipeline run in Supabase or logs
            pass
        
        return execution_results
    
    def _validate_configuration(self) -> None:
        """Validate pipeline configuration."""
        self.logger.info("Validating configuration...")
        
        validation = Config.validate_config()
        
        if not validation['supabase_url_present']:
            raise ValueError("Supabase URL is not configured")
        if not validation['supabase_anon_key_present']:
            raise ValueError("Supabase anon key is not configured")
        if not validation['stock_symbols_present']:
            raise ValueError("No stock symbols configured")
        if not validation['directories_exist']:
            self.logger.warning("Some directories were missing and have been created")
        
        self.logger.info("Configuration validation completed")
    
    def _acquire_data(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """Acquire data from Yahoo Finance."""
        self.logger.info(f"Acquiring data for {len(symbols)} symbols...")
        raw_data = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching data for {symbol} from Yahoo Finance")
                df = self.api_client.get_daily_prices(symbol, period='100d')
                if not df.empty:
                    raw_data[symbol] = df
                    # Save raw data
                    filename = f"{symbol}_daily_{timestamp}.csv"
                    filepath = f"{Config.DATA_RAW_DIR}/{filename}"
                    df.to_csv(filepath)
                    self.logger.info(f"Saved raw data: {filepath}")
                else:
                    self.logger.warning(f"No data received for {symbol}")
                    raw_data[symbol] = df
            except Exception as e:
                self.logger.error(f"Failed to acquire data for {symbol}: {e}")
                raw_data[symbol] = pd.DataFrame()  # Empty DataFrame
        total_records = sum(len(df) for df in raw_data.values())
        self.logger.info(f"Data acquisition completed: {total_records} total records")
        return raw_data
    
    def _process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean the raw data."""
        self.logger.info("Processing and cleaning data...")
        processed_data = {}
        for symbol, df in raw_data.items():
            try:
                if df.empty:
                    self.logger.warning(f"Skipping {symbol}: no data available")
                    processed_data[symbol] = df
                    continue
                # Clean the data
                cleaned_df = self.data_processor.clean_stock_data(df)
                # Calculate technical indicators
                enhanced_df = self.data_processor.calculate_technical_indicators(cleaned_df)
                processed_data[symbol] = enhanced_df
                self.logger.info(f"Processed {len(enhanced_df)} records for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to process data for {symbol}: {e}")
                processed_data[symbol] = pd.DataFrame()
        # Create comparison DataFrame
        if len(processed_data) >= 2:
            try:
                comparison_df = self.data_processor.create_comparison_dataframe(processed_data)
                processed_data['_comparison'] = comparison_df
                self.logger.info(f"Created comparison dataset with {len(comparison_df)} records")
            except Exception as e:
                self.logger.error(f"Failed to create comparison data: {e}")
        total_processed = sum(len(df) for symbol, df in processed_data.items() 
                            if not symbol.startswith('_'))
        self.logger.info(f"Data processing completed: {total_processed} total processed records")
        return processed_data
    
    def _store_data(self, processed_data: Dict[str, Any]) -> Dict[str, int]:
        """Store processed data in Supabase."""
        self.logger.info("Storing data in Supabase...")
        storage_results = {}
        for symbol, df in processed_data.items():
            if symbol.startswith('_'):
                continue
            try:
                if df.empty:
                    storage_results[symbol] = 0
                    continue
                # Save stock price data
                records_saved = self.db_manager.save_stock_data(df, symbol)
                storage_results[symbol] = records_saved
            except Exception as e:
                self.logger.error(f"Failed to store data for {symbol}: {e}")
                storage_results[symbol] = 0
        total_stored = sum(storage_results.values())
        self.logger.info(f"Data storage completed: {total_stored} total records stored")
        return storage_results
    
    def _perform_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis."""
        self.logger.info("Performing statistical analysis...")
        analysis_results = {
            'statistics': {},
            'comparison_data': processed_data.get('_comparison', pd.DataFrame())
        }
        for symbol, df in processed_data.items():
            if symbol.startswith('_'):
                continue
            try:
                if df.empty:
                    analysis_results['statistics'][symbol] = {}
                    continue
                stats = self.data_processor.generate_summary_statistics(df, symbol)
                analysis_results['statistics'][symbol] = stats
            except Exception as e:
                self.logger.error(f"Failed to analyze data for {symbol}: {e}")
                analysis_results['statistics'][symbol] = {}
        symbols_analyzed = len([s for s in analysis_results['statistics'] if analysis_results['statistics'][s]])
        self.logger.info(f"Analysis completed for {symbols_analyzed} symbols")
        return analysis_results
    
    def _generate_reports(self, processed_data: Dict[str, Any], 
                         analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate reports in multiple formats."""
        self.logger.info("Generating reports...")
        try:
            # Filter out special DataFrames
            data_for_reports = {symbol: df for symbol, df in processed_data.items() 
                              if not symbol.startswith('_')}
            reports = self.report_generator.generate_comprehensive_report(
                data_dict=data_for_reports,
                comparison_df=analysis_results['comparison_data'],
                stats_dict=analysis_results['statistics'],
                timestamp=datetime.now()
            )
            if reports:
                self.logger.info(f"Generated {len(reports)} report files:")
                for report_type, filepath in reports.items():
                    self.logger.info(f"  - {report_type.upper()}: {filepath}")
            else:
                self.logger.warning("No reports were generated")
            return reports
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            return {}
    
    def _log_execution_summary(self, results: Dict[str, Any]) -> None:
        """Log execution summary."""
        self.logger.info("="*60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
        self.logger.info(f"Symbols Processed: {', '.join(results['symbols_processed'])}")
        self.logger.info(f"Records Processed: {results['records_processed']:,}")
        self.logger.info(f"Reports Generated: {', '.join(results['reports_generated']) if results['reports_generated'] else 'None'}")
        self.logger.info(f"Execution Time: {results['execution_time']:.2f} seconds")
        if not results['success'] and results.get('error_message'):
            self.logger.info(f"Error: {results['error_message']}")
        self.logger.info("="*60)


def main():
    """Main entry point for the pipeline."""
    pipeline = DataAutomationPipeline()
    
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='Data Automation Pipeline')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh of all data regardless of last update')
    args = parser.parse_args()
    
    # Run the pipeline
    results = pipeline.run_pipeline(force_refresh=args.force_refresh)
    
    # Exit with appropriate code
    exit_code = 0 if results['success'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 