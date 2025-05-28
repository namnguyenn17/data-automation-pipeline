"""
Configuration module for the Data Automation Pipeline.

This module handles all configuration settings including:
- API configurations (Yahoo Finance)
- Stock symbol list
- Supabase settings
- Report formats
- Logging configuration
"""

import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Main configuration class for the pipeline."""
    
    # Yahoo Finance does not require API key
    YAHOO_FINANCE_ENABLED = True
    
    # OpenAI Configuration for AI Analysis
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    
    # Stock symbols (comma-separated in .env)
    STOCK_SYMBOLS = os.getenv('STOCK_SYMBOLS', 'AMD,NVDA,BYD,TSLA').split(',')

    # Technical Indicators Configuration
    TECHNICAL_INDICATORS = {
        'SMA': [20, 50],  # Simple Moving Average periods
        'EMA': [12, 26],  # Exponential Moving Average periods
        'RSI': 14,        # RSI period
        'MACD': [12, 26, 9],  # MACD fast, slow, signal periods
        'BBANDS': [20, 2]     # Bollinger Bands period, std deviation
    }

    # Data directories
    DATA_RAW_DIR = "data/raw"
    DATA_PROCESSED_DIR = "data/processed"
    REPORTS_DIR = os.getenv('REPORT_OUTPUT_DIR', 'reports')
    TEMPLATES_DIR = "templates"
    LOGS_DIR = "logs"

    # Report Configuration
    REPORT_FORMATS = os.getenv('REPORT_FORMATS', 'html,excel').split(',')

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/pipeline.log')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Report styling
    CHART_STYLE = 'plotly_white'
    CHART_WIDTH = 1200
    CHART_HEIGHT = 600

    @classmethod
    def get_stock_symbols(cls) -> List[str]:
        """Get list of stock symbols for analysis."""
        return [s.strip().upper() for s in cls.STOCK_SYMBOLS if s.strip()]

    @classmethod
    def get_current_stock_pair(cls) -> List[str]:
        """Get current stock symbols as a pair/list for comparison."""
        symbols = cls.get_stock_symbols()
        return symbols[:2] if len(symbols) >= 2 else symbols

    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate configuration settings."""
        validation = {
            'stock_symbols_present': bool(cls.get_stock_symbols()),
            'directories_exist': all([
                os.path.exists(cls.DATA_RAW_DIR) or os.makedirs(cls.DATA_RAW_DIR, exist_ok=True),
                os.path.exists(cls.DATA_PROCESSED_DIR) or os.makedirs(cls.DATA_PROCESSED_DIR, exist_ok=True),
                os.path.exists(cls.REPORTS_DIR) or os.makedirs(cls.REPORTS_DIR, exist_ok=True),
                os.path.exists(cls.LOGS_DIR) or os.makedirs(cls.LOGS_DIR, exist_ok=True)
            ])
        }
        return validation

    @classmethod
    def setup_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR, 
            cls.REPORTS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True) 