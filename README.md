# Data Automation Pipeline (Yahoo Finance + Supabase)

A comprehensive, automated data pipeline for quantitative research that pulls stock data from Yahoo Finance, processes it with technical indicators, stores it in Supabase (cloud PostgreSQL), and generates professional reports.

## 🎯 **Project Overview**

This pipeline demonstrates **modern data engineering best practices** for financial data analysis:

- **Extract**: Pull real-time stock data from Yahoo Finance (no API key required)
- **Transform**: Clean, validate, and calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Load**: Store processed data in Supabase for persistence and historical analysis
- **Report**: Generate interactive HTML and Excel reports with charts and insights

## 🏗️ **Architecture & Data Flow**

### **Why This Architecture?**

1. **Raw Data Storage** (`data/raw/`): 
   - Preserves original data from Yahoo Finance
   - Enables data lineage and debugging
   - Allows reprocessing without re-fetching

2. **Processed Data** (`data/processed/`):
   - Cleaned, validated data with technical indicators
   - Ready for analysis and reporting
   - Optimized for performance

3. **Supabase Storage**:
   - **Persistence**: Data survives between runs
   - **Historical Analysis**: Build months/years of data
   - **Performance**: Don't re-fetch same data every time
   - **Collaboration**: Multiple users/applications can access
   - **Backup**: Cloud-based, reliable storage
   - **Scalability**: PostgreSQL can handle large datasets

4. **Report Generation**:
   - **HTML**: Interactive charts with Plotly
   - **Excel**: Structured data for further analysis
   - **Professional**: Ready for stakeholders

## 🚀 **Quick Start**

### **1. Prerequisites**
- Python 3.8+
- Supabase account (free tier available)

### **2. Setup Environment**
```bash
# Clone and navigate to project
git clone <your-repo>
cd data-automation-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Configure Supabase**

#### **3.1 Create Supabase Project**
1. Go to [supabase.com](https://supabase.com)
2. Create a new project
3. Note your project URL and API keys

#### **3.2 Create Database Table**
1. Go to your Supabase dashboard → SQL Editor
2. Run the SQL from `supabase_table.sql`:

```sql
-- This creates the stock_data table with all necessary columns
-- See supabase_table.sql for the complete script
```

#### **3.3 Configure Environment Variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual Supabase credentials:
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### **4. Run the Pipeline**

#### **Test Without Supabase (Recommended First)**
```bash
# Test data processing and report generation
python test_pipeline_no_supabase.py
```

#### **Full Pipeline with Supabase**
```bash
# Run the complete pipeline
python scripts/main.py
```

## 📊 **Features**

### **Data Sources**
- **Yahoo Finance**: Real-time stock data via `yfinance`
- **Symbols**: AMD, NVDA, BYD, TSLA (configurable)
- **Period**: Last 100 days (configurable)

### **Technical Analysis**
- **Moving Averages**: SMA (20, 50), EMA (12, 26)
- **Momentum**: RSI (14-period)
- **Trend**: MACD (12, 26, 9)
- **Volatility**: Bollinger Bands (20, 2σ)
- **Risk Metrics**: Daily returns, volatility, Sharpe ratio

### **Data Quality**
- **Missing Data**: Forward/backward fill, median imputation
- **Outliers**: 3-sigma capping
- **Validation**: Price consistency checks
- **Types**: Automatic data type conversion

### **Reports**
- **HTML**: Interactive Plotly charts, responsive design
- **Excel**: Multiple sheets (Summary, Individual stocks, Technical analysis)
- **Insights**: Automated performance comparisons
- **Scheduling**: Ready for cron/Airflow automation

## 🔧 **Configuration**

### **Stock Symbols**
```bash
# In .env file
STOCK_SYMBOLS=AMD,NVDA,BYD,TSLA,AAPL,GOOGL
```

### **Report Formats**
```bash
# Choose which reports to generate
REPORT_FORMATS=html,excel  # pdf available but requires WeasyPrint
```

### **Technical Indicators**
```python
# In scripts/config.py
TECHNICAL_INDICATORS = {
    'SMA': [20, 50],        # Simple Moving Average periods
    'EMA': [12, 26],        # Exponential Moving Average periods
    'RSI': 14,              # RSI period
    'MACD': [12, 26, 9],    # MACD fast, slow, signal
    'BBANDS': [20, 2]       # Bollinger Bands period, std dev
}
```

## 🤖 **Automation**

### **Cron (Linux/macOS)**
```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/project && source .venv/bin/activate && python scripts/main.py >> logs/cron.log 2>&1
```

### **Airflow (Mock Setup)**
```python
# Example DAG structure
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG('stock_analysis_pipeline', schedule_interval='@daily')

acquire_data = PythonOperator(task_id='acquire_data', python_callable=acquire_data_func)
process_data = PythonOperator(task_id='process_data', python_callable=process_data_func)
store_data = PythonOperator(task_id='store_data', python_callable=store_data_func)
generate_reports = PythonOperator(task_id='generate_reports', python_callable=generate_reports_func)

acquire_data >> process_data >> store_data >> generate_reports
```

## 📁 **Project Structure**

```
data-automation-pipeline/
├── scripts/
│   ├── config.py              # Configuration management
│   ├── api_client.py          # Yahoo Finance API client
│   ├── data_processor.py      # Data cleaning & technical analysis
│   ├── storage.py             # Supabase data storage
│   ├── report_generator.py    # HTML/Excel report generation
│   └── main.py               # Main pipeline orchestrator
├── data/
│   ├── raw/                  # Raw data from Yahoo Finance
│   └── processed/            # Cleaned & enhanced data
├── reports/                  # Generated HTML/Excel reports
├── logs/                     # Pipeline execution logs
├── supabase_table.sql        # Database schema
├── test_pipeline_no_supabase.py  # Test without Supabase
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md               # This file
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"Invalid Supabase URL"**
   - Check your `.env` file has correct Supabase credentials
   - Test with `test_pipeline_no_supabase.py` first

2. **"Excel timezone error"**
   - Fixed in latest version - Excel reports now handle timezones correctly

3. **"No data for symbol"**
   - Yahoo Finance may have temporary issues
   - Check symbol spelling (use uppercase: AMD, not amd)

4. **"Permission denied" on Supabase**
   - Verify your service role key has correct permissions
   - Check RLS policies in Supabase dashboard

### **Testing Components**
```bash
# Test individual components
python test_components.py      # Test data acquisition & processing
python test_pipeline_no_supabase.py  # Test full pipeline without Supabase
```

## 📈 **Sample Output**

The pipeline generates:
- **HTML Report**: Interactive charts showing price movements, technical indicators, and performance comparisons
- **Excel Report**: Multiple sheets with raw data, summary statistics, and technical analysis
- **Logs**: Detailed execution logs for monitoring and debugging

## 🎯 **Use Cases**

- **Quantitative Research**: Historical data analysis and backtesting
- **Portfolio Management**: Daily monitoring of stock performance
- **Risk Management**: Volatility and correlation analysis
- **Automated Reporting**: Daily/weekly reports for stakeholders
- **Data Engineering Portfolio**: Demonstrates ETL, API integration, and automation skills

## 🔮 **Future Enhancements**

- **More Data Sources**: Alpha Vantage, Polygon, IEX Cloud
- **Advanced Analytics**: Machine learning predictions, sentiment analysis
- **Real-time Processing**: WebSocket connections for live data
- **Alerting**: Email/Slack notifications for significant events
- **Web Dashboard**: React/Streamlit frontend for interactive analysis

---

**Built with**: Python, Yahoo Finance, Supabase, Plotly, Pandas, OpenPyXL
