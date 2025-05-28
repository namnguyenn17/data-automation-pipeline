# 🚀 Interactive Stock Analysis Dashboard

## Overview

This project includes **two powerful Streamlit dashboards** that transform your static data pipeline into an **interactive, web-based analysis platform**. No more hardcoded stock symbols in `.env` files - now you can dynamically select any stocks and get real-time analysis!

## 🎯 **Why Build a Dashboard?**

### **Problems with Static Pipeline:**
- ❌ Hardcoded symbols in `.env` files
- ❌ Need to restart pipeline for different stocks
- ❌ No real-time interaction
- ❌ Limited to predetermined analysis

### **Dashboard Benefits:**
- ✅ **Dynamic Stock Selection** - Pick any stocks on-demand
- ✅ **Real-time Analysis** - Instant results without pipeline restarts
- ✅ **Interactive Charts** - Zoom, pan, hover for details
- ✅ **Financial Statements** - Income, balance sheet, cash flow
- ✅ **Peer Comparison** - Side-by-side stock analysis
- ✅ **Risk Analysis** - VaR, volatility, drawdown analysis
- ✅ **Data Export** - Download processed data instantly

---

## 📊 **Two Dashboard Options**

### **1. Basic Dashboard** (`dashboard.py`)
**Best for**: Quick analysis, learning, or lightweight use

**Features:**
- 📈 Dynamic stock selection (dropdown or text input)
- 📊 Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- 💰 Financial statements (Income, Balance Sheet, Cash Flow)
- 🔍 Multi-stock comparison
- 💾 Data export (CSV, Excel, JSON)
- ⚡ Fast and lightweight

### **2. Advanced Dashboard** (`advanced_dashboard.py`)
**Best for**: Professional analysis, portfolio management, research

**Features:**
- 🚀 **Full Pipeline Integration** - Uses your existing components
- ☁️ **Supabase Storage** - Save data to cloud database
- 📈 **Advanced Technical Analysis** - 15+ indicators with professional charts
- 💼 **Comprehensive Financial Analysis** - 20+ financial ratios and metrics
- 🔍 **Sector-based Stock Selection** - Browse by industry
- ⚠️ **Risk Analysis** - VaR, maximum drawdown, return distributions
- 📊 **Correlation Analysis** - Portfolio correlation matrices
- 📋 **Analyst Recommendations** - Real-time analyst data
- 🎯 **Professional Reports** - Generate HTML/Excel reports
- 📤 **CSV Upload** - Bulk import stock symbols

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install streamlit>=1.28.0
# (Other dependencies already in requirements.txt)
```

### **2. Launch Dashboard**

#### **Option A: Use the Launcher**
```bash
python run_dashboard.py
```

#### **Option B: Direct Launch**
```bash
# Basic Dashboard
streamlit run dashboard.py

# Advanced Dashboard  
streamlit run advanced_dashboard.py
```

### **3. Open in Browser**
- Dashboard will automatically open at `http://localhost:8501`
- If not, click the link in your terminal

---

## 📈 **Dashboard Features Deep Dive**

### **🎛️ Dynamic Stock Selection**

#### **Multiple Input Methods:**
1. **Browse by Sector** - Technology, Finance, Healthcare, Consumer, Energy
2. **Enter Custom Symbols** - Type any stock symbols
3. **Upload CSV File** - Bulk import from spreadsheet

#### **Popular Stocks by Sector:**
- **Technology**: AAPL, GOOGL, MSFT, NVDA, AMD, TSLA
- **Finance**: JPM, BAC, WFC, GS, MS, C
- **Healthcare**: JNJ, PFE, UNH, ABBV, MRK, BMY
- **Consumer**: AMZN, WMT, HD, MCD, NKE, SBUX
- **Energy**: XOM, CVX, COP, EOG, SLB, PSX

### **📊 Technical Analysis**

#### **Price Charts:**
- **Candlestick Charts** - OHLC data with volume
- **Moving Averages** - SMA (20, 50), EMA (12, 26)
- **Bollinger Bands** - Price channel analysis
- **Volume Analysis** - Color-coded volume bars

#### **Technical Indicators:**
- **RSI** - Relative Strength Index (overbought/oversold)
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Volatility bands
- **Volatility** - 20-day rolling volatility
- **Daily Returns** - Percentage changes

### **💰 Financial Statements Analysis**

#### **Company Overview:**
- Company name, sector, industry
- Market cap, P/E ratio, dividend yield
- Employee count, headquarters, website

#### **Financial Ratios (20+ Metrics):**
- **Valuation**: P/E, P/B, EV/EBITDA, Market Cap
- **Profitability**: ROE, ROA, Profit Margins, Operating Margins
- **Financial Health**: Current Ratio, Debt/Equity, Interest Coverage
- **Growth**: Revenue Growth, Earnings Growth, Asset Turnover

#### **Financial Statements:**
- **Income Statement** - Revenue, expenses, net income trends
- **Balance Sheet** - Assets, liabilities, equity
- **Cash Flow** - Operating, investing, financing cash flows
- **Earnings History** - Quarterly and annual earnings
- **Analyst Recommendations** - Buy/Hold/Sell ratings

### **🔍 Peer Comparison**

#### **Performance Comparison:**
- **Normalized Price Charts** - Starting point = 100
- **Performance Metrics Table** - Side-by-side comparison
- **Correlation Analysis** - How stocks move together
- **Risk-Return Analysis** - Volatility vs returns

### **⚠️ Risk Analysis**

#### **Risk Metrics:**
- **Annual Volatility** - Price fluctuation measure
- **Value at Risk (VaR)** - Potential losses at 95% and 99% confidence
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Sharpe Ratio** - Risk-adjusted returns

#### **Return Distribution:**
- **Histogram** - Daily return frequency distribution
- **Statistical Analysis** - Mean, standard deviation, skewness

### **💾 Data Export & Reports**

#### **Export Formats:**
- **CSV** - Raw data for further analysis
- **Excel** - Multi-sheet workbooks with formatted data
- **JSON** - Structured data for APIs

#### **Report Generation:**
- **HTML Reports** - Interactive charts and analysis
- **Excel Reports** - Professional formatted spreadsheets
- **Comprehensive Analysis** - Technical + fundamental combined

---

## 🛠️ **Advanced Features**

### **☁️ Supabase Integration** (Advanced Dashboard Only)

```python
# Enable cloud storage in sidebar
save_to_cloud = st.sidebar.checkbox("Save to Supabase", value=False)
```

**Benefits:**
- **Persistent Storage** - Data survives between sessions
- **Historical Analysis** - Build long-term datasets
- **Collaboration** - Share data across team
- **Backup** - Cloud-based data protection

### **📤 CSV Upload for Bulk Analysis**

```python
# Upload a CSV file with stock symbols
uploaded_file = st.sidebar.file_uploader("Upload CSV with symbols", type=['csv'])
```

**Use Cases:**
- Portfolio analysis (upload your holdings)
- Sector analysis (upload industry stocks)
- Custom watchlists
- Academic research

### **🎯 Professional Report Generation**

```python
# Generate comprehensive reports
if st.button("🎯 Generate Comprehensive Report"):
    reports = report_generator.generate_comprehensive_report(...)
```

**Report Contents:**
- Executive summary with key insights
- Technical analysis charts
- Financial statement analysis
- Risk assessment
- Peer comparison
- Data tables

---

## 🎨 **Dashboard Screenshots**

### **Main Interface:**
```
🚀 Advanced Stock Analysis Dashboard
├── 📊 Sidebar Controls
│   ├── Stock Selection (Sector/Custom/Upload)
│   ├── Time Period (1mo to 5y)
│   ├── Analysis Options
│   └── Cloud Storage Settings
├── 📈 Technical Analysis Tab
│   ├── Key Metrics (Price, Change %, RSI, Volatility)
│   ├── Candlestick Charts with Indicators
│   └── Volume Analysis
├── 💰 Financial Analysis Tab
│   ├── Company Overview
│   ├── Financial Ratios Dashboard
│   ├── Financial Trends Charts
│   └── Detailed Statements
├── 🔍 Peer Comparison Tab
│   ├── Normalized Performance Charts
│   ├── Metrics Comparison Table
│   └── Correlation Analysis
├── ⚠️ Risk Analysis Tab
│   ├── Risk Metrics
│   └── Return Distribution
└── 💾 Data Export Tab
    ├── Format Selection (CSV/Excel/JSON)
    ├── Data Preview
    └── Download Buttons
```

---

## 🔧 **Customization & Configuration**

### **Adding New Indicators**

```python
# In calculate_technical_indicators function
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Add your custom indicator
    df['custom_indicator'] = your_calculation(df)
    return df
```

### **Adding New Sectors**

```python
# In main() function
popular_stocks = {
    "Technology": ["AAPL", "GOOGL", "MSFT"],
    "Your_Sector": ["SYMBOL1", "SYMBOL2", "SYMBOL3"],  # Add here
}
```

### **Customizing Charts**

```python
# Modify chart appearance
fig.update_layout(
    template="plotly_dark",  # Dark theme
    title_font_size=20,
    height=800
)
```

---

## 🚀 **Performance & Caching**

### **Data Caching:**
```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol: str, period: str = "1y"):
    # Data fetching logic
```

**Benefits:**
- ⚡ **Faster Loading** - Cached data loads instantly
- 💰 **API Efficiency** - Reduces Yahoo Finance API calls
- 🔄 **Auto Refresh** - TTL ensures fresh data

### **Session State:**
```python
# Preserve data across tab switches
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
```

---

## 🎯 **Use Cases & Examples**

### **1. Portfolio Monitoring**
```python
# Upload CSV with your holdings
# Get real-time analysis of your portfolio
# Compare performance against benchmarks
```

### **2. Sector Analysis**
```python
# Select "Technology" sector
# Compare AAPL, GOOGL, MSFT, NVDA
# Analyze correlation and performance
```

### **3. Individual Stock Research**
```python
# Enter "TSLA"
# Get comprehensive analysis:
# - Technical indicators
# - Financial ratios
# - Risk metrics
# - Analyst recommendations
```

### **4. Academic Research**
```python
# Upload list of stocks for study
# Export processed data
# Generate professional reports
```

---

## 🔮 **Future Enhancements**

### **Near-term (Easy to Add):**
- 📊 **More Indicators** - Stochastic, Williams %R, CCI
- 🌍 **International Markets** - European, Asian stocks
- 📱 **Mobile Optimization** - Responsive design
- 🎨 **Themes** - Dark mode, custom color schemes

### **Medium-term (Moderate Effort):**
- 🤖 **Machine Learning** - Price prediction models
- 📈 **Backtesting** - Strategy performance testing
- 📊 **Portfolio Optimization** - Efficient frontier
- 🔔 **Alerts** - Price/indicator notifications

### **Long-term (Advanced Features):**
- 📰 **News Integration** - Sentiment analysis
- 🌐 **Real-time Data** - WebSocket connections
- 👥 **Multi-user** - User authentication
- 🏗️ **Strategy Builder** - Visual strategy creation

---

## 🆚 **Dashboard vs Pipeline Comparison**

| Feature | Static Pipeline | Streamlit Dashboard |
|---------|----------------|-------------------|
| **Stock Selection** | Hardcoded in `.env` | Dynamic, any stocks |
| **Interaction** | Command-line only | Web-based GUI |
| **Real-time** | Scheduled runs | Instant analysis |
| **Sharing** | File sharing | Web URL sharing |
| **Visualization** | Static images | Interactive charts |
| **Data Export** | Pre-defined format | Multiple formats |
| **User Experience** | Technical users | Anyone can use |
| **Deployment** | Server/cron | Web application |

---

## 🎉 **Conclusion**

The **Streamlit dashboards** transform your data automation pipeline from a **technical tool** into a **user-friendly, interactive platform**. Whether you choose the **Basic Dashboard** for quick analysis or the **Advanced Dashboard** for comprehensive research, you now have:

✅ **Dynamic stock selection** - No more `.env` file editing
✅ **Real-time analysis** - Instant results
✅ **Professional visualization** - Interactive charts
✅ **Comprehensive financial data** - Income statements, ratios, risk metrics
✅ **Easy sharing** - Send web links instead of files
✅ **Export capabilities** - Get data in any format you need

**Ready to get started?** Run `python run_dashboard.py` and experience the power of interactive stock analysis! 🚀 