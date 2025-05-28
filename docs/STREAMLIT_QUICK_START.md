# 🚀 Streamlit Dashboard Quick Start Guide

## 🎯 **What You Get**

Transform your static data pipeline into an **interactive web dashboard** with:

- ✅ **Dynamic Stock Selection** - No more hardcoded symbols!
- ✅ **Real-time Analysis** - Instant technical indicators & financial statements
- ✅ **Interactive Charts** - Zoom, pan, hover for details
- ✅ **Financial Statements** - Income, balance sheet, cash flow analysis
- ✅ **Risk Analysis** - VaR, volatility, drawdown metrics
- ✅ **Data Export** - Download CSV, Excel, JSON instantly

---

## 🚀 **Setup (5 Minutes)**

### **1. Install Dependencies**

```bash
# Activate your virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Streamlit
pip install streamlit

# Verify installation
streamlit hello
```

### **2. Launch Dashboard**

#### **Option A: Quick Launcher**
```bash
python run_dashboard.py
```

#### **Option B: Direct Launch**
```bash
# Basic Dashboard (lightweight, fast)
streamlit run dashboard.py

# Advanced Dashboard (full features + pipeline integration)
streamlit run advanced_dashboard.py
```

### **3. Open in Browser**
- Streamlit will automatically open `http://localhost:8501`
- If not, click the link shown in your terminal

---

## 📊 **Dashboard Options**

### **🏃‍♂️ Basic Dashboard** (`dashboard.py`)
**Perfect for**: Quick analysis, learning, lightweight use

**Features:**
- Simple stock selection (dropdown/text input)
- Core technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Financial statements (Income, Balance Sheet, Cash Flow)
- Multi-stock comparison
- Data export (CSV, Excel, JSON)

### **🚀 Advanced Dashboard** (`advanced_dashboard.py`)
**Perfect for**: Professional analysis, research, portfolio management

**Features:**
- Full pipeline integration (uses your existing components)
- Sector-based stock selection
- 15+ technical indicators with professional charts
- 20+ financial ratios and metrics
- Risk analysis (VaR, drawdown, correlation)
- Supabase cloud storage integration
- Professional report generation
- CSV upload for bulk analysis

---

## 🎛️ **Using the Dashboard**

### **Stock Selection Methods:**

1. **Browse by Sector**
   - Technology: AAPL, GOOGL, MSFT, NVDA, AMD, TSLA
   - Finance: JPM, BAC, WFC, GS, MS, C
   - Healthcare: JNJ, PFE, UNH, ABBV, MRK, BMY
   - Consumer: AMZN, WMT, HD, MCD, NKE, SBUX
   - Energy: XOM, CVX, COP, EOG, SLB, PSX

2. **Enter Custom Symbols**
   ```
   AAPL
   GOOGL
   TSLA
   ```

3. **Upload CSV File** (Advanced Dashboard)
   ```csv
   Symbol
   AAPL
   MSFT
   GOOGL
   ```

### **Analysis Options:**

- **Time Periods**: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
- **Technical Analysis**: Moving averages, RSI, MACD, Bollinger Bands
- **Financial Statements**: Income statement, balance sheet, cash flow
- **Peer Comparison**: Normalized performance charts
- **Risk Analysis**: VaR, volatility, drawdown
- **Data Export**: CSV, Excel, JSON formats

---

## 📈 **Dashboard Features**

### **📊 Technical Analysis Tab**
- **Key Metrics**: Current price, change %, volatility, RSI, Sharpe ratio
- **Candlestick Charts**: OHLC data with volume
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands
- **Volume Analysis**: Color-coded volume bars

### **💰 Financial Analysis Tab**
- **Company Overview**: Name, sector, industry, market cap
- **Financial Ratios**: P/E, P/B, ROE, ROA, profit margins
- **Financial Statements**: Income, balance sheet, cash flow
- **Earnings Data**: Historical and quarterly earnings
- **Analyst Recommendations**: Buy/Hold/Sell ratings

### **🔍 Peer Comparison Tab**
- **Normalized Performance**: Starting point = 100
- **Metrics Comparison**: Side-by-side performance table
- **Correlation Analysis**: How stocks move together
- **Risk-Return Analysis**: Volatility vs returns

### **⚠️ Risk Analysis Tab**
- **Risk Metrics**: Annual volatility, VaR (95%, 99%), max drawdown
- **Return Distribution**: Histogram of daily returns
- **Statistical Analysis**: Mean, std dev, skewness

### **💾 Data Export Tab**
- **Format Options**: CSV, Excel (multi-sheet), JSON
- **Data Preview**: Last 10 records
- **Download Buttons**: Instant file downloads
- **Report Generation**: Comprehensive HTML/Excel reports

---

## 🎯 **Example Use Cases**

### **1. Quick Stock Analysis**
```
1. Select "Technology" sector
2. Choose "AAPL"
3. Set period to "1y"
4. View technical analysis + financial statements
5. Export data as needed
```

### **2. Portfolio Comparison**
```
1. Enter multiple symbols: "AAPL,GOOGL,MSFT,TSLA"
2. Compare normalized performance
3. Check correlation matrix
4. Analyze risk metrics
5. Download comparison report
```

### **3. Sector Analysis**
```
1. Browse "Technology" sector
2. Select multiple tech stocks
3. Compare financial ratios
4. Export comprehensive Excel report
```

### **4. Risk Assessment**
```
1. Select high-volatility stocks
2. View risk analysis tab
3. Check VaR and drawdown metrics
4. Compare return distributions
```

---

## 🛠️ **Advanced Features**

### **☁️ Supabase Integration** (Advanced Dashboard)
- Enable "Save to Supabase" in sidebar
- Persistent cloud storage
- Historical data accumulation
- Team collaboration

### **📤 CSV Upload** (Advanced Dashboard)
- Upload portfolio holdings
- Bulk analysis of custom stock lists
- Academic research datasets

### **🎯 Professional Reports**
- Click "Generate Comprehensive Report"
- HTML reports with interactive charts
- Excel reports with multiple sheets
- Professional formatting and insights

---

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **"Module not found" errors**
   ```bash
   # Make sure virtual environment is activated
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **"Port already in use"**
   ```bash
   # Use different port
   streamlit run dashboard.py --server.port 8502
   ```

3. **"No data for symbol"**
   - Check symbol spelling (use uppercase)
   - Yahoo Finance may have temporary issues
   - Try a different stock symbol

4. **Slow loading**
   - Data is cached for 5 minutes
   - First load may be slower
   - Subsequent loads will be faster

### **Performance Tips:**

- Use **Basic Dashboard** for quick analysis
- **Advanced Dashboard** for comprehensive research
- Data is cached automatically
- Close unused browser tabs to free memory

---

## 🎨 **Customization**

### **Add New Stock Symbols**
```python
# In dashboard.py, modify the popular_stocks dictionary
popular_stocks = {
    "Technology": ["AAPL", "GOOGL", "MSFT", "YOUR_SYMBOL"],
    "Your_Sector": ["SYMBOL1", "SYMBOL2", "SYMBOL3"]
}
```

### **Add New Technical Indicators**
```python
# In calculate_technical_indicators function
df['your_indicator'] = your_calculation(df['close'])
```

### **Customize Chart Appearance**
```python
# Modify chart styling
fig.update_layout(
    template="plotly_dark",  # Dark theme
    title_font_size=20,
    height=800
)
```

---

## 🚀 **Next Steps**

1. **Start with Basic Dashboard** - Get familiar with the interface
2. **Try Advanced Dashboard** - Explore full features
3. **Integrate with Pipeline** - Set up Supabase for data persistence
4. **Customize Analysis** - Add your own indicators and metrics
5. **Share with Team** - Deploy to cloud for team access

---

## 📚 **Resources**

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Yahoo Finance Data**: Powered by `yfinance` library
- **Plotly Charts**: Interactive visualization library
- **Technical Analysis**: Industry-standard indicators

---

## 🎉 **Ready to Start?**

```bash
# 1. Launch the dashboard
python run_dashboard.py

# 2. Choose your option:
#    - Basic Dashboard (option 1)
#    - Advanced Dashboard (option 2)

# 3. Open http://localhost:8501 in your browser

# 4. Start analyzing stocks! 🚀
```

**No more hardcoded `.env` files - now you have dynamic, interactive stock analysis at your fingertips!** 📈 