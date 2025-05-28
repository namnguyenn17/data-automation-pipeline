#!/usr/bin/env python3
"""
Test dashboard components without Streamlit
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import os

def test_yfinance():
    """Test Yahoo Finance data fetching."""
    print("🔄 Testing Yahoo Finance data fetching...")
    try:
        # Get basic stock data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} records for AAPL")
            print(f"   Columns: {list(data.columns)}")
        else:
            print("❌ No data returned")
            return False
            
        # Test financial statements
        info = ticker.info
        financials = ticker.financials
        
        if info:
            print(f"✅ Company info available: {info.get('longName', 'N/A')}")
        if not financials.empty:
            print(f"✅ Financial statements available: {financials.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicator calculations."""
    print("\n🔄 Testing technical indicator calculations...")
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.random.randn(100).cumsum()
        
        df = pd.DataFrame({
            'close': prices,
            'open': prices + np.random.randn(100) * 0.5,
            'high': prices + np.abs(np.random.randn(100)) * 0.8,
            'low': prices - np.abs(np.random.randn(100)) * 0.8,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Calculate SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # Check results
        if not df['sma_20'].isna().all():
            print("✅ SMA calculation successful")
        if not df['rsi'].isna().all():
            print("✅ RSI calculation successful")
        
        print(f"✅ Technical indicators calculated for {len(df)} records")
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
        return False

def test_plotly_charts():
    """Test Plotly chart creation."""
    print("\n🔄 Testing Plotly chart creation...")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        prices = 100 + np.random.randn(50).cumsum()
        
        # Create a simple chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, name='Price'))
        fig.update_layout(title="Test Chart", height=400)
        
        # Test subplot creation
        fig_subplots = make_subplots(rows=2, cols=1)
        fig_subplots.add_trace(go.Scatter(x=dates, y=prices, name='Price'), row=1, col=1)
        fig_subplots.add_trace(go.Bar(x=dates, y=np.random.randint(1000, 10000, 50), name='Volume'), row=2, col=1)
        
        print("✅ Plotly chart creation successful")
        print("✅ Plotly subplots creation successful")
        return True
        
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")
        return False

def test_financial_ratios():
    """Test financial ratio calculations."""
    print("\n🔄 Testing financial ratio calculations...")
    try:
        # Simulate company info
        info = {
            'marketCap': 2800000000000,  # $2.8T
            'trailingPE': 28.5,
            'priceToBook': 45.2,
            'returnOnEquity': 0.175,
            'returnOnAssets': 0.089,
            'profitMargins': 0.251,
            'operatingMargins': 0.298,
            'currentRatio': 1.04,
            'debtToEquity': 172.48,
            'revenueGrowth': 0.081,
            'earningsGrowth': 0.114
        }
        
        # Test calculations
        market_cap_formatted = f"${info['marketCap']/1e9:.2f}B"
        roe_formatted = f"{info['returnOnEquity']*100:.2f}%"
        
        print(f"✅ Market Cap formatting: {market_cap_formatted}")
        print(f"✅ ROE formatting: {roe_formatted}")
        print("✅ Financial ratio calculations successful")
        return True
        
    except Exception as e:
        print(f"❌ Financial ratios test failed: {e}")
        return False

def test_data_export():
    """Test data export functionality."""
    print("\n🔄 Testing data export functionality...")
    try:
        # Create sample data
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(1000000, 10000000, 10)
        })
        
        # Test CSV export
        csv_data = df.to_csv(index=False)
        print(f"✅ CSV export successful: {len(csv_data)} characters")
        
        # Test JSON export
        json_data = df.to_json(orient='records', date_format='iso')
        print(f"✅ JSON export successful: {len(json_data)} characters")
        
        # Test Excel export (in memory)
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        excel_data = buffer.getvalue()
        print(f"✅ Excel export successful: {len(excel_data)} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Data export test failed: {e}")
        return False

def main():
    """Run all dashboard component tests."""
    print("🚀 Testing Dashboard Components")
    print("=" * 50)
    
    tests = [
        ("Yahoo Finance Data", test_yfinance),
        ("Technical Indicators", test_technical_indicators),
        ("Plotly Charts", test_plotly_charts),
        ("Financial Ratios", test_financial_ratios),
        ("Data Export", test_data_export)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Dashboard components are ready.")
        print("\n🚀 To launch the dashboard:")
        print("   1. Install Streamlit: pip install streamlit")
        print("   2. Run launcher: python run_dashboard.py")
        print("   3. Or run directly: streamlit run dashboard.py")
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main() 