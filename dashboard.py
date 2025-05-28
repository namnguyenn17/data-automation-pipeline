#!/usr/bin/env python3
"""
Advanced Streamlit Dashboard for Stock Analysis Pipeline
Integrates with existing pipeline components + financial statements + Supabase
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import sys
import os
import io
import logging

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import our existing pipeline components
from scripts.config import Config
from scripts.api_client import YahooFinanceClient
from scripts.data_processor import DataProcessor
from scripts.report_generator import ReportGenerator
from scripts.ai_analyzer import AIStockAnalyzer

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = {}
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .financial-metric {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Logging Setup (adapted from scripts/main.py)
def setup_dashboard_logging():
    """Setup logging configuration for the dashboard."""
    logger = logging.getLogger('data_automation_pipeline') # Use the same logger name
    if not logger.handlers: # Setup handlers only if not already configured
        logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        
        # Clear existing handlers just in case, though the check above should prevent duplicates
        logger.handlers = [] 
        
        formatter = logging.Formatter(Config.LOG_FORMAT)
        
        Config.setup_directories() # Ensure log directory exists
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler() # Log to console as well for Streamlit
        console_handler.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.info("Dashboard logging configured.")
    return logger

# Initialize logging when the script runs
dashboard_logger = setup_dashboard_logging()

# Initialize pipeline components
@st.cache_resource
def init_pipeline_components():
    """Initialize pipeline components."""
    try:
        api_client = YahooFinanceClient()
        data_processor = DataProcessor()
        report_generator = ReportGenerator()
        ai_analyzer = AIStockAnalyzer()
        return api_client, data_processor, report_generator, ai_analyzer
    except Exception as e:
        st.error(f"Failed to initialize pipeline components: {e}")
        dashboard_logger.error(f"Failed to initialize pipeline components: {e}", exc_info=True)
        return None, None, None, None

# Cache data functions
@st.cache_data(ttl=300)
def get_comprehensive_stock_data(symbol: str, period: str = "1y"):
    """Get comprehensive stock data using our pipeline components."""
    components = init_pipeline_components()
    if not all(components):
        return None, None
    api_client, data_processor, _, _ = components
    
    try:
        # Get raw data
        raw_df = api_client.get_daily_prices(symbol, period)
        if raw_df.empty:
            return None, None
        
        # Process data with technical indicators
        processed_df = data_processor.clean_stock_data(raw_df)
        enhanced_df = data_processor.calculate_technical_indicators(processed_df)
        
        # Generate statistics
        stats = data_processor.generate_summary_statistics(enhanced_df, symbol)
        
        return enhanced_df, stats
    except Exception as e:
        st.error(f"Error processing data for {symbol}: {e}")
        dashboard_logger.error(f"Error processing data for {symbol}: {e}", exc_info=True)
        return None, None

@st.cache_data(ttl=3600)
def get_financial_statements(symbol: str):
    """Get comprehensive financial statements."""
    try:
        ticker = yf.Ticker(symbol)
        
        return {
            'info': ticker.info,
            'income_statement': ticker.financials,
            'quarterly_income': ticker.quarterly_financials,
            'balance_sheet': ticker.balance_sheet,
            'quarterly_balance': ticker.quarterly_balance_sheet,
            'cash_flow': ticker.cashflow,
            'quarterly_cashflow': ticker.quarterly_cashflow,
            'dividends': ticker.dividends,
            'splits': ticker.splits,
            'actions': ticker.actions,
            'recommendations': ticker.recommendations,
            'analyst_price_targets': ticker.analyst_price_targets if hasattr(ticker, 'analyst_price_targets') else None,
            'earnings_dates': ticker.earnings_dates
        }
    except Exception as e:
        st.error(f"Error fetching financial data for {symbol}: {e}")
        dashboard_logger.error(f"Error fetching financial data for {symbol}: {e}", exc_info=True)
        return None

def create_advanced_price_chart(df: pd.DataFrame, symbol: str):
    """Create advanced price chart with technical indicators."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'{symbol} Stock Price with Technical Indicators',
            'Volume',
            'RSI',
            'MACD'
        ),
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.175, 0.175]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Moving averages
    for ma in ['sma_20', 'sma_50', 'ema_12', 'ema_26']:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[ma], 
                    name=ma.upper(),
                    line=dict(width=2)
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash'), opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'), opacity=0.5,
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' 
              for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', 
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, row=3, col=1)
    
    # MACD
    if all(col in df.columns for col in ['macd', 'macd_signal', 'macd_histogram']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', 
                      line=dict(color='red', width=2)),
            row=4, col=1
        )
        
        # MACD histogram with colors
        histogram_colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
                   marker_color=histogram_colors, opacity=0.6),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f"{symbol} Comprehensive Technical Analysis",
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_financial_analysis_dashboard(financials: dict, symbol: str):
    """Create comprehensive financial analysis dashboard."""
    if not financials:
        return None
    
    # Key financial ratios and metrics
    info = financials.get('info', {})
    income_stmt = financials.get('income_statement', pd.DataFrame())
    balance_sheet = financials.get('balance_sheet', pd.DataFrame())
    
    # Create financial metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📊 Valuation Metrics")
        if info:
            market_cap = info.get('marketCap', 0)
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            st.metric("P/B Ratio", f"{info.get('priceToBook', 'N/A')}")
            st.metric("EV/EBITDA", f"{info.get('enterpriseToEbitda', 'N/A')}")
    
    with col2:
        st.markdown("### 💰 Profitability")
        if info:
            st.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A")
            st.metric("ROA", f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "N/A")
            st.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A")
            st.metric("Operating Margin", f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A")
    
    with col3:
        st.markdown("### 🏦 Financial Health")
        if info:
            st.metric("Current Ratio", f"{info.get('currentRatio', 'N/A')}")
            st.metric("Quick Ratio", f"{info.get('quickRatio', 'N/A')}")
            st.metric("Debt/Equity", f"{info.get('debtToEquity', 'N/A')}")
            st.metric("Interest Coverage", f"{info.get('interestCoverage', 'N/A')}")
    
    with col4:
        st.markdown("### 📈 Growth & Efficiency")
        if info:
            st.metric("Revenue Growth", f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "N/A")
            st.metric("Earnings Growth", f"{info.get('earningsGrowth', 0)*100:.2f}%" if info.get('earningsGrowth') else "N/A")
            st.metric("Asset Turnover", f"{info.get('assetTurnover', 'N/A')}")
            st.metric("Inventory Turnover", f"{info.get('inventoryTurnover', 'N/A')}")
    
    # Financial trends chart
    if not income_stmt.empty:
        fig = create_financial_trends_chart(financials, symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    return True

def create_financial_trends_chart(financials: dict, symbol: str):
    """Create financial trends chart with error handling."""
    income_stmt = financials.get('income_statement', pd.DataFrame())
    
    if income_stmt.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue & Net Income', 'Profit Margins', 'Assets & Liabilities', 'Cash Flow'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue and Net Income
    if 'Total Revenue' in income_stmt.index:
        revenue = income_stmt.loc['Total Revenue'].dropna() / 1e9
        fig.add_trace(
            go.Scatter(x=revenue.index, y=revenue.values, name='Revenue ($B)', 
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
    
    if 'Net Income' in income_stmt.index:
        net_income = income_stmt.loc['Net Income'].dropna() / 1e9
        fig.add_trace(
            go.Scatter(x=net_income.index, y=net_income.values, name='Net Income ($B)', 
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
    
    # Profit margins with error handling
    if all(metric in income_stmt.index for metric in ['Total Revenue', 'Gross Profit', 'Net Income']):
        revenue = income_stmt.loc['Total Revenue']
        gross_profit = income_stmt.loc['Gross Profit']
        net_income = income_stmt.loc['Net Income']
        
        # Safe division function to handle zeros
        def safe_divide(a, b):
            return np.where(b != 0, a / b * 100, np.nan)
        
        try:
            # Convert to numpy arrays for safe division
            gross_margin = pd.Series(
                safe_divide(gross_profit.values, revenue.values),
                index=revenue.index
            ).dropna()
            
            net_margin = pd.Series(
                safe_divide(net_income.values, revenue.values),
                index=revenue.index
            ).dropna()
            
            if not gross_margin.empty:
                fig.add_trace(
                    go.Scatter(x=gross_margin.index, y=gross_margin.values, name='Gross Margin %', 
                              line=dict(color='orange', width=2)),
                    row=1, col=2
                )
            
            if not net_margin.empty:
                fig.add_trace(
                    go.Scatter(x=net_margin.index, y=net_margin.values, name='Net Margin %', 
                              line=dict(color='red', width=2)),
                    row=1, col=2
                )
        except Exception as e:
            dashboard_logger.warning(f"Error calculating margins for {symbol}: {e}")
    
    # Add more financial metrics based on available data
    balance_sheet = financials.get('balance_sheet', pd.DataFrame())
    if not balance_sheet.empty:
        if 'Total Assets' in balance_sheet.index:
            assets = balance_sheet.loc['Total Assets'].dropna() / 1e9
            fig.add_trace(
                go.Scatter(x=assets.index, y=assets.values, name='Total Assets ($B)', 
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
        
        if 'Total Debt' in balance_sheet.index:
            debt = balance_sheet.loc['Total Debt'].dropna() / 1e9
            fig.add_trace(
                go.Scatter(x=debt.index, y=debt.values, name='Total Debt ($B)', 
                          line=dict(color='red', width=2)),
                row=2, col=1
            )
    
    # Cash flow
    cash_flow = financials.get('cash_flow', pd.DataFrame())
    if not cash_flow.empty:
        if 'Operating Cash Flow' in cash_flow.index:
            ocf = cash_flow.loc['Operating Cash Flow'].dropna() / 1e9
            fig.add_trace(
                go.Scatter(x=ocf.index, y=ocf.values, name='Operating Cash Flow ($B)', 
                          line=dict(color='green', width=2)),
                row=2, col=2
            )
        
        if 'Free Cash Flow' in cash_flow.index:
            fcf = cash_flow.loc['Free Cash Flow'].dropna() / 1e9
            fig.add_trace(
                go.Scatter(x=fcf.index, y=fcf.values, name='Free Cash Flow ($B)', 
                          line=dict(color='blue', width=2)),
                row=2, col=2
            )
    
    fig.update_layout(
        title=f"{symbol} Financial Trends Analysis",
        height=700,
        showlegend=True
    )
    
    return fig

def generate_ai_analysis_pdf(symbol: str, analysis: dict, result: dict) -> bytes:
    """Generate PDF report for AI analysis."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import io
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Add caption style for metadata
        caption_style = ParagraphStyle(
            'CustomCaption',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"🤖 AI Investment Analysis Report", title_style))
        story.append(Paragraph(f"Stock Symbol: {symbol}", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Recommendation Summary
        rec = analysis.get('recommendation', 'HOLD')
        confidence = analysis.get('confidence_level', 'MEDIUM')
        rec_color = colors.green if rec == 'BUY' else colors.red if rec == 'SELL' else colors.orange
        
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Create recommendation table
        rec_data = [
            ['Recommendation:', rec],
            ['Confidence Level:', confidence],
            ['Target Timeframe:', analysis.get('target_timeframe', 'N/A')]
        ]
        
        if 'price_target_range' in analysis and isinstance(analysis['price_target_range'], dict):
            target = analysis['price_target_range']
            if target.get('low') and target.get('high'):
                rec_data.append(['Price Target:', f"${target['low']:.2f} - ${target['high']:.2f}"])
        
        rec_table = Table(rec_data, colWidths=[2*inch, 3*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (1, 0), (1, 0), rec_color),
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 20))
        
        # Investment Thesis
        if 'investment_thesis' in analysis:
            story.append(Paragraph("Investment Thesis", heading_style))
            story.append(Paragraph(analysis['investment_thesis'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Key Strengths
        if 'key_strengths' in analysis:
            story.append(Paragraph("Key Strengths", heading_style))
            for strength in analysis['key_strengths']:
                story.append(Paragraph(f"• {strength}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Key Risks
        if 'key_risks' in analysis:
            story.append(Paragraph("Key Risks", heading_style))
            for risk in analysis['key_risks']:
                story.append(Paragraph(f"• {risk}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Technical Outlook
        if 'technical_outlook' in analysis:
            story.append(Paragraph("Technical Outlook", heading_style))
            story.append(Paragraph(analysis['technical_outlook'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Fundamental Outlook
        if 'fundamental_outlook' in analysis:
            story.append(Paragraph("Fundamental Outlook", heading_style))
            story.append(Paragraph(analysis['fundamental_outlook'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Sector Considerations
        if 'sector_considerations' in analysis:
            story.append(Paragraph("Sector Considerations", heading_style))
            story.append(Paragraph(analysis['sector_considerations'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Risk Factors
        if 'risk_factors' in analysis:
            story.append(Paragraph("Risk Factors to Monitor", heading_style))
            story.append(Paragraph(analysis['risk_factors'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Catalysts
        if 'catalysts' in analysis:
            story.append(Paragraph("Potential Catalysts", heading_style))
            story.append(Paragraph(analysis['catalysts'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Disclaimer", heading_style))
        disclaimer = """This analysis is for educational and informational purposes only and does not constitute financial advice. 
        Always conduct your own research and consult with qualified financial professionals before making investment decisions. 
        Past performance does not guarantee future results."""
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Metadata
        story.append(Spacer(1, 20))
        metadata = f"Report generated: {result['timestamp'][:19]} | AI Model: {result.get('model_used', 'GPT-4')}"
        story.append(Paragraph(metadata, caption_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("📄 PDF generation requires reportlab package. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"❌ Failed to generate PDF: {str(e)}")
        dashboard_logger.error(f"PDF generation error: {e}", exc_info=True)
        return None

def generate_comparative_analysis_pdf(symbols: list, comparison: dict, result: dict) -> bytes:
    """Generate PDF report for comparative analysis."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import io
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Add caption style for metadata
        caption_style = ParagraphStyle(
            'CustomCaption',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        
        # Build story
        story = []
        
        # Title
        story.append(Paragraph(f"🤖 AI Comparative Analysis Report", title_style))
        story.append(Paragraph(f"Stocks: {', '.join(symbols)}", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Rankings Table
        if 'ranking' in comparison:
            story.append(Paragraph("Investment Rankings", heading_style))
            
            # Create ranking table
            ranking_data = [['Rank', 'Symbol', 'Score', 'Recommendation', 'Rationale']]
            
            for rank_item in comparison['ranking']:
                ranking_data.append([
                    str(rank_item.get('rank', 'N/A')),
                    rank_item.get('symbol', 'N/A'),
                    str(rank_item.get('score', 'N/A')),
                    rank_item.get('recommendation', 'N/A'),
                    rank_item.get('rationale', 'N/A')[:50] + '...' if len(rank_item.get('rationale', '')) > 50 else rank_item.get('rationale', 'N/A')
                ])
            
            ranking_table = Table(ranking_data, colWidths=[0.5*inch, 0.8*inch, 0.7*inch, 1*inch, 2.5*inch])
            ranking_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(ranking_table)
            story.append(Spacer(1, 20))
        
        # Portfolio Allocations
        if 'portfolio_allocation' in comparison:
            story.append(Paragraph("Portfolio Allocation Recommendations", heading_style))
            
            alloc = comparison['portfolio_allocation']
            
            # Conservative
            if 'conservative_investor' in alloc:
                story.append(Paragraph("Conservative Investor:", styles['Heading3']))
                for stock, pct in alloc['conservative_investor'].items():
                    story.append(Paragraph(f"• {stock}: {pct}%", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Moderate
            if 'moderate_investor' in alloc:
                story.append(Paragraph("Moderate Investor:", styles['Heading3']))
                for stock, pct in alloc['moderate_investor'].items():
                    story.append(Paragraph(f"• {stock}: {pct}%", styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Aggressive
            if 'aggressive_investor' in alloc:
                story.append(Paragraph("Aggressive Investor:", styles['Heading3']))
                for stock, pct in alloc['aggressive_investor'].items():
                    story.append(Paragraph(f"• {stock}: {pct}%", styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Comparative Strengths
        if 'comparative_strengths' in comparison:
            story.append(Paragraph("Comparative Strengths", heading_style))
            strengths = comparison['comparative_strengths']
            
            if 'growth_potential' in strengths:
                story.append(Paragraph(f"Growth Potential: {strengths['growth_potential']}", styles['Normal']))
            if 'financial_stability' in strengths:
                story.append(Paragraph(f"Financial Stability: {strengths['financial_stability']}", styles['Normal']))
            if 'valuation' in strengths:
                story.append(Paragraph(f"Valuation: {strengths['valuation']}", styles['Normal']))
            if 'risk_profile' in strengths:
                story.append(Paragraph(f"Risk Profile: {strengths['risk_profile']}", styles['Normal']))
            
            story.append(Spacer(1, 15))
        
        # Key Insights
        if 'key_insights' in comparison:
            story.append(Paragraph("Key Insights", heading_style))
            for insight in comparison['key_insights']:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Risk Considerations
        if 'risk_considerations' in comparison:
            story.append(Paragraph("Risk Considerations", heading_style))
            story.append(Paragraph(comparison['risk_considerations'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Disclaimer", heading_style))
        disclaimer = """This comparative analysis is for educational and informational purposes only and does not constitute financial advice. 
        Always conduct your own research and consult with qualified financial professionals before making investment decisions. 
        Past performance does not guarantee future results. Diversification does not guarantee against loss."""
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Metadata
        story.append(Spacer(1, 20))
        metadata = f"Report generated: {result['timestamp'][:19]} | AI Model: {result.get('model_used', 'GPT-4')}"
        story.append(Paragraph(metadata, caption_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("📄 PDF generation requires reportlab package. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"❌ Failed to generate PDF: {str(e)}")
        dashboard_logger.error(f"PDF generation error: {e}", exc_info=True)
        return None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Stock selection
    st.sidebar.subheader("📊 Stock Selection")
    
    popular_stocks = {
        "Electric Vehicles": ["TSLA", "BYDDF", "RIVN", "LCID"],
        "Technology": ["AAPL", "GOOGL", "MSFT", "NVDA", "AMD"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "BMY"],
        "Consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX"],
        "Real Estate": ["SPY", "QQQ", "IWM", "VOO", "VGT", "VNQ"],
        "Consumer Staples": ["PG", "KO", "PEP", "WMT", "MCD", "WBA"],
        "Industrials": ["GE", "MMM", "BA", "CAT", "CSCO", "IBM"],
        "Utilities": ["DUK", "ED", "NEE", "SO", "D", "EDV"],
        "Communication Services": ["GOOGL", "META", "AMZN", "NFLX", "CMCSA", "DIS"],
    }
    
    # Selection method
    selection_method = st.sidebar.radio(
        "How would you like to select stocks?",
        ["Browse by Sector", "Enter Custom Symbols"]
    )
    
    symbols = []
    
    if selection_method == "Browse by Sector":
        # Use a key for the sector selectbox to detect changes
        sector = st.sidebar.selectbox("Choose sector:", list(popular_stocks.keys()), key="sector_select")

        # Initialize session state for selected symbols if it doesn't exist
        if 'selected_symbols_for_sector' not in st.session_state:
            st.session_state.selected_symbols_for_sector = popular_stocks[sector][:2]

        # If sector changes, update the default for multiselect
        # This part is tricky with direct callbacks for selectbox,
        # Streamlit's execution model makes it simpler to manage defaults based on current state.
        # We will manage the default of the multiselect.
        # A more robust way might involve a callback and resetting, but let's try managing the default first.

        # Check if the sector has changed or if the selection for the current sector is not set
        if 'current_sector_for_multiselect' not in st.session_state or \
           st.session_state.current_sector_for_multiselect != sector:
            st.session_state.current_sector_for_multiselect = sector
            # Set the default for the new sector
            st.session_state.selected_symbols_for_sector = popular_stocks[sector][:2]


        selected_symbols = st.sidebar.multiselect(
            f"Select {sector} stocks:",
            popular_stocks[sector],
            default=st.session_state.selected_symbols_for_sector, # Use session state for default
            key=f"multiselect_{sector}" # Dynamic key for multiselect based on sector
        )
        # Update session state if user changes selection within the same sector
        st.session_state.selected_symbols_for_sector = selected_symbols
        symbols = selected_symbols
        
    elif selection_method == "Enter Custom Symbols":
        symbols_text = st.sidebar.text_area(
            "Enter stock symbols (one per line or comma-separated):",
            "AAPL\nGOOGL\nTSLA"
        )
        symbols = [s.strip().upper() for s in symbols_text.replace(',', '\n').split('\n') if s.strip()]
        
    
    
    # Analysis settings
    st.sidebar.subheader("⚙️ Analysis Settings")
    period = st.sidebar.selectbox(
        "Time Period:",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    analysis_options = st.sidebar.multiselect(
        "Analysis Types:",
        ["Technical Analysis", "Financial Statements", "Peer Comparison", "Risk Analysis"],
        default=["Technical Analysis", "Financial Statements"]
    )
    
    if not symbols:
        st.info("👆 Please select some stocks from the sidebar to get started!")
        st.markdown("""
        ### 🎯 Welcome to the Stock Analysis Dashboard!
        
        This dashboard provides:
        - **Dynamic Stock Selection**: Choose from popular stocks or enter your own
        - **Technical Analysis**: 15+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands
        - **Financial Statements**: Income statement, balance sheet, cash flow analysis
        - **Peer Comparison**: Side-by-side stock performance comparison
        - **Data Export**: Download processed data in multiple formats
        
        Get started by selecting stocks from the sidebar! 🚀
        """)
        return
    
    # Main content tabs
    tabs = st.tabs([
        "📈 Technical Analysis", 
        "💰 Financial Analysis", 
        "🔍 Peer Comparison", 
        "⚠️ Risk Analysis",
        "🤖 AI Analysis",
        "💾 Data Export"
    ])
    
    # Tab 1: Technical Analysis
    with tabs[0]:
        if "Technical Analysis" in analysis_options:
            st.header("📈 Technical Analysis")
            
            for symbol in symbols:
                with st.expander(f"📊 {symbol} Technical Analysis", expanded=len(symbols) == 1):
                    with st.spinner(f"Loading and processing data for {symbol}..."):
                        df, stats = get_comprehensive_stock_data(symbol, period)
                    
                    if df is None:
                        st.error(f"❌ Could not load data for {symbol}")
                        continue
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        current_price = stats.get('current_price', 0)
                        price_change = stats.get('price_change_1d', 0)
                        st.metric("Price", f"${current_price:.2f}", f"{price_change:+.2f}")
                    
                    with col2:
                        change_pct = stats.get('price_change_pct_1d', 0)
                        st.metric("Change %", f"{change_pct:+.2f}%")
                    
                    with col3:
                        volatility = stats.get('volatility', 0)
                        st.metric("Volatility", f"{volatility:.2f}%")
                    
                    with col4:
                        rsi = stats.get('current_rsi', 0)
                        if rsi:
                            rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                            st.metric("RSI", f"{rsi:.1f}", rsi_signal)
                        else:
                            st.metric("RSI", "N/A")
                    
                    with col5:
                        sharpe = stats.get('sharpe_ratio', 0)
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    # Technical chart
                    fig = create_advanced_price_chart(df, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.processed_data[symbol] = df
    
    # Tab 2: Financial Analysis
    with tabs[1]:
        if "Financial Statements" in analysis_options:
            st.header("💰 Financial Analysis")
            
            for symbol in symbols:
                with st.expander(f"💼 {symbol} Financial Analysis", expanded=len(symbols) == 1):
                    with st.spinner(f"Loading financial data for {symbol}..."):
                        financials = get_financial_statements(symbol)
                    
                    if not financials:
                        st.error(f"❌ Could not load financial data for {symbol}")
                        continue
                    
                    # Company overview
                    info = financials.get('info', {})
                    if info:
                        st.markdown(f"### 🏢 {info.get('longName', symbol)} Overview")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Sector**: {info.get('sector', 'N/A')}")
                            st.write(f"**Industry**: {info.get('industry', 'N/A')}")
                            st.write(f"**Country**: {info.get('country', 'N/A')}")
                            st.write(f"**Employees**: {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees**: N/A")
                        
                        with col2:
                            st.write(f"**Website**: {info.get('website', 'N/A')}")
                            st.write(f"**CEO**: {info.get('companyOfficers', [{}])[0].get('name', 'N/A') if info.get('companyOfficers') else 'N/A'}")
                            st.write(f"**Founded**: {info.get('founded', 'N/A')}")
                            st.write(f"**Market**: {info.get('exchange', 'N/A')}")
                    
                    # Financial dashboard
                    create_financial_analysis_dashboard(financials, symbol)
                    
                    # Detailed financial statements
                    st.markdown("### 📋 Detailed Financial Statements")
                    statement_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Earnings", "Recommendations"])
                    
                    with statement_tabs[0]:
                        income_stmt = financials.get('income_statement', pd.DataFrame())
                        if not income_stmt.empty:
                            st.dataframe(income_stmt.head(20), use_container_width=True)
                        else:
                            st.info("Income statement data not available")
                    
                    with statement_tabs[1]:
                        balance_sheet = financials.get('balance_sheet', pd.DataFrame())
                        if not balance_sheet.empty:
                            st.dataframe(balance_sheet.head(20), use_container_width=True)
                        else:
                            st.info("Balance sheet data not available")
                    
                    with statement_tabs[2]:
                        cash_flow = financials.get('cash_flow', pd.DataFrame())
                        if not cash_flow.empty:
                            st.dataframe(cash_flow.head(20), use_container_width=True)
                        else:
                            st.info("Cash flow data not available")
                    
                    with statement_tabs[3]:
                        earnings_data = financials.get('earnings_dates')

                        if earnings_data is not None and not earnings_data.empty:
                            st.dataframe(earnings_data)
                        else:
                            st.write("Earnings dates data not available.")
                    
                    with statement_tabs[4]:
                        recommendations = financials.get('recommendations', pd.DataFrame())
                        if not recommendations.empty:
                            st.dataframe(recommendations.tail(10), use_container_width=True)
                        else:
                            st.info("Analyst recommendations not available")
                    
                    # Store in session state
                    st.session_state.financial_data[symbol] = financials
    
            # Tab 3: Peer Comparison
    with tabs[2]:
        if "Peer Comparison" in analysis_options:
            if len(symbols) < 2:
                st.info("👥 Peer Comparison requires at least 2 stocks. Please select more stocks from the sidebar.")
                st.markdown("""
                ### How to use Peer Comparison:
                1. Select 2 or more stocks from the sidebar
                2. Make sure "Peer Comparison" is checked in Analysis Types
                3. The comparison will show relative performance, correlations, and key metrics
                """)
            else:
                st.header("🔍 Peer Comparison")
                
                comparison_data = {}
                failed_symbols = []
                for symbol in symbols:
                    df, stats = get_comprehensive_stock_data(symbol, period)
                    if df is not None and not df.empty and stats is not None:
                        comparison_data[symbol] = {'data': df, 'stats': stats}
                    else:
                        failed_symbols.append(symbol)
                
                if failed_symbols:
                    st.error(f"Could not load data for the following symbols (they will be excluded from comparison): {', '.join(failed_symbols)}")

                if len(comparison_data) < 2:
                    st.warning("Need at least 2 valid stocks with data for comparison. Please check selections or try different symbols.")
                else:
                    # Normalized performance comparison
                    st.subheader("📊 Performance Comparison")
                    fig = go.Figure()
                    
                    for symbol, data in comparison_data.items():
                        df = data['data']
                        normalized_price = (df['close'] / df['close'].iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=normalized_price,
                            name=symbol,
                            line=dict(width=3)
                        ))
                    
                    fig.update_layout(
                        title="Normalized Performance Comparison (Starting Point = 100)",
                        xaxis_title="Date",
                        yaxis_title="Normalized Price",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics table
                    st.subheader("📈 Performance Metrics Comparison")
                    metrics_data = []
                    for symbol, data in comparison_data.items():
                        stats = data['stats']
                        metrics_data.append({
                            'Symbol': symbol,
                            'Current Price': f"${stats.get('current_price', 0):.2f}",
                            '1D Change %': f"{stats.get('price_change_pct_1d', 0):+.2f}%",
                            '7D Change %': f"{stats.get('price_change_7d', 0):+.2f}%",
                            '30D Change %': f"{stats.get('price_change_30d', 0):+.2f}%",
                            'Volatility %': f"{stats.get('volatility', 0):.2f}%",
                            'Sharpe Ratio': f"{stats.get('sharpe_ratio', 0):.2f}",
                            'RSI': f"{stats.get('current_rsi', 0):.1f}" if stats.get('current_rsi') else "N/A"
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Correlation analysis
                    st.subheader("🔗 Correlation Analysis")
                    returns_data = {}
                    for symbol, data in comparison_data.items():
                        df = data['data']
                        if 'daily_return' in df.columns:
                            returns_data[symbol] = df['daily_return']
                    
                    if len(returns_data) > 1:
                        returns_df = pd.DataFrame(returns_data).dropna()
                        correlation_matrix = returns_df.corr()
                        
                        fig = px.imshow(
                            correlation_matrix,
                            text_auto=True,
                            color_continuous_scale="RdBu",
                            title="Daily Returns Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Risk Analysis
    with tabs[3]:
        if "Risk Analysis" in analysis_options:
            st.header("⚠️ Risk Analysis")
            
            for symbol in symbols:
                # Get data for the symbol if not already processed
                if symbol not in st.session_state.processed_data:
                    with st.spinner(f"Loading data for {symbol}..."):
                        df, stats = get_comprehensive_stock_data(symbol, period)
                        if df is not None:
                            st.session_state.processed_data[symbol] = df
                
                if symbol in st.session_state.processed_data:
                    df = st.session_state.processed_data[symbol]
                    
                    with st.expander(f"⚠️ {symbol} Risk Analysis", expanded=len(symbols) == 1):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Risk Metrics")
                            
                            if 'daily_return' in df.columns:
                                returns = df['daily_return'].dropna()
                                
                                # Calculate risk metrics
                                volatility = returns.std() * np.sqrt(252) * 100
                                var_95 = returns.quantile(0.05) * 100
                                var_99 = returns.quantile(0.01) * 100
                                
                                st.metric("Annual Volatility", f"{volatility:.2f}%")
                                st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
                                st.metric("Value at Risk (99%)", f"{var_99:.2f}%")
                                
                                # Maximum drawdown
                                cumulative_returns = (1 + returns).cumprod()
                                rolling_max = cumulative_returns.cummax()
                                drawdown = (cumulative_returns - rolling_max) / rolling_max
                                max_drawdown = drawdown.min() * 100
                                st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
                        
                        with col2:
                            st.subheader("📈 Return Distribution")
                            
                            if 'daily_return' in df.columns:
                                returns = df['daily_return'].dropna() * 100
                                
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=returns,
                                    nbinsx=50,
                                    name="Daily Returns",
                                    marker_color='lightblue',
                                    opacity=0.7
                                ))
                                
                                fig.update_layout(
                                    title=f"{symbol} Daily Returns Distribution",
                                    xaxis_title="Daily Return (%)",
                                    yaxis_title="Frequency",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: AI Analysis
    with tabs[4]:
        st.header("🤖 AI-Powered Investment Analysis")
        
        # Initialize AI analyzer
        components = init_pipeline_components()
        if not all(components):
            st.error("Failed to initialize AI analyzer.")
        else:
            _, _, _, ai_analyzer = components
            
            if not ai_analyzer.is_available():
                st.warning("🔑 AI Analysis requires OpenAI API key. Please set OPENAI_API_KEY in your environment variables.")
                st.markdown("""
                ### How to enable AI Analysis:
                1. Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
                2. Set the environment variable: `export OPENAI_API_KEY=your_key_here`
                3. Restart the dashboard
                
                The AI analysis provides:
                - **Long-term investment recommendations** (6+ months horizon)
                - **Buy/Hold/Sell signals** with confidence levels
                - **Risk assessment** and key factors to monitor
                - **Comparative analysis** for portfolio allocation
                """)
            else:
                st.success("✅ AI Analysis is ready! Using OpenAI for long-term investment insights.")
                
                # Analysis mode selection
                analysis_mode = st.radio(
                    "Choose Analysis Mode:",
                    ["Individual Stock Analysis", "Comparative Portfolio Analysis"]
                )
                
                if analysis_mode == "Individual Stock Analysis":
                    # Individual stock analysis
                    st.subheader("📊 Individual Stock Analysis")
                    
                    for symbol in symbols:
                        with st.expander(f"🤖 {symbol} AI Investment Analysis", expanded=len(symbols) == 1):
                            col1, col2 = st.columns([2, 1])
                            
                            with col2:
                                if st.button(f"🔍 Analyze {symbol}", key=f"analyze_{symbol}"):
                                    # Get data
                                    with st.spinner(f"Analyzing {symbol} with AI..."):
                                        # Ensure we have the data
                                        if symbol not in st.session_state.processed_data:
                                            df, stats = get_comprehensive_stock_data(symbol, period)
                                            if df is not None:
                                                st.session_state.processed_data[symbol] = df
                                        
                                        if symbol not in st.session_state.financial_data:
                                            financials = get_financial_statements(symbol)
                                            if financials:
                                                st.session_state.financial_data[symbol] = financials
                                        
                                        # Prepare data for AI analysis
                                        df = st.session_state.processed_data.get(symbol)
                                        financials = st.session_state.financial_data.get(symbol)
                                        
                                        if df is not None:
                                            # Generate stats
                                            _, data_processor, _, _ = init_pipeline_components()
                                            stats = data_processor.generate_summary_statistics(df, symbol)
                                            
                                            # Run AI analysis
                                            analysis_result = ai_analyzer.analyze_stock_longterm(
                                                symbol, df, stats, financials
                                            )
                                            
                                            # Store in session state
                                            st.session_state.ai_analysis[symbol] = analysis_result
                            
                            with col1:
                                # Display existing analysis if available
                                if symbol in st.session_state.ai_analysis:
                                    result = st.session_state.ai_analysis[symbol]
                                    
                                    if result['available']:
                                        analysis = result['analysis']
                                        
                                        # Recommendation header
                                        rec = analysis.get('recommendation', 'HOLD')
                                        confidence = analysis.get('confidence_level', 'MEDIUM')
                                        
                                        # Color coding for recommendations
                                        rec_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(rec, "🟡")
                                        conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "🟡")
                                        
                                        st.markdown(f"### {rec_color} **{rec}** | Confidence: {conf_color} {confidence}")
                                        
                                        # Key metrics
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            if 'price_target_range' in analysis:
                                                target = analysis['price_target_range']
                                                if isinstance(target, dict) and target.get('low') and target.get('high'):
                                                    st.metric("Price Target Range", f"${target['low']:.2f} - ${target['high']:.2f}")
                                        
                                        with col_b:
                                            if 'target_timeframe' in analysis:
                                                st.metric("Timeframe", analysis['target_timeframe'])
                                        
                                        # Investment thesis
                                        if 'investment_thesis' in analysis:
                                            st.markdown("**💡 Investment Thesis:**")
                                            st.write(analysis['investment_thesis'])
                                        
                                        # Detailed analysis sections (using tabs instead of nested expander)
                                        st.markdown("---")
                                        st.markdown("**📋 Detailed Analysis**")
                                        tabs_detail = st.tabs(["Strengths & Risks", "Technical", "Fundamental", "Catalysts"])
                                        
                                        with tabs_detail[0]:
                                            col_str, col_risk = st.columns(2)
                                            with col_str:
                                                st.markdown("**🚀 Key Strengths:**")
                                                if 'key_strengths' in analysis:
                                                    for strength in analysis['key_strengths']:
                                                        st.write(f"• {strength}")
                                            
                                            with col_risk:
                                                st.markdown("**⚠️ Key Risks:**")
                                                if 'key_risks' in analysis:
                                                    for risk in analysis['key_risks']:
                                                        st.write(f"• {risk}")
                                        
                                        with tabs_detail[1]:
                                            if 'technical_outlook' in analysis:
                                                st.write(analysis['technical_outlook'])
                                        
                                        with tabs_detail[2]:
                                            if 'fundamental_outlook' in analysis:
                                                st.write(analysis['fundamental_outlook'])
                                        
                                        with tabs_detail[3]:
                                            if 'catalysts' in analysis:
                                                st.write(analysis['catalysts'])
                                        
                                        # Analysis metadata
                                        st.caption(f"Analysis generated: {result['timestamp'][:19]} | Model: {result.get('model_used', 'GPT-4')}")
                                        
                                        # Add PDF download functionality
                                        st.markdown("---")
                                        if st.button(f"📄 Download {symbol} AI Analysis as PDF", key=f"pdf_{symbol}"):
                                            with st.spinner("Generating PDF report..."):
                                                pdf_content = generate_ai_analysis_pdf(symbol, analysis, result)
                                                if pdf_content:
                                                    st.download_button(
                                                        f"💾 Download PDF Report",
                                                        pdf_content,
                                                        f"{symbol}_AI_Analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                        "application/pdf",
                                                        key=f"download_pdf_{symbol}"
                                                    )
                                    
                                    else:
                                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                                
                                else:
                                    st.info(f"👆 Click 'Analyze {symbol}' to generate AI-powered investment analysis")
                
                elif analysis_mode == "Comparative Portfolio Analysis":
                    # Comparative analysis
                    st.subheader("🔄 Comparative Portfolio Analysis")
                    
                    if len(symbols) < 2:
                        st.info("📊 Comparative analysis requires at least 2 stocks. Please select more stocks from the sidebar.")
                    else:
                        col_comp1, col_comp2 = st.columns([3, 1])
                        
                        with col_comp2:
                            if st.button("🔄 Compare Stocks", key="compare_all"):
                                with st.spinner("Running AI comparative analysis..."):
                                    # Prepare data for all symbols
                                    comparison_data = {}
                                    
                                    for symbol in symbols:
                                        # Ensure we have data
                                        if symbol not in st.session_state.processed_data:
                                            df, stats = get_comprehensive_stock_data(symbol, period)
                                            if df is not None:
                                                st.session_state.processed_data[symbol] = df
                                        
                                        if symbol not in st.session_state.financial_data:
                                            financials = get_financial_statements(symbol)
                                            if financials:
                                                st.session_state.financial_data[symbol] = financials
                                        
                                        # Prepare data dict
                                        df = st.session_state.processed_data.get(symbol)
                                        financials = st.session_state.financial_data.get(symbol)
                                        
                                        if df is not None:
                                            _, data_processor, _, _ = init_pipeline_components()
                                            stats = data_processor.generate_summary_statistics(df, symbol)
                                            
                                            comparison_data[symbol] = {
                                                'data': df,
                                                'stats': stats,
                                                'financials': financials
                                            }
                                    
                                    # Run comparative analysis
                                    if len(comparison_data) >= 2:
                                        comparison_result = ai_analyzer.compare_stocks_ai(symbols, comparison_data)
                                        st.session_state.ai_analysis['comparison'] = comparison_result
                        
                        with col_comp1:
                            # Display comparison results
                            if 'comparison' in st.session_state.ai_analysis:
                                result = st.session_state.ai_analysis['comparison']
                                
                                if result['available']:
                                    comparison = result['comparison']
                                    
                                    # Rankings
                                    if 'ranking' in comparison:
                                        st.markdown("### 🏆 Investment Ranking")
                                        ranking_df = pd.DataFrame(comparison['ranking'])
                                        st.dataframe(ranking_df, use_container_width=True)
                                    
                                    # Portfolio allocations
                                    if 'portfolio_allocation' in comparison:
                                        st.markdown("### 💼 Portfolio Allocation Recommendations")
                                        
                                        alloc = comparison['portfolio_allocation']
                                        col_cons, col_mod, col_agg = st.columns(3)
                                        
                                        with col_cons:
                                            st.markdown("**Conservative Investor**")
                                            if 'conservative_investor' in alloc:
                                                for stock, pct in alloc['conservative_investor'].items():
                                                    st.write(f"{stock}: {pct}%")
                                        
                                        with col_mod:
                                            st.markdown("**Moderate Investor**")
                                            if 'moderate_investor' in alloc:
                                                for stock, pct in alloc['moderate_investor'].items():
                                                    st.write(f"{stock}: {pct}%")
                                        
                                        with col_agg:
                                            st.markdown("**Aggressive Investor**")
                                            if 'aggressive_investor' in alloc:
                                                for stock, pct in alloc['aggressive_investor'].items():
                                                    st.write(f"{stock}: {pct}%")
                                    
                                    # Key insights
                                    if 'key_insights' in comparison:
                                        st.markdown("### 💡 Key Insights")
                                        for insight in comparison['key_insights']:
                                            st.write(f"• {insight}")
                                    
                                    # Risk considerations
                                    if 'risk_considerations' in comparison:
                                        st.markdown("### ⚠️ Risk Considerations")
                                        st.write(comparison['risk_considerations'])
                                
                                else:
                                    st.error(f"❌ Comparison failed: {result.get('error', 'Unknown error')}")
                            
                            else:
                                st.info("👆 Click 'Compare Stocks' to generate AI-powered comparative analysis")
                        
                        # Add PDF download for comparative analysis
                        if 'comparison' in st.session_state.ai_analysis:
                            result = st.session_state.ai_analysis['comparison']
                            if result['available']:
                                st.markdown("---")
                                if st.button("📄 Download Comparative Analysis as PDF", key="pdf_comparison"):
                                    with st.spinner("Generating comparative analysis PDF..."):
                                        pdf_content = generate_comparative_analysis_pdf(symbols, result['comparison'], result)
                                        if pdf_content:
                                            st.download_button(
                                                "💾 Download Comparative PDF Report",
                                                pdf_content,
                                                f"Comparative_Analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                                                "application/pdf",
                                                key="download_comparative_pdf"
                                            )
    
    # Tab 6: Data Export
    with tabs[5]:
        st.header("💾 Data Export & Reports")
        
        if st.session_state.processed_data:
            # Export options
            export_format = st.selectbox("Choose export format:", ["CSV", "Excel", "JSON"])
            include_technical = st.checkbox("Include technical indicators", True)
            include_financial = st.checkbox("Include financial data", True)
            
            for symbol in st.session_state.processed_data:
                df = st.session_state.processed_data[symbol]
                
                st.subheader(f"📊 {symbol} Data Export")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Preview:**")
                    display_df = df.tail(10) if include_technical else df[['open', 'high', 'low', 'close', 'volume']].tail(10)
                    st.dataframe(display_df, use_container_width=True)
                
                with col2:
                    st.write("**Export Options:**")
                    
                    if export_format == "CSV":
                        csv = df.to_csv()
                        st.download_button(
                            f"📥 Download {symbol} CSV",
                            csv,
                            f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )
                    
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Price_Data')
                            if include_financial and symbol in st.session_state.financial_data:
                                financials = st.session_state.financial_data[symbol]
                                if not financials['income_statement'].empty:
                                    financials['income_statement'].to_excel(writer, sheet_name='Income_Statement')
                                if not financials['balance_sheet'].empty:
                                    financials['balance_sheet'].to_excel(writer, sheet_name='Balance_Sheet')
                        
                        st.download_button(
                            f"📥 Download {symbol} Excel",
                            buffer.getvalue(),
                            f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif export_format == "JSON":
                        json_data = df.to_json(orient='records', date_format='iso')
                        st.download_button(
                            f"📥 Download {symbol} JSON",
                            json_data,
                            f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.json",
                            "application/json"
                        )
            
            # Generate comprehensive report
            if st.button("🎯 Generate Comprehensive Report"):
                components = init_pipeline_components()
                if not all(components):
                    st.error("Failed to initialize report generator.")
                    return
                _, _, report_generator, _ = components
                
                if report_generator:
                    with st.spinner("Generating comprehensive report..."):
                        # Prepare data for report generation
                        data_dict = st.session_state.processed_data
                        
                        # Create comparison DataFrame
                        if len(data_dict) > 1:
                            _, data_processor, _, _ = init_pipeline_components()
                            comparison_df = data_processor.create_comparison_dataframe(data_dict)
                        else:
                            comparison_df = pd.DataFrame()
                        
                        # Generate statistics
                        stats_dict = {}
                        for symbol, df in data_dict.items():
                            _, data_processor, _, _ = init_pipeline_components()
                            stats_dict[symbol] = data_processor.generate_summary_statistics(df, symbol)
                        
                        # Generate reports
                        reports = report_generator.generate_comprehensive_report(
                            data_dict=data_dict,
                            comparison_df=comparison_df,
                            stats_dict=stats_dict,
                            timestamp=datetime.now()
                        )
                        
                        if reports:
                            st.success("✅ Reports generated successfully!")
                            for report_type, filepath in reports.items():
                                st.write(f"📄 **{report_type.upper()}**: `{filepath}`")
                        else:
                            st.error("❌ Failed to generate reports")

if __name__ == "__main__":
    main() 