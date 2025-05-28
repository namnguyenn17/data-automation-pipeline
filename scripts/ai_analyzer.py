"""
AI-powered Stock Analysis Module

This module uses OpenAI GPT to provide intelligent stock analysis
for long-term investment decisions (6+ months horizon).
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from openai import OpenAI
from datetime import datetime
from .config import Config

logger = logging.getLogger('data_automation_pipeline')

class AIStockAnalyzer:
    """AI-powered stock analyzer using OpenAI for long-term investment insights."""
    
    def __init__(self):
        """Initialize the AI analyzer with OpenAI client."""
        self.client = None
        if Config.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    def is_available(self) -> bool:
        """Check if AI analysis is available."""
        return self.client is not None
    
    def prepare_technical_summary(self, df: pd.DataFrame, stats: Dict) -> str:
        """Prepare technical analysis summary for AI input."""
        try:
            current_price = stats.get('current_price', 0)
            price_change_1d = stats.get('price_change_pct_1d', 0)
            price_change_7d = stats.get('price_change_7d', 0)
            price_change_30d = stats.get('price_change_30d', 0)
            volatility = stats.get('volatility', 0)
            rsi = stats.get('current_rsi', 0)
            sharpe_ratio = stats.get('sharpe_ratio', 0)
            
            # Calculate additional metrics
            recent_data = df.tail(20)
            sma_20 = recent_data['close'].mean() if len(recent_data) > 0 else 0
            sma_50 = df.tail(50)['close'].mean() if len(df) >= 50 else 0
            
            # Price position relative to moving averages
            price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
            price_vs_sma50 = ((current_price - sma_50) / sma_50 * 100) if sma_50 > 0 else 0
            
            # Volume analysis
            avg_volume = df['volume'].tail(20).mean()
            recent_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
            volume_ratio = (recent_volume / avg_volume) if avg_volume > 0 else 1
            
            # Support and resistance levels
            high_52w = df['high'].tail(252).max() if len(df) >= 252 else df['high'].max()
            low_52w = df['low'].tail(252).min() if len(df) >= 252 else df['low'].min()
            
            technical_summary = f"""
Technical Analysis Summary:
- Current Price: ${current_price:.2f}
- Price Changes: 1D: {price_change_1d:+.2f}%, 7D: {price_change_7d:+.2f}%, 30D: {price_change_30d:+.2f}%
- Volatility (Annualized): {volatility:.2f}%
- RSI (14-period): {rsi:.1f}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Price vs 20-day SMA: {price_vs_sma20:+.2f}%
- Price vs 50-day SMA: {price_vs_sma50:+.2f}%
- Volume Ratio (vs 20-day avg): {volume_ratio:.2f}x
- 52-week High: ${high_52w:.2f}
- 52-week Low: ${low_52w:.2f}
- Distance from 52w High: {((current_price - high_52w) / high_52w * 100):+.2f}%
- Distance from 52w Low: {((current_price - low_52w) / low_52w * 100):+.2f}%
"""
            return technical_summary
        except Exception as e:
            logger.error(f"Error preparing technical summary: {e}")
            return "Technical analysis data unavailable"
    
    def prepare_financial_summary(self, financials: Dict) -> str:
        """Prepare financial analysis summary for AI input."""
        try:
            info = financials.get('info', {})
            income_stmt = financials.get('income_statement', pd.DataFrame())
            balance_sheet = financials.get('balance_sheet', pd.DataFrame())
            
            # Company basics
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            market_cap = info.get('marketCap', 0)
            
            # Key financial metrics
            pe_ratio = info.get('trailingPE', 0)
            forward_pe = info.get('forwardPE', 0)
            peg_ratio = info.get('pegRatio', 0)
            price_to_book = info.get('priceToBook', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            roe = info.get('returnOnEquity', 0)
            profit_margin = info.get('profitMargins', 0)
            revenue_growth = info.get('revenueGrowth', 0)
            earnings_growth = info.get('earningsGrowth', 0)
            
            # Financial health indicators
            current_ratio = info.get('currentRatio', 0)
            quick_ratio = info.get('quickRatio', 0)
            cash_per_share = info.get('totalCashPerShare', 0)
            
            financial_summary = f"""
Financial Analysis Summary:
Company Profile:
- Sector: {sector}
- Industry: {industry}
- Market Cap: ${market_cap:,.0f} if {market_cap} > 0 else 'N/A'

Valuation Metrics:
- P/E Ratio (Trailing): {pe_ratio:.2f} if {pe_ratio} > 0 else 'N/A'
- P/E Ratio (Forward): {forward_pe:.2f} if {forward_pe} > 0 else 'N/A'
- PEG Ratio: {peg_ratio:.2f} if {peg_ratio} > 0 else 'N/A'
- Price-to-Book: {price_to_book:.2f} if {price_to_book} > 0 else 'N/A'

Profitability & Growth:
- Profit Margin: {profit_margin*100:.2f}% if {profit_margin} > 0 else 'N/A'
- Return on Equity: {roe*100:.2f}% if {roe} > 0 else 'N/A'
- Revenue Growth: {revenue_growth*100:+.2f}% if {revenue_growth} != 0 else 'N/A'
- Earnings Growth: {earnings_growth*100:+.2f}% if {earnings_growth} != 0 else 'N/A'

Financial Health:
- Debt-to-Equity: {debt_to_equity:.2f} if {debt_to_equity} > 0 else 'N/A'
- Current Ratio: {current_ratio:.2f} if {current_ratio} > 0 else 'N/A'
- Quick Ratio: {quick_ratio:.2f} if {quick_ratio} > 0 else 'N/A'
- Cash per Share: ${cash_per_share:.2f} if {cash_per_share} > 0 else 'N/A'
"""
            return financial_summary
        except Exception as e:
            logger.error(f"Error preparing financial summary: {e}")
            return "Financial analysis data unavailable"
    
    def analyze_stock_longterm(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        stats: Dict, 
        financials: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Provide comprehensive AI analysis for long-term investment (6+ months).
        
        Returns:
            Dict containing analysis results, recommendation, and confidence level
        """
        if not self.is_available():
            return {
                'available': False,
                'error': 'OpenAI API not configured. Please set OPENAI_API_KEY environment variable.'
            }
        
        try:
            # Prepare data summaries
            technical_summary = self.prepare_technical_summary(df, stats)
            financial_summary = self.prepare_financial_summary(financials) if financials else "Financial data not available"
            
            # Create the analysis prompt
            prompt = f"""
You are a professional financial analyst providing long-term investment analysis (6+ months horizon). 

Analyze the following stock data for {symbol} and provide a comprehensive investment recommendation:

{technical_summary}

{financial_summary}

Please provide your analysis in the following JSON format:
{{
    "recommendation": "BUY|HOLD|SELL",
    "confidence_level": "HIGH|MEDIUM|LOW",
    "target_timeframe": "6-12 months",
    "price_target_range": {{"low": 0, "high": 0}},
    "key_strengths": ["strength1", "strength2", "strength3"],
    "key_risks": ["risk1", "risk2", "risk3"],
    "investment_thesis": "2-3 sentence summary of why you recommend this action",
    "technical_outlook": "Summary of technical indicators and chart patterns",
    "fundamental_outlook": "Summary of financial health and business prospects",
    "sector_considerations": "Industry trends and competitive positioning",
    "risk_factors": "Main risks to monitor",
    "catalysts": "Potential positive catalysts in next 6-12 months"
}}

Focus on:
1. Long-term value creation potential
2. Financial stability and growth prospects
3. Industry positioning and competitive advantages
4. Risk-adjusted returns for 6+ month holding period
5. Consider current market conditions and macroeconomic factors

Be objective and provide actionable insights for long-term investors.
"""

            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst specializing in long-term investment analysis. Provide detailed, objective analysis based on the data provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1500
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON content between braces
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = analysis_text[start_idx:end_idx]
                    analysis_data = json.loads(json_str)
                else:
                    # Fallback if JSON extraction fails
                    analysis_data = {
                        "recommendation": "HOLD",
                        "confidence_level": "MEDIUM",
                        "analysis_text": analysis_text
                    }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis_data = {
                    "recommendation": "HOLD",
                    "confidence_level": "MEDIUM", 
                    "analysis_text": analysis_text
                }
            
            return {
                'available': True,
                'symbol': symbol,
                'analysis': analysis_data,
                'timestamp': datetime.now().isoformat(),
                'model_used': Config.OPENAI_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return {
                'available': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def compare_stocks_ai(self, symbols: List[str], data_dict: Dict) -> Dict[str, any]:
        """
        AI-powered comparison of multiple stocks for long-term investment.
        
        Args:
            symbols: List of stock symbols to compare
            data_dict: Dictionary containing data for each symbol
            
        Returns:
            Dict containing comparative analysis and ranking
        """
        if not self.is_available():
            return {
                'available': False,
                'error': 'OpenAI API not configured.'
            }
        
        if len(symbols) < 2:
            return {
                'available': False,
                'error': 'Need at least 2 stocks for comparison.'
            }
        
        try:
            # Prepare comparison data
            comparison_summary = f"Comparative Analysis of {', '.join(symbols)} for Long-term Investment:\n\n"
            
            for symbol in symbols:
                if symbol in data_dict:
                    stock_data = data_dict[symbol]
                    df = stock_data.get('data')
                    stats = stock_data.get('stats')
                    financials = stock_data.get('financials')
                    
                    if df is not None and stats is not None:
                        comparison_summary += f"\n--- {symbol} ---\n"
                        comparison_summary += self.prepare_technical_summary(df, stats)
                        if financials:
                            comparison_summary += self.prepare_financial_summary(financials)
                        comparison_summary += "\n"
            
            # Create comparison prompt
            prompt = f"""
As a professional financial analyst, compare these stocks for long-term investment (6+ months horizon):

{comparison_summary}

Provide a comparative analysis in JSON format:
{{
    "ranking": [
        {{
            "symbol": "STOCK1",
            "rank": 1,
            "score": 85,
            "recommendation": "BUY|HOLD|SELL",
            "rationale": "Why this stock ranks here"
        }}
    ],
    "comparative_strengths": {{
        "growth_potential": "Which stock has best growth prospects",
        "financial_stability": "Which stock is most financially stable",
        "valuation": "Which stock offers best value",
        "risk_profile": "Which stock has best risk-adjusted returns"
    }},
    "portfolio_allocation": {{
        "conservative_investor": {{"STOCK1": 60, "STOCK2": 40}},
        "moderate_investor": {{"STOCK1": 50, "STOCK2": 50}},
        "aggressive_investor": {{"STOCK1": 40, "STOCK2": 60}}
    }},
    "key_insights": ["insight1", "insight2", "insight3"],
    "risk_considerations": "Main risks when investing in this combination"
}}

Focus on long-term value creation, diversification benefits, and risk-adjusted returns.
"""

            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional portfolio analyst specializing in comparative stock analysis for long-term investments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = analysis_text[start_idx:end_idx]
                    comparison_data = json.loads(json_str)
                else:
                    comparison_data = {"analysis_text": analysis_text}
            except json.JSONDecodeError:
                comparison_data = {"analysis_text": analysis_text}
            
            return {
                'available': True,
                'symbols': symbols,
                'comparison': comparison_data,
                'timestamp': datetime.now().isoformat(),
                'model_used': Config.OPENAI_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error in AI comparison: {e}")
            return {
                'available': False,
                'error': f'Comparison failed: {str(e)}'
            } 