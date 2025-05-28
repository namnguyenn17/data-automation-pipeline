# 🤖 AI-Powered Stock Analysis Setup Guide

## Overview

Your Data Automation Pipeline now includes AI-powered investment analysis using OpenAI's GPT models. This feature provides **long-term investment recommendations** (6+ months horizon) with detailed analysis of technical indicators, financial data, and market conditions.

## Features

- **Individual Stock Analysis**: Get detailed buy/hold/sell recommendations with confidence levels
- **Comparative Portfolio Analysis**: Compare multiple stocks and get portfolio allocation recommendations
- **Long-term Focus**: All analysis is designed for 6+ month investment horizons
- **Professional Insights**: Technical, fundamental, and sector analysis
- **Risk Assessment**: Comprehensive risk evaluation and monitoring factors

## Setup Instructions

### 1. Install Dependencies

First, make sure you have the OpenAI package installed:

```bash
pip install openai>=1.0.0
```

Or if using the requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Create a new API key
4. Copy the key (it starts with `sk-`)

### 3. Configure Environment Variables

Set your OpenAI API key as an environment variable:

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_MODEL=gpt-4  # Optional, defaults to gpt-4
```

**On Windows:**
```cmd
set OPENAI_API_KEY=your_api_key_here
set OPENAI_MODEL=gpt-4
```

**Or create a .env file:**
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
STOCK_SYMBOLS=AMD,NVDA,BYD,TSLA
```

### 4. Run the Dashboard

```bash
streamlit run dashboard.py
```

## Usage Guide

### Individual Stock Analysis

1. Select stocks from the sidebar
2. Navigate to the "🤖 AI Analysis" tab
3. Choose "Individual Stock Analysis"
4. Click "🔍 Analyze [SYMBOL]" for any stock
5. View the comprehensive analysis including:
   - Buy/Hold/Sell recommendation with confidence level
   - Price target range
   - Investment thesis
   - Key strengths and risks
   - Technical and fundamental outlook
   - Potential catalysts

### Comparative Portfolio Analysis

1. Select 2 or more stocks from the sidebar
2. Navigate to the "🤖 AI Analysis" tab
3. Choose "Comparative Portfolio Analysis"
4. Click "🔄 Compare Stocks"
5. View comparative analysis including:
   - Investment ranking
   - Portfolio allocation recommendations for different investor types
   - Key insights and risk considerations

## Important Notes

### Cost Considerations

- Each AI analysis call uses OpenAI tokens
- Individual stock analysis: ~1,000-1,500 tokens
- Comparative analysis: ~1,200-2,000 tokens
- Current GPT-4 pricing: ~$0.03 per 1,000 tokens
- Consider using GPT-3.5-turbo for lower costs (set `OPENAI_MODEL=gpt-3.5-turbo`)

### Investment Disclaimer

**⚠️ Important:** The AI analysis is for educational and informational purposes only. It is NOT financial advice. Always:

- Do your own research
- Consult with financial professionals
- Consider your risk tolerance
- Diversify your investments
- Never invest more than you can afford to lose

### Data Sources

The AI analysis combines:
- **Technical Data**: From Yahoo Finance (price, volume, indicators)
- **Financial Data**: From Yahoo Finance (income statements, balance sheets)
- **Market Context**: The AI model's training data (up to its knowledge cutoff)

## Troubleshooting

### "AI Analysis requires OpenAI API key"

- Ensure your `OPENAI_API_KEY` environment variable is set
- Restart the Streamlit app after setting environment variables
- Check that your API key is valid and has credits

### "Analysis failed" Error

- Check your internet connection
- Verify your OpenAI API key has available credits
- Try with a smaller number of stocks for comparative analysis
- Check the logs for detailed error messages

### Analysis Taking Too Long

- The AI analysis can take 10-30 seconds per stock
- For multiple stocks, consider analyzing them one at a time
- Using GPT-3.5-turbo is faster than GPT-4

## Example Analysis Output

```
🟢 BUY | Confidence: 🟢 HIGH

Price Target Range: $95.00 - $115.00
Timeframe: 6-12 months

💡 Investment Thesis:
Strong fundamentals with robust revenue growth and expanding market share in the semiconductor industry. Technical indicators suggest oversold conditions with potential for recovery.

🚀 Key Strengths:
• Strong revenue growth in AI and data center segments
• Solid balance sheet with low debt-to-equity ratio
• Market leadership in key growth areas

⚠️ Key Risks:
• Cyclical nature of semiconductor industry
• Geopolitical tensions affecting supply chain
• High valuation multiples compared to historical averages
```

## Support

If you encounter issues:

1. Check this guide first
2. Review the logs in `logs/pipeline.log`
3. Ensure all dependencies are installed
4. Verify your OpenAI API key is valid

Happy investing! 🚀📈 