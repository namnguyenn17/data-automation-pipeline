#!/usr/bin/env python3
"""
Test script to verify PDF generation functionality
"""

import sys
import os
from datetime import datetime

# Add the scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.dirname(__file__))

def test_pdf_generation():
    """Test the PDF generation with mock data."""
    try:
        from dashboard import generate_ai_analysis_pdf
        
        # Mock analysis data
        mock_analysis = {
            'recommendation': 'BUY',
            'confidence_level': 'HIGH',
            'target_timeframe': '6-12 months',
            'price_target_range': {'low': 95.0, 'high': 115.0},
            'investment_thesis': 'Strong fundamentals with robust revenue growth and expanding market share in the semiconductor industry.',
            'key_strengths': [
                'Strong revenue growth in AI and data center segments',
                'Solid balance sheet with low debt-to-equity ratio',
                'Market leadership in key growth areas'
            ],
            'key_risks': [
                'Cyclical nature of semiconductor industry',
                'Geopolitical tensions affecting supply chain',
                'High valuation multiples compared to historical averages'
            ],
            'technical_outlook': 'Technical indicators suggest oversold conditions with potential for recovery.',
            'fundamental_outlook': 'Solid financial performance with strong cash generation.',
            'sector_considerations': 'Semiconductor sector showing signs of recovery.',
            'risk_factors': 'Monitor trade tensions and inventory levels.',
            'catalysts': 'New product launches and datacenter expansion.'
        }
        
        mock_result = {
            'timestamp': datetime.now().isoformat(),
            'model_used': 'GPT-4'
        }
        
        # Generate PDF
        print("Testing PDF generation...")
        pdf_content = generate_ai_analysis_pdf('NVDA', mock_analysis, mock_result)
        
        if pdf_content:
            # Save test PDF
            with open('test_ai_analysis.pdf', 'wb') as f:
                f.write(pdf_content)
            print("✅ PDF generated successfully: test_ai_analysis.pdf")
            print(f"📄 File size: {len(pdf_content):,} bytes")
            return True
        else:
            print("❌ PDF generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1) 