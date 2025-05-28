#!/usr/bin/env python3
"""
Dashboard Launcher for Stock Analysis Pipeline
Choose between basic or advanced dashboard
"""

import subprocess
import sys
import os

def main():
    """Launches the Streamlit stock analysis dashboard."""
    dashboard_script = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    print("🚀 Launching Stock Analysis Dashboard...")
    print(f"Features: Dynamic stock input, technical analysis, financial statements, EV stock focus")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_script], check=True)
    except FileNotFoundError:
        print("Error: Streamlit not found. Please ensure Streamlit is installed and in your PATH.")
        print("You can install it by running: pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit dashboard: {e}")
    except KeyboardInterrupt:
        print("👋 Dashboard launcher exited.")

if __name__ == "__main__":
    main() 