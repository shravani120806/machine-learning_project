#!/usr/bin/env python3
"""
ITT App Launcher
================

This script launches the Intelligent Transportation Technology (ITT) application.

Usage:
    python run_itt_app.py

Or directly:
    streamlit run itt_app.py

Features:
- 🏠 Dashboard: Overview of EV station data and key metrics
- 📊 Data Analysis: Comprehensive data exploration and visualization
- 🔮 Demand Prediction: AI-powered station demand forecasting
- 🗺️ Station Locator: Interactive map with station locations
- 📈 Market Insights: Growth trends and investment opportunities

Requirements:
- Python 3.7+
- All packages listed in requirements.txt
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'scikit-learn', 'joblib', 
        'numpy', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def launch_app():
    """Launch the ITT Streamlit application."""
    if not os.path.exists('itt_app.py'):
        print("❌ itt_app.py not found in current directory!")
        return False
    
    print("🚀 Launching ITT - Intelligent Transportation Technology App...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⭐ Features available:")
    print("   🏠 Dashboard - Key metrics and overview")
    print("   📊 Data Analysis - Explore the data")
    print("   🔮 Demand Prediction - AI forecasting")
    print("   🗺️ Station Locator - Interactive maps")
    print("   📈 Market Insights - Trends and opportunities")
    print("\n⌨️  Press Ctrl+C to stop the app")
    print("="*50)
    
    try:
        subprocess.run(['streamlit', 'run', 'itt_app.py'], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit app")
        return False
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return True
    
    return True

if __name__ == "__main__":
    print("🚗 ITT - Intelligent Transportation Technology")
    print("=" * 50)
    
    if check_requirements():
        print()
        launch_app()
    else:
        print("\n❌ Cannot launch app due to missing dependencies")
        sys.exit(1)