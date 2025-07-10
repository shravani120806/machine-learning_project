# 🚗 ITT - Intelligent Transportation Technology

A comprehensive web application for analyzing EV charging station data and predicting future demand using machine learning.

## 🌟 Features

### 🏠 Dashboard
- **Key Metrics**: Overview of total states, stations, population, and model accuracy
- **Interactive Charts**: Station count by state and population vs stations analysis
- **Real-time Statistics**: Live updates of key performance indicators

### 📊 Data Analysis
- **Dataset Overview**: Complete data exploration with interactive tables
- **Statistical Analysis**: Comprehensive statistical summaries
- **Correlation Analysis**: Heatmap visualization of feature relationships
- **Distribution Analysis**: Histograms and box plots for data distribution insights

### 🔮 Demand Prediction
- **AI-Powered Forecasting**: Uses Linear Regression and Random Forest models
- **Multi-Factor Prediction**: Considers population, vehicle density, GDP, and urban percentage
- **Model Comparison**: Side-by-side comparison of different ML algorithms
- **Interactive Input**: Real-time predictions with custom parameters

### 🗺️ Station Locator
- **Interactive Maps**: Plotly-powered geographic visualization
- **Station Filtering**: Filter by state and station type
- **Utilization Metrics**: Visual representation of station usage
- **Location Intelligence**: Smart location analysis for optimal placement

### 📈 Market Insights
- **Growth Projections**: 10-year forecasting (2020-2030)
- **Investment Analysis**: Priority scoring for optimal investment decisions
- **Market Opportunities**: Gap analysis and potential identification
- **Trend Visualization**: Interactive trend analysis across multiple states

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup

1. **Clone or download the files**
   ```bash
   # Ensure you have these files:
   # - itt_app.py
   # - requirements.txt
   # - run_itt_app.py
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**
   ```bash
   # Option 1: Use the launcher script
   python run_itt_app.py
   
   # Option 2: Direct Streamlit command
   streamlit run itt_app.py
   ```

## 🚀 Usage

### Running the App

1. **Start the application** using one of the methods above
2. **Open your browser** and navigate to `http://localhost:8501`
3. **Explore the features** using the sidebar navigation

### Navigation Guide

#### 🏠 Dashboard
- View overall system metrics
- Analyze station distribution across states
- Explore population vs station relationships

#### 📊 Data Analysis
- Examine the complete dataset
- Review statistical summaries
- Analyze feature correlations
- Study data distributions

#### 🔮 Demand Prediction
- Input area characteristics:
  - Population (100K - 20M)
  - Vehicle Density (500 - 1200)
  - GDP Per Capita ($30K - $100K)
  - Urban Percentage (50% - 100%)
- Compare predictions from multiple models
- Visualize prediction results

#### 🗺️ Station Locator
- Select states for analysis
- Filter by station types:
  - Fast Charging
  - Level 2
  - Level 1
- View utilization rates
- Analyze geographic distribution

#### 📈 Market Insights
- Review growth projections
- Identify investment opportunities
- Analyze market potential
- Study state-by-state trends

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Visualization**: Plotly for interactive charts and maps
- **Machine Learning**: Scikit-learn for predictive modeling
- **Data Processing**: Pandas and NumPy for data manipulation

### Models Used
1. **Linear Regression**: Fast, interpretable baseline model
2. **Random Forest**: Advanced ensemble method for improved accuracy

### Data Features
- **Population**: Total area population
- **Vehicle Density**: Vehicles per square mile
- **GDP Per Capita**: Economic indicator
- **Urban Percentage**: Urbanization level
- **Station Count**: Target variable for prediction

### Performance Metrics
- **R² Score**: Model accuracy measurement
- **Mean Squared Error (MSE)**: Prediction error analysis
- **Mean Absolute Error (MAE)**: Average prediction deviation

## 📊 Sample Data

The app uses synthetic data based on real-world patterns for demonstration:
- **20 US States**: Representative sample across different regions
- **Multiple Features**: Economic, demographic, and transportation data
- **Station Types**: Fast charging, Level 2, and Level 1 stations
- **Realistic Correlations**: Data reflects real-world relationships

## 🎯 Use Cases

### For Transportation Planners
- Predict future charging infrastructure needs
- Identify underserved areas
- Optimize station placement strategies

### For Investors
- Evaluate market opportunities
- Assess investment priorities
- Analyze growth potential

### For Policymakers
- Plan infrastructure development
- Allocate resources efficiently
- Support EV adoption strategies

### For Researchers
- Analyze transportation trends
- Study urbanization impacts
- Model infrastructure demand

## 🔍 Key Insights

The ITT app reveals several important trends:

- **Urban Correlation**: Higher urbanization strongly correlates with station demand
- **Economic Factors**: GDP per capita influences infrastructure investment
- **Growth Projections**: 300% growth expected in EV infrastructure by 2030
- **Regional Variations**: Significant differences in adoption rates across states

## 🛠️ Customization

### Adding Your Own Data
1. Replace the `load_sample_data()` function with your data source
2. Ensure your data has the required columns:
   - State, Population, Vehicle_Density, Station_Count, GDP_Per_Capita, Urban_Percentage
3. Adjust feature selection in the `train_models()` function if needed

### Modifying Models
1. Add new models in the `train_models()` function
2. Update the prediction page to include new models
3. Adjust visualization components accordingly

## 📝 License

This project is created for educational and demonstration purposes. Feel free to modify and adapt for your specific needs.

## 🤝 Support

If you encounter any issues or have questions:

1. Check that all dependencies are properly installed
2. Ensure you're running Python 3.7 or higher
3. Verify that all required files are present
4. Try running `python run_itt_app.py` for diagnostic information

## 🌍 Future Enhancements

Potential improvements for future versions:
- Real-time data integration
- Advanced ML models (XGBoost, Neural Networks)
- Mobile-responsive design
- User authentication and data persistence
- API integration with real EV station databases
- Advanced geospatial analysis
- Custom reporting and export features

---

**Built with ❤️ using Streamlit, Plotly, and Scikit-learn**

*Empowering Smart Transportation Decisions with Data Science*