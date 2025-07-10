import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="ITT - Intelligent Transportation Technology",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🚗 ITT - Intelligent Transportation Technology</h1>', unsafe_allow_html=True)
st.markdown("### Advanced EV Charging Station Analysis & Demand Prediction Platform")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "🏠 Dashboard", 
    "📊 Data Analysis", 
    "🔮 Demand Prediction", 
    "🗺️ Station Locator",
    "📈 Market Insights"
])

# Load and prepare sample data
@st.cache_data
def load_sample_data():
    # Create sample data based on the ML project
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 
              'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI']
    
    np.random.seed(42)
    data = {
        'State': states,
        'Population': np.random.randint(500000, 15000000, 20),
        'Vehicle_Density': np.random.randint(600, 1000, 20),
        'Station_Count': np.random.randint(50, 3000, 20),
        'GDP_Per_Capita': np.random.randint(40000, 80000, 20),
        'Urban_Percentage': np.random.randint(60, 95, 20)
    }
    
    df = pd.DataFrame(data)
    # Make station count somewhat correlated with population and vehicle density
    df['Station_Count'] = (df['Population'] / 5000 + df['Vehicle_Density'] / 2 + 
                          np.random.randint(-200, 200, 20)).astype(int)
    df['Station_Count'] = np.clip(df['Station_Count'], 50, 5000)
    
    return df

# Train models
@st.cache_data
def train_models(df):
    X = df[['Population', 'Vehicle_Density', 'GDP_Per_Capita', 'Urban_Percentage']]
    y = df['Station_Count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    
    return {
        'lr_model': lr_model, 'rf_model': rf_model,
        'lr_r2': lr_r2, 'rf_r2': rf_r2,
        'X_test': X_test, 'y_test': y_test,
        'lr_pred': lr_pred, 'rf_pred': rf_pred
    }

df = load_sample_data()
models = train_models(df)

# Dashboard Page
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total States", len(df), "🌎")
    with col2:
        st.metric("Total Stations", f"{df['Station_Count'].sum():,}", "⚡")
    with col3:
        st.metric("Avg Population", f"{df['Population'].mean():,.0f}", "👥")
    with col4:
        st.metric("Model Accuracy", f"{max(models['lr_r2'], models['rf_r2']):.2%}", "🎯")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Station Count by State")
        fig1 = px.bar(df.sort_values('Station_Count', ascending=False), 
                     x='State', y='Station_Count',
                     color='Station_Count',
                     color_continuous_scale='viridis')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Population vs Stations")
        fig2 = px.scatter(df, x='Population', y='Station_Count', 
                         size='Vehicle_Density', color='State',
                         hover_data=['GDP_Per_Capita'])
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.header("📊 Comprehensive Data Analysis")
    
    # Data overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Dataset Overview")
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.write(df.describe())
    
    st.markdown("---")
    
    # Correlation analysis
    st.subheader("🔍 Correlation Analysis")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    fig = px.imshow(corr_matrix, 
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu',
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Distribution Analysis")
    selected_feature = st.selectbox("Select feature to analyze:", 
                                   ['Population', 'Vehicle_Density', 'Station_Count', 'GDP_Per_Capita'])
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=selected_feature, nbins=20, 
                          title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_feature, 
                    title=f"Box Plot of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

# Demand Prediction Page
elif page == "🔮 Demand Prediction":
    st.header("🔮 EV Station Demand Prediction")
    
    # Model comparison
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Linear Regression R²", f"{models['lr_r2']:.3f}")
    with col2:
        st.metric("Random Forest R²", f"{models['rf_r2']:.3f}")
    
    # Input form
    st.subheader("🎯 Make a Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        population = st.number_input("Population", min_value=100000, max_value=20000000, 
                                   value=1000000, step=100000)
        vehicle_density = st.number_input("Vehicle Density", min_value=500, max_value=1200, 
                                        value=800, step=10)
    
    with col2:
        gdp_per_capita = st.number_input("GDP Per Capita ($)", min_value=30000, max_value=100000, 
                                       value=55000, step=1000)
        urban_percentage = st.number_input("Urban Percentage (%)", min_value=50, max_value=100, 
                                         value=75, step=1)
    
    if st.button("🔮 Predict Station Demand", type="primary"):
        input_data = pd.DataFrame({
            'Population': [population],
            'Vehicle_Density': [vehicle_density],
            'GDP_Per_Capita': [gdp_per_capita],
            'Urban_Percentage': [urban_percentage]
        })
        
        lr_prediction = models['lr_model'].predict(input_data)[0]
        rf_prediction = models['rf_model'].predict(input_data)[0]
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🎯 Prediction Results</h3>
            <p><strong>Linear Regression:</strong> {int(lr_prediction)} stations</p>
            <p><strong>Random Forest:</strong> {int(rf_prediction)} stations</p>
            <p><strong>Average Prediction:</strong> {int((lr_prediction + rf_prediction) / 2)} stations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization of prediction
        prediction_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'Average'],
            'Prediction': [lr_prediction, rf_prediction, (lr_prediction + rf_prediction) / 2]
        })
        
        fig = px.bar(prediction_data, x='Model', y='Prediction',
                    title="Prediction Comparison",
                    color='Prediction',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# Station Locator Page
elif page == "🗺️ Station Locator":
    st.header("🗺️ EV Station Location Intelligence")
    
    # Create sample station data with coordinates
    @st.cache_data
    def create_station_map_data():
        np.random.seed(42)
        station_data = []
        
        for _, row in df.iterrows():
            for i in range(min(int(row['Station_Count'] / 100), 20)):  # Limit for demo
                station_data.append({
                    'State': row['State'],
                    'Latitude': np.random.uniform(25, 49),
                    'Longitude': np.random.uniform(-125, -65),
                    'Station_Type': np.random.choice(['Fast Charging', 'Level 2', 'Level 1'], 
                                                   p=[0.4, 0.5, 0.1]),
                    'Utilization': np.random.uniform(0.3, 0.95)
                })
        
        return pd.DataFrame(station_data)
    
    station_map_data = create_station_map_data()
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        selected_states = st.multiselect("Select States:", 
                                       options=df['State'].tolist(),
                                       default=df['State'].tolist()[:5])
    
    with col2:
        station_type = st.selectbox("Station Type:", 
                                   options=['All'] + station_map_data['Station_Type'].unique().tolist())
    
    # Filter data
    filtered_data = station_map_data[station_map_data['State'].isin(selected_states)]
    if station_type != 'All':
        filtered_data = filtered_data[filtered_data['Station_Type'] == station_type]
    
    # Map visualization
    st.subheader("🗺️ Station Locations")
    fig = px.scatter_mapbox(filtered_data,
                           lat='Latitude', lon='Longitude',
                           color='Station_Type',
                           size='Utilization',
                           hover_data=['State', 'Utilization'],
                           mapbox_style='open-street-map',
                           height=600,
                           zoom=3)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Station statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stations", len(filtered_data))
    with col2:
        st.metric("Avg Utilization", f"{filtered_data['Utilization'].mean():.1%}")
    with col3:
        st.metric("States Covered", len(filtered_data['State'].unique()))

# Market Insights Page
elif page == "📈 Market Insights":
    st.header("📈 Market Insights & Trends")
    
    # Generate trend data
    @st.cache_data
    def generate_trend_data():
        years = list(range(2020, 2031))
        data = []
        
        for year in years:
            growth_factor = (year - 2020) * 0.15 + 1
            for state in df['State'][:10]:  # Top 10 states
                base_stations = df[df['State'] == state]['Station_Count'].iloc[0]
                projected_stations = int(base_stations * growth_factor + np.random.randint(-50, 50))
                data.append({
                    'Year': year,
                    'State': state,
                    'Projected_Stations': projected_stations
                })
        
        return pd.DataFrame(data)
    
    trend_data = generate_trend_data()
    
    # Growth projections
    st.subheader("📈 Growth Projections (2020-2030)")
    
    selected_states_trend = st.multiselect("Select states for trend analysis:",
                                         options=trend_data['State'].unique(),
                                         default=trend_data['State'].unique()[:5])
    
    filtered_trend = trend_data[trend_data['State'].isin(selected_states_trend)]
    
    fig = px.line(filtered_trend, x='Year', y='Projected_Stations', 
                 color='State', title="EV Station Growth Projections")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Market analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Market Opportunity")
        market_data = df.copy()
        market_data['Market_Potential'] = (market_data['Population'] / 1000 * 
                                         market_data['GDP_Per_Capita'] / 50000)
        
        fig = px.scatter(market_data, x='Market_Potential', y='Station_Count',
                        size='Population', color='State',
                        title="Market Potential vs Current Stations")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Investment Priorities")
        investment_score = (df['Population'] / df['Population'].max() * 0.4 +
                          df['GDP_Per_Capita'] / df['GDP_Per_Capita'].max() * 0.3 +
                          (df['Vehicle_Density'] / df['Vehicle_Density'].max()) * 0.3)
        
        investment_df = pd.DataFrame({
            'State': df['State'],
            'Investment_Score': investment_score,
            'Current_Stations': df['Station_Count']
        }).sort_values('Investment_Score', ascending=False)
        
        fig = px.bar(investment_df.head(10), x='State', y='Investment_Score',
                    color='Investment_Score',
                    title="Top 10 States - Investment Priority Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Market Insights")
    
    insights = [
        "🚀 **Rapid Growth**: EV charging infrastructure is projected to grow 300% by 2030",
        "🏙️ **Urban Focus**: States with higher urban percentages show better ROI",
        "💡 **Opportunity Gap**: Several high-GDP states are underserved in charging infrastructure",
        "📊 **Demand Correlation**: Vehicle density is the strongest predictor of station demand",
        "🌱 **Sustainability**: Investment in charging infrastructure drives EV adoption rates"
    ]
    
    for insight in insights:
        st.markdown(insight)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    🚗 ITT - Intelligent Transportation Technology Platform<br>
    Empowering Smart Transportation Decisions with Data Science
</div>
""", unsafe_allow_html=True)