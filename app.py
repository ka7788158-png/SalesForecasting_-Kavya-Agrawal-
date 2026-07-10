import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Page Config & Custom CSS ---
st.set_page_config(page_title="Demand Intelligence System", page_icon="📈", layout="wide")

# Custom HTML/CSS injection for a professional web app feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #3498db; }
    .metric-label { font-size: 1rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading & Caching ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    
    # Fix Sales data type if needed
    if df['Sales'].dtype == 'object':
        df['Sales'] = df['Sales'].astype(str).str.replace(',', '').astype(float)
        
    return df

df = load_and_prep_data()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Demand Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", 
                        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# ==========================================
# PAGE 1: Sales Overview Dashboard
# ==========================================
if page == "Sales Overview":
    st.markdown("<h1>Sales Overview Dashboard</h1>", unsafe_allow_html=True)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
    with col2:
        cat_filter = st.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
        
    filtered_df = df[(df['Region'].isin(region_filter)) & (df['Category'].isin(cat_filter))]
    
    # Top Metrics UI Cards
    total_sales = filtered_df['Sales'].sum()
    total_orders = filtered_df.shape[0]
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
            <div class="card">
                <div class="metric-label">Total Revenue</div>
                <div class="metric-value">${total_sales:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="card">
                <div class="metric-label">Total Orders</div>
                <div class="metric-value">{total_orders:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    # Charts
    c1, c2 = st.columns(2)
    with c1:
        yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig_bar = px.bar(yearly_sales, x='Year', y='Sales', title="Total Sales by Year", 
                         color_discrete_sequence=['#3498db'], text_auto='.2s')
        fig_bar.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        monthly_sales = filtered_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        fig_line = px.line(monthly_sales, x='Order Date', y='Sales', title="Monthly Sales Trend",
                           markers=True, line_shape="spline", color_discrete_sequence=['#e74c3c'])
        fig_line.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# PAGE 2: Forecast Explorer (XGBoost)
# ==========================================
elif page == "Forecast Explorer":
    st.markdown("<h1>Forecast Explorer (XGBoost)</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        segment_type = st.selectbox("Select Segment Type", ["Category", "Region"])
        segment_value = st.selectbox(f"Select {segment_type}", df[segment_type].unique())
        horizon = st.slider("Forecast Horizon (Months)", 1, 3, 3)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # Dynamic XGBoost Training
        seg_df = df[df[segment_type] == segment_value]
        seg_time = seg_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
        
        # Features
        seg_time['Lag_1'] = seg_time['Sales'].shift(1)
        seg_time['Lag_2'] = seg_time['Sales'].shift(2)
        seg_time['Lag_3'] = seg_time['Sales'].shift(3)
        seg_time['Rolling_Mean'] = seg_time['Sales'].rolling(3).mean()
        seg_time['Month'] = seg_time['Order Date'].dt.month
        seg_time['Quarter'] = seg_time['Order Date'].dt.quarter
        
        seg_clean = seg_time.dropna().reset_index(drop=True)
        features = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean', 'Month', 'Quarter']
        X = seg_clean[features]
        y = seg_clean['Sales']
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        
        # Metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Forecast
        future_dates = [seg_time['Order Date'].max() + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
        forecasts = []
        last_seq = seg_time['Sales'].values.tolist()
        
        for date in future_dates:
            curr_X = pd.DataFrame([[last_seq[-1], last_seq[-2], last_seq[-3], np.mean(last_seq[-3:]), date.month, date.quarter]], columns=features)
            pred = model.predict(curr_X)[0]
            forecasts.append(pred)
            last_seq.append(pred)
            
        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seg_time['Order Date'], y=seg_time['Sales'], mode='lines+markers', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=forecasts, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"Sales Forecast for {segment_value}", plot_bgcolor="white", hovermode="x unified")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
            <div style="display:flex; justify-content:space-around;" class="card">
                <div><b>Model MAE:</b> {mae:.2f}</div>
                <div><b>Model RMSE:</b> {rmse:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: Anomaly Report
# ==========================================
elif page == "Anomaly Report":
    st.markdown("<h1>Anomaly Report (Isolation Forest)</h1>", unsafe_allow_html=True)
    
    weekly_sales = df.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_sales['Anomaly'] = iso.fit_predict(weekly_sales[['Sales']])
    weekly_sales['Is_Anomaly'] = weekly_sales['Anomaly'] == -1
    
    anomalies = weekly_sales[weekly_sales['Is_Anomaly']]
    
    fig = px.line(weekly_sales, x='Order Date', y='Sales', title="Detected Sales Anomalies")
    fig.add_trace(go.Scatter(x=anomalies['Order Date'], y=anomalies['Sales'], mode='markers', 
                             marker=dict(color='red', size=10, symbol='x'), name='Anomaly'))
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🚨 Detected Anomaly Dates")
    display_anomalies = anomalies[['Order Date', 'Sales']].copy()
    display_anomalies['Order Date'] = display_anomalies['Order Date'].dt.strftime('%Y-%m-%d')
    display_anomalies['Sales'] = display_anomalies['Sales'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_anomalies, use_container_width=True)

# ==========================================
# PAGE 4: Product Demand Segments
# ==========================================
elif page == "Product Demand Segments":
    st.markdown("<h1>Product Demand Segmentation</h1>", unsafe_allow_html=True)
    
    # Feature Engineering
    stats = df.groupby('Sub-Category')['Sales'].agg(['sum', 'mean']).rename(columns={'sum': 'Volume', 'mean': 'Avg_Order'})
    volatility = df.set_index('Order Date').groupby('Sub-Category').resample('ME')['Sales'].sum().reset_index().groupby('Sub-Category')['Sales'].std().rename('Volatility')
    
    seg_df = pd.concat([stats, volatility], axis=1).fillna(0)
    scaled = StandardScaler().fit_transform(seg_df)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    seg_df['Cluster'] = kmeans.fit_predict(scaled)
    
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled)
    seg_df['PCA1'] = pca_res[:, 0]
    seg_df['PCA2'] = pca_res[:, 1]
    seg_df = seg_df.reset_index()
    
    # Strategy Map for labels
    strategy_map = {
        0: "Rising Stars",
        1: "Declining Demand",
        2: "High Volatility (JIT)",
        3: "Stable Demand (Core)"
    }
    seg_df['Strategy Label'] = seg_df['Cluster'].map(strategy_map)
    
    fig = px.scatter(seg_df, x='PCA1', y='PCA2', color='Strategy Label', text='Sub-Category',
                     title="2D Cluster Projection of Product Demand", size_max=60)
    fig.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(plot_bgcolor="white", height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Sub-Category Assignments")
    st.dataframe(seg_df.sort_values('Cluster')[['Sub-Category', 'Strategy Label', 'Volume', 'Avg_Order', 'Volatility']], use_container_width=True)
