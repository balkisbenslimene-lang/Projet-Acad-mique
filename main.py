import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# ======================================
# 🎨 Custom Theme & Configuration
# ======================================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Target the sidebar section directly */
    section[data-testid="stSidebar"] > div {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 1rem;
    }

    /* Style all checkbox containers in the sidebar */
    section[data-testid="stSidebar"] div[role="checkbox"] {
        display: block;
        width: 100%;
        margin: 5px 0;
    }

    /* Style the checkbox labels */
    section[data-testid="stSidebar"] div[role="checkbox"] label {
        display: block;
        width: 100%;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
        border: 2px solid #ff4b4b;
        background: white;
        color: #ff4b4b;
        font-weight: bold;
        transition: all 0.3s;
        margin: 0 !important;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="checkbox"] label:hover {
        background: #ff4b4b !important;
        color: white !important;
    }

    /* Hide actual checkboxes */
    section[data-testid="stSidebar"] div[role="checkbox"] input {
        display: none;
    }

    /* Checked state */
    section[data-testid="stSidebar"] div[role="checkbox"] input:checked + label {
        background: #ff4b4b !important;
        color: white !important;
    }

    /* Rest of your existing CSS... */
    









</style>
""", unsafe_allow_html=True)

# ======================================
# 🔐 Authentication System
# ======================================
users = {
    "Nadhir": "Admin",
    "Samar": "Admin",
    "Balkis": "Admin",
    "Ahmed": "Admin",
    "Sarrah": "Admin"
}

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.container():
        st.markdown("<div class='login-form'>", unsafe_allow_html=True)
        st.title("🔒 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if users.get(username) == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ======================================
# 📊 Main Dashboard
# ======================================

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Power BI", "🔮 Predict", "📈 Previsions"],
    horizontal=True,
    label_visibility="collapsed"
)
# Add logout button to sidebar (persistent across all pages)
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.rerun()
# --- Power BI Page (Now First) ---
if page == "📊 Power BI":
    st.title("📊 Live Business Intelligence")
    try:
        with open("powerbi_embed.html", "r") as f:
            html_code = f.read()
            st.components.v1.html(html_code, height=850, scrolling=True)
    except FileNotFoundError:
        st.warning("⚠️ Power BI dashboard not found. Please ensure 'powerbi_embed.html' is in your directory.")

# --- Predict Page ---
if page == "🔮 Predict":
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("🔮 Prediction Dashboard")

    # 1️⃣ Load Random Forest model
    @st.cache_resource
    def load_model(path="random_forest_model.pkl"):
        try:
            model = joblib.load(path)
            if hasattr(model, 'feature_names_in_'):
                st.success(f"Model loaded! Expected features: {list(model.feature_names_in_)}")
            else:
                st.success("Model loaded")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None

    model = load_model()

    # 2️⃣ Load and preprocess data
    @st.cache_data
    def load_data(path="df2_traité.xlsx"):
        try:
            df = pd.read_excel(path)
            
            expected_features = [
                'total_nb_recharge', 'total_rechage', 'total_u_data', 'total_rev_option',
                'total_u_out', 'total_u_in', 'usage_op3', 'nb_cont_out', 'nb_cont_in',
                'nb_cell_visite_out', 'nb_cell_visite_in', 'nbr_contrat', 'nbr_actif'
            ]
            
            missing = [col for col in expected_features if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            df = df[expected_features]
            
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = pd.factorize(df[col])[0]
                
            return df
            
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame()

    df = load_data()

    # 3️⃣ Display data preview
    if not df.empty:
        st.subheader("📌 Data Preview")
        st.dataframe(df.head())
        
        # 4️⃣ Prediction button
        if st.button("Predict", type="primary"):
            try:
                if hasattr(model, 'n_features_in_'):
                    if len(df.columns) != model.n_features_in_:
                        st.error(f"Feature mismatch. Model expects {model.n_features_in_} features, got {len(df.columns)}")
                
                preds = model.predict(df)
                results = df.copy()
                results["Prediction"] = preds
                
                st.success("✅ Prediction successful!")
                st.dataframe(results)
                
                # --------------------------------------------
                # NEW: Improved visualizations (2 columns)
                # --------------------------------------------
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. Prediction Distribution (Pie Chart)
                    st.subheader("📊 Prediction Distribution")
                    fig1, ax1 = plt.subplots(figsize=(5, 5))
                    results["Prediction"].value_counts().plot(
                        kind='pie',
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=['#66b3ff', '#99ff99'],  # Custom colors
                        wedgeprops={'edgecolor': 'white'},  # Add white edges
                        ax=ax1
                    )
                    ax1.set_ylabel("")
                    st.pyplot(fig1, use_container_width=True)  # Responsive sizing
                
                with col2:
                    # 2. NEW: Actual vs. Predicted (Bar Plot)
                    st.subheader("🔍 Prediction Breakdown")
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.countplot(
                        x="Prediction",
                        data=results,
                        palette="pastel",
                        edgecolor=".2",
                        ax=ax2
                    )
                    plt.xlabel("Predicted Class")
                    plt.ylabel("Count")
                    st.pyplot(fig2, use_container_width=True)
                
                # --------------------------------------------
                # NEW: Additional Insight (Single-row section)
                # --------------------------------------------
                st.subheader("📈 Prediction Confidence")
                
                # For Random Forest, show prediction probabilities (if available)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df)
                    proba_df = pd.DataFrame(proba, columns=[f"Class_{i}" for i in range(proba.shape[1])])
                    st.bar_chart(proba_df.mean())  # Avg confidence per class
                
                # --------------------------------------------
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning("No data loaded - check error messages above")

# --- Previsions Page ---
elif page == "📈 Previsions":
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    st.title("📈 Future Time Series Forecasting")
    
    # 1️⃣ Evaluation function
    def evaluate_model(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return round(mae, 4), round(rmse, 4), round(mape, 2)
    
    # 2️⃣ Create sequences function
    def create_sequences(data, n_lags):
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i - n_lags:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    # 3️⃣ Load data
    @st.cache_data
    def load_data():
        try:
            # Replace with your actual data loading logic
            # Example: ts_log = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
            # For demo, creating synthetic data
            np.random.seed(42)
            dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
            values = np.cumsum(np.random.randn(200)) + 100
            ts_log = pd.Series(values, index=dates)
            st.success("✅ Data loaded successfully")
            return ts_log
        except Exception as e:
            st.error(f"❌ Data loading failed: {str(e)}")
            return None
    
    ts_log = load_data()
    
    if ts_log is not None:
        # 4️⃣ Parameters
        st.sidebar.header("Forecast Parameters")
        n_lags = st.sidebar.slider("Number of Lags", 1, 20, 10)
        forecast_steps = st.sidebar.slider("Forecast Steps", 1, 365, 30)
        n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.1)
        
        # 5️⃣ Train on full dataset
        train_values = ts_log.values
        X_train, y_train = create_sequences(train_values, n_lags)
        
        # 6️⃣ Train model
        with st.spinner("Training forecasting model..."):
            model_xgb = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            model_xgb.fit(X_train, y_train)
            
            # 7️⃣ Generate future forecast
            last_seq = list(train_values[-n_lags:])
            future_preds = []
            
            for _ in range(forecast_steps):
                input_seq = np.array(last_seq[-n_lags:]).reshape(1, -1)
                next_pred = model_xgb.predict(input_seq)[0]
                future_preds.append(next_pred)
                last_seq.append(next_pred)
            
            # 8️⃣ Create future dates
            last_date = ts_log.index[-1]
            freq = ts_log.index.freq or pd.infer_freq(ts_log.index)
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq=freq
            )
            
            # 9️⃣ Visualize results
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=ts_log.index,
                y=ts_log.values,
                name="Historical Data",
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_preds,
                name="Future Forecast",
                line=dict(color='red', dash='dot'),
                mode='lines+markers'
            ))
            
            # Add confidence interval (example)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=np.array(future_preds) * 1.1,  # Upper bound (example)
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=np.array(future_preds) * 0.9,  # Lower bound (example)
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name="Confidence Interval"
            ))
            
            fig.update_layout(
                title=f"Future Forecast (Next {forecast_steps} Periods)",
                xaxis_title="Date",
                yaxis_title="Value",
                legend_title="Series",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 🔟 Show forecast table
            st.subheader("Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': future_preds,
                'Upper Bound': np.array(future_preds) * 1.1,
                'Lower Bound': np.array(future_preds) * 0.9
            }).set_index('Date')
            
            st.dataframe(forecast_df.style.format("{:.2f}"))
            
            # Download button
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Forecast",
                data=csv,
                file_name=f"future_forecast_{pd.Timestamp.now().date()}.csv",
                mime='text/csv'
            )
    
 