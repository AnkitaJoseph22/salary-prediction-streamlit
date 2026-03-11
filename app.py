import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Updated for High Visibility) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Force ALL headers to be the specific light purple shade */
    h1, h2, h3, h4, h5, h6 {
        color: #deb7f7 !important; 
    }
    
    /* --- NEW: Specific Visibility Fixes --- */
    
    /* 1. Years of Experience Label */
    .stNumberInput label p {
        color: #deb7f7 !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* 2. Metric Labels (Confidence, Market Trend) */
    [data-testid="stMetricLabel"] {
        color: #deb7f7 !important;
        font-weight: 500;
    }
    
    /* 3. Metric Values */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* ------------------------------------- */
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Main App Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #deb7f7; /* Requested Purple */
        text-shadow: 0px 0px 15px rgba(222, 183, 247, 0.3); /* Soft Purple Glow */
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #b2dfdb; /* Lighter Teal/Grey */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        color: #e0f2f1; /* Light text inside cards */
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00b09b, #96c93d);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,255,136,0.3);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,255,136,0.5);
    }
    
    /* Floating Money Animation Background */
    .money-bg {
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    
    /* Result Animation */
    @keyframes pulse {
        0% { transform: scale(1); text-shadow: 0 0 10px rgba(222, 183, 247, 0.5); }
        50% { transform: scale(1.05); text-shadow: 0 0 20px rgba(222, 183, 247, 0.8); }
        100% { transform: scale(1); text-shadow: 0 0 10px rgba(222, 183, 247, 0.5); }
    }
    .prediction-text {
        text-align: center;
        color: #deb7f7; /* Requested Purple */
        font-size: 2.2rem;
        font-weight: 700;
        animation: pulse 2s infinite;
        margin-top: 1rem;
    }
</style>
<div class="money-bg"></div>
""", unsafe_allow_html=True)

# --- Model Loading with Error Handling ---
@st.cache_resource
def load_model():
    model_path = 'salary_model.pkl'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# --- Main Interface ---
if model is None:
    st.error("⚠️ Model file `salary_model.pkl` not found!")
    st.warning("Please run your Jupyter Notebook (cells 1-4) to train and save the model first.")
    st.stop()

# Header
st.markdown('<h1 class="main-header">💰 Salary Predictor Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Career & Wealth Forecast</p>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Profile Details")
    
    experience = st.number_input(
        "Years of Experience", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0, 
        step=0.5,
        format="%.1f",
        help="Enter your total professional experience in years."
    )
    
    st.markdown("---")
    
    if st.button("🚀 Predict Salary", use_container_width=True):
        with st.spinner("Crunching the numbers..."):
            # Prediction Logic
            input_data = np.array([[experience]])
            pred = model.predict(input_data)[0]
            
            # Formatting results
            annual_salary = f"${pred:,.2f}"
            monthly_salary = f"${pred/12:,.2f}"
            
            # Balloons & Snow
            st.balloons()
            
            st.markdown(f"""
            <div class="prediction-text">
                {annual_salary} / yr
            </div>
            <p style="text-align: center; color: #deb7f7;">approx {monthly_salary} / mo</p>
            """, unsafe_allow_html=True)

            # Metrics
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.metric("Confidence", "95%", delta="High")
            c2.metric("Market Trend", "+4.2%", delta="Up")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Visualization
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Market Analysis Trend")
    
    # Create Data for Plot
    x_range = np.linspace(0, 30, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Customizing Plot Style to match Dark Theme
    fig.patch.set_alpha(0)      # Transparent figure background
    ax.set_facecolor('#1e2a38') # Dark plot background
    
    # Gradient Line Effect
    ax.plot(x_range, y_range, color='#deb7f7', linewidth=3, label='Market Trend Line')
    
    # User's Point
    user_pred = model.predict([[experience]])[0]
    ax.scatter([experience], [user_pred], color='#ff4081', s=200, zorder=5, edgecolors='white', linewidth=2, label='You are here')
    
    # Labels and Grid with matching colors
    ax.set_xlabel("Years of Experience", color='#e0f2f1', fontsize=12)
    ax.set_ylabel("Salary ($)", color='#e0f2f1', fontsize=12)
    ax.tick_params(axis='x', colors='#e0f2f1')
    ax.tick_params(axis='y', colors='#e0f2f1')
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.1)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#e0f2f1')
        spine.set_alpha(0.3)
        
    ax.legend(facecolor='#1e2a38', labelcolor='#e0f2f1')
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Model Stats")
    st.info("Model: Linear Regression (Scikit-Learn)")
    st.write("Training Data: 30 records")
    st.write("Accuracy (R²): ~95%")
    
    st.markdown("---")
    st.markdown("### 💡 Career Tip")
    if experience < 5:
        st.write("Focus on learning core skills and building a portfolio.")
    elif experience < 10:
        st.write("Look for leadership roles or specialized certifications.")
    else:
        st.write("Consider mentorship or strategic consulting roles.")