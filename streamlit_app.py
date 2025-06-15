import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.predict import LoanPredictor, check_model_files
import os

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .approved {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .rejected {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Approval Predictor</h1>', unsafe_allow_html=True)
    
    # Check if model files exist
    model_exists, missing_files = check_model_files()
    
    if not model_exists:
        st.error("⚠️ Model files not found!")
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"- {file}")
        st.info("📝 Please run the training script first: `python src/train_model.py`")
        return
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        predictor = LoanPredictor()
        predictor.load_model()
        return predictor
    
    predictor = load_predictor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Info", "About"])
    
    if page == "Prediction":
        prediction_page(predictor)
    elif page == "Model Info":
        model_info_page(predictor)
    else:
        about_page()

def prediction_page(predictor):
    st.header("🔮 Make a Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        st.subheader("Financial Information")
        income_annum = st.number_input("Annual Income (₹)", min_value=0, value=100000, step=10000)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=5000000, step=50000)
        loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=20)
    
    with col2:
        st.subheader("Credit & Assets")
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=350)
        
        st.subheader("Asset Values (₹)")
        residential_assets = st.number_input("Residential Assets Value", min_value=0, value=100000, step=100000)
        commercial_assets = st.number_input("Commercial Assets Value", min_value=0, value=100000, step=50000)
        luxury_assets = st.number_input("Luxury Assets Value", min_value=0, value=100000, step=50000)
        bank_assets = st.number_input("Bank Asset Value", min_value=0, value=100000, step=50000)
    
    # Prediction button
    if st.button("🔍 Predict Loan Approval", type="primary"):
        # Prepare input data
        input_data = {
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets,
            'commercial_assets_value': commercial_assets,
            'luxury_assets_value': luxury_assets,
            'bank_asset_value': bank_assets
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            prediction, confidence = predictor.predict(input_data)
        
        if prediction is not None:
            # Display result
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            if prediction == "Approved" or prediction == 1:
                st.markdown(f"""
                <div class="prediction-box approved">
                    <h3>✅ Loan Approved!</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            elif prediction == "Rejected" or prediction == 0:
                st.markdown(f"""
                <div class="prediction-box rejected">
                    <h3>❌ Loan Rejected</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Unexpected prediction value: {prediction}")
            
            # Display input summary
            st.subheader("📋 Input Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.write(f"**Income:** ₹{income_annum:,}")
                st.write(f"**Loan Amount:** ₹{loan_amount:,}")
                st.write(f"**CIBIL Score:** {cibil_score}")
                st.write(f"**Dependents:** {no_of_dependents}")
            
            with summary_col2:
                st.write(f"**Education:** {education}")
                st.write(f"**Self Employed:** {self_employed}")
                st.write(f"**Loan Term:** {loan_term} years")
                st.write(f"**Total Assets:** ₹{(residential_assets + commercial_assets + luxury_assets + bank_assets):,}")
            
            # Create visualization
            create_prediction_chart(input_data, prediction, confidence)
        else:
            st.error("❌ Error making prediction. Please check your inputs.")

def create_prediction_chart(input_data, prediction, confidence):
    """Create visualization for the prediction"""
    st.subheader("📈 Financial Overview")
    
    # Create asset distribution pie chart
    assets = {
        'Residential': input_data['residential_assets_value'],
        'Commercial': input_data['commercial_assets_value'],
        'Luxury': input_data['luxury_assets_value'],
        'Bank': input_data['bank_asset_value']
    }
    
    # Filter out zero values
    assets = {k: v for k, v in assets.items() if v > 0}
    
    if assets:
        fig_pie = px.pie(values=list(assets.values()), names=list(assets.keys()), 
                        title="Asset Distribution")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Create income vs loan amount comparison
    fig_bar = go.Figure(data=[
        go.Bar(name='Amount', x=['Annual Income', 'Loan Amount'], 
               y=[input_data['income_annum'], input_data['loan_amount']],
               marker_color=['#2E8B57', '#FF6347'])
    ])
    fig_bar.update_layout(title="Income vs Loan Amount Comparison",
                         yaxis_title="Amount (₹)")
    st.plotly_chart(fig_bar, use_container_width=True)

def model_info_page(predictor):
    st.header("📊 Model Information")
    
    # Display feature importance
    importance_df = predictor.get_feature_importance()
    
    if importance_df is not None:
        st.subheader("🎯 Feature Importance")
        
        # Create horizontal bar chart
        fig = px.bar(importance_df, x='importance', y='feature', 
                    orientation='h', title="Feature Importance in Loan Approval")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table
        st.subheader("📋 Importance Values")
        st.dataframe(importance_df, use_container_width=True)
    
    # Model statistics
    st.subheader("🔧 Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Algorithm:** Random Forest Classifier")
        st.info("**Features:** 11 input features")
        st.info("**Scaling:** StandardScaler applied")
    
    with col2:
        st.info("**Training Split:** 80% train, 20% test")
        st.info("**Validation:** Stratified sampling")
        st.info("**Output:** Binary classification (Approved/Rejected)")

def about_page():
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🏦 Loan Approval Predictor
    
    This application uses machine learning to predict loan approval decisions based on various financial and personal factors.
    
    ### 🎯 Features
    - **Real-time Predictions**: Get instant loan approval predictions
    - **Interactive Interface**: User-friendly input forms
    - **Visual Analytics**: Charts and graphs for better understanding
    - **Model Transparency**: View feature importance and model details
    
    ### 📊 Input Features
    The model considers the following factors:
    
    1. **Personal Information**
       - Number of dependents
       - Education level
       - Employment status
    
    2. **Financial Information**
       - Annual income
       - Requested loan amount
       - Loan term
    
    3. **Credit & Assets**
       - CIBIL score
       - Residential assets value
       - Commercial assets value
       - Luxury assets value
       - Bank asset value
    
    ### 🤖 How It Works
    1. Input your financial and personal information
    2. The model processes your data using advanced algorithms
    3. Get an instant prediction with confidence score
    4. View detailed analysis and recommendations
    
    ### 🔒 Privacy & Security
    - No data is stored permanently
    - All predictions are made locally
    - Your information is not shared with third parties
    
    ---
    
    **Note:** This is a predictive model for educational purposes. Actual loan decisions may vary and depend on additional factors not included in this model.
    """)

if __name__ == "__main__":
    main()