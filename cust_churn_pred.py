import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Loading model and preprocessor with caching
@st.cache_resource
def load_assets():
    model = joblib.load('xgboost_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_assets()

# App title and description
st.title('Telco Customer Churn Predictor')
st.markdown("""
Predict likelihood of customer churn based on service details
""")

# Input widgets organized in columns
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Customer Profile")
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        
    with col2:
        st.subheader("Service Details")
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        
    with col3:
        st.subheader("Account Details")
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly_charges = st.number_input('Monthly Charges ($)', 0.0, 200.0, 70.0)
        total_charges = st.number_input('Total Charges ($)', 0.0, 10000.0, 2000.0)
    
    submitted = st.form_submit_button("Predict Churn Probability")

# When form is submitted
if submitted:
    # Map inputs to DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Preprocess and prediction
    with st.spinner('Analyzing...'):
        processed_data = preprocessor.transform(input_data)
        churn_prob = model.predict_proba(processed_data)[0][1]
    
    # Display results
    st.subheader("Prediction Result")
    st.metric(label="Churn Probability", value=f"{churn_prob:.1%}")
    
    # Visual indicator
    if churn_prob > 0.75:
        st.error("High churn risk! Customer needs immediate retention actions")
    elif churn_prob > 0.5:
        st.warning("Moderate churn risk. Recommend proactive engagement")
    else:
        st.success("Low churn risk. Customer appears stable")
    
    # Feature importance visualization
    st.subheader("Top Factors Influencing Prediction")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        feature_names = preprocessor.get_feature_names_out()
        importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importance.sort_values(ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
        ax.set_title('Top Predictive Features')
        st.pyplot(fig)
        
    except Exception as e:
        st.info("Enable feature importance visualization by installing matplotlib and seaborn")

# Adding sidebar with info
st.sidebar.header("About")
st.sidebar.info("""
- **Model**: XGBoost classifier
- **Training data**: Telco Customer Churn dataset
- **Accuracy**: ~81% (test set)
- **Target**: Predict customer churn probability
""")
st.sidebar.markdown("[Dataset Source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")