# Customer-Churn-Pred
This Streamlit application predicts customer churn probability for telecom companies using machine learning. Based on customer attributes and service details, it identifies high-risk customers and highlights key factors driving churn.

Access the live app from here - https://customer-churn-pred-hm7e3ipexcstvt7qkzw36q.streamlit.app/

🚀 FEATURES
- Real-time churn probability prediction

- Risk level classification (High/Medium/Low)

- Top influencing factors visualization

- Mobile-responsive interface

- Example customer profiles for quick testing

🧠 MODEL DETAILS
- Algorithm: XGBoost Classifier

- Accuracy: ~81% on test data

- Training Data: Telco Customer Churn Dataset

- Key Features: Contract type, tenure, payment method, service details

💻 LOCAL INSTALLATION
1. Clone repository:
   git clone https://https://github.com/yashydv9/Customer-Churn-Pred
   
   cd telco-churn-app

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run cust_churn_pred.py

🛠️ Technical Requirements
- Python 3.8+

- Streamlit

- XGBoost

- scikit-learn

- pandas

- numpy

- matplotlib

- seaborn

📊 EXAMPLE PREDICTIONS
Customer Profile - Churn Probability - Risk Level

Senior citizen, month-to-month contract, fiber internet - 92% - 🔴 High

Long-term customer, two-year contract, automatic payment - 8% - 🟢 Low

Mid-tenure, one-year contract, high monthly charges - 42% - 🟡 Medium

📂 FILE STRUCTURE
telco-churn-app/
├── app.py              
# Main Streamlit application

├── xgboost_churn_model.pkl  
# Trained XGBoost model

├── preprocessor.pkl     
# Data preprocessing pipeline

├── requirements.txt     
# Python dependencies

└── README.md           
# This documentation

🤝 CONTRIBUTING
1. Fork the repository

2. Create your feature branch (git checkout -b feature/your-feature)

3. Commit your changes (git commit -am 'Add some feature')

4. Push to the branch (git push origin feature/your-feature)

5. Open a pull request
