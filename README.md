# Rossman-Store-Sales
Project Overview-
Rossmann operates thousands of drug stores across Europe. Sales vary a lot based on factors like promotions, holidays, competition, store type, and time.
This project analyzes historical sales data and builds machine learning and deep learning models to predict daily store sales accurately.
The final solution also includes a deployed web application where users can upload data and get sales predictions instantly.

Objectives-
Understand sales patterns using exploratory data analysis (EDA)
Study the impact of promotions, holidays, competition, and store characteristics
Build accurate sales prediction models
Compare traditional ML with deep learning approaches
Deploy the trained model using a web interface

Dataset-
Source: Rossmann Store Sales
Key Files
train.csv – Historical sales data (with target variable)
test.csv – Future dates without sales (used for prediction)
store.csv – Store-level metadata

Exploratory Data Analysis (EDA)-
Key insights observed from the data:
Promotions significantly increase sales and customer count
Sales and number of customers are highly correlated
Store types respond differently to promotions
Stores open all weekdays perform better on weekends
Competition distance impacts sales differently across stores
December shows strong seasonal effects due to holidays
Assortment type “b” has the highest average sales
These insights guided feature engineering and model selection.

Feature Engineering-
Several new features were created to capture sales behavior better:
Date-based features: Year, Month, Day, WeekOfYear
Weekend indicator: IsWeekend
Promotion impact timing
Days since competition opened
Days since Promo2 started
Categorical encoding for store type, assortment, holidays
Missing values were handled using logical defaults and statistical imputations.

Models Used-
1️⃣ Random Forest Regressor (Primary Model)
Handles non-linear relationships well
Works effectively with mixed feature types
Feature importance analysis performed
Evaluation Results:
RMSLE: 0.2358
MSE: 2,358,893
This indicates strong predictive performance for daily sales forecasting.

2️⃣ Deep Learning – LSTM (Time Series)
Built using historical sales sequences for a sample store
Captures temporal dependencies in sales
Trained on sliding windows of past sales
This model demonstrates how deep learning can model sales trends over time, especially useful for long-term forecasting.

Feature Importance (Random Forest)-
Top drivers of sales:
Store Open / Closed status
Promotion
Competition Distance
Store ID
Day of Week
Competition open duration
Promo2 duration
This confirms that operational and promotional factors dominate sales performance.

Deployment (Streamlit App)-
A web application was created using Streamlit:
App Features
Upload a CSV file
Automatically generate required features
Predict sales using trained model
View predictions in table and chart format
Download predictions as CSV
This makes the model usable by non-technical users.

Project Structure-
Rossmann-Store-Sales/
│
├── Rossmann Store Sales.ipynb
├── app.py
├── requirements.txt
├── rf_sales_model_*.pkl
├── store_test.csv
├── README.md

How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the App
streamlit run app.py

Handling Large Model Files-
The trained .pkl file exceeds GitHub’s size limit.

Recommended solutions:
GitHub Releases
Google Drive link
Git LFS (Large File Storage)

Key Learnings-
Promotions have the strongest influence on sales
Store-level behavior varies significantly
Feature engineering is more impactful than model complexity
Random Forest performs extremely well for tabular retail data
LSTM adds value for time-series understanding
Deployment is crucial to make models business-ready

Final Deliverables-
Jupyter Notebook with full analysis
Trained ML & DL models
Deployed Streamlit application
GitHub repository
Screenshots of results and app

Conclusion-
This project successfully demonstrates an end-to-end data science workflow — from raw data and insights to machine learning, deep learning, and deployment — providing a practical and business-ready solution for retail sales forecasting.
