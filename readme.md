# Predicting Customer Spending from Demographics and Purchase Frequency

## Project Summary
This project predicts annual customer spending using supervised regression.

- Target: `spending`
- Numeric features: `age`, `income`, `purchase_frequency`
- Categorical features: `gender`, `education`, `country`
- Identifier dropped: `name`

Two models are trained and compared:
1. Linear Regression
2. Random Forest Regressor

The web dashboard also:
- compares both model predictions,
- highlights the better-performing model,
- assigns customer segment (Low/Medium/High value),
- and shows personalized marketing recommendations.

## Files
- `EE7209_Customer_Spending_Regression.ipynb`: Main analysis notebook.
- `EE7209_Customer_Spending_Regression.executed.ipynb`: Executed snapshot notebook (optional artifact).
- `customer_data.csv`: Dataset used by notebook and app.
- `app.py`: Streamlit web application.
- `requirements.txt`: Python dependencies.

## Run the Web App
From the project folder:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Verify Everything is Working
1. App loads without errors.
2. You can submit the form and receive:
- Linear Regression prediction
- Random Forest prediction
- Best model indicator
- Segment label
- Marketing recommendation
3. Model quality table appears with CV metrics (RMSE, MAE, R2).

## Notes
- Country is transformed using Top-N + `Other` grouping.
- Numeric features are standardized.
- Full preprocessing + model are encapsulated in scikit-learn pipeline logic.
