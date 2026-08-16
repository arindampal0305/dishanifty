# DishaNifty: Indian Stock Market Direction Prediction

Predicts the next trading day direction (UP / DOWN / NEUTRAL) of the NIFTY 50 index using Machine Learning and Deep Learning.

## Research Question
Does increasing model complexity (Random Forest to XGBoost to LSTM) improve predictive accuracy and trading performance on the Indian stock market?

## Team
| Member | Role |
|--------|------|
| Arindam Pal | Pipeline Foundation |
| Sujal | Random Forest + XGBoost |
| Khushi | LSTM |
| Sneha | EDA + Backtesting + Dashboard + Documentation |

## Tech Stack
- **Language:** Python 3.12
- **Data:** yfinance
- **ML:** scikit-learn, xgboost, tensorflow
- **Indicators:** pandas-ta
- **Dashboard:** Streamlit

## Setup
```bash
git clone https://github.com/arindampal0305/DishaNifty.git
cd DishaNifty
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Pipeline
```bash
python src/data_loader.py
python src/data_cleaner.py
python src/features.py
python src/target.py
python src/validation.py
```

## Dataset
- **Tickers:** NIFTY 50, India VIX, USD/INR via yfinance
- **Period:** January 2015 to present
- **Features:** 22 technical indicators

## Target Variable
- **UP (2):** next day return > +0.25%
- **DOWN (0):** next day return < -0.25%
- **NEUTRAL (1):** within 0.25%

## Academic Context
BIT Mesra, Lalpur Campus: Semester 5 AI/ML Project (2026-2027)
Mentor: Dr. Binod Kumar Sharma Sir