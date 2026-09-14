# AI ETF Portfolio Optimizer

**ETF Portfolio Analytics & Quantitative Decision Platform** -- performance analysis, portfolio
optimization, risk analytics, simulation, and machine learning, with AI-assisted interpretation.

A professional FinTech web application built with Python and Streamlit, designed as a portfolio project for UK Master's programme applications in Business Analytics, Finance Analytics, Financial Technology, and Data Analytics.

---

## Overview

The AI ETF Portfolio Optimizer is a comprehensive investment analytics platform that enables users to:

- Analyse historical ETF performance with advanced risk metrics
- Optimise portfolio allocations using mean-variance and risk parity methods
- Simulate long-term investment outcomes with Monte Carlo methods
- Assess portfolio risk with stress testing and scenario analysis
- Apply machine learning models for educational return direction prediction
- Generate AI-powered portfolio explanations with OpenAI GPT integration
- Save and compare portfolios using a local SQLite database

---

## Features

| Feature | Description |
|---|---|
| **ETF Analysis** | Historical prices, return distributions, correlation heatmaps, technical indicators |
| **Portfolio Optimizer** | 5 optimization methods, efficient frontier, Monte Carlo simulation, backtesting |
| **Investment Simulator** | Monte Carlo projection, compound growth, scenario comparison |
| **Risk Analytics** | VaR, CVaR, Beta, Alpha, Tracking Error, stress testing |
| **Machine Learning** | Logistic Regression & Random Forest for direction prediction |
| **AI Portfolio Analyst** | OpenAI-powered synthesis of already-computed portfolio metrics, with rule-based fallback |
| **Portfolio History** | SQLite storage, portfolio comparison, CSV export |

---

## Screenshots

> *Deploy the application and add screenshots here.*

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend / App** | Streamlit 1.32+ |
| **Data Processing** | Pandas 2.0+, NumPy 1.26+ |
| **Visualisation** | Plotly 5.18+ |
| **Optimisation** | SciPy 1.11+ |
| **Machine Learning** | Scikit-learn 1.3+ |
| **Market Data** | yfinance 0.2.36+ |
| **Database** | SQLite + SQLAlchemy 2.0+ |
| **PDF Reports** | ReportLab 4.0+ |
| **AI Integration** | OpenAI 1.12+ |

---

## Local Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-ETF-Portfolio-Optimizer.git
cd AI-ETF-Portfolio-Optimizer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## How to Run

```bash
streamlit run app.py
```

The main entry file is `app.py` in the project root. All pages are in the `pages/` directory and are automatically discovered by Streamlit.

---

## Streamlit Cloud Deployment

### Steps

1. Push the project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**.
4. Select your repository, branch, and set the main file to `app.py`.
5. Click **Deploy**.

### Important Notes

- Ensure `requirements.txt` is in the project root.
- Do **not** commit `.streamlit/secrets.toml` to GitHub.
- The SQLite database is created automatically on first run.
- The application works without an OpenAI API key (uses rule-based fallback).

---

## OpenAI Setup (Secret Key Configuration)

All OpenAI calls in this app go through a single module, `src/openai_service.py`
(the current **Responses API** -- `client.responses.create`, not the legacy
Chat Completions API). It reads `OPENAI_API_KEY` (required) and an optional
`OPENAI_MODEL` override from Streamlit secrets or the environment -- **never**
hard-code a key in source. The app works fully without a key configured:
every AI-backed feature (ETF Analysis "AI Interpretation", AI Portfolio
Analyst synthesis, Market Intelligence "Today's Market Summary") falls back
to a clearly-labeled Rule-Based result instead of failing.

### Local Development

Create `.streamlit/secrets.toml` (already excluded via `.gitignore` -- never
commit this file):

```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
OPENAI_MODEL = "gpt-5.6-luna"  # optional -- defaults to a cost-conscious current model
```

### Streamlit Community Cloud

1. Go to your app settings on Streamlit Cloud.
2. Click **Secrets**.
3. Add:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
OPENAI_MODEL = "gpt-5.6-luna"  # optional
```

Never commit the real key to the repository. The application functions
fully without an OpenAI key -- AI Portfolio Analyst, ETF Analysis's "AI
Interpretation", and Market Intelligence's market summary all use
rule-based analysis as a fallback, and the UI labels which one produced
the result you're looking at ("AI-Generated" vs "Rule-Based").

---

## Project Structure

```
AI-ETF-Portfolio-Optimizer/
│
├── app.py                          # Main landing page
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme and server settings
│
├── pages/
│   ├── 1_ETF_Analysis.py           # ETF price and risk analysis
│   ├── 2_Portfolio_Optimizer.py    # Mean-variance optimization
│   ├── 3_Investment_Simulator.py   # Monte Carlo simulation
│   ├── 4_Risk_Analytics.py         # Portfolio risk metrics
│   ├── 5_Machine_Learning.py       # ML direction prediction
│   ├── 6_AI_Advisor.py             # AI portfolio explanation
│   └── 7_Portfolio_History.py      # Saved portfolio management
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # yfinance data download with caching
│   ├── data_cleaner.py             # Data validation and preprocessing
│   ├── financial_metrics.py        # Sharpe, Sortino, VaR, CVaR, etc.
│   ├── technical_indicators.py     # SMA, EMA, RSI, MACD, Bollinger Bands
│   ├── portfolio_optimizer.py      # Mean-variance, risk parity, Monte Carlo
│   ├── simulator.py                # Long-term investment simulation
│   ├── machine_learning.py         # ML pipeline with time-series splitting
│   ├── ai_advisor.py               # OpenAI integration with fallback
│   ├── database.py                 # SQLite/SQLAlchemy ORM
│   ├── report_generator.py         # ReportLab PDF generation
│   ├── charts.py                   # Plotly chart functions
│   └── utils.py                    # Helper utilities and CSS loader
│
├── assets/
│   └── style.css                   # Custom CSS styling
│
├── data/
│   └── sample_etf_data.csv         # Fallback sample data
│
├── database/
│   └── portfolio.db                # SQLite database (auto-created)
│
├── reports/                        # Generated PDF reports
└── images/                         # Chart exports
```

---

## Educational Disclaimer

This platform is developed as a **portfolio project for academic purposes**, specifically for applications to UK Master's programmes in:

- Business Analytics
- Finance Analytics
- Financial Technology (FinTech)
- Data Analytics

**This application is for educational and demonstration purposes only.** It does not constitute financial advice, investment recommendations, or a solicitation to buy or sell any securities. Past performance is not indicative of future results. Always consult a qualified financial adviser before making investment decisions.

---

## Future Improvements

- Real-time price streaming with WebSocket integration
- Multi-currency support and FX hedging analysis
- ESG scoring and sustainable investing filters
- Factor model analysis (Fama-French 3/5 factor)
- Options pricing and Greeks calculation
- Portfolio rebalancing scheduler
- Email alerts for portfolio threshold breaches
- Advanced backtesting with transaction cost modelling
- Integration with broker APIs for live portfolio tracking
- Mobile-responsive Progressive Web App (PWA) version

---

## Author

Portfolio project for UK Master's programme applications.
Built with Python, Streamlit, and modern FinTech data science tools.
