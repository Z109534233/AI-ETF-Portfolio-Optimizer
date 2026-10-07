# AI ETF Portfolio Optimizer

[![Tests](https://github.com/Z109534233/AI-ETF-Portfolio-Optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/Z109534233/AI-ETF-Portfolio-Optimizer/actions/workflows/tests.yml)

**Quantitative ETF portfolio analytics, optimization, risk validation, simulation, and model evaluation in one Streamlit dashboard.**

[Live Demo](https://ai-etf-portfolio-optimizer.onrender.com) · [Source Code](https://github.com/Z109534233/AI-ETF-Portfolio-Optimizer) · MIT License

**Author:** Tzu-Hsin Tseng · National Taipei University of Business (NTUB)

---

## Why this project

This project explores a practical question:

> How can an ETF portfolio dashboard make optimization results useful **without overstating the reliability of historical estimates**?

Instead of presenting a single optimizer output as the “answer,” the dashboard adds uncertainty analysis, validation, and methodology disclosure around the result. The emphasis is on **quantitative reasoning, reproducibility, and transparent limitations**.

The application supports Taiwan, U.S., and U.K. ETF workflows and provides a bilingual **English / Traditional Chinese** interface.

---

## Quick review path

If you are reviewing this project for a technical or academic portfolio, the fastest path is:

1. Open **Portfolio Optimizer** and run a Maximum Sharpe portfolio.
2. Open **Backtest & Risk** to compare the strategy with Equal Weight and Minimum Volatility using a **3-year trailing / quarterly walk-forward out-of-sample backtest with turnover-based transaction costs**.
3. Inspect **Bootstrap Weight Stability** to see how allocation weights change under joint historical resampling.
4. Open **Risk Analytics** for historical VaR/CVaR, drawdown, stress testing, and rolling VaR backtesting.
5. Open **Machine Learning** to inspect chronological holdout results, embargo gaps, and expanding-window validation.
6. Open **Investment Simulator / Goal Planner** to compare deterministic assumptions with Monte Carlo outcome distributions.

**Live dashboard:** https://ai-etf-portfolio-optimizer.onrender.com

---

## Key quantitative methods

### Portfolio optimization

The optimizer currently supports:

- Equal Weight
- Maximum Sharpe Ratio
- Minimum Volatility
- Target Return
- Risk Parity

For the mean-variance methods, expected returns and covariance are estimated from the **same common-date observation panel**. This avoids treating one market's holiday as another asset's artificial 0% return and prevents expected-return / covariance sample drift in mixed-market portfolios. The default covariance estimator is **Ledoit-Wolf shrinkage**, which is more stable than an unregularized sample covariance matrix when assets are correlated or the sample is limited.

Mixed-market portfolios are converted to a selected base currency before the common-date return panel is constructed. The dashboard discloses the active assumptions, constraints, historical window, risk-free-rate source, covariance estimator, and numerical diagnostics instead of hiding them behind the final weights.

### Walk-forward out-of-sample backtest

The strategy-evaluation chart uses a rolling out-of-sample procedure:

- use the trailing **3 years** of data to estimate portfolio weights;
- default to a **10-year historical window** in Portfolio Optimizer when the selected ETFs have sufficient history;
- rebalance every **3 months**;
- hold the resulting weights over the next period without daily reset;
- allow weights to drift between rebalances;
- charge **10 bps × one-way turnover** at each subsequent rebalance;
- use the **historical FRED DGS3MO rate available at each rebalance** for Maximum Sharpe rather than today's rate; the reported out-of-sample Sharpe also uses the historical rate series;
- compare the current strategy against Equal Weight, Maximum Sharpe, and Minimum Volatility where applicable;
- surface optimizer failures instead of silently relabeling a fallback portfolio.

The first allocation is not charged because the transaction-cost comparison focuses on rebalancing turnover.

### Bootstrap weight stability

Maximum Sharpe results can be sensitive to estimation error, especially in expected returns.

The dashboard therefore includes an on-demand bootstrap sensitivity analysis that:

- performs **200 joint-row bootstrap resamples** of historical daily returns;
- preserves same-day cross-asset dependence by resampling rows jointly;
- re-runs the same Maximum Sharpe objective and constraints on each successful resample;
- excludes failed optimizer solves rather than replacing them with artificial fallback weights;
- reports median, IQR, 5th–95th percentile ranges, inclusion frequency, and top-holding frequency;
- overlays the original point-estimate portfolio on the bootstrap weight distribution.

This is presented as a **sensitivity analysis**, not as a forecast and not as an implementation of proprietary resampled-efficiency methods.

### Risk analytics and VaR validation

Risk Analytics includes:

- annualized volatility and maximum drawdown;
- Sharpe, Sortino, and Calmar ratios;
- historical VaR and CVaR;
- diversification and concentration diagnostics;
- benchmark-relative Beta, Alpha, Tracking Error, and Information Ratio;
- historical stress scenarios;
- rolling one-step-ahead historical VaR exception backtesting;
- **Kupiec Proportion-of-Failures unconditional coverage test**.

The Kupiec p-value is presented as evidence about compatibility between observed exception frequency and the stated VaR confidence level; it is not treated as proof that the model is “correct.”

### Monte Carlo simulation and goal planning

The simulation layer separates in-sample historical optimizer statistics from forward-looking educational assumptions.

Features include:

- Monte Carlo long-horizon wealth paths;
- nominal and inflation-adjusted outcomes;
- P10 / median / P90 terminal values;
- target-attainment path share;
- deterministic contribution estimates shown separately from stochastic results;
- an estimated monthly contribution associated with approximately **80% of simulated paths** reaching a nominal goal.

Scenario assumptions are explicitly labeled as educational assumptions rather than return forecasts.

### Machine learning evaluation

The ML page provides an educational ETF return-direction classification workflow using:

- Logistic Regression;
- Random Forest;
- chronological train/test splitting with an **embargo gap equal to the forward-label horizon**;
- Accuracy, Precision, Recall, F1, ROC AUC, and confusion matrix;
- a training-set majority-class baseline;
- **expanding-window walk-forward validation** inside the pre-holdout training period.

The final holdout set is kept separate from the walk-forward folds to reduce leakage. Fold dispersion is described as temporal instability, not evidence of future predictability.

---

## Data and model safeguards

Several design choices were added specifically to reduce misleading outputs:

- **Risk-free rate provenance:** current single-run analysis uses FRED DGS3MO as the primary source, Yahoo Finance `^IRX` as a labeled secondary live proxy, and a fixed fallback only if both live sources fail. Walk-forward evaluation separately retrieves historical FRED DGS3MO and uses the observation available at each rebalance.
- **Aligned estimation sample:** every selected asset uses the same observed dates for return and covariance estimation.
- **Shrinkage covariance:** Ledoit-Wolf is the default covariance estimator for portfolio construction and the efficient frontier.
- **Covariance diagnostics:** rank, condition number, and average pairwise correlation are surfaced before interpreting concentrated optimizer results.
- **No fabricated live holdings prices:** cross-page analysis from guest holdings is disabled if a required live price cannot be obtained.
- **Guest-session isolation:** public holdings and watchlist entries are stored only in the current Streamlit session and are not shared between visitors.
- **AI separation:** AI-generated text interprets already-computed metrics; the financial calculations themselves are deterministic Python code. A rule-based fallback is available when no OpenAI API key is configured.
- **Bilingual parity:** English and Traditional Chinese UI strings are covered by regression checks.

---

## Main dashboard modules

| Module | Purpose |
|---|---|
| **ETF Analysis** | Historical performance, distributions, rolling metrics, correlations, technical indicators |
| **Portfolio Optimizer** | Mean-variance optimization, Risk Parity, Ledoit-Wolf covariance, efficient frontier, walk-forward OOS backtesting, bootstrap weight stability |
| **Investment Simulator** | Monte Carlo investment projection and long-horizon scenario analysis |
| **Risk Analytics** | VaR/CVaR, drawdown, benchmark metrics, stress tests, VaR exception backtesting |
| **Machine Learning** | Direction classification, holdout testing, baseline comparison, walk-forward validation |
| **AI Advisor** | Structured interpretation of computed portfolio metrics with rule-based fallback |
| **My Portfolio** | Goal Planner, session-only holdings/watchlist, saved portfolio comparison |
| **Market Intelligence** | Major-index snapshot, curated market news, and rule-based/AI-assisted summaries |

---

## Technology stack

| Area | Technology |
|---|---|
| Application | Python, Streamlit |
| Data | pandas, NumPy, yfinance |
| Optimization | SciPy |
| Machine Learning | scikit-learn |
| Visualization | Plotly |
| Statistical / Risk Logic | Custom Python modules |
| Persistence | SQLite / SQLAlchemy for saved optimizer history |
| Reports | ReportLab |
| AI Interpretation | OpenAI API with deterministic rule-based fallback |
| Deployment | Render |

---

## Repository structure

```text
AI-ETF-Portfolio-Optimizer/
├── app.py
├── pages/
│   ├── 1_ETF_Analysis.py
│   ├── 2_Portfolio_Optimizer.py
│   ├── 3_Investment_Simulator.py
│   ├── 4_Risk_Analytics.py
│   ├── 5_Machine_Learning.py
│   ├── 6_AI_Advisor.py
│   ├── 7_Portfolio_History.py
│   └── 8_Market_Intelligence.py
├── src/
│   ├── portfolio_optimizer.py
│   ├── financial_metrics.py
│   ├── risk_analytics.py
│   ├── simulator.py
│   ├── machine_learning.py
│   ├── risk_free_rate.py
│   ├── ai_advisor.py
│   ├── data_loader.py
│   ├── charts.py
│   ├── database.py
│   └── ...
├── tests/
├── assets/
├── data/
└── requirements.txt
```

The repository currently contains a broad automated test suite across financial logic, Streamlit page behavior, internationalization, cross-page state handoff, and regression cases.

---

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Z109534233/AI-ETF-Portfolio-Optimizer.git
cd AI-ETF-Portfolio-Optimizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The application will normally open at `http://localhost:8501`.

---

## Optional OpenAI configuration

The dashboard works without an OpenAI API key. AI-assisted explanations fall back to deterministic rule-based output when the key is unavailable.

For local development, create a private `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-key-here"
```

Do not commit secrets to the repository.

---

## Methodological limitations

This project intentionally surfaces limitations rather than treating model output as investment advice.

Key limitations include:

- historical mean returns are noisy estimates and can produce unstable optimizer weights;
- walk-forward results remain sample- and specification-dependent and transaction costs are modeled as a simplified turnover charge rather than a full execution model;
- bootstrap dispersion measures sensitivity to the historical sample, not future return probabilities;
- Monte Carlo results depend on the stated return, volatility, fee, inflation, and distribution assumptions;
- historical VaR can fail during regime shifts and tail events;
- ML classification performance is period-dependent and does not establish persistent predictive skill;
- market data quality and availability depend on external providers;
- cross-market daily returns can be asynchronous because Taiwan, U.K., and U.S. markets close at different times; same-calendar-date alignment does not fully eliminate non-synchronous trading effects and may understate or distort measured correlations;
- Goal Planner currency labels do not imply that all foreign-exchange risk is modeled.

These limitations are part of the project design and are shown in the dashboard where relevant.

---

## Development attribution

This project was designed, integrated, and validated by **Tzu-Hsin Tseng**. Generative AI tools were used as development assistants for tasks such as code review, debugging, refactoring suggestions, test generation, and documentation support. Financial-methodology choices, feature scope, integration decisions, validation criteria, and the final submitted implementation were reviewed and directed by the author.

AI-generated text inside the application is kept separate from deterministic financial calculations; the application can fall back to rule-based interpretation when no OpenAI API key is configured.

## Educational use

This repository is an independent quantitative finance / FinTech portfolio project.

It is intended for **educational and analytical demonstration purposes only**. Nothing in the dashboard constitutes financial advice, a recommendation, or a solicitation to buy or sell securities.

---

## License

MIT License.
