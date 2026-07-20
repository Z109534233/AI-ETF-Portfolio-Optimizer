"""
Charts Module
Creates professional Plotly charts for the ETF Portfolio Optimizer.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Color palette
COLORS = {
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "bg": "#0B1220",
    "panel": "#111827",
    "border": "#1F2937",
    "text": "#F8FAFC",
    "muted": "#9CA3AF",
}

CHART_COLORS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1"
]

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], showgrid=True),
        legend=dict(bgcolor=COLORS["panel"], bordercolor=COLORS["border"], borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=50),
    )
)


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        legend=dict(bgcolor=COLORS["panel"], bordercolor=COLORS["border"], borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def price_chart(prices_df: pd.DataFrame, title: str = "Historical Price") -> go.Figure:
    """Line chart of historical prices."""
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        fig.add_trace(go.Scatter(
            x=prices_df.index, y=prices_df[col],
            name=col, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Price: $%{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price (USD)")
    return apply_dark_theme(fig)


def normalized_price_chart(prices_df: pd.DataFrame, base: float = 100.0) -> go.Figure:
    """Normalized price comparison chart (all starting at base value)."""
    normalized = prices_df.div(prices_df.iloc[0]) * base
    fig = go.Figure()
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col],
            name=col, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Normalized: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        title=f"Normalized Price Comparison (Base={base})",
        xaxis_title="Date", yaxis_title=f"Normalized Price (Base={base})"
    )
    return apply_dark_theme(fig)


def cumulative_return_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Cumulative return chart."""
    returns = prices_df.pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    fig = go.Figure()
    for i, col in enumerate(cum_returns.columns):
        fig.add_trace(go.Scatter(
            x=cum_returns.index, y=cum_returns[col] * 100,
            name=col, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Return: %{{y:.2f}}%<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
    fig.update_layout(title="Cumulative Return (%)", xaxis_title="Date", yaxis_title="Cumulative Return (%)")
    return apply_dark_theme(fig)


def drawdown_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Drawdown chart for each ETF."""
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        prices = prices_df[col].dropna()
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown,
            name=col, fill="tozeroy",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
            fillcolor=f"rgba({int(CHART_COLORS[i % len(CHART_COLORS)][1:3], 16)}, "
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][3:5], 16)}, "
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][5:7], 16)}, 0.15)",
            hovertemplate=f"<b>{col}</b><br>Date: %{{x|%Y-%m-%d}}<br>Drawdown: %{{y:.2f}}%<extra></extra>"
        ))
    fig.update_layout(title="Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)")
    return apply_dark_theme(fig)


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Correlation", tickfont=dict(color=COLORS["text"]))
    ))
    fig.update_layout(title="Correlation Heatmap")
    return apply_dark_theme(fig)


def return_distribution_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Daily return distribution histogram."""
    returns = prices_df.pct_change().dropna() * 100
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Histogram(
            x=returns[col], name=col, opacity=0.7, nbinsx=50,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            hovertemplate=f"<b>{col}</b><br>Return: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>"
        ))
    fig.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        barmode="overlay"
    )
    return apply_dark_theme(fig)


def efficient_frontier_chart(mc_df: pd.DataFrame, optimal_weights: dict = None,
                              min_vol_weights: dict = None,
                              mean_returns: np.ndarray = None,
                              cov_matrix: np.ndarray = None) -> go.Figure:
    """Efficient frontier with Monte Carlo scatter."""
    fig = go.Figure()

    # Monte Carlo scatter
    fig.add_trace(go.Scatter(
        x=mc_df["Volatility"] * 100, y=mc_df["Return"] * 100,
        mode="markers",
        marker=dict(
            color=mc_df["Sharpe"], colorscale="Viridis",
            size=4, opacity=0.6,
            colorbar=dict(title="Sharpe Ratio", tickfont=dict(color=COLORS["text"]))
        ),
        name="Monte Carlo Portfolios",
        hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>"
    ))

    # Max Sharpe point
    if optimal_weights is not None and mean_returns is not None and cov_matrix is not None:
        from src.financial_metrics import portfolio_return, portfolio_volatility
        w = np.array(list(optimal_weights.values()))
        ret = portfolio_return(w, mean_returns) * 100
        vol = portfolio_volatility(w, cov_matrix) * 100
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers",
            marker=dict(color=COLORS["accent"], size=14, symbol="star"),
            name="Max Sharpe Ratio",
            hovertemplate=f"<b>Max Sharpe</b><br>Volatility: {vol:.2f}%<br>Return: {ret:.2f}%<extra></extra>"
        ))

    fig.update_layout(
        title="Efficient Frontier & Monte Carlo Simulation",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)"
    )
    return apply_dark_theme(fig)


def allocation_donut_chart(weights: dict, title: str = "Portfolio Allocation") -> go.Figure:
    """Donut chart for portfolio allocation."""
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5, textinfo="label+percent",
        marker=dict(colors=CHART_COLORS[:len(labels)],
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>"
    )])
    fig.update_layout(
        title=title,
        legend=dict(orientation="v", x=1.05, y=0.5)
    )
    return apply_dark_theme(fig)


def risk_return_scatter(prices_df: pd.DataFrame, periods_per_year: int = 252) -> go.Figure:
    """Risk vs Return scatter plot for multiple ETFs."""
    from src.financial_metrics import annualized_return, annualized_volatility
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        prices = prices_df[col].dropna()
        ret = annualized_return(prices, periods_per_year) * 100
        vol = annualized_volatility(prices, periods_per_year) * 100
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text",
            name=col, text=[col], textposition="top center",
            marker=dict(color=CHART_COLORS[i % len(CHART_COLORS)], size=12),
            hovertemplate=f"<b>{col}</b><br>Volatility: {vol:.2f}%<br>Return: {ret:.2f}%<extra></extra>"
        ))
    fig.update_layout(
        title="Risk vs Return",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        showlegend=False
    )
    return apply_dark_theme(fig)


def portfolio_growth_chart(backtest_df: pd.DataFrame,
                            benchmark_df: pd.DataFrame = None,
                            title: str = "Portfolio Growth") -> go.Figure:
    """Portfolio value over time with optional benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df["Portfolio Value"],
        name="Portfolio", line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>"
    ))
    if benchmark_df is not None and not benchmark_df.empty:
        initial = backtest_df["Portfolio Value"].iloc[0]
        bench_norm = benchmark_df / benchmark_df.iloc[0] * initial
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm,
            name="Benchmark", line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Benchmark: $%{y:,.2f}<extra></extra>"
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    return apply_dark_theme(fig)


def monte_carlo_paths_chart(paths_df: pd.DataFrame, title: str = "Monte Carlo Simulation") -> go.Figure:
    """Monte Carlo simulation paths chart."""
    fig = go.Figure()
    cols = paths_df.columns[:100]  # Limit for performance
    for col in cols:
        fig.add_trace(go.Scatter(
            x=paths_df.index, y=paths_df[col],
            mode="lines", line=dict(width=0.5, color=COLORS["primary"]),
            opacity=0.15, showlegend=False,
            hoverinfo="skip"
        ))
    # Median line
    median = paths_df.median(axis=1)
    fig.add_trace(go.Scatter(
        x=median.index, y=median,
        name="Median", line=dict(color=COLORS["accent"], width=2.5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Median: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    return apply_dark_theme(fig)


def rolling_metrics_chart(prices: pd.Series, window: int = 63,
                           periods_per_year: int = 252) -> go.Figure:
    """Rolling volatility and return chart."""
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year) * 100
    rolling_ret = returns.rolling(window).mean() * periods_per_year * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Rolling Annualized Return (%)", "Rolling Annualized Volatility (%)"],
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(
        x=rolling_ret.index, y=rolling_ret,
        name="Rolling Return", line=dict(color=COLORS["secondary"], width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        name="Rolling Volatility", line=dict(color=COLORS["danger"], width=2)
    ), row=2, col=1)
    fig.update_layout(title=f"Rolling Metrics ({window}-Day Window)", showlegend=True)
    return apply_dark_theme(fig)


def monthly_heatmap(monthly_returns: pd.DataFrame) -> go.Figure:
    """Monthly returns heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values * 100,
        x=monthly_returns.columns.tolist(),
        y=monthly_returns.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(monthly_returns.values * 100, 1),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return (%)", tickfont=dict(color=COLORS["text"]))
    ))
    fig.update_layout(title="Monthly Returns Heatmap (%)")
    return apply_dark_theme(fig)


def feature_importance_chart(feature_importance: pd.Series, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for ML feature importance."""
    top_features = feature_importance.head(top_n)
    fig = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )
    return apply_dark_theme(fig)


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Confusion matrix heatmap."""
    labels = ["Down (0)", "Up (1)"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label"
    )
    return apply_dark_theme(fig)


def future_value_distribution_chart(final_values: np.ndarray,
                                     total_contributed: float) -> go.Figure:
    """Distribution of final portfolio values from Monte Carlo."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_values, nbinsx=50,
        marker_color=COLORS["primary"], opacity=0.8, name="Final Values",
        hovertemplate="Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(x=total_contributed, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="Total Contributed", annotation_font_color=COLORS["danger"])
    fig.add_vline(x=float(np.median(final_values)), line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text="Median", annotation_font_color=COLORS["secondary"])
    fig.update_layout(
        title="Distribution of Final Portfolio Values",
        xaxis_title="Final Value ($)", yaxis_title="Frequency"
    )
    return apply_dark_theme(fig)
