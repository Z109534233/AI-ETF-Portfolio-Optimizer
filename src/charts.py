"""
Charts Module
Creates professional Plotly charts for the ETF Portfolio Optimizer.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.theme import COLORS, CHART_COLORS, FONT_FAMILY
from src.i18n import t

DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family=FONT_FAMILY, size=12),
        xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
)


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared design-system theme to any Plotly figure.

    Backgrounds are transparent so charts blend seamlessly into the
    surrounding chart_card() container rather than showing a mismatched panel.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_secondary"], family=FONT_FAMILY, size=12),
        title=dict(font=dict(color=COLORS["text"], size=14)),
        xaxis=dict(
            gridcolor=COLORS["border"], linecolor=COLORS["border"],
            zerolinecolor=COLORS["border"], tickfont=dict(color=COLORS["text_secondary"], size=11),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"], linecolor=COLORS["border"],
            zerolinecolor=COLORS["border"], tickfont=dict(color=COLORS["text_secondary"], size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"], size=11),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["surface_2"], bordercolor=COLORS["border"],
            font=dict(color=COLORS["text"], family=FONT_FAMILY, size=12),
        ),
        margin=dict(l=48, r=24, t=44, b=44),
        colorway=CHART_COLORS,
    )
    return fig


def price_chart(prices_df: pd.DataFrame, title: str = None) -> go.Figure:
    """Line chart of historical prices."""
    title = title if title is not None else t("chart_historical_prices")
    fig = go.Figure()
    for i, col in enumerate(prices_df.columns):
        fig.add_trace(go.Scatter(
            x=prices_df.index, y=prices_df[col],
            name=col, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_price')}: $%{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(title=title, xaxis_title=t("chart_date"), yaxis_title=t("chart_price_usd"))
    return apply_dark_theme(fig)


def normalized_price_chart(prices_df: pd.DataFrame, base: float = 100.0) -> go.Figure:
    """Normalized price comparison chart (all starting at base value)."""
    normalized = prices_df.div(prices_df.iloc[0]) * base
    fig = go.Figure()
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col],
            name=col, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{col}</b><br>{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_normalized')}: %{{y:.1f}}<extra></extra>"
        ))
    fig.update_layout(
        title=t("chart_normalized_price_comparison", base=base),
        xaxis_title=t("chart_date"), yaxis_title=t("chart_normalized_price_base", base=base)
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
            hovertemplate=f"<b>{col}</b><br>{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_return')}: %{{y:.2f}}%<extra></extra>"
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
    fig.update_layout(title=t("chart_cumulative_return_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_cumulative_return_pct"))
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
            hovertemplate=f"<b>{col}</b><br>{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_return')}: %{{y:.2f}}%<extra></extra>"
        ))
    fig.update_layout(title=t("chart_drawdown_pct"), xaxis_title=t("chart_date"), yaxis_title=t("chart_drawdown_pct"))
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
        hovertemplate=f"<b>%{{y}} vs %{{x}}</b><br>{t('chart_correlation')}: %{{z:.3f}}<extra></extra>",
        colorbar=dict(title=t("chart_correlation"), tickfont=dict(color=COLORS["text"]))
    ))
    fig.update_layout(title=t("chart_correlation_heatmap"))
    return apply_dark_theme(fig)


def return_distribution_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Daily return distribution histogram."""
    returns = prices_df.pct_change().dropna() * 100
    fig = go.Figure()
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Histogram(
            x=returns[col], name=col, opacity=0.7, nbinsx=50,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            hovertemplate=f"<b>{col}</b><br>{t('chart_return')}: %{{x:.2f}}%<br>{t('chart_frequency')}: %{{y}}<extra></extra>"
        ))
    fig.update_layout(
        title=t("chart_daily_return_distribution"),
        xaxis_title=t("chart_daily_return_pct"), yaxis_title=t("chart_frequency"),
        barmode="overlay"
    )
    return apply_dark_theme(fig)


# Distinct shape+color per core strategy so the three markers never rely on
# color alone to be told apart (Round 2B-2 requirement); any method key not
# in this map (shouldn't happen -- callers only ever pass the 3 core
# strategies) falls back to a plain circle.
_STRATEGY_MARKER_STYLE = {
    "Equal Weight": {"symbol": "circle", "color": COLORS["purple"]},
    "Maximum Sharpe Ratio": {"symbol": "star", "color": COLORS["success"]},
    "Minimum Volatility": {"symbol": "diamond", "color": COLORS["primary"]},
}


def efficient_frontier_chart(mc_df: pd.DataFrame,
                              frontier_df: pd.DataFrame = None,
                              strategy_results: dict = None,
                              strategy_labels: dict = None,
                              current_method: str = None) -> go.Figure:
    """Efficient frontier decision-support chart (Round 2B-2).

    `strategy_results` must be the SAME dict Strategy Comparison already
    computes (one entry per core method -- Equal Weight / Maximum Sharpe
    Ratio / Minimum Volatility -- each with expected_return/
    expected_volatility/sharpe_ratio/largest_ticker/largest_weight). This
    function never recalculates those numbers itself, so the markers here
    can never drift from the Strategy Comparison table (single source of
    truth). `frontier_df` is the deterministic constrained-optimization
    curve from compute_efficient_frontier() -- the Monte Carlo cloud is
    background context only and is never treated as the frontier itself.
    `current_method` marks whichever of the three core methods (if any) is
    the user's actual current selection; that marker is emphasized in
    place rather than duplicated as a 4th point.
    """
    fig = go.Figure()

    # Monte Carlo cloud -- background context only: small, semi-transparent
    # markers so it never visually competes with the frontier curve or the
    # strategy markers on top of it.
    fig.add_trace(go.Scatter(
        x=mc_df["Volatility"] * 100, y=mc_df["Return"] * 100,
        mode="markers",
        marker=dict(
            color=mc_df["Sharpe"], colorscale="Viridis",
            size=3.5, opacity=0.45,
            colorbar=dict(
                title=t("metric_sharpe_ratio"), tickfont=dict(color=COLORS["text"]),
                thickness=12, len=0.4, y=0.18, yanchor="bottom", x=1.02,
            ),
        ),
        name=t("chart_monte_carlo_portfolios"),
        hovertemplate=f"{t('chart_volatility')}: %{{x:.2f}}%<br>{t('chart_return')}: %{{y:.2f}}%<br>{t('chart_sharpe')}: %{{marker.color:.2f}}<extra></extra>"
    ))

    # Deterministic Efficient Frontier curve -- NOT the outer edge of the
    # random Monte Carlo cloud; only drawn once there are enough feasible
    # points to look like a curve rather than a single dot.
    if frontier_df is not None and len(frontier_df) >= 2:
        frontier_sorted = frontier_df.sort_values("Volatility")
        fig.add_trace(go.Scatter(
            x=frontier_sorted["Volatility"] * 100, y=frontier_sorted["Return"] * 100,
            mode="lines",
            line=dict(color=COLORS["text"], width=2.5),
            name=t("opt_efficient_frontier_card"),
            hovertemplate=f"{t('chart_volatility')}: %{{x:.2f}}%<br>{t('chart_return')}: %{{y:.2f}}%<extra></extra>"
        ))

    # Three core strategy markers, sourced from strategy_results -- never
    # independently recomputed here.
    if strategy_results:
        for method_key, data in strategy_results.items():
            style = _STRATEGY_MARKER_STYLE.get(method_key, {"symbol": "circle", "color": COLORS["text_secondary"]})
            label = (strategy_labels or {}).get(method_key, method_key)
            is_current = method_key == current_method
            ret = data["expected_return"] * 100
            vol = data["expected_volatility"] * 100
            current_line = f"{t('opt_current_strategy_label')}<br>" if is_current else ""
            largest_line = ""
            if data.get("largest_ticker"):
                largest_line = f"<br>{t('opt_col_largest_position')}: {data['largest_ticker']} {data['largest_weight']:.2%}"
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers",
                marker=dict(
                    symbol=style["symbol"], color=style["color"],
                    size=22 if is_current else 15,
                    line=dict(
                        width=3 if is_current else 1.5,
                        color=COLORS["warning"] if is_current else COLORS["text"],
                    ),
                ),
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>{current_line}"
                    f"{t('metric_expected_annual_return')}: {ret / 100:.2%}<br>"
                    f"{t('metric_expected_volatility')}: {vol / 100:.2%}<br>"
                    f"{t('metric_sharpe_ratio')}: {data['sharpe_ratio']:.2f}"
                    f"{largest_line}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=t("chart_efficient_frontier_mc"),
        xaxis_title=t("chart_annualized_volatility_pct"),
        yaxis_title=t("chart_ef_yaxis_return"),
        # Legend pinned to the upper portion of the right-side column,
        # colorbar (set above) pinned to the lower portion -- non-overlapping
        # y-bands so the two never collide regardless of how many strategy
        # markers are in the legend.
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    return apply_dark_theme(fig)


def allocation_donut_chart(weights: dict, title: str = None) -> go.Figure:
    """Donut chart for portfolio allocation."""
    title = title if title is not None else t("chart_portfolio_allocation")
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5, textinfo="label+percent",
        marker=dict(colors=CHART_COLORS[:len(labels)],
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate=f"<b>%{{label}}</b><br>{t('chart_weight')}: %{{value:.1f}}%<extra></extra>"
    )])
    fig.update_layout(
        title=title,
        legend=dict(orientation="v", x=1.05, y=0.5)
    )
    return apply_dark_theme(fig)


def sentiment_donut_chart(bullish_pct: float, neutral_pct: float, bearish_pct: float) -> go.Figure:
    """Donut chart for aggregate news-headline sentiment (Market Intelligence page)."""
    labels = [t("chart_bullish"), t("chart_neutral"), t("chart_bearish")]
    values = [bullish_pct, neutral_pct, bearish_pct]
    colors = [COLORS["success"], COLORS["text_muted"], COLORS["danger"]]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.6, textinfo="label+percent",
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>"
    )])
    fig.update_layout(showlegend=False)
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
            hovertemplate=f"<b>{col}</b><br>{t('chart_volatility')}: {vol:.2f}%<br>{t('chart_return')}: {ret:.2f}%<extra></extra>"
        ))
    fig.update_layout(
        title=t("chart_risk_vs_return"),
        xaxis_title=t("chart_annualized_volatility_pct"),
        yaxis_title=t("chart_annualized_return_pct"),
        showlegend=False
    )
    return apply_dark_theme(fig)


def portfolio_growth_chart(backtest_df: pd.DataFrame,
                            benchmark_df: pd.DataFrame = None,
                            title: str = None) -> go.Figure:
    """Portfolio value over time with optional benchmark."""
    title = title if title is not None else t("home_chart_growth_title")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=backtest_df.index, y=backtest_df["Portfolio Value"],
        name=t("chart_portfolio"), line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate=f"{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('metric_final_value')}: $%{{y:,.2f}}<extra></extra>"
    ))
    if benchmark_df is not None and not benchmark_df.empty:
        initial = backtest_df["Portfolio Value"].iloc[0]
        bench_norm = benchmark_df / benchmark_df.iloc[0] * initial
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm,
            name=t("chart_benchmark"), line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
            hovertemplate=f"{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_benchmark')}: $%{{y:,.2f}}<extra></extra>"
        ))
    fig.update_layout(title=title, xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"))
    return apply_dark_theme(fig)


def historical_growth_chart(history_df: pd.DataFrame, currency_symbol: str = "$",
                             title: str = None) -> go.Figure:
    """Historical Simulation growth chart (Investment Simulator Round 2):
    Portfolio Value vs Cumulative Contributions over the actual backtest
    period, so the gap between the two lines visually IS the investment
    growth (as opposed to money the investor put in). `history_df` must
    have columns "Portfolio Value" and "Cumulative Contributions"
    (src/simulator.py's historical_backtest() return shape) -- this is
    real historical data, never Monte Carlo paths.
    """
    title = title if title is not None else t("hist_growth_chart_title")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df["Portfolio Value"],
        name=t("hist_chart_portfolio_value"), line=dict(color=COLORS["primary"], width=2.5),
        fill="tonexty", fillcolor="rgba(59,130,246,0.10)",
        hovertemplate=f"{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('hist_chart_portfolio_value')}: {currency_symbol}%{{y:,.0f}}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=history_df.index, y=history_df["Cumulative Contributions"],
        name=t("hist_cumulative_contributions_line"), line=dict(color=COLORS["text_muted"], width=1.5, dash="dash"),
        hovertemplate=f"{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('hist_cumulative_contributions_line')}: {currency_symbol}%{{y:,.0f}}<extra></extra>"
    ))
    fig.update_layout(
        title=title, xaxis_title=t("chart_date"),
        yaxis_title=f"{t('hist_chart_value_label')} ({currency_symbol})",
    )
    return apply_dark_theme(fig)


def monte_carlo_paths_chart(paths_df: pd.DataFrame, title: str = None) -> go.Figure:
    """Monte Carlo simulation paths chart."""
    title = title if title is not None else t("chart_monte_carlo_simulation")
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
        name=t("chart_median"), line=dict(color=COLORS["accent"], width=2.5),
        hovertemplate=f"{t('chart_date')}: %{{x|%Y-%m-%d}}<br>{t('chart_median')}: $%{{y:,.0f}}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title=t("chart_date"), yaxis_title=t("chart_portfolio_value_usd"))
    return apply_dark_theme(fig)


def rolling_metrics_chart(prices: pd.Series, window: int = 63,
                           periods_per_year: int = 252) -> go.Figure:
    """Rolling volatility and return chart."""
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window).std() * np.sqrt(periods_per_year) * 100
    rolling_ret = returns.rolling(window).mean() * periods_per_year * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[t("chart_rolling_annualized_return_pct"), t("chart_rolling_annualized_volatility_pct")],
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(
        x=rolling_ret.index, y=rolling_ret,
        name=t("chart_rolling_return"), line=dict(color=COLORS["success"], width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        name=t("chart_rolling_volatility"), line=dict(color=COLORS["danger"], width=2)
    ), row=2, col=1)
    fig.update_layout(title=t("chart_rolling_metrics_window", window=window), showlegend=True)
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
        hovertemplate=f"{t('chart_year')}: %{{y}} / %{{x}}<br>{t('chart_return')}: %{{z:.2f}}%<extra></extra>",
        colorbar=dict(title=t("chart_return"), tickfont=dict(color=COLORS["text"]))
    ))
    fig.update_layout(title=t("chart_monthly_returns_heatmap"))
    return apply_dark_theme(fig)


def feature_importance_chart(feature_importance: pd.Series, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for ML feature importance."""
    top_features = feature_importance.head(top_n)
    fig = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate=f"<b>%{{y}}</b><br>{t('chart_importance_score')}: %{{x:.4f}}<extra></extra>"
    ))
    fig.update_layout(
        title=t("chart_top_n_feature_importance", n=top_n),
        xaxis_title=t("chart_importance_score"),
        yaxis_title=t("chart_feature"),
        yaxis=dict(autorange="reversed")
    )
    return apply_dark_theme(fig)


def confusion_matrix_chart(cm: np.ndarray) -> go.Figure:
    """Confusion matrix heatmap."""
    labels = [t("chart_down_label"), t("chart_up_label")]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color="white"),
        hovertemplate=f"{t('chart_actual_label')}: %{{y}}<br>{t('chart_predicted_label')}: %{{x}}<br>{t('chart_frequency')}: %{{z}}<extra></extra>",
        showscale=False
    ))
    fig.update_layout(
        title=t("chart_confusion_matrix"),
        xaxis_title=t("chart_predicted_label"),
        yaxis_title=t("chart_actual_label")
    )
    return apply_dark_theme(fig)


def future_value_distribution_chart(final_values: np.ndarray,
                                     total_contributed: float) -> go.Figure:
    """Distribution of final portfolio values from Monte Carlo."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_values, nbinsx=50,
        marker_color=COLORS["primary"], opacity=0.8, name=t("chart_final_values"),
        hovertemplate=f"{t('metric_final_value')}: $%{{x:,.0f}}<br>{t('chart_frequency')}: %{{y}}<extra></extra>"
    ))
    fig.add_vline(x=total_contributed, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=t("chart_total_contributed"), annotation_font_color=COLORS["danger"])
    fig.add_vline(x=float(np.median(final_values)), line_dash="dash", line_color=COLORS["success"],
                  annotation_text=t("chart_median"), annotation_font_color=COLORS["success"])
    fig.update_layout(
        title=t("chart_distribution_final_values"),
        xaxis_title=t("chart_final_value_usd"), yaxis_title=t("chart_frequency")
    )
    return apply_dark_theme(fig)
