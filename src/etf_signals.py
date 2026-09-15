"""
Shared ETF quant-signal computations for pages/1_ETF_Analysis.py (Issue #20
section 2 -- "Fix interpretation inconsistency").

Before this module existed, the ETF Analysis page computed a ticker's
score/trend/recommendation in THREE independently-tuned places (the
Overview "AI Score" card, the Overview "Key Observations" block, and the
Compare workspace's "ETF Compare Score" table) with different weightings
and thresholds -- so the same ticker could legitimately show Trend =
Bullish in one card and a contradictory recommendation in another. Every
workspace now calls compute_quant_signals() for a given price series and
gets back the SAME Trend Signal / Quant Score / Portfolio View / Signal
Agreement, so the numbers can never disagree with each other.

Three explicitly distinct, non-overlapping constructs (per the issue):
  - Trend Signal: Bullish / Neutral / Bearish -- based ONLY on the trailing
    TREND_LOOKBACK_DAYS-trading-day return (recent_trend_return()), never
    the full selected-date-range annualized return. A ticker with a strong
    multi-year annualized_return but a negative last-quarter move must be
    able to show Bearish here -- that is the whole point of a "recent
    direction" label, and pinning it to the full-window return (as an
    earlier version of this module did) made the label false for any
    window longer than the recent move it claims to describe.
  - Quant Score: 0-100, combining return, risk and momentum.
  - Portfolio View: Underweight / Neutral / Overweight -- a research framing
    derived from the Quant Score, NOT a Buy/Sell instruction.
  - Signal Agreement: the fraction of the underlying indicators (return,
    Sharpe, momentum, price-vs-long-MA) that agree with the overall score's
    direction, expressed as a percentage. This is genuinely what the old
    "Confidence" label measured -- it is not a statistical/probabilistic
    confidence interval, so it must never be re-labeled "Confidence".
"""

import pandas as pd

from src.financial_metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, maximum_drawdown,
)
from src.technical_indicators import sma, momentum
from src.openai_service import cached_generate, generate_text, fingerprint as _fingerprint

TREND_BULLISH, TREND_NEUTRAL, TREND_BEARISH = "Bullish", "Neutral", "Bearish"
VIEW_OVERWEIGHT, VIEW_NEUTRAL, VIEW_UNDERWEIGHT = "Overweight", "Neutral", "Underweight"
RISK_LOW, RISK_MEDIUM, RISK_HIGH = "Low", "Medium", "High"
RETURN_POOR, RETURN_FAIR, RETURN_GOOD, RETURN_VERY_GOOD, RETURN_EXCELLENT = (
    "Poor", "Fair", "Good", "Very Good", "Excellent",
)

# Below this many valid price points, the 20/50-day moving averages behind
# the sign-agreement calc are still NaN (or nearly so) and annualized_return/
# sharpe_ratio are numerically unstable -- compute_quant_signals() still
# returns a value (never raises) rather than crash on a thin-history
# ticker, but callers displaying multiple tickers side by side (e.g. ETF
# Analysis's Compare workspace) should disclose that a ticker under this
# threshold has an unreliable Quant Score, not silently present it at the
# same confidence as a fully-populated series.
MIN_RELIABLE_HISTORY_POINTS = 10

# Trend Signal's lookback: ~1 trading quarter, deliberately much shorter
# than the Quant Score's annualized_return window so the two can genuinely
# disagree (e.g. a ticker up strongly over the full selected range but down
# over the last quarter shows Bullish score inputs alongside a Bearish
# Trend Signal -- exactly the "these signals may differ" case the page's
# semantics note discloses, not a bug).
TREND_LOOKBACK_DAYS = 60


def has_sufficient_history(p) -> bool:
    return len(p) >= MIN_RELIABLE_HISTORY_POINTS


def recent_trend_return(p: pd.Series) -> float:
    """Trailing ~TREND_LOOKBACK_DAYS-trading-day price return -- the ONLY
    input to Trend Signal. Distinct from annualized_return(p) (used for the
    Quant Score and the Overview return KPI), which reflects the full
    user-selected date range and can span years.

    Falls back to the longest window the series actually has (down to a
    single period) rather than returning NaN on thin-history tickers --
    has_sufficient_history() already gates whether the result is reliable
    enough to display.
    """
    if len(p) < 2:
        return 0.0
    window = min(TREND_LOOKBACK_DAYS, len(p) - 1)
    value = momentum(p, window).iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def trend_signal_from_return(recent_ret: float) -> str:
    """Same thresholds as the page's KPI-row Trend chip -- this is the ONE
    definition of Trend Signal used everywhere on the page. `recent_ret`
    must be a recent-window return (recent_trend_return()), not the full
    selected-range annualized return."""
    if recent_ret > 0.05:
        return TREND_BULLISH
    if recent_ret < -0.05:
        return TREND_BEARISH
    return TREND_NEUTRAL


def portfolio_view_from_score(score: float) -> str:
    if score >= 65:
        return VIEW_OVERWEIGHT
    if score <= 35:
        return VIEW_UNDERWEIGHT
    return VIEW_NEUTRAL


def risk_level_from_vol(vol: float) -> str:
    """The ONE per-ETF Risk Level classification used everywhere ETF
    Analysis shows a per-ticker risk bucket (Issue #39: before this, the
    Rankings table and the now-removed "ETF Compare Score" table each
    independently thresholded annualized volatility -- 0.15/0.28 vs
    0.12/0.25 -- so the same ETF could show a different Risk Level in each
    table). Preserves the original Rankings thresholds."""
    if vol < 0.15:
        return RISK_LOW
    if vol < 0.28:
        return RISK_MEDIUM
    return RISK_HIGH


def expected_return_label_from_ann_ret(ann_ret: float) -> str:
    """Per-ETF Expected Return label shown alongside risk_level_from_vol()
    in the same canonical table."""
    if ann_ret < 0:
        return RETURN_POOR
    if ann_ret < 0.08:
        return RETURN_FAIR
    if ann_ret < 0.15:
        return RETURN_GOOD
    if ann_ret < 0.25:
        return RETURN_VERY_GOOD
    return RETURN_EXCELLENT


def compute_quant_signals(p: pd.Series, risk_free_rate: float) -> dict:
    """`p` is one ticker's cleaned, already-.dropna()'d price Series.

    Returns the canonical dict every workspace on the ETF Analysis page
    reads from -- score/trend/portfolio_view/signal_agreement plus the raw
    inputs (ret_ann, ret_recent, vol, sharpe, mdd, mom) so callers can build
    their own rule-based commentary without recomputing any of them
    differently.
    """
    ann_ret = annualized_return(p)
    recent_ret = recent_trend_return(p)
    vol = annualized_volatility(p)
    sr = sharpe_ratio(p, risk_free_rate)
    mdd = maximum_drawdown(p)
    ma_short = sma(p, 20).iloc[-1]
    ma_long = sma(p, 50).iloc[-1]
    mom_last = momentum(p, 10).iloc[-1]
    mom = mom_last if pd.notna(mom_last) else 0.0
    price_now = p.iloc[-1]

    score = 50.0
    score += max(-22, min(22, ann_ret * 140))
    score += max(-18, min(18, sr * 11))
    score += max(-12, min(12, mom * 200))
    score -= max(-8, min(22, (vol - 0.15) * 90))
    score -= max(0, min(22, abs(mdd) * 55))
    score = int(round(max(0, min(100, score))))

    trend = trend_signal_from_return(recent_ret)
    portfolio_view = portfolio_view_from_score(score)

    # Signal Agreement: the literal fraction of these four Quant-Score
    # inputs that agree with the score's own direction, expressed as a
    # percentage (0-100) -- NOT a statistical confidence interval, and
    # never rescaled/floored, or the label would stop meaning what it says.
    signs = [
        1 if ann_ret > 0 else (-1 if ann_ret < 0 else 0),
        1 if sr > 0 else (-1 if sr < 0 else 0),
        1 if mom > 0 else (-1 if mom < 0 else 0),
        1 if price_now > ma_long else (-1 if price_now < ma_long else 0),
    ]
    overall_sign = 1 if score >= 50 else -1
    agreement = sum(1 for s in signs if s == overall_sign) / len(signs)
    signal_agreement = round(agreement * 100)

    return {
        "score": score, "trend": trend, "portfolio_view": portfolio_view,
        "signal_agreement": signal_agreement, "ret_ann": ann_ret,
        "ret_recent": recent_ret, "vol": vol,
        "sharpe": sr, "mdd": mdd, "mom": mom, "price": price_now,
        "ma_short": ma_short, "ma_long": ma_long,
    }


# ── OpenAI "AI Interpretation" (Issue #20 section 2C) ───────────────────────
# The LLM is only ever shown the numbers compute_quant_signals() already
# produced -- it explains the tension between Trend Signal / Quant Score /
# Portfolio View in natural language, it never recomputes or invents one of
# them. Grounding rule identical to src/ai_advisor.py.

_INTERPRETATION_SYSTEM_INSTRUCTIONS = """You are a financial analytics explainer embedded in an ETF analytics platform.
You will be given a ticker and a set of ALREADY-COMPUTED deterministic values: annualized return, volatility,
Sharpe ratio, maximum drawdown, momentum, a Quant Score (0-100), a Trend Signal (Bullish/Neutral/Bearish), and a
Portfolio View (Underweight/Neutral/Overweight).

Your job is ONLY to explain, in 2-4 short sentences, why these signals might agree or disagree with each other
(e.g. a bullish recent trend but a middling Quant Score because volatility or drawdown is elevated), and what
that tension means for how a reader should interpret the page. Do NOT invent, recompute, or restate any number
that was not given to you. Do NOT issue a buy/sell/hold instruction or personalized financial advice. Do NOT
introduce any new quantitative claim. If the signals are aligned, say so plainly instead of manufacturing tension.
Respond in the same language as the user message (Traditional Chinese if the input is in Traditional Chinese,
English otherwise)."""


def build_interpretation_prompt(ticker: str, lang: str, window_start, window_end, signals: dict) -> str:
    if lang == "zh-TW":
        return (
            f"標的：{ticker}\n"
            f"資料區間：{window_start} 至 {window_end}\n"
            f"年化報酬率：{signals['ret_ann']:.2%}\n"
            f"近 {TREND_LOOKBACK_DAYS} 個交易日報酬率（Trend Signal 依據）：{signals['ret_recent']:.2%}\n"
            f"年化波動度：{signals['vol']:.2%}\n"
            f"夏普比率：{signals['sharpe']:.2f}\n"
            f"最大回撤：{signals['mdd']:.2%}\n"
            f"動能（10 日）：{signals['mom']:.2%}\n"
            f"Quant Score（量化評分，0-100）：{signals['score']}\n"
            f"Trend Signal（趨勢訊號）：{signals['trend']}\n"
            f"Portfolio View（投資組合觀點）：{signals['portfolio_view']}\n"
            f"Signal Agreement（訊號一致性）：{signals['signal_agreement']}%\n"
            "請用繁體中文，以 2-4 句話說明上述訊號之間為何一致或存在張力，並說明讀者應如何解讀這些差異。"
        )
    return (
        f"Ticker: {ticker}\n"
        f"Data window: {window_start} to {window_end}\n"
        f"Annualized return: {signals['ret_ann']:.2%}\n"
        f"Trailing {TREND_LOOKBACK_DAYS}-trading-day return (basis for Trend Signal): {signals['ret_recent']:.2%}\n"
        f"Annualized volatility: {signals['vol']:.2%}\n"
        f"Sharpe ratio: {signals['sharpe']:.2f}\n"
        f"Maximum drawdown: {signals['mdd']:.2%}\n"
        f"10-day momentum: {signals['mom']:.2%}\n"
        f"Quant Score (0-100): {signals['score']}\n"
        f"Trend Signal: {signals['trend']}\n"
        f"Portfolio View: {signals['portfolio_view']}\n"
        f"Signal Agreement: {signals['signal_agreement']}%\n"
        "In 2-4 sentences, explain why these signals agree or are in tension with each other, "
        "and what that means for how the reader should interpret this page."
    )


def interpretation_fingerprint(ticker: str, lang: str, window_start, window_end, signals: dict) -> str:
    return _fingerprint(
        ticker, lang, window_start, window_end, signals["score"], signals["trend"],
        signals["portfolio_view"], signals["signal_agreement"],
        round(signals["ret_ann"], 4), round(signals["ret_recent"], 4), round(signals["vol"], 4),
        round(signals["sharpe"], 4), round(signals["mdd"], 4), round(signals["mom"], 4),
    )


def generate_etf_interpretation(ticker: str, lang: str, window_start, window_end, signals: dict,
                                 session_state=None) -> dict:
    """Returns {"text": str|None, "source": "ai"|"rule_based", "error": str|None}.

    Caches by interpretation_fingerprint() when `session_state` is given
    (any dict-like -- st.session_state in the app, a plain dict in tests),
    so an unrelated widget rerun never re-spends an OpenAI call, and only
    a material change to the ticker/window/signals triggers a new one.
    """
    prompt = build_interpretation_prompt(ticker, lang, window_start, window_end, signals)
    if session_state is not None:
        fp = interpretation_fingerprint(ticker, lang, window_start, window_end, signals)
        result = cached_generate(
            session_state, f"_etf_ai_interp_cache_{ticker}", fp,
            _INTERPRETATION_SYSTEM_INSTRUCTIONS, prompt, max_output_tokens=400,
        )
    else:
        result = generate_text(_INTERPRETATION_SYSTEM_INSTRUCTIONS, prompt, max_output_tokens=400)

    if result["available"]:
        return {"text": result["text"], "source": "ai", "error": None}
    return {"text": None, "source": "rule_based", "error": result.get("error")}
