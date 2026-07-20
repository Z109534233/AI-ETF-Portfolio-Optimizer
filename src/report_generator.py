"""
Report Generator Module
Generates professional PDF reports using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# Color palette
DARK_BLUE = colors.HexColor("#0B1220")
NAVY = colors.HexColor("#111827")
ACCENT_BLUE = colors.HexColor("#3B82F6")
LIGHT_BLUE = colors.HexColor("#DBEAFE")
WHITE = colors.white
GRAY = colors.HexColor("#6B7280")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
GREEN = colors.HexColor("#10B981")
RED = colors.HexColor("#EF4444")


def generate_portfolio_report(
    portfolio_name: str,
    weights: dict,
    metrics: dict,
    investment_amount: float = 10000.0,
    optimization_method: str = "N/A",
    notes: str = ""
) -> bytes:
    """
    Generate a professional PDF portfolio report.
    Returns the PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=ACCENT_BLUE,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=GRAY,
        spaceAfter=4,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=ACCENT_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
        borderPad=4
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=4,
        leading=14
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=GRAY,
        spaceAfter=4,
        leading=12,
        alignment=TA_CENTER
    )

    # Header
    story.append(Paragraph("AI ETF Portfolio Optimizer", title_style))
    story.append(Paragraph("Portfolio Analysis Report", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M UTC')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_BLUE, spaceAfter=12))

    # Portfolio Overview
    story.append(Paragraph("Portfolio Overview", heading_style))
    overview_data = [
        ["Portfolio Name", portfolio_name],
        ["Optimization Method", optimization_method],
        ["Investment Amount", f"${investment_amount:,.2f}"],
        ["Number of Holdings", str(len(weights))],
        ["Report Date", datetime.now().strftime("%d %B %Y")],
    ]
    overview_table = Table(overview_data, colWidths=[6 * cm, 10 * cm])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.4 * cm))

    # Portfolio Allocation
    story.append(Paragraph("Portfolio Allocation", heading_style))
    alloc_header = ["Ticker", "Weight (%)", "Allocation ($)"]
    alloc_rows = [alloc_header]
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        alloc_rows.append([
            ticker,
            f"{weight:.2%}",
            f"${weight * investment_amount:,.2f}"
        ])
    # Total row
    alloc_rows.append(["TOTAL", "100.00%", f"${investment_amount:,.2f}"])

    alloc_table = Table(alloc_rows, colWidths=[4 * cm, 5 * cm, 7 * cm])
    alloc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [WHITE, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(alloc_table)
    story.append(Spacer(1, 0.4 * cm))

    # Performance Metrics
    story.append(Paragraph("Performance Metrics", heading_style))
    metric_rows = [["Metric", "Value"]]
    key_metrics = [
        "Expected Annual Return", "Expected Annual Volatility", "Sharpe Ratio",
        "Maximum Drawdown", "Sortino Ratio", "Calmar Ratio",
        "VaR (95%)", "CVaR (95%)", "Beta", "Alpha",
        "Annualized Return", "Annualized Volatility",
        "Total Return", "Diversification Ratio",
    ]
    for key in key_metrics:
        if key in metrics:
            metric_rows.append([key, str(metrics[key])])

    if len(metric_rows) > 1:
        metrics_table = Table(metric_rows, colWidths=[9 * cm, 7 * cm])
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(metrics_table)

    # Notes
    if notes:
        story.append(Paragraph("Notes", heading_style))
        story.append(Paragraph(notes, body_style))

    # Risk Summary
    story.append(Paragraph("Risk Summary", heading_style))
    risk_text = (
        "This portfolio analysis is based on historical data and quantitative optimization models. "
        "Portfolio metrics such as expected return, volatility, and Sharpe Ratio are derived from "
        "historical price data and do not guarantee future performance. Investors should consider "
        "their personal risk tolerance, investment horizon, and financial circumstances before "
        "making any investment decisions."
    )
    story.append(Paragraph(risk_text, body_style))

    # Disclaimer
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY, spaceAfter=8))
    disclaimer_text = (
        "EDUCATIONAL DISCLAIMER: This report is generated by the AI ETF Portfolio Optimizer for "
        "educational and demonstration purposes only. It does not constitute financial advice, "
        "investment recommendations, or a solicitation to buy or sell any securities. "
        "Past performance is not indicative of future results. Always consult a qualified "
        "financial adviser before making investment decisions. "
        "AI ETF Portfolio Optimizer — Portfolio Project for Academic Purposes."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
