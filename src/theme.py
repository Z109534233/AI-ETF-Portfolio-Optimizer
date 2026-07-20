"""
Design System Tokens
Single source of truth for colors, spacing, and chart styling shared by
assets/style.css (via st.markdown), src/charts.py (Plotly), and src/ui.py.
"""

# ── Core Palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg": "#0B1220",            # app background — deep navy
    "bg_alt": "#0E1729",        # secondary background
    "surface": "#131B2E",       # card / panel surface
    "surface_2": "#182238",     # elevated surface (hover, nested cards)
    "border": "#232E45",        # default border
    "border_strong": "#2E3B57", # hover / focus border
    "primary": "#3B82F6",       # brand blue
    "primary_dark": "#2563EB",
    "primary_soft": "rgba(59, 130, 246, 0.12)",
    "success": "#34D399",       # soft green — positive values
    "success_soft": "rgba(52, 211, 153, 0.12)",
    "danger": "#F87171",        # soft red — negative values
    "danger_soft": "rgba(248, 113, 113, 0.12)",
    "warning": "#FBBF24",
    "warning_soft": "rgba(251, 191, 36, 0.12)",
    "purple": "#A78BFA",
    "cyan": "#22D3EE",
    "text": "#F1F5F9",          # primary text — near white
    "text_secondary": "#94A3B8",# secondary / body text
    "text_muted": "#64748B",    # captions, disabled
    "bg_legacy": "#0B1220",     # kept for chart bg compatibility
    "panel": "#131B2E",         # alias used by legacy chart code
    "muted": "#94A3B8",         # alias used by legacy chart code
    "accent": "#FBBF24",        # alias used by legacy chart code
    "pink": "#F472B6",
}

CHART_COLORS = [
    "#3B82F6", "#34D399", "#FBBF24", "#F87171", "#A78BFA",
    "#22D3EE", "#F472B6", "#84CC16", "#FB923C", "#818CF8",
]

# ── Spacing Scale (px) ──────────────────────────────────────────────────────
SPACE = {
    "xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "3xl": 48,
}

# ── Radius / Shadow ─────────────────────────────────────────────────────────
RADIUS = {"sm": "6px", "md": "10px", "lg": "14px", "xl": "18px"}

SHADOW = {
    "sm": "0 1px 2px rgba(0,0,0,0.24)",
    "md": "0 4px 16px rgba(0,0,0,0.28)",
    "lg": "0 12px 32px rgba(0,0,0,0.36)",
    "glow": "0 0 0 1px rgba(59,130,246,0.4), 0 8px 24px rgba(59,130,246,0.15)",
}

FONT_FAMILY = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"


def color_for(value: float, positive_is_good: bool = True) -> str:
    """Return the design-system green/red for a signed metric value."""
    good = value >= 0 if positive_is_good else value < 0
    return COLORS["success"] if good else COLORS["danger"]


# Simple inline icon paths (24x24 viewBox, Feather-style strokes).
ICONS = {
    "dollar": '<path d="M12 1v22M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "trending-down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "shield": '<path d="M12 2 3 6v6c0 5 3.8 9.4 9 10 5.2-.6 9-5 9-10V6z"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "bar-chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "pie-chart": '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/>',
}


def icon_svg(name: str, size: int = 18, color: str = None) -> str:
    """Return an inline SVG icon string for use inside markdown HTML."""
    color = color or COLORS["text_secondary"]
    body = ICONS.get(name, ICONS["activity"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
