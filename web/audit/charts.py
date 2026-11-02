"""Vega-Lite specifications for the dashboard's charts, rendered in the browser by vega-embed.
The same charts as the Streamlit dashboard, written as plain dictionaries so the web
application needs no plotting library."""
from __future__ import annotations

from collections import Counter

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
RED, BLUE, GREEN = "#e5484d", "#3aa8f0", "#13914f"  # the design palette (static/src/app.css)


def _spec(values: list[dict], **spec) -> dict:
    return {"$schema": SCHEMA, "width": "container", "data": {"values": values},
            "config": {"view": {"stroke": None}, "axis": {"labelLimit": 220}}, **spec}


def token(name: str) -> str:
    """A colour named by its design token; static/js/charts.js swaps in the token's value for
    the current theme, so a category keeps its meaning (and contrast) in light and dark."""
    return f"token(--{name})"


def donut(segments: list[dict], field: str = "label") -> dict:
    """segments: [{"label", "reports", "color" (a token name)}]. Colours are fixed per category,
    not taken from the palette by order, and the page lists the same segments with counts, so
    the chart never relies on colour alone."""
    values = [{field: s["label"], "reports": s["reports"]} for s in segments if s["reports"]]
    scale = {"domain": [s["label"] for s in segments], "range": [token(s["color"]) for s in segments]}
    return _spec(values, height=200, mark={"type": "arc", "innerRadius": 62, "cornerRadius": 4, "padAngle": 0.02},
                 encoding={"theta": {"field": "reports", "type": "quantitative"},
                           "color": {"field": field, "type": "nominal", "scale": scale, "legend": None},
                           "tooltip": [{"field": field, "title": "Category"}, {"field": "reports", "title": "Reports"}]})


def horizontal_bars(values: list[dict], y: str, x: str, x_title: str, tooltip: list | None = None,
                    x_format: str | None = None, domain: list | None = None, color: dict | None = None,
                    sort: str | None = "-x", log: bool = False, height: int | None = None) -> dict:
    x_enc = {"field": x, "type": "quantitative", "title": x_title}
    if x_format:
        x_enc["axis"] = {"format": x_format}
    if domain or log:
        x_enc["scale"] = {**({"domain": domain} if domain else {}), **({"type": "symlog"} if log else {})}
    encoding = {"x": x_enc, "y": {"field": y, "type": "nominal", "sort": sort, "title": None},
                "tooltip": tooltip or [{"field": y}, {"field": x}]}
    if color:
        encoding["color"] = color
    return _spec(values, height=height or max(120, 26 * len(values)), mark="bar", encoding=encoding)


def anomaly_histogram(values: list[dict]) -> dict:
    return _spec(values, height=220, mark="bar", encoding={
        "x": {"field": "anomaly_score", "type": "quantitative", "bin": {"maxbins": 20}, "title": "Anomaly score"},
        "y": {"aggregate": "count", "type": "quantitative", "title": "Claims"},
        "color": {"field": "verdict", "type": "nominal", "title": "Verdict", "legend": {"orient": "top"},
                  "scale": {"domain": ["inconsistent", "consistent", "none"],
                            "range": [token("red"), token("sky"), token("faint")]}},
        "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "claims"}]})


def shap_frequency(attributions: list[dict]) -> dict:
    """How often each feature is among a claim's top 3 SHAP attributions."""
    counts, magnitude = Counter(), Counter()
    for attribution in attributions:
        for feature, value in (attribution or {}).items():
            counts[feature] += 1
            magnitude[feature] += abs(value)
    values = [{"feature": f, "claims": n, "mean_abs_shap": round(magnitude[f] / n, 4)} for f, n in counts.items()]
    return horizontal_bars(values, "feature", "claims", "Claims",
                           tooltip=[{"field": "feature"}, {"field": "claims"},
                                    {"field": "mean_abs_shap", "format": ".3f", "title": "mean |SHAP|"}])


def shap_contributions(attribution: dict) -> dict:
    """One claim's top contributions: red pushes towards anomalous, blue away."""
    values = [{"feature": f, "shap": round(v, 4)} for f, v in attribution.items()]
    return horizontal_bars(values, "feature", "shap", "Contribution to the anomaly score", height=120,
                           color={"condition": {"test": "datum.shap > 0", "value": RED}, "value": BLUE},
                           tooltip=[{"field": "feature"}, {"field": "shap", "format": ".3f"}])


def funnel(summary: dict) -> dict:
    values = [{"stage": "Screened", "claims": summary["claims"]},
              {"stage": "Flagged (stage 1)", "claims": summary["flagged_stage1"]},
              {"stage": "Inconsistent (classifier)", "claims": summary["inconsistent"]},
              {"stage": "Narrative (LLM)", "claims": summary["narratives"]}]
    return horizontal_bars(values, "stage", "claims", "Claims (log scale)", sort=None, log=True)


def share_by_typology(rows: list[dict], title: str, color: dict | None = None) -> dict:
    return horizontal_bars(rows, "typology", "share", title, x_format="%", domain=[0, 1], color=color,
                           tooltip=[{"field": "typology"}, {"field": "flagged"}, {"field": "claims"}])


SHOULD_FLAG_COLOR = {"field": "should be flagged", "type": "nominal",
                     "scale": {"domain": ["yes", "no"], "range": [GREEN, RED]}}
