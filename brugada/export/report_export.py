from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import io
import math
import re
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from brugada.ui.components import (
    DEFAULT_LEAD_NAMES,
    _plot_12_lead,
    _plot_decision_margin,
    _plot_evidence_heatmap,
)


def _result_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        candidate = float(value)
    except Exception:  # noqa: BLE001
        return default
    if math.isnan(candidate) or math.isinf(candidate):
        return default
    return candidate


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = _safe_str(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _build_evidence_summary(result: Any) -> str:
    clinician_explain = _result_get(result, "clinician_explain", {})
    if not isinstance(clinician_explain, dict):
        return "S0/M0/W0"

    counts = clinician_explain.get("evidence_counts", {})
    if not isinstance(counts, dict):
        return "S0/M0/W0"

    def to_int(v):
        try:
            return int(float(v))
        except Exception:  # noqa: BLE001
            return 0

    strong = to_int(counts.get("strong", 0))
    moderate = to_int(counts.get("moderate", 0))
    weak = to_int(counts.get("weak", 0))
    return f"S{strong}/M{moderate}/W{weak}"


def _sanitize_filename(name: str, fallback: str = "record") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", _safe_str(name, fallback))
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _html_table_from_rows(columns: list[str], rows: list[list[str]]) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _build_html_document(title: str, body: str) -> str:
    escaped_title = html.escape(title)
    return f"""
<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{escaped_title}</title>
<style>
:root {{
  --ink: #1e293b;
  --muted: #475569;
  --card: #ffffff;
  --bg: #f8fafc;
  --border: #cbd5e1;
  --accent: #0f766e;
}}
body {{
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  color: var(--ink);
  background: var(--bg);
  margin: 0;
  padding: 1rem;
}}
.wrap {{
  max-width: 1120px;
  margin: 0 auto;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.9rem;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.7rem;
}}
.metric {{
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem;
  background: #f8fafc;
}}
.metric .label {{
  font-size: 0.82rem;
  color: var(--muted);
}}
.metric .value {{
  font-size: 1.35rem;
  font-weight: 700;
}}
h1, h2, h3 {{
  margin: 0.2rem 0 0.6rem 0;
}}
h1 {{ font-size: 1.35rem; }}
h2 {{ font-size: 1.1rem; }}
h3 {{ font-size: 0.98rem; }}
small {{ color: var(--muted); }}
img {{
  max-width: 100%;
  border: 1px solid var(--border);
  border-radius: 8px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.87rem;
}}
th, td {{
  border: 1px solid var(--border);
  padding: 0.4rem 0.5rem;
  text-align: left;
  vertical-align: top;
}}
thead {{ background: #f1f5f9; }}
ul {{ margin: 0.4rem 0 0.1rem 1.1rem; }}
@media print {{
  body {{ padding: 0; background: white; }}
  .card {{ break-inside: avoid; page-break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class=\"wrap\">
{body}
</div>
</body>
</html>
""".strip()


def build_single_case_html_report(
    result: Any,
    record_name: str | None = None,
    patient_id: str | None = None,
    generated_at_utc: str | None = None,
) -> str:
    label = _safe_str(_result_get(result, "label", "Unknown"), "Unknown")

    probability_raw = _safe_float(_result_get(result, "probability", 0.0), 0.0)
    probability_display = _safe_float(_result_get(result, "display_probability", probability_raw), probability_raw)

    threshold_raw = _safe_float(_result_get(result, "decision_threshold", 0.05), 0.05)
    threshold_display = _safe_float(_result_get(result, "display_threshold", 0.35), 0.35)

    stability_raw = _safe_float(_result_get(result, "decision_stability", 0.0), 0.0)
    stability_display = _safe_float(_result_get(result, "display_decision_stability", stability_raw), stability_raw)

    class_support = _safe_float(_result_get(result, "class_support", 0.0), 0.0)
    gray_zone = _safe_bool(_result_get(result, "gray_zone", False), False)

    clinician_explain = _result_get(result, "clinician_explain", {})
    if not isinstance(clinician_explain, dict):
        clinician_explain = {}

    recommendation_tier = _safe_str(
        clinician_explain.get("recommendation_tier", _result_get(result, "recommendation_tier", "")),
        "routine_clinical_correlation",
    )
    recommendation_text = _safe_str(
        clinician_explain.get("recommendation_text", _result_get(result, "recommendation_text", "")),
        "Clinical correlation is recommended.",
    )

    evidence_summary = _safe_str(_result_get(result, "evidence_summary", ""), "")
    if not evidence_summary:
        evidence_summary = _build_evidence_summary(result)

    explanation = _safe_str(
        _result_get(result, "explanation", "Model prediction generated. Clinical correlation is recommended."),
        "Model prediction generated. Clinical correlation is recommended.",
    )

    clinical_evidence = _result_get(result, "clinical_evidence", [])
    if not isinstance(clinical_evidence, list):
        clinical_evidence = []
    clinical_evidence = [item for item in clinical_evidence if isinstance(item, dict)]

    model_contributions = _result_get(result, "model_contributions", {})
    if not isinstance(model_contributions, dict):
        model_contributions = {}

    signal_plot = _result_get(result, "signal_for_plot", _result_get(result, "signal", None))
    fs_plot = _safe_float(_result_get(result, "fs", 500.0), 500.0)
    lead_names = _result_get(result, "lead_names", DEFAULT_LEAD_NAMES)
    if not isinstance(lead_names, list):
        lead_names = list(DEFAULT_LEAD_NAMES)
    highlights = _result_get(result, "highlighted_segments", {})
    if not isinstance(highlights, dict):
        highlights = {}

    display_record_name = _safe_str(record_name, "") or _safe_str(
        _result_get(result, "record_name", _result_get(result, "record", "ECG record")),
        "ECG record",
    )
    display_patient_id = _safe_str(patient_id, "") or _safe_str(_result_get(result, "patient_id", ""), "N/A")
    display_record_uid = _safe_str(_result_get(result, "record_uid", ""), "N/A")
    generated = generated_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    risk_pct = probability_display * 100.0
    confidence_pct = _safe_float(_result_get(result, "display_confidence", _result_get(result, "confidence", 0.0)), 0.0)

    margin_fig = _plot_decision_margin(probability_display, threshold_display, _result_get(result, "display_gray_zone_upper", None))
    margin_b64 = _fig_to_base64(margin_fig)

    heatmap_b64 = ""
    if clinical_evidence:
        evidence_df = pd.DataFrame(clinical_evidence)
        heatmap_fig = _plot_evidence_heatmap(evidence_df)
        heatmap_b64 = _fig_to_base64(heatmap_fig)

    ecg_b64 = ""
    if signal_plot is not None:
        try:
            ecg_fig = _plot_12_lead(signal_plot, lead_names, fs_plot, highlights)
            ecg_b64 = _fig_to_base64(ecg_fig)
        except Exception:  # noqa: BLE001
            ecg_b64 = ""

    evidence_rows = []
    for item in clinical_evidence:
        evidence_rows.append(
            [
                _safe_str(item.get("lead", ""), ""),
                _safe_str(item.get("tier", ""), ""),
                _safe_str(item.get("reliability", ""), ""),
                f"{_safe_float(item.get('score', 0.0), 0.0):.3f}",
                str(_safe_int_like(item.get("segments", 0))),
            ]
        )

    contrib_rows = [
        [view, f"{_safe_float(weight, 0.0):.2f}%"]
        for view, weight in sorted(model_contributions.items(), key=lambda item: _safe_float(item[1], 0.0), reverse=True)
    ]

    actions = clinician_explain.get("next_actions", [])
    if not isinstance(actions, list):
        actions = []

    metrics_html = f"""
<div class=\"card\">
  <h2>Diagnostic Snapshot</h2>
  <div class=\"grid\">
    <div class=\"metric\"><div class=\"label\">Brugada Risk Score</div><div class=\"value\">{risk_pct:.2f}%</div></div>
    <div class=\"metric\"><div class=\"label\">Decision Confidence</div><div class=\"value\">{confidence_pct:.1f}%</div></div>
    <div class=\"metric\"><div class=\"label\">Prediction Support</div><div class=\"value\">{class_support:.1f}%</div></div>
    <div class=\"metric\"><div class=\"label\">Decision Stability</div><div class=\"value\">{stability_display:.2f} pp</div></div>
  </div>
</div>
"""

    evidence_table_html = ""
    if evidence_rows:
        evidence_table_html = _html_table_from_rows(
            ["Lead", "Tier", "Reliability", "Score", "Segments"],
            evidence_rows,
        )

    contrib_table_html = ""
    if contrib_rows:
        contrib_table_html = _html_table_from_rows(["Model View", "Contribution"], contrib_rows)

    actions_html = "".join(f"<li>{html.escape(_safe_str(item, ''))}</li>" for item in actions if _safe_str(item, ""))
    if not actions_html:
        actions_html = "<li>Clinical correlation is recommended.</li>"

    body = f"""
<div class=\"card\">
  <h1>Brugada Clinical Report</h1>
  <small>Generated at {html.escape(generated)}</small>
  <div class=\"grid\" style=\"margin-top:0.6rem;\">
    <div><strong>Record:</strong> {html.escape(display_record_name)}</div>
    <div><strong>Patient ID:</strong> {html.escape(display_patient_id)}</div>
    <div><strong>Record UID:</strong> {html.escape(display_record_uid)}</div>
    <div><strong>Label:</strong> {html.escape(label)}</div>
    <div><strong>Recommendation Tier:</strong> {html.escape(recommendation_tier)}</div>
    <div><strong>Evidence Summary:</strong> {html.escape(evidence_summary)}</div>
    <div><strong>Gray Zone:</strong> {"Yes" if gray_zone else "No"}</div>
    <div><strong>Raw Decision Threshold:</strong> {threshold_raw:.3f}</div>
    <div><strong>Display Threshold:</strong> {threshold_display:.3f}</div>
  </div>
</div>

{metrics_html}

<div class=\"card\">
  <h2>Recommendation</h2>
  <p>{html.escape(recommendation_text)}</p>
  <h3>Recommended Next Actions</h3>
  <ul>{actions_html}</ul>
</div>

<div class=\"card\">
  <h2>Interpretation Narrative</h2>
  <p>{html.escape(explanation)}</p>
</div>

<div class=\"card\">
  <h2>Decision Margin Figure</h2>
  <img alt=\"Decision margin\" src=\"data:image/png;base64,{margin_b64}\" />
</div>
"""

    if heatmap_b64:
        body += f"""
<div class=\"card\">
  <h2>V1-V3 Morphology Evidence Heatmap</h2>
  <img alt=\"Evidence heatmap\" src=\"data:image/png;base64,{heatmap_b64}\" />
</div>
"""

    if evidence_table_html:
        body += f"""
<div class=\"card\">
  <h2>Clinical Evidence Table</h2>
  {evidence_table_html}
</div>
"""

    if contrib_table_html:
        body += f"""
<div class=\"card\">
  <h2>Model Contribution Summary</h2>
  {contrib_table_html}
</div>
"""

    if ecg_b64:
        body += f"""
<div class=\"card\">
  <h2>12-lead ECG Overview</h2>
  <img alt=\"12 lead ECG\" src=\"data:image/png;base64,{ecg_b64}\" />
</div>
"""

    return _build_html_document(
        title=f"Brugada Clinical Report - {display_record_name}",
        body=body,
    )


def _safe_int_like(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return 0


def build_batch_html_zip(batch_results: list[dict], batch_name: str = "batch_reports") -> bytes:
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_rows = []

        for idx, item in enumerate(batch_results, start=1):
            if not isinstance(item, dict):
                continue

            raw = item.get("raw", item)
            record_name = _safe_str(item.get("record", f"record_{idx}"), f"record_{idx}")

            report_html = build_single_case_html_report(
                result=raw,
                record_name=record_name,
            )
            file_stem = _sanitize_filename(record_name, fallback=f"record_{idx}")
            file_name = f"{idx:03d}_{file_stem}.html"
            zf.writestr(file_name, report_html)

            label = _safe_str(_result_get(raw, "label", _result_get(item, "label", "Unknown")), "Unknown")
            prob = _safe_float(
                _result_get(raw, "display_probability", _result_get(item, "probability", 0.0)),
                0.0,
            )
            manifest_rows.append((file_name, record_name, label, f"{prob * 100.0:.2f}%"))

        if manifest_rows:
            manifest_lines = ["file_name,record_name,label,display_probability"]
            for row in manifest_rows:
                safe = [cell.replace(",", " ") for cell in row]
                manifest_lines.append(",".join(safe))
            zf.writestr("manifest.csv", "\n".join(manifest_lines))

    buffer.seek(0)
    return buffer.getvalue()
