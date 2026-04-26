"""
Shared CSS for all Epsilon Fund Streamlit pages.
Matches streamlit_app.py exactly — do not duplicate there.

Usage:
    from utils.styles import apply_styles
    apply_styles()
"""

_CSS = """
<style>
  /* Base font */
  html, body, [class*="css"] { font-size: 13px !important; }

  /* Page background */
  .stApp            { background-color: #f8f8f7 !important; }
  /* Force wide layout via CSS so the page never reverts to Streamlit's
     centered ~46rem max-width on auto-refresh / rerun, even if
     set_page_config(layout="wide") fails to re-apply for any reason. */
  .block-container,
  [data-testid="stMainBlockContainer"] {
      background-color: #f8f8f7 !important;
      padding-top: 5rem !important;
      padding-bottom: 2rem !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      max-width: 100% !important;
  }

  /* Remove horizontal padding that Streamlit injects inside tab panels so
     content inside tabs fills the same full width as standalone pages. */
  [data-baseweb="tab-panel"] {
      padding-left: 0 !important;
      padding-right: 0 !important;
  }

  /* Expander header typography — compact hint style */
  .streamlit-expanderHeader {
      font-size: 11px !important;
      font-weight: 400 !important;
      letter-spacing: 0 !important;
      text-transform: none !important;
      color: #888780 !important;
  }

  /* Card wrapper */
  .dashboard-card {
      background: white;
      border: 1px solid #d3d1c7;
      border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      padding: 0;
      margin-bottom: 14px;
      overflow: hidden;
  }
  .card-header {
      padding: 10px 16px 8px;
      border-bottom: 1px solid #d3d1c7;
      display: flex; align-items: center; gap: 12px;
  }
  .section-label {
      font-size: 11px !important; font-weight: 500;
      letter-spacing: 0.06em; text-transform: uppercase;
      color: #444441; margin: 0;
  }
  .section-note { font-size: 11px; color: #888780; }

  /* Base table */
  table { font-size: 13px !important; }
  .dash-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .dash-table th {
      font-size: 11px !important; font-weight: 500; letter-spacing: 0.05em;
      text-transform: uppercase; color: #444441;
      padding: 7px 12px 5px; text-align: center !important;
      border-bottom: 1px solid #d3d1c7; white-space: nowrap;
  }
  .dash-table th.r { text-align: center !important; }
  .dash-table th.c { text-align: center !important; }
  .dash-table td {
      padding: 8px 12px; border-bottom: 1px solid #e4e4e1;
      vertical-align: middle; text-align: center !important;
  }
  .dash-table tr:last-child td { border-bottom: none; }
  .dash-table td.r { text-align: center !important; }
  .dash-table td.c { text-align: center !important; }
  .dash-table td[style] { text-align: center !important; }
  .dash-table th[style] { text-align: center !important; }

  /* Remove vertical column lines */
  .dash-table td, .dash-table th {
      border-left: none !important;
      border-right: none !important;
  }
  [data-testid="stExpander"] .dash-table td.field-label {
      border-right: 1px solid #d3d1c7 !important;
  }

  /* Grey shading on first column for labeled tables */
  .dash-table-labeled td:first-child {
      background: #fafaf9 !important;
      font-weight: 500;
      border-right: 1px solid #d3d1c7 !important;
  }

  /* Asset cell */
  .asset-name  { font-weight: 500; font-size: 12px; }
  .asset-alloc { font-size: 11px; color: #888780; }

  /* Decision row tints */
  .row-ENTRY td { background: #f0f8e8 !important; }
  .row-EXIT  td { background: #fdeaea !important; }

  /* Decision badges */
  .badge {
      display: inline-block; padding: 2px 10px; border-radius: 4px;
      font-size: 11px; font-weight: 500; white-space: nowrap;
  }
  .badge-ENTRY { background: #EAF3DE; color: #3B6D11; }
  .badge-HOLD  { background: #FAEEDA; color: #854F0B; }
  .badge-EXIT  { background: #FCEBEB; color: #A32D2D; }
  .badge-FLAT  { background: #F1EFE8; color: #5F5E5A; }

  /* Bool text */
  .t { color: #1a5c2a; }
  .f { color: #8a1a1a; }

  .caution { background: #FAEEDA; color: #854F0B; font-weight: 500; }
  .entry-t { background: #EAF3DE; color: #3B6D11; font-weight: 600; }
  .entry-f { background: #FCEBEB; color: #A32D2D; font-weight: 600; }

  .stop-up   { color: #1a5c2a; font-size: 11px; font-weight: 600; }
  .stop-prev { color: #888780; font-size: 11px; }

  .dash-table td.field-label {
      font-size: 12px; font-weight: 500; color: #888780;
      white-space: nowrap; background: #fafaf9;
      border-right: 1px solid #d3d1c7;
      width: 180px; min-width: 180px; max-width: 180px;
      text-align: center !important;
  }
  .divider-row td {
      background: #f5f5f3; font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.07em;
      color: #888780; padding: 4px 12px;
      border-bottom: 1px solid #d3d1c7;
  }

  .badge-fixed {
      background: #e4e4e1; color: #5F5E5A; font-size: 9px;
      padding: 1px 4px; border-radius: 3px; margin-left: 5px;
      display: inline-block; vertical-align: middle;
  }

  .badge-ent_normal  { background: #EAF3DE; color: #3B6D11; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_caution { background: #FCEBEB; color: #A32D2D; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-ent_both    { background: #FAEEDA; color: #854F0B; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_normal  { background: #EAF3DE; color: #3B6D11; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }
  .badge-pos_caution { background: #FCEBEB; color: #A32D2D; display:inline-block; padding:2px 7px; border-radius:3px; font-size:11px; font-weight:600; }

  .upd-yes { color: #1a5c2a; font-weight: 600; }
  .upd-no  { color: #888780; }

  .dash-meta {
      font-size: 12px; color: #888780;
      margin-bottom: 18px; line-height: 1.8;
  }
  .dash-meta strong { color: #444441; font-weight: 500; }

  .table-scroll { overflow-x: auto; }

  .row-total td {
      background: #f0efea !important;
      font-weight: 600;
      border-top: 2px solid #d3d1c7 !important;
  }

  .form-card {
      background: white; border: 1px solid #d3d1c7; border-radius: 6px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 14px 16px 10px;
      margin-bottom: 14px;
  }
  .form-card-header {
      font-size: 12px; font-weight: 600; color: #444441;
      margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
  }

  /* ── Journal-specific additions ───────────────────────────────────────── */

  /* Stats table — left-aligned label column */
  .stats-table {
      width: 100%; border-collapse: collapse;
      font-size: 13px; font-variant-numeric: tabular-nums;
  }
  .stats-table td {
      padding: 6px 16px;
      border-bottom: 1px solid #e4e4e1;
      vertical-align: middle;
  }
  .stats-table tr:last-child td { border-bottom: none; }
  .stats-table td.stat-label {
      font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
      text-transform: uppercase; color: #888780;
      width: 55%; white-space: nowrap;
  }
  .stats-table td.stat-value {
      font-weight: 500; color: #444441; text-align: right;
  }
  .stat-pos { color: #3B6D11 !important; }
  .stat-neg { color: #A32D2D !important; }
  .stat-amb { color: #854F0B !important; }   /* amber — MAE warning */

  /* "no data" muted message inside a section */
  .no-data-msg {
      font-size: 12px; color: #888780; font-style: italic;
      padding: 16px 0;
  }

  /* Legacy-data warning icon inline */
  .legacy-warn { color: #854F0B; font-size: 11px; }
</style>
"""


def apply_styles() -> None:
    """Inject the shared CSS block. Call once at the top of each page."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
