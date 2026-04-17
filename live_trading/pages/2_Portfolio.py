import os
import sys

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_LT_DIR    = os.path.dirname(_PAGES_DIR)
if _LT_DIR not in sys.path:
    sys.path.insert(0, _LT_DIR)

import streamlit as st
from utils.data_loader import load_trades, build_trade_pairs, load_config
from utils.charts import equity_curve_chart, drawdown_chart

st.set_page_config(page_title="Portfolio", layout="wide")
st.title("Portfolio")

pairs = build_trade_pairs()
st.write(f"{len(pairs['closed'])} closed trades available for analysis")
