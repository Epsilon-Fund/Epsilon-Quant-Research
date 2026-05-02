"""
Centralised data loading for all Streamlit dashboards.

Every public function accepts a ``data_dir`` argument — the directory that
contains the dashboard's trades.json, positions.json, live_params.json,
mae_cache.json, and config.py.  Pass it as:

    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    trades   = load_trades(DATA_DIR)

This lets the same shared module serve momentum/, statarb/, bbbreakout/,
and any future dashboard without duplicating file-read or config logic.
"""

import importlib.util
import json
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd

# ── Optional Streamlit cache (no-op when run outside a Streamlit server) ──────
try:
    import streamlit as _st
    _cache_data = _st.cache_data(ttl=300)
except Exception:
    _cache_data = lambda f: f  # passthrough in plain Python scripts

# ── Path setup (for binance_client in compute_mae) ────────────────────────────
_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))   # live_trading/shared/
_LT_DIR     = os.path.dirname(_SHARED_DIR)                  # live_trading/
_ROOT       = os.path.dirname(_LT_DIR)                      # repo root
_INFRA_DATA = os.path.join(_ROOT, 'infrastructure', 'data')
if _INFRA_DATA not in sys.path:
    sys.path.append(_INFRA_DATA)

# ── FIFO position_id regex ─────────────────────────────────────────────────────
_FIFO_PID_RE = re.compile(r'^([A-Z0-9]+)_\d{8}_\d{3}$')


def _pid_to_symbol(pid: str) -> str:
    """Extract symbol from a FIFO position_id ('BTCUSDT_20260415_001' -> 'BTCUSDT').
    If pid is already a plain symbol returns it unchanged."""
    m = _FIFO_PID_RE.match(pid)
    return m.group(1) if m else pid


# ── Config loader (importlib, cached per data_dir) ────────────────────────────

_CONFIG_CACHE: dict = {}


def _load_config(data_dir: str):
    """Load and cache config.py from data_dir, returning the module object."""
    abs_dir = os.path.abspath(data_dir)
    if abs_dir not in _CONFIG_CACHE:
        config_path = os.path.join(abs_dir, 'config.py')
        spec = importlib.util.spec_from_file_location(
            f'_cfg_{abs_dir.replace(os.sep, "_")}', config_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONFIG_CACHE[abs_dir] = mod
    return _CONFIG_CACHE[abs_dir]


# ── Low-level file helpers ────────────────────────────────────────────────────

def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        raw = f.read().strip()
    return json.loads(raw) if raw else default


def _save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Schema normalisation ──────────────────────────────────────────────────────

def _leverage(t: dict, key_new: str, key_old: str) -> float:
    """Return leverage from new key, falling back to old key. Never returns None."""
    for k in (key_new, key_old):
        v = t.get(k)
        if v is not None:
            return float(v)
    return 0.0


def _normalize_trade(t: dict) -> dict:
    """
    Unify three historical schemas into one consistent dict:
      v1 — no position_id, uses 'asset'; size in 'actual_size_pct'
      v2 — has position_id; size still in 'actual_size_pct'
      v3 — has position_id; size in 'actual_leverage' (current _write_trade)
    """
    position_id = t.get('position_id') or t.get('asset', '')

    date_str = t.get('date', '')
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception:
        parsed_date = None

    return {
        'position_id':          position_id,
        'action':               t.get('action', ''),
        'strategy':             t.get('strategy', ''),
        'date':                 parsed_date,
        'actual_price':         float(t.get('actual_price', 0)),
        'theoretical_price':    float(t.get('theoretical_price', 0)),
        'actual_leverage':      _leverage(t, 'actual_leverage',      'actual_size_pct'),
        'theoretical_leverage': _leverage(t, 'theoretical_leverage', 'theoretical_size_pct'),
        'slippage_pct':         float(t.get('slippage_pct', 0)),
        'theoretical_stop':     t.get('theoretical_stop'),
        'discretion_note':      t.get('discretion_note'),
        # Trade direction — 'long' / 'short'.  Defaults to 'long' for legacy
        # records and long-only strategies (momentum).  Critical for PnL sign
        # in build_trade_pairs() — a missing direction silently turns a
        # losing short into a winning one.
        'direction':            (t.get('direction') or 'long'),
        # ── new fields (null for legacy records) ──────────────────────────────
        'entry_close':          t.get('entry_close'),
        'exit_close':           t.get('exit_close'),
        'exit_reason':          t.get('exit_reason'),
        'signal_snapshot':      t.get('signal_snapshot'),
        # ── entry classification (explicit or derived) ────────────────────────
        'entry_type':           t.get('entry_type'),   # 'Strategy' | 'Discretionary' | None
        # ── capital snapshot (frozen at entry time; None for legacy records) ───
        'coin_capital':         t.get('coin_capital'),
        'size_usd':             t.get('size_usd'),
        'capital_total':        t.get('capital_total'),
        'coin_weight':          t.get('coin_weight'),
    }


# ── Public loaders ────────────────────────────────────────────────────────────

def load_trades(data_dir: str) -> list:
    """
    Return all trade entries as a list of normalised dicts,
    sorted by date ascending (oldest first).
    """
    path = os.path.join(data_dir, 'trades.json')
    raw  = _read_json(path, [])
    trades = [_normalize_trade(t) for t in raw]
    trades.sort(key=lambda t: t['date'] or datetime.min.date())
    return trades


def load_positions(data_dir: str) -> dict:
    """Return positions.json as a dict keyed by position_id."""
    return _read_json(os.path.join(data_dir, 'positions.json'), {})


def load_live_params(data_dir: str) -> dict:
    """Return live_params.json as a dict."""
    return _read_json(os.path.join(data_dir, 'live_params.json'), {})


def load_realised_capital(data_dir: str) -> float:
    """
    Return the current realised capital, ALWAYS derived from trades.json.

    realised_capital = config CAPITAL + sum of every closed trade's P&L.

    Computing this from trade history (rather than reading a separately-
    maintained realised_capital.json) means the value can never drift from
    the actual closed-trade record — including for trades that pre-date
    the realised-capital tracking system, or trades pulled in via a git
    merge of trades.json.

    Falls back to config CAPITAL if trades.json is unreadable or empty.
    Never returns 0 or None.
    """
    try:
        capital = float(_load_config(data_dir).CAPITAL)
    except Exception:
        capital = 100_000.0
    try:
        pairs      = build_trade_pairs(data_dir)
        closed_pnl = sum(p['actual_pnl_usd'] for p in pairs.get('closed', []))
        return capital + closed_pnl
    except Exception:
        return capital


def update_realised_capital(data_dir: str, pnl_usd: float, trade_id) -> float:
    """
    Audit-log helper: write the post-exit realised capital to
    realised_capital.json so we have a per-trade running record.

    The file is now only a log — `load_realised_capital` derives the live
    value from trades.json, so this write is no longer load-bearing for
    sizing or display.

    Note on timing: `load_realised_capital` reads from a CACHED
    build_trade_pairs() at this point in the call sequence (the cache is
    cleared later by invalidate_trade_caches()), so the value it returns
    here is the pre-write total — adding pnl_usd produces the post-write
    total without re-reading trades.json.
    """
    old_capital = load_realised_capital(data_dir)
    new_capital = old_capital + pnl_usd
    path = os.path.join(data_dir, 'realised_capital.json')
    _save_json(path, {
        'realised_capital': new_capital,
        'last_updated':     date.today().isoformat(),
        'last_trade_id':    trade_id,
        'last_pnl_usd':     pnl_usd,
    })
    return new_capital


def invalidate_trade_caches() -> None:
    """
    Clear ONLY caches whose values depend on trades.json / positions.json /
    realised_capital.json.  Market-data, signal, and live-price caches are
    left alone — they don't change when a trade is logged, and rebuilding
    them is expensive.

    Call this after writing a trade (or confirming a stop) instead of the
    blanket st.cache_data.clear() — that one also wipes the market-data
    cache and forces a full Binance/parquet re-read on the next render.
    """
    try:
        build_trade_pairs.clear()
        build_equity_curve.clear()
        build_capital_deployment.clear()
    except Exception:
        pass
    # Component-level caches imported lazily to avoid circular imports.
    try:
        from shared.portfolio_components import _load_portfolio, _load_fund_data
        _load_portfolio.clear()
        _load_fund_data.clear()
    except Exception:
        pass
    try:
        from shared.trade_log_components import _load_single, _load_all
        _load_single.clear()
        _load_all.clear()
    except Exception:
        pass


def load_config(data_dir: str) -> dict:
    """Return dashboard config as a plain dict."""
    cfg = _load_config(data_dir)
    return {
        'capital':           cfg.CAPITAL,
        'active_assets':     cfg.ACTIVE_ASSETS,
        'coin_weights':      cfg.COIN_WEIGHTS,
        'execution_hour':    cfg.EXECUTION_HOUR,
        'trading_cost_pct':  getattr(cfg, 'TRADING_COST_PCT', 0.0),
    }


def get_coin_capital(symbol: str, data_dir: str) -> float:
    """Dollar capital allocated to this coin, using config from data_dir."""
    cfg          = _load_config(data_dir)
    capital      = cfg.CAPITAL
    coin_weights = cfg.COIN_WEIGHTS
    active       = cfg.ACTIVE_ASSETS

    if symbol in coin_weights:
        weight = coin_weights[symbol]
    else:
        allocated    = sum(coin_weights.values())
        remaining    = 1.0 - allocated
        n_unweighted = sum(1 for a in active if a not in coin_weights)
        weight       = remaining / n_unweighted if n_unweighted > 0 else 0.0
    return capital * weight


# ── Entry reason derivation ───────────────────────────────────────────────────

def derive_entry_reason(signal_snapshot, discretion_note=None) -> str:
    """
    Classify entry as 'Strategy' or 'Discretionary'.

    Priority order:
      1. If signal_snapshot was captured: use entry_long flag directly.
      2. If no snapshot: fall back to discretion_note presence.
    """
    if signal_snapshot is not None:
        return 'Strategy' if signal_snapshot.get('entry_long') else 'Discretionary'
    return 'Discretionary' if discretion_note else 'Strategy'


# ── MAE computation with file cache ──────────────────────────────────────────

def compute_mae(position_id: str, symbol: str, entry_date, exit_date,
                entry_price: float, data_dir: str):
    """
    Maximum Adverse Excursion (%) for a trade.

    MAE % = (min(Low) - entry_price) / entry_price * 100

    Closed trades (exit_date is not None) are cached in mae_cache.json inside
    data_dir.  Open trades (exit_date is None) are computed fresh — not cached.

    Returns None on fetch error or if fewer than 1 bar is available.
    """
    from datetime import date as _date

    mae_cache_path = os.path.join(data_dir, 'mae_cache.json')
    is_closed      = (exit_date is not None)

    if is_closed:
        cache = _read_json(mae_cache_path, {})
        if position_id in cache:
            return cache[position_id]

    try:
        from shared.cache_manager import get_daily_ohlcv_range

        today    = _date.today()
        end_dt   = exit_date if exit_date is not None else today

        df = get_daily_ohlcv_range(symbol, entry_date, end_dt)

        if df is None or df.empty or len(df) < 1:
            return None

        min_low = float(df['Low'].min())
        mae_pct = (min_low - entry_price) / entry_price * 100

        if is_closed:
            cache[position_id] = mae_pct
            _save_json(mae_cache_path, cache)

        return mae_pct

    except Exception as e:
        print(f"  compute_mae({position_id}): {e}")
        return None


# ── Trade pair builder ────────────────────────────────────────────────────────

@_cache_data
def build_trade_pairs(data_dir: str) -> dict:
    """
    Match ENTRY and EXIT trades chronologically by symbol.

    Groups by symbol (extracted from position_id) so that old-format entries
    (pid == plain symbol, e.g. 'BTCUSDT') correctly match new-format exits
    (pid == FIFO format, e.g. 'BTCUSDT_20260415_001').

    Returns:
        {
          "closed": list of closed-pair dicts,
          "open":   list of unmatched ENTRY dicts (open positions)
        }
    """
    trades    = load_trades(data_dir)
    _cfg      = _load_config(data_dir)
    cost_pct  = getattr(_cfg, 'TRADING_COST_PCT', 0.0)  # fraction per leg

    from collections import defaultdict
    entries_by_sym: dict = defaultdict(list)
    exits_by_sym:   dict = defaultdict(list)

    for t in trades:
        sym = _pid_to_symbol(t['position_id'])
        if t['action'] == 'ENTRY':
            entries_by_sym[sym].append(t)
        elif t['action'] == 'EXIT':
            exits_by_sym[sym].append(t)

    closed:      list = []
    open_trades: list = []

    for sym, entry_list in entries_by_sym.items():
        exit_list = exits_by_sym.get(sym, [])

        for i, entry in enumerate(entry_list):
            if i < len(exit_list):
                ex = exit_list[i]

                act_entry  = entry['actual_price']
                act_exit   = ex['actual_price']
                theo_entry = entry['theoretical_price']
                theo_exit  = ex['theoretical_price']
                act_lev    = entry['actual_leverage']
                theo_lev   = entry['theoretical_leverage']

                # ── Capital snapshot (frozen at entry time) ──────────────────
                # Use values stored in the ENTRY record when available.
                # For legacy records that pre-date the snapshot, fall back to
                # current config and mark the pair as legacy so the UI can warn.
                coin_cap_snap = entry.get('coin_capital')
                size_usd_snap = entry.get('size_usd')
                is_legacy     = (coin_cap_snap is None or size_usd_snap is None)

                if is_legacy:
                    coin_cap     = get_coin_capital(sym, data_dir)
                    act_size_usd = coin_cap * act_lev
                else:
                    coin_cap     = coin_cap_snap
                    act_size_usd = size_usd_snap   # frozen at entry time

                theo_size_usd = coin_cap * theo_lev

                # ── Close-based P&L ─────────────────────────────────────────
                eff_entry_close = entry.get('entry_close') or theo_entry or act_entry
                eff_exit_close  = ex.get('exit_close')  or theo_exit  or act_exit
                has_close_data  = (entry.get('entry_close') is not None
                                   and ex.get('exit_close') is not None)

                # Direction sign: long profits when price rises, short profits
                # when it falls.  Defaults to 'long' for legacy records and for
                # long-only strategies (momentum) — neutral, no behaviour change.
                direction = (entry.get('direction') or 'long').lower()
                dir_sign  = 1 if direction == 'long' else -1

                close_to_close_pct = (
                    (eff_exit_close / eff_entry_close - 1) * 100 * dir_sign
                    if eff_entry_close else 0.0
                )

                pnl_usd      = act_size_usd * (close_to_close_pct / 100)
                act_ret_pct  = ((act_exit  / act_entry  - 1) * 100 * dir_sign) if act_entry  else 0.0
                theo_ret_pct = ((theo_exit / theo_entry - 1) * 100 * dir_sign) if theo_entry else 0.0
                act_pnl_usd  = act_size_usd  * (act_ret_pct  / 100)
                theo_pnl_usd = theo_size_usd * (theo_ret_pct / 100)

                # ── Trading costs (round-trip: entry leg + exit leg) ─────────
                # cost_pct is per-leg (e.g. 0.001 = 0.1%); multiply by 2 for
                # the full round trip. Set TRADING_COST_PCT = 0.0 in config.py
                # to disable until the platform fee schedule is known.
                act_cost_usd  = act_size_usd  * cost_pct * 2
                theo_cost_usd = theo_size_usd * cost_pct * 2
                act_pnl_usd   -= act_cost_usd
                theo_pnl_usd  -= theo_cost_usd

                # Net return % — P&L as a fraction of position size, after costs.
                # Differs from act_ret_pct (gross price move) when costs > 0.
                act_net_ret_pct  = act_pnl_usd  / act_size_usd  * 100 if act_size_usd  else 0.0
                theo_net_ret_pct = theo_pnl_usd / theo_size_usd * 100 if theo_size_usd else 0.0

                # ── Slippage vs prior-day close ──────────────────────────────
                entry_slip = (
                    (act_entry - eff_entry_close) / eff_entry_close * 100
                    if eff_entry_close else 0.0
                )
                exit_slip = (
                    (act_exit - eff_exit_close) / eff_exit_close * 100
                    if eff_exit_close else 0.0
                )

                holding_days = (
                    (ex['date'] - entry['date']).days
                    if ex['date'] and entry['date'] else 0
                )

                # ── Signal metadata ──────────────────────────────────────────
                signal_snapshot = entry.get('signal_snapshot')
                entry_type = (
                    entry.get('entry_type')
                    or derive_entry_reason(signal_snapshot, entry.get('discretion_note'))
                )
                entry_reason = entry_type   # kept for backward compat

                # ── MAE (cached for closed trades) ───────────────────────────
                mae_pct = compute_mae(
                    entry['position_id'], sym,
                    entry['date'], ex['date'],
                    act_entry, data_dir,
                )

                closed.append({
                    'position_id':            entry['position_id'],
                    'symbol':                 sym,
                    'strategy':               entry['strategy'],
                    'direction':              direction,
                    'entry_date':             entry['date'],
                    'exit_date':              ex['date'],
                    'holding_days':           holding_days,
                    'entry_close':            entry.get('entry_close'),
                    'exit_close':             ex.get('exit_close'),
                    'has_close_data':         has_close_data,
                    'close_to_close_pct':     close_to_close_pct,
                    'actual_entry':           act_entry,
                    'actual_exit':            act_exit,
                    'theoretical_entry':      theo_entry,
                    'theoretical_exit':       theo_exit,
                    'entry_slippage_pct':     entry_slip,
                    'exit_slippage_pct':      exit_slip,
                    'slippage_pct':           (entry_slip + exit_slip) / 2,
                    'leverage':               act_lev,
                    'actual_size_pct':        act_lev,
                    'theoretical_size_pct':   theo_lev,
                    'coin_capital':           coin_cap,
                    'size_usd':               act_size_usd,
                    'actual_size_usd':        act_size_usd,
                    'theoretical_size_usd':   theo_size_usd,
                    'pnl_usd':                    pnl_usd,
                    'actual_pnl_usd':             act_pnl_usd,       # net of costs
                    'actual_return_pct':           act_ret_pct,       # gross price move (no costs)
                    'actual_net_return_pct':       act_net_ret_pct,   # net of round-trip costs
                    'theoretical_return_pct':      theo_ret_pct,      # gross
                    'theoretical_net_return_pct':  theo_net_ret_pct,  # net of costs
                    'theoretical_pnl_usd':         theo_pnl_usd,      # net of costs
                    'cost_usd':                    act_cost_usd,
                    'mae_pct':                mae_pct,
                    'exit_reason':            ex.get('exit_reason') or '—',
                    'signal_snapshot':        signal_snapshot,
                    'entry_reason':           entry_reason,
                    'entry_type':             entry_type,
                    'discretion_note':        entry.get('discretion_note'),
                    # True when coin_capital/size_usd were missing from the ENTRY
                    # record and had to be recomputed from current config.
                    'legacy':                 is_legacy,
                })
            else:
                open_trades.append(entry)

    return {'closed': closed, 'open': open_trades}


# ── Equity curve builders ─────────────────────────────────────────────────────

_EQUITY_COLS = ['date', 'actual_pnl', 'theoretical_pnl',
                'actual_cumulative', 'theoretical_cumulative']


def _equity_df_from_closed(closed: list,
                           start_date: date,
                           end_date: date) -> pd.DataFrame:
    """Shared helper: build daily equity step-function from a list of closed pairs.

    Both start_date and end_date must be derived from trades data by the caller —
    never from date.today() or any config value.
    """
    pnl_by_date: dict = {}
    for p in closed:
        d = p['exit_date']
        if d is None:
            continue
        if d not in pnl_by_date:
            pnl_by_date[d] = {'actual_pnl': 0.0, 'theoretical_pnl': 0.0}
        # Use actual execution-price P&L so the equity curve matches the trade log.
        # pnl_usd (close-to-close) is unreliable for legacy records where
        # entry_close is null — mixing execution prices with close prices
        # overstates gains/losses significantly.
        pnl_by_date[d]['actual_pnl']      += p['actual_pnl_usd']
        pnl_by_date[d]['theoretical_pnl'] += p['theoretical_pnl_usd']

    rows = []
    current = start_date
    while current <= end_date:
        day = pnl_by_date.get(current, {'actual_pnl': 0.0, 'theoretical_pnl': 0.0})
        rows.append({'date': current, **day})
        current += timedelta(days=1)

    df = pd.DataFrame(rows, columns=['date', 'actual_pnl', 'theoretical_pnl'])
    df['actual_cumulative']      = df['actual_pnl'].cumsum()
    df['theoretical_cumulative'] = df['theoretical_pnl'].cumsum()
    return df


def _add_execution_pnl(
    df: pd.DataFrame,
    closed: list,
    execution_hour: int,
    end_date: date,
    data_dir: str,
) -> pd.DataFrame:
    """
    Compute execution-hour P&L for each closed pair and add two columns:

      execution_pnl         — daily P&L booked at exit_date + 1 day
      execution_cumulative  — cumulative sum of execution_pnl

    Execution model:
      Signal fires on the close of day T.
      Entry  executed at the Open of the EXECUTION_HOUR UTC bar on T+1.
      Exit   executed at the Open of the EXECUTION_HOUR UTC bar on exit_date+1.

    Falls back to flagging the pair with exec_approx=True and skipping it
    when the hourly bar is unavailable (cache not yet populated).
    """
    try:
        from shared.cache_manager import get_hourly_ohlcv
    except ImportError:
        return df   # cache module unavailable — skip silently

    _cfg      = _load_config(data_dir)
    cost_pct  = getattr(_cfg, 'TRADING_COST_PCT', 0.0)

    exec_pnl_by_date: dict[date, float] = {}
    n_approx = 0

    for p in closed:
        entry_date = p['entry_date']
        exit_date  = p['exit_date']
        symbol     = p['symbol']
        size_usd   = p.get('actual_size_usd') or p.get('size_usd') or 0.0

        if not entry_date or not exit_date or not size_usd:
            continue

        try:
            hourly = get_hourly_ohlcv(symbol, entry_date, exit_date)
            if hourly is None or hourly.empty:
                p['exec_approx'] = True
                n_approx += 1
                continue

            # Build tz-naive T+1 execution timestamps
            exec_entry_ts = (
                pd.Timestamp(str(entry_date + timedelta(days=1)))
                + pd.Timedelta(hours=execution_hour)
            )
            exec_exit_ts = (
                pd.Timestamp(str(exit_date + timedelta(days=1)))
                + pd.Timedelta(hours=execution_hour)
            )

            # Normalise index timezone so the lookup works regardless of
            # whether the cache stored tz-naive or tz-aware timestamps.
            idx = hourly.index
            if idx.tz is not None:
                # Make both the index and our lookup keys tz-naive UTC
                hourly = hourly.copy()
                hourly.index = idx.tz_convert('UTC').tz_localize(None)

            exec_entry_price = (
                float(hourly.loc[exec_entry_ts, 'Open'])
                if exec_entry_ts in hourly.index else None
            )
            exec_exit_price = (
                float(hourly.loc[exec_exit_ts, 'Open'])
                if exec_exit_ts  in hourly.index else None
            )

            if exec_entry_price is None or exec_exit_price is None:
                p['exec_approx'] = True
                n_approx += 1
                continue

            _dir = (p.get('direction') or 'long').lower()
            _ds  = 1 if _dir == 'long' else -1
            exec_return  = _ds * (exec_exit_price - exec_entry_price) / exec_entry_price
            exec_pnl_usd = exec_return * size_usd - size_usd * cost_pct * 2

            # Book at exit_date + 1 day (when the exit execution actually occurs)
            book_date = exit_date + timedelta(days=1)
            exec_pnl_by_date[book_date] = (
                exec_pnl_by_date.get(book_date, 0.0) + exec_pnl_usd
            )

        except Exception as e:
            print(f"  exec_pnl {symbol} ({entry_date}->{exit_date}): {e}")
            p['exec_approx'] = True
            n_approx += 1

    if not exec_pnl_by_date:
        if n_approx:
            print(f"  build_equity_curve: execution P&L unavailable for "
                  f"{n_approx} pair(s) — run backfill_cache.py")
        return df

    # Extend date range if execution bookings fall after end_date
    exec_end = max(exec_pnl_by_date.keys())
    if exec_end > end_date:
        last_actual_cum  = float(df['actual_cumulative'].iloc[-1])
        last_theo_cum    = float(df['theoretical_cumulative'].iloc[-1])
        extra_rows = []
        current = end_date + timedelta(days=1)
        while current <= exec_end:
            extra_rows.append({
                'date':                   current,
                'actual_pnl':             0.0,
                'theoretical_pnl':        0.0,
                'actual_cumulative':      last_actual_cum,
                'theoretical_cumulative': last_theo_cum,
            })
            current += timedelta(days=1)
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df = df.copy()
    df['execution_pnl']        = df['date'].map(exec_pnl_by_date).fillna(0.0)
    df['execution_cumulative'] = df['execution_pnl'].cumsum()

    if n_approx:
        print(f"  build_equity_curve: {n_approx} pair(s) fell back to approx "
              f"(hourly cache missing for those dates)")

    return df


@_cache_data
def build_equity_curve(data_dir: str) -> pd.DataFrame:
    """
    Daily step-function equity curve from closed trade pairs.

    Date range: min(entry_date) -> max(exit_date) across all closed pairs.
    Derived entirely from trades data — never uses date.today().

    Columns: date | actual_pnl | theoretical_pnl |
             actual_cumulative | theoretical_cumulative |
             execution_pnl | execution_cumulative   ← added when hourly cache exists
    """
    pairs  = build_trade_pairs(data_dir)
    closed = pairs.get('closed', [])

    if not closed:
        return pd.DataFrame(columns=_EQUITY_COLS)

    entry_dates = [p['entry_date'] for p in closed if p['entry_date']]
    exit_dates  = [p['exit_date']  for p in closed if p['exit_date']]

    if not entry_dates or not exit_dates:
        return pd.DataFrame(columns=_EQUITY_COLS)

    start_date = min(entry_dates)
    end_date   = max(max(exit_dates), date.today())

    print(f"build_equity_curve: start_date={start_date}  end_date={end_date}")

    df = _equity_df_from_closed(closed, start_date, end_date)

    # ── Execution-hour P&L (added when hourly cache is populated) ─────────────
    try:
        _cfg           = _load_config(data_dir)
        execution_hour = getattr(_cfg, 'EXECUTION_HOUR', 8)
        df = _add_execution_pnl(df, closed, execution_hour, end_date, data_dir)
    except Exception as e:
        print(f"build_equity_curve: execution P&L skipped — {e}")

    return df


def build_coin_equity_curves(data_dir: str) -> dict:
    """
    Per-symbol equity curves.

    Returns dict[symbol -> DataFrame] with the same columns as build_equity_curve,
    filtered to trades for that coin only.
    """
    from collections import defaultdict

    pairs  = build_trade_pairs(data_dir)
    closed = pairs.get('closed', [])

    if not closed:
        return {}

    by_symbol: dict = defaultdict(list)
    for p in closed:
        by_symbol[p['symbol']].append(p)

    today  = date.today()
    result = {}
    for symbol, sym_closed in by_symbol.items():
        entry_dates = [p['entry_date'] for p in sym_closed if p['entry_date']]
        exit_dates  = [p['exit_date']  for p in sym_closed if p['exit_date']]
        if not entry_dates or not exit_dates:
            continue
        # Extend to today so coins with no recent trades show a flat tail
        # rather than ending abruptly at their last exit date.
        result[symbol] = _equity_df_from_closed(
            sym_closed, min(entry_dates), today
        )

    return result


# ── Capital deployment builder ────────────────────────────────────────────────

@_cache_data
def build_capital_deployment(data_dir: str, coins=None) -> pd.DataFrame:
    """
    Daily capital deployment series.

    Uses size_usd frozen in the ENTRY record; positions without a stored
    size_usd (legacy records) are excluded to preserve snapshot integrity.

    coins: optional list of symbols to include. None = all coins.
    Columns: date | deployed_usd | deployment_pct
    """
    trades    = load_trades(data_dir)
    positions = load_positions(data_dir)
    capital   = float(_load_config(data_dir).CAPITAL)
    _coins    = set(coins) if coins is not None else None

    windows = []
    for t in trades:
        if t['action'] != 'ENTRY' or t.get('size_usd') is None:
            continue

        pid    = t['position_id']
        symbol = _pid_to_symbol(pid)

        if _coins is not None and symbol not in _coins:
            continue

        entry_date = t['date']
        size_usd   = float(t['size_usd'])

        pos          = positions.get(pid, {})
        exit_date_s  = pos.get('exit_date')
        exit_date    = None
        if exit_date_s:
            try:
                exit_date = datetime.strptime(exit_date_s, '%Y-%m-%d').date()
            except Exception:
                pass

        windows.append({
            'entry_date': entry_date,
            'exit_date':  exit_date,
            'size_usd':   size_usd,
        })

    if not windows:
        return pd.DataFrame(columns=['date', 'deployed_usd', 'deployment_pct'])

    entry_dates = [w['entry_date'] for w in windows if w['entry_date']]
    if not entry_dates:
        return pd.DataFrame(columns=['date', 'deployed_usd', 'deployment_pct'])

    start_date = min(entry_dates)
    today      = date.today()

    rows    = []
    current = start_date
    while current <= today:
        deployed = sum(
            w['size_usd']
            for w in windows
            if w['entry_date'] and w['entry_date'] <= current
            and (w['exit_date'] is None or w['exit_date'] >= current)
        )
        rows.append({
            'date':           current,
            'deployed_usd':   deployed,
            'deployment_pct': deployed / capital * 100 if capital else 0.0,
        })
        current += timedelta(days=1)

    return pd.DataFrame(rows)
