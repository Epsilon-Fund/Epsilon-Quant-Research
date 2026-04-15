"""
strategies.py — Strategy callables shared by optimise.py and dashboard.py.

Imported at module level in both files so neither depends on the other.

momentum_swing   : ETH, SOL, BNB, ADA, XRP  — includes Vol_MA filter
momentum_no_vol  : AVAX, BTC                 — Vol_MA lines removed
                   (confirmed structurally identical from both notebooks)
"""

import numpy as np
import pandas as pd


def momentum_swing(df_slice: pd.DataFrame, params: dict):
    """Momentum swing with volume-MA entry filter. ETH, SOL, BNB, ADA, XRP.
    Source: topics/momentum/strategies/wf_testing_2/momentumETH_wf.ipynb
    """
    df = df_slice.copy()

    df['EMA']          = df['Close'].ewm(span=params['ema_span'], adjust=False).mean()
    df['Swing_Hi_Cau'] = df['High'].rolling(params['swing_caution']).max()
    df['Swing_Lo_Cau'] = df['Low'].rolling(params['swing_caution']).min()
    df['Swing_Hi_Stp'] = df['High'].rolling(params['swing_stop']).max()

    def atr(period):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low']  - df['Close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()

    df['ATR_Cau'] = atr(params['atr_caution'])
    df['ATR_Stp'] = atr(params['atr_stop'])
    df['ATR_Sz']  = atr(params['atr_size'])

    up    = df['High'].diff();  down = -df['Low'].diff()
    pdm   = up.where((up > down) & (up > 0), 0.0)
    ndm   = down.where((down > up) & (down > 0), 0.0)
    atr14 = atr(14)
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(span=14, adjust=False).mean() / atr14
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    df['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    df['Vol_MA'] = df['Volume'].rolling(params['vol_ma_period']).mean()
    direction    = df['Close'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['OBV']    = (df['Volume'] * direction).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(params['obv_ma_period']).mean()

    df['Caution_OBV']   = (df['Close'] > df['Close'].shift(params['obv_lookback'])) & (df['OBV'] < df['OBV_MA'])
    df['Caution_Long']  = ((df['Swing_Hi_Cau'] - df['Low']) > 1.5 * df['ATR_Cau']) | df['Caution_OBV']
    df['Caution_Short'] = ((df['High'] - df['Swing_Lo_Cau']) > 1.5 * df['ATR_Cau']) | (df['Close'] > df['EMA'])

    _valid = (
        df['Swing_Hi_Stp'].notna() & df['ATR_Stp'].notna() &
        df['ATR_Sz'].notna() & df['OBV_MA'].notna() & df['Vol_MA'].notna()
    )
    df['Entry_Long'] = (
        (df['Close'] > df['EMA']) &
        (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) &
        (df['Volume'] > df['Vol_MA']) &
        _valid
    )
    df['position_size_raw'] = (
        params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])
    ).clip(0.1, params['max_leverage'])

    n             = len(df)
    position      = [0]      * n
    position_size = [0.0]    * n
    stop_arr      = [np.nan] * n
    in_position   = 0
    stop_loss     = np.nan
    current_size  = 0.0

    for i in range(1, n):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if in_position == 0:
            if curr['Entry_Long']:
                in_position  = 1
                current_size = curr['position_size_raw']
                cl = curr['Caution_Long']; cs = curr['Caution_Short']
                if cl and cs: sm = params['stop_mult_ent_both']
                elif cl:      sm = params['stop_mult_ent_caution']
                else:         sm = params['stop_mult_ent_normal']
                stop_loss = curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale']
        else:
            if prev['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                stop_loss    = np.nan
            else:
                sm        = params['stop_mult_pos_caution'] if curr['Caution_Long'] else params['stop_mult_pos_normal']
                stop_loss = max(stop_loss, curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale'])

        position[i]      = in_position
        position_size[i] = current_size
        stop_arr[i]      = stop_loss

    df['position']      = position
    df['position_size'] = position_size
    df['stop_loss']     = stop_arr

    indicator_cols = ['EMA', 'ATR_Cau', 'ADX_14', 'Swing_Hi_Cau', 'Vol_MA', 'OBV_MA']
    df['position']      = df['position'].fillna(0).astype(int)
    df['position_size'] = df['position_size'].fillna(0.0)
    df['stop_loss']     = df['stop_loss'].fillna(0.0)

    return df, indicator_cols


def momentum_no_vol(df_slice: pd.DataFrame, params: dict):
    """Momentum swing without volume-MA filter. AVAX, BTC.
    Source: topics/momentum/strategies/wf_testing_2/momentumBTC_wf.ipynb
    """
    df = df_slice.copy()

    df['EMA']          = df['Close'].ewm(span=params['ema_span'], adjust=False).mean()
    df['Swing_Hi_Cau'] = df['High'].rolling(params['swing_caution']).max()
    df['Swing_Lo_Cau'] = df['Low'].rolling(params['swing_caution']).min()
    df['Swing_Hi_Stp'] = df['High'].rolling(params['swing_stop']).max()

    def atr(period):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift(1)).abs()
        lc = (df['Low']  - df['Close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=period, adjust=False).mean()

    df['ATR_Cau'] = atr(params['atr_caution'])
    df['ATR_Stp'] = atr(params['atr_stop'])
    df['ATR_Sz']  = atr(params['atr_size'])

    up    = df['High'].diff();  down = -df['Low'].diff()
    pdm   = up.where((up > down) & (up > 0), 0.0)
    ndm   = down.where((down > up) & (down > 0), 0.0)
    atr14 = atr(14)
    pdi   = 100 * pdm.ewm(span=14, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(span=14, adjust=False).mean() / atr14
    dx    = (100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)).fillna(0)
    df['ADX_14'] = dx.ewm(span=14, adjust=False).mean()

    direction    = df['Close'].diff().apply(lambda x: 1 if x > 0 else -1)
    df['OBV']    = (df['Volume'] * direction).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(params['obv_ma_period']).mean()

    df['Caution_OBV']   = (df['Close'] > df['Close'].shift(params['obv_lookback'])) & (df['OBV'] < df['OBV_MA'])
    df['Caution_Long']  = ((df['Swing_Hi_Cau'] - df['Low']) > 1.5 * df['ATR_Cau']) | df['Caution_OBV']
    df['Caution_Short'] = ((df['High'] - df['Swing_Lo_Cau']) > 1.5 * df['ATR_Cau']) | (df['Close'] > df['EMA'])

    _valid = (
        df['Swing_Hi_Stp'].notna() & df['ATR_Stp'].notna() &
        df['ATR_Sz'].notna() & df['OBV_MA'].notna()
    )
    df['Entry_Long'] = (
        (df['Close'] > df['EMA']) &
        (~df['Caution_Long'] | (df['ADX_14'] > params['adx_override'])) &
        _valid
    )
    df['position_size_raw'] = (
        params['risk_per_trade'] / (df['ATR_Sz'] / df['Close'])
    ).clip(0.1, params['max_leverage'])

    n             = len(df)
    position      = [0]      * n
    position_size = [0.0]    * n
    stop_arr      = [np.nan] * n
    in_position   = 0
    stop_loss     = np.nan
    current_size  = 0.0

    for i in range(1, n):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if in_position == 0:
            if curr['Entry_Long']:
                in_position  = 1
                current_size = curr['position_size_raw']
                cl = curr['Caution_Long']; cs = curr['Caution_Short']
                if cl and cs: sm = params['stop_mult_ent_both']
                elif cl:      sm = params['stop_mult_ent_caution']
                else:         sm = params['stop_mult_ent_normal']
                stop_loss = curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale']
        else:
            if prev['Close'] < stop_loss:
                in_position  = 0
                current_size = 0.0
                stop_loss    = np.nan
            else:
                sm        = params['stop_mult_pos_caution'] if curr['Caution_Long'] else params['stop_mult_pos_normal']
                stop_loss = max(stop_loss, curr['Swing_Hi_Stp'] - curr['ATR_Stp'] * sm * params['stop_atr_scale'])

        position[i]      = in_position
        position_size[i] = current_size
        stop_arr[i]      = stop_loss

    df['position']      = position
    df['position_size'] = position_size
    df['stop_loss']     = stop_arr

    indicator_cols = ['EMA', 'ATR_Cau', 'ADX_14', 'Swing_Hi_Cau', 'OBV_MA']
    df['position']      = df['position'].fillna(0).astype(int)
    df['position_size'] = df['position_size'].fillna(0.0)
    df['stop_loss']     = df['stop_loss'].fillna(0.0)

    return df, indicator_cols


# ── Registry — callables only, no numerical config ────────────────────────────
STRATEGY_REGISTRY = {
    "momentum_swing":  momentum_swing,   # ETH, SOL, BNB, ADA, XRP
    "momentum_no_vol": momentum_no_vol,  # AVAX, BTC
}
