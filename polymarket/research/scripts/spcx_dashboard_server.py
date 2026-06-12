"""SPCX interactive localhost dashboard server (Block S5e).

A thin display layer over the spcx_pm_pdf_monitor engine: the engine's poll loop stays
the single source of truth (polling, survivor fit, parquet logging untouched); this module
serves a localhost page and pushes each poll's payload to connected browsers over a
websocket. Read-only and advisory: no order placement, no venue auth, no signals beyond
the gameplan rules the playbook panel already quotes.

- Binds 127.0.0.1 only. No auth — acceptable ONLY because it never leaves localhost.
- Front-end: scripts/assets/spcx_dashboard.html + vendored ECharts (no CDN/network).
- Library choice: aiohttp — one small dependency covering HTTP + websockets in a single
  background-thread event loop, no framework, no build step.

Fallback ladder (document for Friday): browser dies → server keeps broadcasting, reload
reconnects with a full snapshot; server dies → the SAME process's terminal output and
optional --html static page continue (engine is independent); process dies → restart
resumes session history from today's parquet shards and day-state from the JSON file.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_PORT = 8642
WIRE_POINTS = 400  # downsample fit curves for the websocket


def _downsample(arr, n: int = WIRE_POINTS) -> list[float]:
    step = max(1, len(arr) // n)
    out = [float(x) for x in arr[::step]]
    if float(arr[-1]) != out[-1]:
        out.append(float(arr[-1]))
    return out


def _stats_lite(rec: dict | None) -> dict | None:
    if rec is None:
        return None
    return {k: rec.get(k) for k in ("mean_ps", "median_ps", "pwin", "perp", "spot", "ts")}


def curves_payload(rep: dict, snap: dict, perp: float | None, spot: float | None) -> dict:
    """The survivor/PDF/lognormal blocks drawn in panel 04 — shared by the live websocket
    payload and the /api/curve time-scrub endpoint, so a scrubbed historical curve renders
    through exactly the pipeline a live poll does."""
    from scripts.spcx_pm_pdf_monitor import TAIL_STRIKES, basis_price  # no cycles at load

    a = rep["stats_primary"]
    fit = rep["fit"]
    cap_to_ps = 1e12 / a["shares"]
    ln = rep.get("lognormal") or {}
    lognormal = {k: ln.get(k) for k in ("mu", "sig", "median_cap_t", "mean_cap_t", "wrmse_c")}
    ds_grid = _downsample(fit["grid"])
    lognormal["fair_S"] = ([ln["S"](float(k)) for k in ds_grid] if ln.get("S") else None)
    # analytic lognormal density on the same grid (cap-space, same units as fit["pdf"]) —
    # the smooth single-hump construction the audit recommends showing on the PDF chart
    if ln.get("mu") is not None:
        mu, sig = ln["mu"], ln["sig"]
        lognormal["fair_pdf"] = [
            (math.exp(-((math.log(k) - mu) ** 2) / (2 * sig * sig))
             / (k * sig * math.sqrt(2 * math.pi))) if k > 1e-9 else 0.0
            for k in ds_grid]
    else:
        lognormal["fair_pdf"] = None
    clipped = set(fit["monotone_violations"])
    strikes = []
    for row in snap["ladder"]:
        p, _ = basis_price(row, rep["basis"])
        strikes.append({"k": row["strike_t"], "bid": row.get("bid"), "ask": row.get("ask"),
                        "p": p, "clipped": row["strike_t"] in clipped})
    # Tail-sell PDF markers (the violet shaded zone + strike dots on the PDF chart). Derived
    # from the curve's OWN strikes here — sell = the YES bid — so they render identically on
    # the live PDF and on any scrubbed historical curve (the live-only depth/velocity columns
    # of the panel-01 screen stay live-only; the PDF marker only needs strike + sell bid).
    tail_set = set(TAIL_STRIKES)
    tail_pts = [{"strike_t": s["k"], "strike_ps": s["k"] * cap_to_ps, "sell_bid": s["bid"]}
                for s in strikes if s["k"] in tail_set and s["bid"] is not None]
    # raw interval-mass histogram: S[i]−S[i+1] over each strike gap, the assumption-free
    # density the audit recommends showing beside any fitted curve. Survivor points are
    # running-min clipped (same as the fit) so masses are never negative.
    raw_hist, run = [], 1.0
    clipped_pts = []
    for k, p in ((s["k"], s["p"]) for s in strikes if s["p"] is not None):
        run = min(run, p)
        clipped_pts.append((k, run))
    for (k0, p0), (k1, p1) in zip(clipped_pts, clipped_pts[1:]):
        w = k1 - k0
        if w > 0:
            raw_hist.append({"lo_ps": k0 * cap_to_ps, "hi_ps": k1 * cap_to_ps,
                             "dens": (p0 - p1) / w})
    return {
        "offer": rep["offer"],
        "stats": {k: a[k] for k in ("p_win_offer", "mean_ps", "median_ps", "mean_cap_t")},
        "survivor": {"grid": ds_grid, "S": _downsample(fit["S"]), "strikes": strikes},
        "pdf": {"price": [k * cap_to_ps for k in ds_grid],
                "dens": _downsample(fit["pdf"]),
                "raw_hist": raw_hist,
                "p25": a["p25_ps"], "p75": a["p75_ps"], "mode": a["mode_ps"],
                "perp": perp, "spot": spot, "tail_pts": tail_pts},
        "lognormal": lognormal,
    }


def build_ws_payload(dash, pb, rep: dict, snap: dict, avwap_info: dict | None,
                     playbook_html: str, curves=None, classifier=None,
                     halts=None, arb=None, arb_fee_default: float = 0.0) -> dict:
    """One poll → the JSON the browser draws. Everything financial is computed server-side
    (engine/calc); the browser only renders."""
    from scripts.spcx_pm_pdf_monitor import (  # no cycles at load
        TAIL_STRIKES, hedge_chart_data, hedge_ops_eval, infer_node, render_halts,
        render_hedge_ops, render_indications, render_pm_panel, render_tranches,
        tail_trade_eval, tranche_chart_data, tranche_schedule)

    a = rep["stats_primary"]
    now = dash.history[-1]["ts"] if dash.history else time.time()
    hl = snap.get("hl") or {}
    ops = hedge_ops_eval(pb, hl.get("mark"),
                         (rep.get("spot") or {}).get("spot"),
                         hl.get("funding_hourly", 0.0), hl.get("max_leverage"),
                         dash.history, now)
    halts_snap = halts.snapshot() if halts is not None else None  # one call: one-shot alert
    day_shape = (classifier.view(getattr(pb, "shape_override", None), now)
                 if classifier is not None else None)
    shape = day_shape["state"] if day_shape else (getattr(pb, "shape_override", None)
                                                  or "FLAT")
    tranches = tranche_schedule(pb.fill, pb.hedged, pb.sold, pb.cross_ts, now,
                                shape=shape,
                                session_high=(day_shape or {}).get("session_high"))
    # S5j charts — first-green is remembered on the dash for the readiness marker
    if ops and ops["pair_close"]["green"] and getattr(dash, "pair_green_since", None) is None:
        dash.pair_green_since = now
    tranche_chart = tranche_chart_data(tranches, pb.cross_ts, now)
    hedge_chart = hedge_chart_data(ops, dash.history, pb.cross_ts, now,
                                   green_since=getattr(dash, "pair_green_since", None))
    # the tail-sell screen with real repricing velocity (analyze()'s baseline had no history)
    tail = tail_trade_eval(snap, rep.get("lognormal"), dash.history, now)
    cap_to_ps = 1e12 / a["shares"]
    for r in tail["rows"]:  # $/share equivalent of each tail strike, for the PDF markers
        r["strike_ps"] = r["strike_t"] * cap_to_ps
    levels = [{"level": lv, "label": lab,
               "crossed_ts": (dash.alert_first_touch.get(lv) or (None, None))[0],
               "source": (dash.alert_first_touch.get(lv) or (None, None))[1],
               "downside": lv <= dash.offer}
              for lv, lab in dash._levels()]
    meta = rep.get("spot_meta") or {}
    spot_block = rep.get("spot") or {}
    curve_blocks = curves_payload(rep, snap,
                                  perp=(rep.get("hl_gap") or {}).get("hl_mark"),
                                  spot=spot_block.get("spot"))
    payload = {
        "type": "poll",
        "ts": now,
        "ts_iso": snap["fetched_at_utc"],
        "basis": rep["basis"],
        "offer": rep["offer"],
        "stats": {k: a[k] for k in ("p_win_offer", "mean_ps", "median_ps", "mode_ps",
                                    "p25_ps", "p75_ps", "p90_ps", "p95_ps", "mean_cap_t",
                                    "ev_vs_offer_ps")},
        "start": _stats_lite(dash.session_start_rec),
        "prev": _stats_lite(dash.history[-2] if len(dash.history) >= 2 else None),
        "hl": rep.get("hl_gap"),
        "spot": {"price": spot_block.get("spot"), "status": meta.get("status"),
                 "age_s": meta.get("age_s"), "source": meta.get("source"),
                 "pctile": spot_block.get("crowd_pctile_of_spot")},
        "flags": rep["degradation_flags"],
        "violations": rep["monotone_violations"],
        "feeds": {"pm": dash.last_ok.get("pm"), "hl": dash.last_ok.get("hl"),
                  "spot": dash.last_ok.get("spot")},
        "survivor": curve_blocks["survivor"],
        "pdf": curve_blocks["pdf"],
        "no_ipo": rep.get("no_ipo"),
        "tails": {str(k): list(dash.history[-1]["tails"].get(k, (None, None)))
                  if dash.history else [None, None]
                  for k in TAIL_STRIKES},
        "tail_trade": tail,
        "lognormal": curve_blocks["lognormal"],
        "curve_range": curves.range() if curves is not None else None,
        "hedge_ops_html": render_hedge_ops(ops),
        "tranche_html": render_tranches(tranches),
        "tranche_chart": tranche_chart,
        "hedge_chart": hedge_chart,
        "day_shape": day_shape,
        "halts": halts_snap,
        "halts_html": render_halts(halts_snap),
        "avwap": avwap_info,
        "node": infer_node(pb, time.time()),
        "levels": levels,
        "playbook_html": playbook_html,
        # Block S5k PM tab (display only — terminal numbers, dashboard layout)
        "pm_html": render_pm_panel(rep, dash, now, arb=arb,
                                   arb_fee_default=arb_fee_default),
        "indications_html": render_indications(
            pb, (rep.get("hl_gap") or {}).get("hl_mark"), a["mean_ps"]),
    }
    return payload


def snapshot_message(dash, pb, last_payload: dict | None, reference: dict | None) -> dict:
    """Sent to every (re)connecting client: full session history + current state, so a
    fresh browser tab is immediately complete."""
    hist = [{"ts": r["ts"], "mean_ps": r["mean_ps"], "median_ps": r["median_ps"],
             "pwin": r["pwin"], "perp": r["perp"], "spot": r["spot"],
             "shape": r.get("shape"), "avwap": r.get("avwap"),
             "tails": {str(k): list(v) for k, v in (r.get("tails") or {}).items()}}
            for r in dash.history]
    ref = None
    if reference:
        ref = {"label": reference["label"],
               "grid": _downsample(reference["fit"]["grid"]),
               "S": _downsample(reference["fit"]["S"])}
    return {"type": "snapshot", "history": hist,
            "marks": [{"ts": t, "label": lab} for t, lab in dash.marks],
            "state": {f: getattr(pb, f) for f in pb.FIELDS},
            "reference": ref, "last": last_payload}


class DashboardServer:
    """aiohttp app in a daemon thread; the engine thread calls push() per poll and
    update_playbook() after UI state changes. All websocket I/O stays on the loop thread."""

    def __init__(self, dash, pb, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 curves=None, reh_state_path=None):
        self.dash, self.pb = dash, pb
        self.host, self.port = host, port
        self.actual_port: int | None = None
        self.last_payload: dict | None = None
        self.last_ctx: tuple | None = None  # (rep, snap, avwap_info) for re-renders
        self.curves = curves                # CurveIndex | None — the time-scrub source
        self._curve_cache: dict[float, dict] = {}  # entry ts → /api/curve body (FIFO-capped)
        # Block S5i rehearsal: ISOLATED namespace — its own state file, no parquet, no
        # production PlaybookState access. None until /api/rehearsal/load.
        self.rehearsal: dict | None = None
        self._reh_state_path = reh_state_path  # default resolved lazily (import cycle)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        self._ready = threading.Event()
        self._runner = None

    # ---------- engine-thread API ----------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="spcx-dash-server").start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("dashboard server failed to start within 10s")

    def push(self, payload: dict) -> None:
        self.last_payload = payload
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    def broadcast_now(self, message: dict) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    def url(self) -> str:
        return f"http://{self.host}:{self.actual_port or self.port}/"

    # ---------- loop thread ----------
    def _run(self) -> None:
        from aiohttp import web

        async def main():
            app = web.Application()
            app.router.add_get("/", self._page)
            app.router.add_get("/assets/{name}", self._asset)
            app.router.add_get("/ws", self._ws)
            app.router.add_post("/api/state", self._api_state)
            app.router.add_post("/api/mark", self._api_mark)
            app.router.add_get("/api/curve", self._api_curve)
            app.router.add_post("/api/indication", self._api_indication)
            app.router.add_get("/api/rehearsal/load", self._api_reh_load)
            app.router.add_get("/api/rehearsal/panel", self._api_reh_panel)
            app.router.add_post("/api/rehearsal/state", self._api_reh_state)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self.host, self.port)
            await site.start()
            self.actual_port = site._server.sockets[0].getsockname()[1]
            self._ready.set()
            while True:  # daemon thread: lives until process exit
                await asyncio.sleep(3600)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        except Exception:
            self._ready.set()  # never hang the engine thread on a dead server

    async def _broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        dead = []
        for ws in self._clients:
            try:
                await ws.send_str(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _page(self, request):
        from aiohttp import web
        return web.Response(text=(ASSETS_DIR / "spcx_dashboard.html").read_text(),
                            content_type="text/html")

    async def _asset(self, request):
        from aiohttp import web
        name = request.match_info["name"]
        path = (ASSETS_DIR / name).resolve()
        if not path.is_file() or ASSETS_DIR not in path.parents:
            return web.Response(status=404, text="not found")
        ctype = {"js": "application/javascript", "png": "image/png",
                 "svg": "image/svg+xml", "html": "text/html"}.get(
            path.suffix.lstrip("."), "application/octet-stream")
        return web.Response(body=path.read_bytes(), content_type=ctype)

    async def _api_indication(self, request):
        """Operator-pasted 'indicated to open' range (e.g. '148-152'): timestamped,
        appended to PlaybookState.indications, persisted, and re-broadcast immediately."""
        from aiohttp import web
        try:
            body = await request.json()
            text = str(body["text"]).strip()[:60]
            assert text
        except Exception:
            return web.Response(status=400, text="bad json — need {'text': '148-152'}")
        self.pb.indications = list(self.pb.indications or [])
        self.pb.indications.append({"ts": time.time(), "text": text})
        self.pb.save()
        from scripts.spcx_pm_pdf_monitor import render_indications
        rep = self.last_ctx[0] if self.last_ctx else None
        snap = self.last_ctx[1] if self.last_ctx else None
        perp = ((snap or {}).get("hl") or {}).get("mark")
        mean = rep["stats_primary"]["mean_ps"] if rep else None
        self.broadcast_now({"type": "indication",
                            "indications_html": render_indications(self.pb, perp, mean)})
        return web.json_response({"ok": True, "n": len(self.pb.indications)})

    # ---------- Block S5i rehearsal endpoints (isolated namespace) ----------
    def _reh_path(self):
        if self._reh_state_path is None:
            from scripts.spcx_rehearsal import REHEARSAL_STATE_PATH
            self._reh_state_path = REHEARSAL_STATE_PATH
        return self._reh_state_path

    def _reh_sold(self) -> float:
        try:
            return float(json.loads(self._reh_path().read_text()).get("sold", 0.0))
        except Exception:
            return 0.0

    def _reh_save_sold(self, sold: float) -> None:
        p = self._reh_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"sold": sold}))

    async def _api_reh_load(self, request):
        """Fetch + build a rehearsal session (executor — network + 390 classifier steps).
        Touches only rehearsal state; production pb/parquet are never involved."""
        from aiohttp import web
        symbol = (request.query.get("symbol") or "NVDA").upper()[:6]
        day = request.query.get("date") or None

        def work():
            from scripts.spcx_rehearsal import build_rehearsal, load_tape
            bars, d, source = load_tape(symbol, day)
            reh = build_rehearsal(bars)
            reh.update({"symbol": symbol, "date": d, "source": source})
            return reh

        try:
            reh = await asyncio.get_event_loop().run_in_executor(None, work)
        except Exception as e:
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=502)
        self.rehearsal = reh
        if request.query.get("reset"):
            self._reh_save_sold(0.0)
        body = {k: reh[k] for k in ("symbol", "date", "source", "cross_px", "cross_ts",
                                    "offer", "pop", "levels", "t", "close", "avwap",
                                    "session_high", "ribbon", "n", "perp_offset")}
        body["sold"] = self._reh_sold()
        return web.json_response(body)

    async def _api_reh_panel(self, request):
        from aiohttp import web
        if self.rehearsal is None:
            return web.json_response({"error": "no rehearsal loaded"}, status=404)
        try:
            i = int(request.query["i"])
        except (KeyError, ValueError):
            return web.json_response({"error": "i (bar index) required"}, status=400)
        sold = self._reh_sold()
        from scripts.spcx_rehearsal import panel_at
        body = await asyncio.get_event_loop().run_in_executor(
            None, panel_at, self.rehearsal, i, sold, self._reh_path())
        body.pop("tranche", None)   # rows are embedded in tranche_html; keep wire lean
        body["sold"] = sold
        return web.json_response(body)

    async def _api_reh_state(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            sold = float(body["sold"])
            assert sold >= 0
        except Exception:
            return web.Response(status=400, text="bad json — need {'sold': n}")
        self._reh_save_sold(sold)
        return web.json_response({"ok": True, "sold": sold})

    async def _api_curve(self, request):
        """Time-scrub fetch: ?ts=<epoch seconds> → the survivor/PDF/lognormal blocks for
        the nearest logged poll, refit through the live pipeline. The fit runs in the
        default executor (it costs ~50ms uncached) and results are cached per entry ts."""
        from aiohttp import web
        if self.curves is None:
            return web.json_response({"error": "time-scrub index not enabled"}, status=404)
        try:
            ts = float(request.query["ts"])
        except (KeyError, ValueError):
            return web.json_response({"error": "ts query param (epoch seconds) required"},
                                     status=400)
        entry = self.curves.nearest(ts)
        if entry is not None and entry["ts"] in self._curve_cache:
            return web.json_response(self._curve_cache[entry["ts"]])
        got = await asyncio.get_event_loop().run_in_executor(
            None, self.curves.report_at, ts)
        if got is None:
            return web.json_response({"error": "no refittable poll near ts"}, status=404)
        e, snap, rep = got
        body = curves_payload(rep, snap, perp=e.get("perp"), spot=e.get("spot"))
        body.update({"type": "curve", "ts": e["ts"]})
        if len(self._curve_cache) > 600:   # FIFO cap — payloads are small (downsampled)
            self._curve_cache.pop(next(iter(self._curve_cache)))
        self._curve_cache[e["ts"]] = body
        return web.json_response(body)

    async def _ws(self, request):
        from aiohttp import web
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        try:
            from scripts.spcx_pm_pdf_monitor import DashboardState  # typing only
            ref = getattr(self.dash, "reference", None)
            snap_msg = snapshot_message(self.dash, self.pb, self.last_payload, ref)
            snap_msg["curve_range"] = self.curves.range() if self.curves else None
            await ws.send_str(json.dumps(snap_msg))
            async for _msg in ws:
                pass  # client → server messages go over HTTP POST, not the socket
        finally:
            self._clients.discard(ws)
        return ws

    async def _api_state(self, request):
        """UI day-state updates: {field: value} for PlaybookState.FIELDS, or
        {"cross": ["HH:MM"|"now", price]}. Persists and re-renders the playbook card."""
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400, text="bad json")
        from scripts.spcx_pm_pdf_monitor import parse_cross_arg
        changed = []
        for key, val in body.items():
            if key == "cross" and isinstance(val, list) and len(val) == 2:
                self.pb.cross_ts, self.pb.cross_price = parse_cross_arg(
                    [str(val[0]), str(val[1])], time.time())
                changed.append("cross")
            elif key in self.pb.FIELDS and key != "cross_ts":
                if key in self.pb.LIST_FIELDS:
                    continue   # lists have dedicated endpoints (/api/indication)
                if key in self.pb.STR_FIELDS:
                    setattr(self.pb, key, str(val) if val else None)
                else:
                    setattr(self.pb, key, float(val) if val is not None else None)
                changed.append(key)
        if "hedged" in changed and self.pb.hedged and self.pb.hedge_ts is None:
            self.pb.hedge_ts = time.time()   # stamp the short going on (funding accrual)
        if changed:
            self.pb.save()
            self._rerender_playbook()
        return web.json_response({"ok": True, "changed": changed,
                                  "state": {f: getattr(self.pb, f) for f in self.pb.FIELDS}})

    async def _api_mark(self, request):
        from aiohttp import web
        try:
            body = await request.json()
            label = str(body["label"])[:40]
        except Exception:
            return web.Response(status=400, text="bad json")
        self.dash.marks.append((time.time(), label))
        await self._broadcast({"type": "marks",
                               "marks": [{"ts": t, "label": lab}
                                         for t, lab in self.dash.marks]})
        return web.json_response({"ok": True})

    def _rerender_playbook(self) -> None:
        """Recompute the playbook card off the LAST poll context and push immediately,
        so a UI state change reflects without waiting for the next poll."""
        if not self.last_ctx:
            return
        from scripts.spcx_pm_pdf_monitor import render_playbook
        rep, snap, avwap_info = self.last_ctx
        try:
            html = render_playbook(self.pb, self.dash, rep, snap, avwap_info)
        except Exception as e:
            html = f"<div class='pbcard'>playbook unavailable ({type(e).__name__})</div>"
        if self.last_payload:
            self.last_payload["playbook_html"] = html
        self.broadcast_now({"type": "playbook", "playbook_html": html,
                            "state": {f: getattr(self.pb, f) for f in self.pb.FIELDS}})
