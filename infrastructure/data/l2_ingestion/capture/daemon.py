"""WebSocket capture daemon: subscribes to the Polymarket CLOB WS for every
asset in ``live_universe.json`` (hot-reloaded), stamps each message with arrival
timestamps, and appends raw events to hourly-rotated ``*.jsonl.gz`` files with
auto-reconnect and gap logging."""
