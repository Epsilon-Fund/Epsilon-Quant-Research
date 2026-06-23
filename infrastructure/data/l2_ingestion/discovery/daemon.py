"""Market discovery daemon: polls the Polymarket Gamma API on a schedule,
filters active markets to our target universes, and writes ``live_universe.json``
(atomic rename) for the capture daemon to consume."""
