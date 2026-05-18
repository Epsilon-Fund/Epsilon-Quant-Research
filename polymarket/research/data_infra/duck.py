import duckdb


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    return con
