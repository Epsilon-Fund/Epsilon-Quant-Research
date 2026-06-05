from collections.abc import Iterator
from typing import Any

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/orderbook-subgraph/0.0.1/gn"
)

ORDER_FILLED_FIELDS = (
    "id timestamp maker makerAssetId makerAmountFilled "
    "taker takerAssetId takerAmountFilled fee transactionHash"
)


class GoldskyClient:
    def __init__(self, url: str = GOLDSKY_URL, timeout: int = 60) -> None:
        transport = RequestsHTTPTransport(url=url, retries=0, timeout=timeout)
        self._client = Client(transport=transport, fetch_schema_from_transport=False)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _execute(self, query: str) -> dict[str, Any]:
        return self._client.execute(gql(query))

    def iter_order_filled_events(
        self,
        timestamp_gt: int = 0,
        page_size: int = 1000,
    ) -> Iterator[list[dict]]:
        last_ts = max(0, timestamp_gt)
        sticky_ts: int | None = None
        sticky_last_id: str = ""

        while True:
            if sticky_ts is None:
                q = (
                    "{ orderFilledEvents("
                    f"first: {page_size}, orderBy: timestamp, orderDirection: asc, "
                    f'where: {{ timestamp_gt: "{last_ts}" }}'
                    f") {{ {ORDER_FILLED_FIELDS} }} }}"
                )
            else:
                q = (
                    "{ orderFilledEvents("
                    f"first: {page_size}, orderBy: id, orderDirection: asc, "
                    f'where: {{ timestamp: "{sticky_ts}", id_gt: "{sticky_last_id}" }}'
                    f") {{ {ORDER_FILLED_FIELDS} }} }}"
                )
            page = self._execute(q)["orderFilledEvents"]
            if not page:
                if sticky_ts is not None:
                    last_ts = sticky_ts
                    sticky_ts = None
                    sticky_last_id = ""
                    continue
                return
            yield page

            if sticky_ts is not None:
                sticky_last_id = page[-1]["id"]
                if len(page) < page_size:
                    last_ts = sticky_ts
                    sticky_ts = None
                    sticky_last_id = ""
                continue

            page_ts = [int(r["timestamp"]) for r in page]
            if len(page) == page_size and page_ts[0] == page_ts[-1]:
                sticky_ts = page_ts[0]
                sticky_last_id = page[-1]["id"]
            else:
                last_ts = page_ts[-1]
