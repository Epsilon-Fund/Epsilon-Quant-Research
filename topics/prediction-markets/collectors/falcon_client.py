"""
Base Falcon API client with pagination, retry logic, and rate limiting.
All collectors use this client to query the Heisenberg Narrative API.
"""

import time
from typing import Any

import requests
from loguru import logger

import config


class FalconClient:
    """HTTP client for the Falcon (Heisenberg Narrative) API."""

    def __init__(self):
        if not config.FALCON_API_KEY:
            raise RuntimeError(
                "FALCON_API_KEY is not set. "
                "Create a .env file with your key (see .env.example)."
            )
        self.base_url = config.FALCON_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {config.FALCON_API_KEY}",
            "Content-Type": "application/json",
        }

    def query(
        self,
        agent_id: int,
        params: dict[str, Any] | None = None,
        paginate: bool = True,
    ) -> list[dict]:
        """
        Query a Falcon agent and return all results.

        Args:
            agent_id: The Falcon agent ID (574, 556, 579, 581).
            params: Query parameters specific to the agent.
            paginate: If True, loop through all pages. If False, return first page only.

        Returns:
            List of result dicts combined across all pages.
        """
        if params is None:
            params = {}

        all_results = []
        offset = 0

        while True:
            payload = {
                "agent_id": agent_id,
                "params": params,
                "pagination": {"limit": config.PAGINATION_LIMIT, "offset": offset},
                "formatter_config": {"format_type": "raw"},
            }

            data = self._request_with_retry(payload, agent_id, offset)
            if data is None:
                break

            results = data.get("data", {}).get("results", [])
            all_results.extend(results)

            pagination = data.get("pagination", {})
            has_more = pagination.get("has_more", False)

            logger.info(
                "agent_id={} offset={} returned={} has_more={}",
                agent_id,
                offset,
                len(results),
                has_more,
            )

            if not paginate or not has_more:
                break

            offset += config.PAGINATION_LIMIT
            time.sleep(config.REQUEST_DELAY_SECONDS)

        return all_results

    def _request_with_retry(
        self, payload: dict, agent_id: int, offset: int
    ) -> dict | None:
        """Make a single request with exponential backoff retry on 429/5xx."""
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                response = requests.post(
                    self.base_url, headers=self.headers, json=payload, timeout=60
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    wait = config.RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "agent_id={} offset={} HTTP {} — retrying in {}s (attempt {}/{})",
                        agent_id,
                        offset,
                        response.status_code,
                        wait,
                        attempt,
                        config.MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    "agent_id={} offset={} HTTP {} — {}",
                    agent_id,
                    offset,
                    response.status_code,
                    response.text[:500],
                )
                return None

            except requests.RequestException as e:
                wait = config.RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "agent_id={} offset={} request error: {} — retrying in {}s (attempt {}/{})",
                    agent_id,
                    offset,
                    e,
                    wait,
                    attempt,
                    config.MAX_RETRIES,
                )
                time.sleep(wait)

        logger.error(
            "agent_id={} offset={} all {} retries exhausted",
            agent_id,
            offset,
            config.MAX_RETRIES,
        )
        return None
