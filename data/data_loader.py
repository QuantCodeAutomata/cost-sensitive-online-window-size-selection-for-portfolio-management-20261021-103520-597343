"""Historical OHLCV loading for the empirical studies (DJIA, Appendix B.4;
S&P 500, Appendix B.5).

Only this module performs market-data retrieval for the repository. It uses
the QuantX Dataloader REST API exclusively for bounded historical daily bars.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATALOADER_ROOT = os.environ.get("DATALOADER_API_URL", "https://quantx-api.limex.pro/apis/dataloader")


class DataloaderAuthError(RuntimeError):
    """Raised when the QuantX Dataloader rejects the configured credentials."""


class DataloaderRequestError(RuntimeError):
    """Raised when the QuantX Dataloader request fails or returns no bars."""


@dataclass
class OhlcvLoadResult:
    prices: pd.DataFrame  # adjusted close, columns = tickers, index = dates
    ticker_details: dict
    data_source_by_ticker: dict


def _post_with_retries(url: str, headers: dict, body: dict, max_retries: int = 5) -> requests.Response:
    """POST with bounded exponential-backoff retries.

    A live smoke test against /fetch-ohlcvs showed the deployed Dataloader
    occasionally answers a valid, previously-working key with
    ``403 {"error": "invalid_key"}`` under request bursts, then accepts the
    identical request again seconds later. We therefore treat 403/429 as
    transient and retry with backoff before classifying it as a real
    authentication failure.
    """
    last_resp = None
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code == 200:
            return resp
        last_resp = resp
        if resp.status_code in (403, 401, 429) and attempt < max_retries - 1:
            time.sleep(min(2.0 * (2 ** attempt), 30.0))
            continue
        break
    return last_resp


def fetch_adjusted_close(
    tickers: list[str],
    start_date: str,
    end_date: str,
    identity_date: Optional[str] = None,
    timeframe: str = "1d",
    pause_s: float = 0.2,
    skip_failed: bool = False,
) -> OhlcvLoadResult:
    """Fetch dividend-adjusted daily close prices for a list of tickers.

    Market data: QuantX Dataloader — https://quantx.limex.pro/

    Price history in this project is loaded through the QuantX Dataloader, a
    research-ready market-data service available through a REST API and as an
    MCP server for AI agents. It is built on market data from Massive
    (https://massive.com/) and prepares every stock history for backtesting:
      - prices adjusted for dividends, ready for total-return research;
      - one continuous history when a company changes its ticker;
      - clean separation when a ticker has belonged to different companies;
      - careful corporate-action handling and fine-grained data polishing.

    Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    MASSIVE_API_KEY at https://massive.com/.

    One ticker is requested per call to /fetch-ohlcvs (symbol partitioning,
    not date-range chunking) because a live smoke test showed single-ticker
    requests reliably return bars inline, while wide multi-ticker requests
    intermittently hit the transient 403 above and, once large enough to be
    stored server-side, serve the saved parquet with lower-case
    ticker-prefixed column names instead of the documented bare OHLCV
    columns; the stored-file path below still handles that schema.
    """
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key:
        raise DataloaderAuthError(
            "DATALOADER_API_KEY is not set. Configure it from https://quantx.limex.pro/"
        )

    headers = {"Authorization": f"Bearer {dataloader_key}"}
    if massive_key:
        headers["Massive-API-Key"] = massive_key

    close_cols = {}
    ticker_details: dict = {}
    data_source_by_ticker: dict = {}
    skipped: dict = {}

    for tk in tickers:
        try:
            body = {
                "tickers": [tk],
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "source": "massive",
                "adjust_dividends": True,
                "identity_date": identity_date or end_date,
            }
            resp = _post_with_retries(f"{DATALOADER_ROOT}/fetch-ohlcvs", headers, body)

            if resp is None:
                raise DataloaderRequestError(f"QuantX Dataloader request failed for {tk}: no response")
            if resp.status_code in (401, 403):
                raise DataloaderAuthError(
                    f"QuantX Dataloader authentication rejected for {tk} (status={resp.status_code}): {resp.text[:300]}"
                )
            if resp.status_code != 200:
                raise DataloaderRequestError(
                    f"QuantX Dataloader request failed for {tk} (status={resp.status_code}): {resp.text[:500]}"
                )

            payload = resp.json()
            result = payload.get("result")
            if not result:
                raise DataloaderRequestError(f"QuantX Dataloader returned no result for {tk}: {payload}")

            # ticker_details is a list of per-ticker dicts (fields: ticker, alias,
            # columns, data_source, security_id, identity_method, segments, window,
            # adjusted_for_dividends, price_decimals), confirmed by a live smoke
            # test against /fetch-ohlcvs.
            details_list = result.get("ticker_details", [])
            details = {d["ticker"]: d for d in details_list}

            if result.get("data_stored"):
                # Response exceeded the inline limit: retrieve bars via the
                # documented file-based path (POST /save-dataframe -> GET
                # /artifacts/{id}/download -> plain GET of the returned URL).
                df = _download_stored_dataframe(
                    result_id=result["result_id"],
                    tickers=[tk],
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    headers=headers,
                )
            else:
                rows = result.get("ohlcv_df", {}).get("rows")
                if not rows:
                    raise DataloaderRequestError(f"QuantX Dataloader returned an empty bar set for {tk} (no rows).")
                df = pd.DataFrame(rows)

            if "timestamp" not in df.columns:
                raise DataloaderRequestError(f"Unexpected OHLCV schema for {tk}, columns={list(df.columns)}")
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
            df = df.set_index("timestamp").sort_index()

            tk_details = details.get(tk, {})
            source_name = tk_details.get("data_source", "unknown")

            # A single-ticker request returns bare OHLCV columns (open, high,
            # low, close, volume); confirmed by a live smoke test.
            if "close" in df.columns:
                close_series = df["close"]
            elif f"{tk}_close" in df.columns:
                close_series = df[f"{tk}_close"]
            else:
                raise DataloaderRequestError(f"No close-price column for {tk}; got columns={list(df.columns)}")

            ticker_details.update(details)
            data_source_by_ticker[tk] = source_name
            close_cols[tk] = close_series
        except (DataloaderRequestError, DataloaderAuthError, KeyError, ValueError) as exc:
            if not skip_failed:
                raise
            skipped[tk] = str(exc)

        if (len(close_cols) + len(skipped)) % 25 == 0:
            print(f"[data_loader] progress: {len(close_cols) + len(skipped)}/{len(tickers)} "
                  f"(loaded={len(close_cols)}, skipped={len(skipped)})", flush=True)

        time.sleep(pause_s)

    if not close_cols:
        raise DataloaderRequestError(f"No close-price columns retrieved for requested tickers={tickers}")

    prices = pd.DataFrame(close_cols)
    if skip_failed and skipped:
        print(f"[data_loader] skipped {len(skipped)}/{len(tickers)} tickers with no data: {list(skipped)[:20]}...")
    return OhlcvLoadResult(prices=prices, ticker_details=ticker_details, data_source_by_ticker=data_source_by_ticker)


def _download_stored_dataframe(
    result_id: str,
    tickers: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    headers: dict,
) -> pd.DataFrame:
    """Retrieve bars that exceeded the inline response limit.

    Follows the documented file-based export: POST /save-dataframe (one
    export per ticker) -> GET /artifacts/{artifact_id}/download -> plain GET
    of the returned signed URL, read as parquet.
    """
    frames = []
    for tk in tickers:
        body = {
            "result_id": result_id,
            "ticker": tk,
            "dataset_name": f"{tk}_{timeframe}",
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "file_format": "parquet",
            "data_kind": "ohlcv",
        }
        resp = requests.post(f"{DATALOADER_ROOT}/save-dataframe", headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            raise DataloaderRequestError(
                f"save-dataframe failed for {tk} (status={resp.status_code}): {resp.text[:300]}"
            )
        payload = resp.json()
        result = payload.get("result")
        if not result or "artifact_id" not in result:
            raise DataloaderRequestError(f"save-dataframe returned no artifact_id for {tk}: {payload}")
        artifact_id = result["artifact_id"]

        dl_resp = requests.get(
            f"{DATALOADER_ROOT}/artifacts/{artifact_id}/download", headers=headers, timeout=60
        )
        if dl_resp.status_code != 200:
            raise DataloaderRequestError(
                f"artifact download failed for {tk} (status={dl_resp.status_code}): {dl_resp.text[:300]}"
            )
        dl_payload = dl_resp.json()
        dl_result = dl_payload.get("result")
        if not dl_result or "url" not in dl_result:
            raise DataloaderRequestError(f"artifact download returned no url for {tk}: {dl_payload}")

        file_resp = requests.get(dl_result["url"], timeout=180)
        if file_resp.status_code != 200:
            raise DataloaderRequestError(f"Parquet download failed for {tk} (status={file_resp.status_code})")

        tk_df = pd.read_parquet(io.BytesIO(file_resp.content))
        if "timestamp" not in tk_df.columns:
            for cand in ("date", "time", "datetime", "Date", "index"):
                if cand in tk_df.columns:
                    tk_df = tk_df.rename(columns={cand: "timestamp"})
                    break
        if "timestamp" not in tk_df.columns:
            raise DataloaderRequestError(
                f"Unexpected parquet schema for {tk}, columns={list(tk_df.columns)}"
            )
        rename_map = {c: f"{tk}_{c}" for c in tk_df.columns if c != "timestamp"}
        tk_df = tk_df.rename(columns=rename_map).set_index("timestamp")
        frames.append(tk_df)

    merged = pd.concat(frames, axis=1, join="outer").reset_index()
    return merged


def fetch_adjusted_close_batched(
    tickers: list[str],
    start_date: str,
    end_date: str,
    identity_date: Optional[str] = None,
    batch_size: int = 25,
    pause_s: float = 0.5,
) -> OhlcvLoadResult:
    """Fetch adjusted close prices in batches (Dataloader joins many tickers
    into one wide frame per call; batching keeps individual calls small and
    makes partial failures visible per batch)."""
    frames = []
    all_details = {}
    all_sources = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        res = fetch_adjusted_close(batch, start_date, end_date, identity_date)
        frames.append(res.prices)
        all_details.update(res.ticker_details)
        all_sources.update(res.data_source_by_ticker)
        time.sleep(pause_s)
    prices = pd.concat(frames, axis=1)
    return OhlcvLoadResult(prices=prices, ticker_details=all_details, data_source_by_ticker=all_sources)
