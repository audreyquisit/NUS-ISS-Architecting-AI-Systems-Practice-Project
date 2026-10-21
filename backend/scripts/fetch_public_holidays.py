"""Snapshot the MOM public holidays.

The Ministry of Manpower publishes the public holidays of Singapore on
data.gov.sg. The script takes a snapshot of the consolidated dataset and
writes it to ``data/public_holidays.json``.

Usage (from backend/):
    python scripts/fetch_public_holidays.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT_PATH = DATA_DIR / "public_holidays.json"

DATASET_ID = "d_8ef23381f9417e4d4254ee8b4dcdb176"
DATASET_URL = f"https://data.gov.sg/datasets/{DATASET_ID}/view"
DOWNLOAD_URL = (
    f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
)
LICENCE = "Singapore Open Data Licence (data.gov.sg)"
SOURCE_NAME = "MOM public holidays"

HTTP_TIMEOUT_SECONDS = 60


def download_holidays() -> list[dict[str, Any]]:
    """Download the consolidated public holiday list."""
    print(f"Downloading the MOM public holidays dataset {DATASET_ID}...")
    response = requests.get(DOWNLOAD_URL, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    body = response.json()
    if body.get("code") != 0:
        raise RuntimeError(f"MOM download failed: {body.get('errMsg')}")
    payload = requests.get(body["data"]["url"], timeout=HTTP_TIMEOUT_SECONDS).content
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]
            if not names:
                raise RuntimeError("The MOM download held no CSV file.")
            payload = archive.read(names[0])
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig", errors="replace"))))
    holidays = []
    for row in rows:
        date = str(row.get("date") or "").strip()
        name = str(row.get("holiday") or "").strip()
        if not date or not name:
            continue
        holidays.append({"date": date, "name": name})
    holidays.sort(key=lambda entry: entry["date"])
    print(f"  {len(holidays)} public holidays")
    return holidays


def main() -> None:
    holidays = download_holidays()
    if not holidays:
        print("The dataset held no holidays. The snapshot was not written.")
        sys.exit(1)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    SNAPSHOT_PATH.write_text(
        json.dumps({
            "source": SOURCE_NAME,
            "dataset_id": DATASET_ID,
            "dataset_url": DATASET_URL,
            "licence": LICENCE,
            "as_of": retrieved_at[:10],
            "retrieved_at": retrieved_at,
            "holiday_count": len(holidays),
            "holidays": holidays,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SNAPSHOT_PATH.name}")


if __name__ == "__main__":
    main()
