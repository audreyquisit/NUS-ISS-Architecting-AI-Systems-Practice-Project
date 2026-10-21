"""Optional live context from LTA DataMall.

DataMall publishes the crowd density of every train station and the arrival
times of buses. Both need an account key, so the whole module is skipped
unless ``LTA_DATAMALL_KEY`` is set. Nothing here is used to state a queue
wait: it is live context for the area, not an observation of a stall.

Every function returns ``None`` when the key is missing or the call fails,
so a caller can carry on without live context rather than guessing.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

DATAMALL_BASE_URL = os.getenv(
    "LTA_DATAMALL_BASE_URL", "http://datamall.lta.gov.sg"
).rstrip("/")
STATION_CROWD_URL = f"{DATAMALL_BASE_URL}/api/v1/PCDRealTime"
BUS_ARRIVAL_URL = f"{DATAMALL_BASE_URL}/api/v1/ArrivalV2"
TRAIN_STATION_URL = f"{DATAMALL_BASE_URL}/api/v1/TrainStation"

HTTP_TIMEOUT_SECONDS = float(os.getenv("LTA_TIMEOUT_SECONDS", "10"))
SOURCE_NAME = "LTA DataMall"
SOURCE_KEY = "lta_datamall"

# The crowd density codes DataMall uses, in the order of busyness.
CROWD_LABELS = {
    "l1": "l1 (very low)",
    "l2": "l2 (low)",
    "l3": "l3 (moderate)",
    "a": "a (high)",
    "b": "b (very high)",
    "c": "c (extremely high)",
}


def is_enabled() -> bool:
    """Return True when a DataMall key is configured."""
    return bool(os.getenv("LTA_DATAMALL_KEY", "").strip())


def _get(url: str, params: Optional[dict[str, Any]] = None) -> Optional[Any]:
    """Return the decoded body of a DataMall call, or None when it fails."""
    key = os.getenv("LTA_DATAMALL_KEY", "").strip()
    if not key:
        return None
    try:
        response = requests.get(
            url,
            params=params,
            headers={"AccountKey": key},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def get_station_crowd_density(station_code: str) -> Optional[dict[str, Any]]:
    """Return the live crowd density of a train station."""
    if not station_code or not is_enabled():
        return None
    body = _get(STATION_CROWD_URL)
    if not isinstance(body, dict):
        return None
    for record in body.get("Data") or []:
        station = str(record.get("Station") or "").strip().upper()
        if station == station_code.strip().upper():
            return {
                "station_code": station,
                "crowd_density": record.get("Crowd Density"),
                "crowd_density_label": CROWD_LABELS.get(
                    str(record.get("Crowd Density") or "").strip().casefold(),
                    record.get("Crowd Density"),
                ),
                "bus_load": record.get("Bus Load"),
                "source": SOURCE_NAME,
                "source_key": SOURCE_KEY,
                "is_live": True,
            }
    return None


def get_bus_arrivals(stop_id: str) -> Optional[list[dict[str, Any]]]:
    """Return the live bus arrivals of a stop."""
    if not stop_id or not is_enabled():
        return None
    body = _get(BUS_ARRIVAL_URL, params={"Service": 2, "Stop": stop_id})
    if not isinstance(body, dict):
        return None
    arrivals = []
    for service in body.get("BusServices") or []:
        for arrival in service.get("BusArrival") or []:
            arrivals.append({
                "service": service.get("ServiceNo"),
                "destination": service.get("Destination"),
                "load": arrival.get("Load"),
                "arrival_time": arrival.get("EstimatedArrivalTime"),
                "source": SOURCE_NAME,
                "source_key": SOURCE_KEY,
                "is_live": True,
            })
    return arrivals or None


def get_train_stations() -> Optional[list[dict[str, Any]]]:
    """Return the train stations DataMall knows, with their codes."""
    body = _get(TRAIN_STATION_URL)
    if not isinstance(body, dict):
        return None
    stations = []
    for record in body.get("Data") or []:
        stations.append({
            "station_code": record.get("Station"),
            "station_name": record.get("Name"),
        })
    return stations or None
