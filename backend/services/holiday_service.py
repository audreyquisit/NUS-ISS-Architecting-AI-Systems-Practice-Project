"""Read the MOM public holidays.

A hawker centre may be closed on a public holiday, and a centre that is open
may be busier than usual. The service answers both questions from the MOM
snapshot, and says nothing at all when the snapshot is missing, so a missing
snapshot can never turn into a claim that a centre is open.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT_PATH = DATA_DIR / "public_holidays.json"

SOURCE_NAME = "MOM public holidays"
SOURCE_KEY = "mom_public_holidays"


@lru_cache(maxsize=1)
def _snapshot() -> Optional[dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=1)
def _holidays() -> dict[str, str]:
    """Return the public holidays of the snapshot, keyed by date."""
    snapshot = _snapshot()
    if not snapshot:
        return {}
    holidays: dict[str, str] = {}
    for entry in snapshot.get("holidays", []):
        date_value = str(entry.get("date") or "").strip()
        name = str(entry.get("name") or "").strip()
        if date_value and name:
            holidays[date_value] = name
    return holidays


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return date.today()
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return date.today()


def is_public_holiday(when: Any = None) -> bool:
    """Return True when the day is a Singapore public holiday."""
    holidays = _holidays()
    if not holidays:
        return False
    return _as_date(when).isoformat() in holidays


def public_holiday_name(when: Any = None) -> Optional[str]:
    """Return the name of the public holiday of the day, if it is one."""
    return _holidays().get(_as_date(when).isoformat())


def next_public_holiday(when: Any = None) -> Optional[dict[str, Any]]:
    """Return the next public holiday on or after the day."""
    holidays = _holidays()
    if not holidays:
        return None
    today = _as_date(when)
    upcoming = sorted(
        (value for value in holidays if value >= today.isoformat())
    )
    if not upcoming:
        return None
    date_value = upcoming[0]
    return {
        "date": date_value,
        "name": holidays[date_value],
        "days_away": (date.fromisoformat(date_value) - today).days,
    }


def opening_warning(when: Any = None) -> Optional[dict[str, Any]]:
    """Return the warning to show when a day is a public holiday.

    The warning says the centre may be closed. It never says the centre is
    open, because the sources do not say which centres keep trading.
    """
    if not is_public_holiday(when):
        return None
    name = public_holiday_name(when)
    return {
        "date": _as_date(when).isoformat(),
        "holiday": name,
        "warning": (
            f"{name} is a public holiday. Some hawker centres and stalls close "
            "or keep reduced hours, so check with the operator before you go."
        ),
        "source": SOURCE_NAME,
        "source_key": SOURCE_KEY,
    }


def holiday_context(when: Any = None) -> dict[str, Any]:
    """Return the holiday context of the day, for the reply."""
    snapshot = _snapshot()
    context: dict[str, Any] = {
        "is_public_holiday": is_public_holiday(when),
        "holiday_name": public_holiday_name(when),
        "next_holiday": next_public_holiday(when),
        "available": bool(_holidays()),
    }
    warning = opening_warning(when)
    if warning:
        context["warning"] = warning
    if snapshot:
        context["source"] = SOURCE_NAME
        context["source_key"] = SOURCE_KEY
        context["source_url"] = snapshot.get("dataset_url")
        context["as_of"] = snapshot.get("as_of")
    return context
