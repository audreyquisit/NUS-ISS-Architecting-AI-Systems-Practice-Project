"""Snapshot the MUIS halal certification register for the NEA centres.

MUIS runs a public e-Service that lists every halal certified establishment.
The search needs a session and the anti-forgery token of the page, so the
script keeps one session for the whole run.

Only establishments that sit at an NEA hawker centre are kept, so the
snapshot stays small enough to read and to match. The snapshot is written to
``data/muis_halal.json`` and the matched catalogue stalls receive the
source-owned fields ``halal_certified``, ``halal_certificate_no``,
``halal_source`` and ``halal_as_of``.

The script only runs when ``ENABLE_MUIS_HALAL=1`` is set, because it calls a
live public service. When the snapshot is missing the pipeline degrades to
"halal certification could not be verified"; it never guesses.

Usage (from backend/):
    ENABLE_MUIS_HALAL=1 python scripts/fetch_muis_halal.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.source_matching import (  # noqa: E402
    CentreMatcher, distinctive_tokens, extract_postcode, normalise_name,
    normalise_unit,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
SNAPSHOT_PATH = DATA_DIR / "muis_halal.json"

MUIS_BASE_URL = "https://halal.muis.gov.sg"
MUIS_SEARCH_URL = f"{MUIS_BASE_URL}/api/halal/establishments"
MUIS_PAGE_URL = f"{MUIS_BASE_URL}/halal/establishments"
MUIS_SOURCE_NAME = "MUIS halal e-Service establishment directory"

HTTP_TIMEOUT_SECONDS = 45
REQUEST_DELAY_SECONDS = float(os.getenv("MUIS_REQUEST_DELAY_SECONDS", "0.6"))

# Words that carry no meaning when a centre is searched for.
_SEARCH_STOP_WORDS = frozenset({
    "food", "centre", "center", "market", "hawker", "and", "the", "cooked",
    "blk", "block", "cook",
})


def _enabled() -> bool:
    return os.getenv("ENABLE_MUIS_HALAL", "").strip() in {"1", "true", "yes", "on"}


class MuisClient:
    """A small client for the public MUIS establishment search."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SmartHawker/1.0)",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": MUIS_PAGE_URL,
        })
        self.token = self._read_token()

    def _read_token(self) -> str:
        response = self.session.get(MUIS_PAGE_URL, timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        match = re.search(
            r'__RequestVerificationToken[^>]*value="([^"]+)"', response.text
        )
        if not match:
            raise RuntimeError("MUIS did not return an anti-forgery token.")
        return match.group(1)

    def search(self, text: str) -> list[dict[str, Any]]:
        """Return the establishments the register holds for a search text."""
        if len(text.strip()) < 3:
            return []
        response = self.session.post(
            MUIS_SEARCH_URL,
            json={"text": text},
            headers={"X-CSRF-TOKEN": self.token},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        return list(body.get("data") or [])


def _search_terms(centre: dict[str, Any]) -> list[str]:
    """Return the terms the centre is most likely to be listed under."""
    names = [centre.get("official_name", ""), centre.get("name", "")]
    names.extend(centre.get("aliases", []) or [])
    terms: list[str] = []
    for name in names:
        tokens = distinctive_tokens(name) - _SEARCH_STOP_WORDS
        if not tokens:
            continue
        # The street and the block number identify a centre best.
        for token in sorted(tokens, key=len, reverse=True):
            if len(token) >= 4 and token not in terms:
                terms.append(token)
        if len(terms) >= 4:
            break
    street = centre.get("street_name")
    if street:
        for token in distinctive_tokens(street):
            if len(token) >= 4 and token not in terms:
                terms.append(token)
    return terms[:6]


def _unit_of(address: str) -> Optional[str]:
    """Return the unit or stall number of a register address."""
    match = re.search(r"(?:#\s*|stall\s+)(\d+[A-Za-z]*(?:-\d+[A-Za-z]*)?)", address or "")
    return match.group(1) if match else None


def fetch_establishments(centres: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the certified establishments that sit at an NEA centre."""
    client = MuisClient()
    by_postcode: dict[str, list[dict[str, Any]]] = {}
    for centre in centres:
        for postcode in _postcodes(centre):
            by_postcode.setdefault(postcode, []).append(centre)

    found: dict[str, dict[str, Any]] = {}
    for index, centre in enumerate(centres, start=1):
        for term in _search_terms(centre):
            try:
                results = client.search(term)
            except requests.RequestException as error:
                print(f"  search failed for {term!r}: {error}")
                break
            time.sleep(REQUEST_DELAY_SECONDS)
            for record in results:
                postal = str(record.get("postal") or "").strip()
                if postal not in by_postcode:
                    continue
                key = str(record.get("id") or record.get("number"))
                if key in found:
                    continue
                found[key] = {
                    "name": str(record.get("name") or "").strip(),
                    "address": str(record.get("address") or "").strip(),
                    "unit_number": _unit_of(str(record.get("address") or "")),
                    "postal_code": postal,
                    "certificate_no": str(record.get("number") or "").strip(),
                    "scheme": record.get("subSchemeText") or record.get("schemeText"),
                    "scheme_text": record.get("schemeText"),
                    "sub_scheme": record.get("subSchemeText"),
                    "centre_id": centre["id"],
                }
        if index % 20 == 0:
            print(f"  searched {index}/{len(centres)} centres, {len(found)} establishments")
    return list(found.values())


def _postcodes(centre: dict[str, Any]) -> list[str]:
    values = set()
    for key in ("postal_code", "address", "list_address"):
        value = centre.get(key)
        if not value:
            continue
        postcode = extract_postcode(str(value))
        if postcode:
            values.add(postcode)
    return sorted(values)


def link_to_stalls(
    establishments: list[dict[str, Any]],
    stalls: list[dict[str, Any]],
    centres: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Link the certified establishments to the catalogue stalls."""
    matcher = CentreMatcher(centres)
    by_centre: dict[str, list[dict[str, Any]]] = {}
    for stall in stalls:
        centre_id = stall.get("hawker_centre_id")
        if centre_id:
            by_centre.setdefault(centre_id, []).append(stall)

    linked: dict[str, dict[str, Any]] = {}
    for establishment in establishments:
        centre, method = matcher.match(
            postcode=establishment.get("postal_code"),
            centre_name=establishment.get("name"),
        )
        if not centre:
            continue
        candidates = by_centre.get(centre["id"], [])
        if not candidates:
            continue
        hits: list[dict[str, Any]] = []
        wanted = normalise_name(establishment.get("name"))
        for stall in candidates:
            if normalise_name(stall.get("name")) == wanted:
                hits.append(stall)
        if len(hits) != 1:
            unit = establishment.get("unit_number")
            if unit:
                unit_hits = [
                    stall for stall in candidates
                    if unit.lower() in " ".join(sorted(normalise_unit(stall.get("location")))).lower()
                    or unit.lower() in str(stall.get("location", "")).lower()
                ]
                hits = unit_hits if len(unit_hits) == 1 else []
        if len(hits) != 1:
            continue
        stall = hits[0]
        linked[stall["stall_id"]] = {
            "halal_certificate_no": establishment.get("certificate_no"),
            "halal_source": MUIS_SOURCE_NAME,
            "halal_match": "trader_name" if normalise_name(
                stall.get("name")
            ) == normalise_name(establishment.get("name")) else "unit_number",
        }
    return linked


def main() -> None:
    if not _enabled():
        print(
            "MUIS halal fetching is off. Set ENABLE_MUIS_HALAL=1 to snapshot "
            "the public halal register."
        )
        sys.exit(2)

    centres = json.loads(CENTRES_PATH.read_text(encoding="utf-8"))
    stalls = json.loads(STALLS_PATH.read_text(encoding="utf-8")).get("stalls", [])

    print(f"Searching the MUIS register for {len(centres)} NEA centres...")
    establishments = fetch_establishments(centres)
    print(f"  {len(establishments)} certified establishments at NEA centres")

    linked = link_to_stalls(establishments, stalls, centres)
    print(f"  {len(linked)} catalogue stalls matched")

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    as_of = retrieved_at[:10]
    for stall in stalls:
        match = linked.get(stall.get("stall_id"))
        if not match:
            stall.pop("halal_certified", None)
            stall.pop("halal_certificate_no", None)
            stall.pop("halal_source", None)
            stall.pop("halal_as_of", None)
            continue
        stall["halal_certified"] = True
        stall["halal_certificate_no"] = match["halal_certificate_no"]
        stall["halal_source"] = match["halal_source"]
        stall["halal_as_of"] = as_of
        tags = set(stall.get("dietary_tags", []))
        tags.add("halal_certified")
        stall["dietary_tags"] = sorted(tags)

    payload = json.loads(STALLS_PATH.read_text(encoding="utf-8"))
    payload["stalls"] = stalls
    STALLS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SNAPSHOT_PATH.write_text(
        json.dumps({
            "source": MUIS_SOURCE_NAME,
            "source_url": MUIS_PAGE_URL,
            "licence": (
                "Public e-Service directory. Not a data.gov.sg dataset, so it "
                "is not covered by the Open Data Licence."
            ),
            "retrieved_at": retrieved_at,
            "as_of": as_of,
            "establishment_count": len(establishments),
            "linked_stall_count": len(linked),
            "establishments": establishments,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SNAPSHOT_PATH.name}")
    if not establishments:
        print("No certified establishment was found. The register may have changed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
