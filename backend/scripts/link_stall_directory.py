"""Link the DBS PayLah! merchant directory to the NEA centre catalogue.

The directory in ``data/hawker_stall_directory.json`` lists the merchants
taking part in the DBS PayLah! Saturdays promotion: a centre heading, a
merchant name and a unit number, and nothing else. It carries no menu, no
price, no dietary field and no opening hours, so none of those may be read
into it.

The script links each heading to an NEA centre, writes the linked records to
``data/hawker_stall_directory_linked.json``, records the merchants of every
linked centre on the centre itself, and joins the directory onto the stalls
of the Wak-Wak catalogue where the same unit is listed by both.

Usage (from backend/):
    python scripts/link_stall_directory.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.source_matching import (  # noqa: E402
    CentreDirectoryMatcher, centre_names, normalise_name, normalise_unit,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DIRECTORY_PATH = DATA_DIR / "hawker_stall_directory.json"
LINKED_PATH = DATA_DIR / "hawker_stall_directory_linked.json"
CENTRES_PATH = DATA_DIR / "hawker_centres.json"
STALLS_PATH = DATA_DIR / "hawker_stall.json"

MERCHANT_SOURCE = "DBS PayLah! Saturdays participating merchants"
# The source PDF carries no publication date. The date below is the day the
# extract was taken into this repository, which is what the repository can
# vouch for.
MERCHANT_AS_OF = "2026-09-30"

# The link rate the directory must keep to be useful. A drop below it means
# the directory or the catalogue changed shape and the script stops.
MIN_HEADING_LINK_RATE = 0.9
MIN_RECORD_LINK_RATE = 0.9

_LEADING_NUMBER_RE = re.compile(r"^\s*\d+[\s,.-]+")


def _merchant_name(value: Optional[str]) -> str:
    """Return the merchant name without a leading merchant number."""
    if not value:
        return ""
    return _LEADING_NUMBER_RE.sub("", str(value)).strip()


def _unit_key(value: Optional[str]) -> Optional[str]:
    """Return a comparable key for a unit number."""
    forms = normalise_unit(value)
    if not forms:
        return None
    return sorted(forms)[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def link_headings(
    matcher: CentreDirectoryMatcher, headings: list[str]
) -> dict[str, dict[str, Any]]:
    """Return the link of every directory heading."""
    links: dict[str, dict[str, Any]] = {}
    for heading in headings:
        centre, method, confidence = matcher.match(heading)
        links[heading] = {
            "heading": heading,
            "centre_id": centre["id"] if centre else None,
            "centre_name": centre["name"] if centre else None,
            "match_method": method,
            "match_confidence": confidence if centre else 0.0,
        }
    return links


def join_stalls(
    stalls: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Join directory records onto the stalls of the Wak-Wak catalogue.

    A record joins a catalogue stall when both name the same unit of the same
    centre *and* the two listings call the stall the same name. A unit that
    carries a different name in the directory was let to another trader, so
    the record is kept on the centre instead of on the stall.
    """
    by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for stall in stalls:
        centre_id = stall.get("hawker_centre_id")
        if not centre_id:
            continue
        unit = _unit_key(stall.get("location") or stall.get("address"))
        if unit:
            by_unit[(centre_id, unit)].append(stall)
        name = normalise_name(stall.get("name", ""))
        if name:
            by_name[(centre_id, name)].append(stall)

    joins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    joined = 0
    for record in records:
        centre_id = record.get("centre_id")
        if not centre_id:
            continue
        merchant_name = normalise_name(record.get("merchant_name", ""))
        candidates = by_unit.get((centre_id, record.get("unit_key") or ""), [])
        candidates = [
            stall for stall in candidates
            if normalise_name(stall.get("name", "")) == merchant_name
        ]
        if len(candidates) != 1:
            candidates = by_name.get((centre_id, merchant_name), [])
        if len(candidates) != 1:
            continue
        stall = candidates[0]
        joined += 1
        joins[stall["stall_id"]].append(record)
    return joins, joined


def main() -> None:
    directory = load_json(DIRECTORY_PATH)
    centres = load_json(CENTRES_PATH)
    catalogue = load_json(STALLS_PATH)

    records = directory.get("stalls", [])
    headings = sorted({record["hawker_centre"] for record in records})
    matcher = CentreDirectoryMatcher(centres)
    links = link_headings(matcher, headings)

    linked_records: list[dict[str, Any]] = []
    unlinked: list[dict[str, Any]] = []
    records_by_heading: dict[str, list[dict[str, Any]]] = defaultdict(list)
    heading_record_counts: dict[str, int] = defaultdict(int)
    for record in records:
        heading_record_counts[record["hawker_centre"]] += 1
        link = links[record["hawker_centre"]]
        if not link["centre_id"]:
            continue
        linked = {
            "centre_id": link["centre_id"],
            "stall_name": record.get("stall_name"),
            "merchant_name": _merchant_name(record.get("stall_name")),
            "unit_number": record.get("unit_number"),
            "unit_key": _unit_key(record.get("unit_number")),
            "source_page": record.get("source_page"),
        }
        linked_records.append(linked)
        records_by_heading[record["hawker_centre"]].append(linked)

    centre_reports: list[dict[str, Any]] = []
    merchants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for heading in headings:
        link = links[heading]
        heading_records = records_by_heading.get(heading, [])
        heading_count = heading_record_counts.get(heading, 0)
        if not link["centre_id"]:
            unlinked.append({
                "heading": heading,
                "record_count": heading_count,
            })
            continue
        centre_reports.append({
            "centre_id": link["centre_id"],
            "centre_name": link["centre_name"],
            "heading": heading,
            "match_method": link["match_method"],
            "match_confidence": link["match_confidence"],
            "record_count": heading_count,
        })
        for record in heading_records:
            merchants[link["centre_id"]].append({
                "name": record.get("merchant_name") or record.get("stall_name"),
                "unit_number": record.get("unit_number"),
            })

    # Keep the merchant list of every centre free of duplicates. A row that
    # names neither a merchant nor a unit carries no information, so it is
    # dropped rather than listed as an empty stall.
    for centre_id, entries in merchants.items():
        seen = set()
        unique = []
        for entry in entries:
            if not entry["name"] and not entry["unit_number"]:
                continue
            key = (entry["name"], entry["unit_number"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(entry)
        merchants[centre_id] = unique

    joins, joined_count = join_stalls(catalogue.get("stalls", []), linked_records)
    for stall in catalogue.get("stalls", []):
        records_for_stall = joins.get(stall.get("stall_id"), [])
        if not records_for_stall:
            stall.pop("merchant_list_name", None)
            stall.pop("merchant_unit_number", None)
            stall.pop("merchant_source", None)
            continue
        first = records_for_stall[0]
        stall["merchant_list_name"] = first.get("merchant_name") or first.get("stall_name")
        stall["merchant_unit_number"] = first.get("unit_number")
        stall["merchant_source"] = MERCHANT_SOURCE

    for centre in centres:
        centre_id = centre.get("id")
        if centre_id in merchants:
            centre["merchant_stalls"] = merchants[centre_id]
            centre["merchant_source"] = MERCHANT_SOURCE
            centre["merchant_as_of"] = MERCHANT_AS_OF
        else:
            centre.pop("merchant_stalls", None)
            centre.pop("merchant_source", None)
            centre.pop("merchant_as_of", None)

    linked_heading_count = len(centre_reports)
    linked_record_count = len(linked_records)
    heading_rate = linked_heading_count / len(headings) if headings else 0.0
    record_rate = linked_record_count / len(records) if records else 0.0

    output = {
        "source": MERCHANT_SOURCE,
        "source_file": directory.get("source_file"),
        "source_pages": directory.get("source_pages"),
        "as_of": MERCHANT_AS_OF,
        "record_count": len(records),
        "linked_record_count": linked_record_count,
        "heading_count": len(headings),
        "linked_heading_count": linked_heading_count,
        "heading_link_rate": round(heading_rate, 4),
        "record_link_rate": round(record_rate, 4),
        "catalogue_join_count": joined_count,
        "centres": centre_reports,
        "unlinked_headings": unlinked,
        "records": linked_records,
    }
    LINKED_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CENTRES_PATH.write_text(
        json.dumps(centres, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    STALLS_PATH.write_text(
        json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {LINKED_PATH.name}")
    print(f"  headings linked: {linked_heading_count}/{len(headings)} ({heading_rate:.1%})")
    print(f"  records linked:  {linked_record_count}/{len(records)} ({record_rate:.1%})")
    print(f"  catalogue stalls joined: {joined_count}")
    methods: dict[str, int] = defaultdict(int)
    for report in centre_reports:
        methods[report["match_method"]] += 1
    for method, count in sorted(methods.items()):
        print(f"    {method}: {count}")
    if unlinked:
        print(f"  {len(unlinked)} heading(s) could not be linked:")
        for entry in unlinked:
            print(f"    {entry['heading']} ({entry['record_count']} records)")

    if heading_rate < MIN_HEADING_LINK_RATE or record_rate < MIN_RECORD_LINK_RATE:
        print(
            "The directory link rate fell below "
            f"{MIN_HEADING_LINK_RATE:.0%}; the directory or the centre "
            "catalogue has probably changed shape."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
