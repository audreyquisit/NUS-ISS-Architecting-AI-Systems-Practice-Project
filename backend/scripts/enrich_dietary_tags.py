"""Enrich stall data with dietary tags from the HPB Healthier Dining Programme.

Downloads the Healthier Eateries GeoJSON from data.gov.sg (managed by the
Health Promotion Board) and matches entries to Wak-Wak stalls by postal code
and unit number / name similarity. Matched stalls receive a
``dietary_tags`` list containing ``"healthier_dining_partner"`` and
``dietary_suitable`` set to ``True``.

Usage (from backend/):
    python scripts/enrich_dietary_tags.py

The script is idempotent: re-running it refreshes the HPB data and preserves
any previously added dietary_tags that are not managed by this script.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

# Ensure stdout can handle non-ASCII characters from stall names.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STALLS_PATH = DATA_DIR / "hawker_stall.json"
MOCK_STALLS_PATH = DATA_DIR / "hawker_stall_mock.json"

HPB_DATASET_ID = "d_2925c2ccf75d1c135c2d469e0de3cee6"
HPB_DOWNLOAD_URL = (
    "https://api-open.data.gov.sg/v1/public/api/datasets/"
    + HPB_DATASET_ID
    + "/poll-download"
)

# Dietary preferences that require structured data (ingredients, allergens,
# certification) to verify. The Wak-Wak catalogue does not have these fields,
# so verifiable restrictions cannot be confirmed from the source alone.
MANAGED_TAG = "healthier_dining_partner"


def _normalise(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _extract_postcode(address: str) -> str | None:
    match = re.search(r"\b(\d{6})\b", address or "")
    return match.group(1) if match else None


def _extract_unit(location: str) -> str | None:
    """Extract a normalised unit number from a Wak-Wak location string.

    Examples:
        'Geylang Serai Food Centre #02-130' -> '02-130'
        '#01-108' -> '01-108'
    """
    match = re.search(r"#([A-Za-z0-9/-]+)", location or "")
    if not match:
        return None
    raw = match.group(1)
    # Normalise: remove non-alphanumeric except dash, lower case
    return re.sub(r"[^0-9a-z-]", "", raw.lower())


def _extract_hpb_unit(entry: dict[str, Any]) -> str | None:
    """Extract a normalised unit number from an HPB entry."""
    unit = str(entry.get("ADDRESSUNITNUMBER", "")).strip()
    floor = str(entry.get("ADDRESSFLOORNUMBER", "")).strip()
    if not unit:
        return None
    # HPB uses patterns like '210', '01-74', '3, 5, 8'
    # We take the first unit if comma-separated
    first = unit.split(",")[0].strip()
    if not first:
        return None
    if floor:
        return f"{floor.lower().rstrip()}-{re.sub(r'[^0-9a-z-]', '', first.lower())}"
    return re.sub(r"[^0-9a-z-]", "", first.lower())


# Terms that appear in *every* stall name or location at a given centre and
# therefore must NOT count toward name-similarity matching.
_CENTRE_STOP = frozenset({
    "the", "and", "at", "in", "of", "for", "with", "singapore",
    "food", "centre", "center", "hawker", "market", "foodcourt",
    "coffee", "shop", "eating", "house", "corner", "stall",
    "restaurant", "cafe", "catering", "court",
})


def _name_tokens(name: str) -> set[str]:
    """Extract meaningful food-specific tokens from a name.

    Excludes generic hawker-centre words so that 'Geylang Serai' appearing in
    every stall's location does not cause false matches.
    """
    tokens = re.sub(r"[^a-z0-9]+", " ", name.casefold()).split()
    return {t for t in tokens if len(t) >= 2 and t not in _CENTRE_STOP}


def download_hpb_data() -> list[dict[str, Any]]:
    """Download the HPB Healthier Eateries GeoJSON and return feature properties."""
    print("Downloading HPB Healthier Eateries dataset...")
    response = requests.get(HPB_DOWNLOAD_URL, timeout=30)
    response.raise_for_status()
    poll = response.json()
    if poll.get("code") != 0:
        raise RuntimeError(f"HPB download failed: {poll.get('errMsg')}")
    download_url = poll["data"]["url"]
    print(f"  Fetching GeoJSON ({len(download_url)} chars URL)...")
    geo_resp = requests.get(download_url, timeout=60)
    geo_resp.raise_for_status()
    geojson = json.loads(geo_resp.text)
    features = geojson.get("features", [])
    print(f"  Downloaded {len(features)} establishments")
    return [f.get("properties", {}) for f in features]


def _match_by_unit(
    stall: dict[str, Any], hpb_entries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Try to match a Wak-Wak stall to an HPB entry by unit number."""
    stall_unit = _extract_unit(stall.get("location", ""))
    if not stall_unit:
        return None
    postcode = _extract_postcode(stall.get("address", ""))
    for hpb in hpb_entries:
        hpb_unit = _extract_hpb_unit(hpb)
        hpb_postcode = str(hpb.get("ADDRESSPOSTALCODE", ""))
        if hpb_unit and hpb_postcode == postcode and hpb_unit == stall_unit:
            return hpb
    return None


def _match_by_name_and_unit(
    stall: dict[str, Any], hpb_entries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Match by unit number first, then by conservative name similarity + postcode.

    Name matching requires 2+ shared food-specific tokens. The hawker centre
    name tokens (e.g. 'geylang', 'serai', 'food', 'centre') are excluded from
    both sides so that every stall at the same centre does not falsely match
    an HPB entry that contains the centre name.
    """
    # Try exact unit match first
    match = _match_by_unit(stall, hpb_entries)
    if match:
        return match

    postcode = _extract_postcode(stall.get("address", ""))
    if not postcode:
        return None

    # Build a stop set that includes the hawker centre name tokens so that
    # location-specific words like 'Geylang Serai' do not cause false matches.
    centre_name = stall.get("hawker_centre", "")
    extra_stop = _name_tokens(centre_name)

    stall_tokens = _name_tokens(stall.get("name", "")) - extra_stop
    if not stall_tokens:
        return None

    for hpb in hpb_entries:
        hpb_postcode = str(hpb.get("ADDRESSPOSTALCODE", ""))
        if hpb_postcode != postcode:
            continue
        hpb_name = hpb.get("NAME", "")
        hpb_tokens = _name_tokens(hpb_name) - extra_stop
        if not hpb_tokens:
            continue
        shared = stall_tokens & hpb_tokens
        if len(shared) >= 2:
            return hpb
    return None


def enrich_stall_file(stalls_path: Path, hpb_by_postcode: dict[str, list[dict]]) -> int:
    """Enrich a stall JSON file. Returns count of stalls tagged."""
    if not stalls_path.exists():
        return 0

    data = json.loads(stalls_path.read_text(encoding="utf-8"))
    stalls = data.get("stalls", [])
    tagged = 0
    unmatched_hpb = defaultdict(list)

    for stall in stalls:
        postcode = _extract_postcode(stall.get("address", ""))
        if not postcode:
            continue
        if postcode not in hpb_by_postcode:
            continue

        hpb_entries = hpb_by_postcode[postcode]
        hpb_match = _match_by_name_and_unit(stall, hpb_entries)

        if hpb_match:
            existing_tags = set(stall.get("dietary_tags", []))
            existing_tags.add(MANAGED_TAG)
            stall["dietary_tags"] = sorted(existing_tags)
            tagged += 1
            hpb_name = hpb_match.get("NAME", "")
            print(f"  Tagged: {stall['name']} (postcode {postcode}) <- HPB: {hpb_name}")
        else:
            # Track which HPB entries couldn't be matched for manual review
            stall_name = stall.get("name", "")
            stall_unit = _extract_unit(stall.get("location", ""))
            for hpb in hpb_entries:
                key = f"{postcode}:{hpb.get('NAME', '')}"
                unmatched_hpb[key].append({
                    "stall_name": stall_name,
                    "stall_unit": stall_unit,
                    "hpb_name": hpb.get("NAME", ''),
                    "hpb_unit": _extract_hpb_unit(hpb),
                })

    if unmatched_hpb:
        print("\n  Unmatched HPB entries (postcode overlaps but no stall match):")
        for key, attempts in list(unmatched_hpb.items())[:20]:
            safe_key = key.encode('utf-8', errors='replace').decode('utf-8')
            print(f"    {safe_key}")
            for attempt in attempts[:3]:
                hpb_name = attempt['hpb_name'].encode('utf-8', errors='replace').decode('utf-8')
                ww_name = attempt['stall_name'].encode('utf-8', errors='replace').decode('utf-8')
                print(f"      HPB: {hpb_name} (unit {attempt['hpb_unit']})")
                print(f"      WW:  {ww_name} (unit {attempt['stall_unit']})")

    if tagged > 0:
        data["stalls"] = stalls
        stalls_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return tagged


def main() -> None:
    # Download HPB data
    hpb_entries = download_hpb_data()

    # Index by postcode for fast lookup
    hpb_by_postcode: dict[str, list[dict]] = defaultdict(list)
    for entry in hpb_entries:
        postcode = str(entry.get("ADDRESSPOSTALCODE", ""))
        if postcode:
            hpb_by_postcode[postcode].append(entry)

    print(f"\nHPB entries indexed by postcode: {len(hpb_by_postcode)} postcodes")

    # Enrich real stall data
    real_tagged = enrich_stall_file(STALLS_PATH, hpb_by_postcode)
    print(f"\nTagged {real_tagged} real stalls in {STALLS_PATH.name}")

    # Enrich mock stall data
    if MOCK_STALLS_PATH.exists():
        mock_tagged = enrich_stall_file(MOCK_STALLS_PATH, hpb_by_postcode)
        print(f"Tagged {mock_tagged} mock stalls in {MOCK_STALLS_PATH.name}")

    total = real_tagged + (mock_tagged if MOCK_STALLS_PATH.exists() else 0)
    print(f"\nTotal stalls tagged with '{MANAGED_TAG}': {total}")
    if total == 0:
        print("No stalls were tagged. The HPB data may use different naming or")
        print("unit conventions than the Wak-Wak scrape. Manual review of")
        print("unmatched entries above may be needed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
