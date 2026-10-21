"""Shared text normalisation and record matching for the ingest scripts.

Every ingest script links an external record (HPB Healthier Dining, the MUIS
halal directory, the DBS PayLah merchant list, the hawkercentral listing) to
the stall catalogue. The rules live here so the scripts agree on what counts
as the same unit, the same centre, and the same stall name.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Optional

# Words that appear in the name of nearly every record at a centre and so
# cannot be used to tell two stalls apart.
GENERIC_NAME_WORDS = frozenset({
    "the", "and", "at", "in", "of", "for", "with", "singapore", "s",
    "food", "centre", "center", "hawker", "market", "foodcourt", "food court",
    "coffee", "shop", "eating", "house", "corner", "stall", "coffeeshop",
    "restaurant", "cafe", "catering", "court", "blk", "block", "marketplace",
    "cooked",
})

# The words that carry no meaning when two hawker centres are compared. The
# list is deliberately smaller than ``GENERIC_NAME_WORDS``: a trader writes
# "Changi Village Hawker Centre" where the catalogue writes "Changi Village
# Blk 2 and 3", so "hawker" cannot be treated as noise, while "food" and
# "market" appear in the name of almost every centre.
CENTRE_NAME_STOP_WORDS = frozenset({
    "food", "centre", "center", "market", "hawker", "and", "the", "cooked",
    "blk", "block",
})

# Words that describe a street rather than name one place. They are shared by
# too many centres to identify a single centre on their own.
WEAK_LOCATION_WORDS = frozenset({
    "street", "road", "ave", "avenue", "drive", "lane", "way", "walk", "loop",
    "close", "crescent", "terrace", "rise", "place", "link", "view", "square",
    "block", "blks", "blk", "circle", "gardens", "garden", "park", "field",
    "hill", "heights", "village", "estate", "town", "market", "centre",
    "food", "hawker",
})

# Cyrillic letters that look like Latin ones. The NEA centre list PDF prints
# "ABC Brickworks" with a Cyrillic C, so names are folded before they are
# compared.
_CYRILLIC_LOOKALIKE = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I",
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "ѕ": "s", "і": "i", "ј": "j", "һ": "h", "ԁ": "d", "ӏ": "l", "ԛ": "q",
    "ԝ": "w", "ν": "v", "ё": "e", "Ё": "E", "І": "I", "Ѕ": "S",
})

# Unit formats seen across the sources: "#01-29", "01-29", "STALL 101",
# "#B1-K3". The canonical form is the alphanumeric run without separators.
_UNIT_TOKEN_RE = re.compile(r"([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)")
_UNIT_HINT_RE = re.compile(
    r"(?:#\s*|stall\s+|unit\s+|no\.?\s*|#\s*stall\s+)([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)",
    re.I,
)
_POSTCODE_RE = re.compile(r"\b(\d{6})\b")
_PDF_POSTCODE_RE = re.compile(r"S\s*\(\s*(\d{6})\s*\)")


def normalise_name(value: Optional[str]) -> str:
    """Fold a name to a comparable form.

    ``&`` and ``and`` are treated as the same word because the NEA list and
    the NEA geospatial data spell the same centre both ways. ``Blk`` and
    ``Block`` are treated as the same word for the same reason, and Cyrillic
    look-alike letters are folded onto their Latin counterpart.
    """
    if not value:
        return ""
    text = str(value).translate(_CYRILLIC_LOOKALIKE).replace("&", " and ")
    text = re.sub(r"\bblk\b", "block", text.casefold())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def name_tokens(value: Optional[str]) -> set[str]:
    """Return the distinctive words of a name, ignoring centre-wide words."""
    tokens = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split()
    return {token for token in tokens if len(token) >= 2 and token not in GENERIC_NAME_WORDS}


def name_token_set(value: Optional[str]) -> set[str]:
    """Return every comparable token of a name.

    Numbers are kept whatever their length, because a block number ("Blk 79")
    is the most precise link between a merchant listing and a centre. Single
    letters are dropped because they carry no meaning on their own.
    """
    tokens: set[str] = set()
    for token in normalise_name(value).split():
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens


def distinctive_tokens(value: Optional[str]) -> set[str]:
    """Return the tokens of a name that can tell two places apart."""
    return {token for token in name_token_set(value) if token not in GENERIC_NAME_WORDS}


def numeric_tokens(value: Optional[str]) -> set[str]:
    """Return the number-like tokens of a name, such as a block number."""
    return {
        token for token in name_token_set(value)
        if any(character.isdigit() for character in token)
    }


def extract_postcode(value: Optional[str]) -> Optional[str]:
    """Return the first six-digit postcode in a string."""
    if not value:
        return None
    match = _PDF_POSTCODE_RE.search(str(value))
    if match:
        return match.group(1)
    match = _POSTCODE_RE.search(str(value))
    return match.group(1) if match else None


def extract_postcodes(value: Optional[str]) -> list[str]:
    """Return every six-digit postcode in a string, in order."""
    if not value:
        return []
    text = str(value)
    found = [match.group(1) for match in _PDF_POSTCODE_RE.finditer(text)]
    for match in _POSTCODE_RE.finditer(text):
        if match.group(1) not in found:
            found.append(match.group(1))
    return found


def _floor_unit_forms(token: str) -> set[str]:
    """Return the comparable floor-and-unit forms of a unit token.

    A unit is written with a padded floor and a padded unit number in some
    sources and without padding in others, so ``#02-008`` and ``#02-8`` have
    to compare equal.
    """
    text = str(token).strip().casefold()
    if "-" not in text:
        return set()
    floor, _, unit = text.partition("-")
    floor_digits = "".join(character for character in floor if character.isdigit())
    unit_letters = "".join(character for character in unit if character.isalpha())
    unit_digits = "".join(character for character in unit if character.isdigit())
    if not floor_digits:
        return set()
    forms = set()
    if unit_digits:
        forms.add(f"{int(floor_digits)}-{int(unit_digits)}{unit_letters}")
    elif unit_letters:
        forms.add(f"{int(floor_digits)}-{unit_letters}")
    return forms


def normalise_unit(value: Optional[str]) -> set[str]:
    """Return the comparable forms of a unit number.

    A unit is written in several ways across the sources. ``#01-29`` becomes
    ``0129``. A bare stall number such as ``STALL 101`` also yields ``0101``
    because the first digit of a hawker stall number is the floor.
    """
    if not value:
        return set()
    text = str(value)
    match = _UNIT_HINT_RE.search(text)
    token = match.group(1) if match else None
    if token is None:
        tokens = _UNIT_TOKEN_RE.findall(text)
        token = tokens[-1] if tokens else None
    if not token:
        return set()
    canonical = re.sub(r"[^a-z0-9]", "", token.casefold())
    if not canonical:
        return set()
    candidates = {canonical}
    if canonical.isdigit() and len(canonical) == 3:
        candidates.add(f"0{canonical}")
    candidates |= _floor_unit_forms(token)
    return candidates


def unit_matches(left: Optional[str], right: Optional[str]) -> bool:
    """Return True when two unit numbers refer to the same unit."""
    left_units = normalise_unit(left)
    right_units = normalise_unit(right)
    if not left_units or not right_units:
        return False
    return bool(left_units & right_units)


def street_key(value: Optional[str]) -> frozenset[str]:
    """Return the distinctive words of a street name.

    The words that every street shares are dropped, so "Blk 51 Upper Bukit
    Timah Road" and "Upper Bukit Timah Road" compare the same way.
    """
    return frozenset(distinctive_tokens(value))


def _street_hints(address: Optional[str]) -> list[str]:
    """Return the street parts of an address.

    An address is written "<street>, <area>, Singapore <postcode>", so the
    parts before the postcode are the candidates.
    """
    if not address:
        return []
    text = re.sub(r",\s*Singapore\s*\d{6}.*$", "", str(address), flags=re.I)
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts


class CentreMatcher:
    """Link source records to NEA hawker centres.

    A link is only accepted when it is unique. Ambiguous records stay
    unlinked so a stall is never attached to the wrong centre.
    """

    def __init__(self, centres: Iterable[dict[str, Any]]):
        self.centres = list(centres)
        self._by_postcode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_street: dict[frozenset[str], list[dict[str, Any]]] = defaultdict(list)
        for centre in self.centres:
            for postcode in extract_postcodes(centre.get("address", "")):
                self._by_postcode[postcode].append(centre)
            for key in self._centre_names(centre):
                self._by_name[key].append(centre)
            for street in self._centre_streets(centre):
                key = street_key(street)
                if key:
                    self._by_street[key].append(centre)

    @staticmethod
    def _centre_names(centre: dict[str, Any]) -> set[str]:
        names = {centre.get("name", ""), centre.get("official_name", "")}
        names.update(centre.get("aliases", []) or [])
        return {normalise_name(name) for name in names if name}

    @staticmethod
    def _centre_streets(centre: dict[str, Any]) -> set[str]:
        streets = set()
        if centre.get("street_name"):
            streets.add(str(centre["street_name"]))
        for part in _street_hints(centre.get("address", ""))[1:]:
            streets.add(part)
        return streets

    def by_postcode(self, postcode: Optional[str]) -> list[dict[str, Any]]:
        return list(self._by_postcode.get(postcode or "", []))

    def by_name(self, name: Optional[str]) -> list[dict[str, Any]]:
        return list(self._by_name.get(normalise_name(name), []))

    def by_street(self, street: Optional[str]) -> list[dict[str, Any]]:
        key = street_key(street)
        return list(self._by_street.get(key, [])) if key else []

    def match(
        self,
        *,
        postcode: Optional[str] = None,
        centre_name: Optional[str] = None,
        street: Optional[str] = None,
        address: Optional[str] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Return the centre for a source record and how it was matched.

        An exact postcode match wins. A postcode shared by several centres is
        narrowed by the street, and a record whose postcode is unknown is
        matched by a unique centre name. A record whose centre is named the
        way the traders use it, and whose address names the street of one
        centre only, is matched on the street. Ambiguous records stay
        unlinked so a stall is never attached to the wrong centre.
        """
        by_postcode = self.by_postcode(postcode)
        if len(by_postcode) == 1:
            return by_postcode[0], "exact_postcode"
        if len(by_postcode) > 1:
            wanted = normalise_name(centre_name)
            if wanted:
                narrowed = [
                    centre for centre in by_postcode
                    if wanted in self._centre_names(centre)
                ]
                if len(narrowed) == 1:
                    return narrowed[0], "name_unique"
            if street:
                narrowed = [
                    centre for centre in by_postcode
                    if street and street.casefold() in str(
                        centre.get("address", "")
                    ).casefold()
                ]
                if len(narrowed) == 1:
                    return narrowed[0], "unit_postcode"
            return None, None
        by_name = self.by_name(centre_name)
        if len(by_name) == 1:
            return by_name[0], "name_unique"
        if address:
            for hint in _street_hints(address):
                candidates = self.by_street(hint)
                if len(candidates) != 1:
                    continue
                centre = candidates[0]
                if self._shares_name_token(centre, centre_name):
                    return centre, "street_unique"
        return None, None

    @staticmethod
    def _shares_name_token(
        centre: dict[str, Any], centre_name: Optional[str]
    ) -> bool:
        """Return True when the two names share a word that identifies one."""
        wanted = distinctive_tokens(centre_name)
        if not wanted:
            return False
        have = set()
        for name in centre_names(centre):
            have |= distinctive_tokens(name)
        shared = (wanted & have) - WEAK_LOCATION_WORDS
        return bool(shared)


def centre_names(centre: dict[str, Any]) -> set[str]:
    """Return every name a centre is known by.

    The catalogue keeps the official name, the display name and any renamed
    name as an alias. Some display names carry the former name of the centre
    in brackets, which is the name a merchant directory is likely to use.
    """
    names = {centre.get("name", ""), centre.get("official_name", "")}
    names.update(centre.get("aliases", []) or [])
    for value in list(names):
        match = re.search(r"\(([^)]+)\)", value or "")
        if match:
            names.add(match.group(1))
    return {name for name in names if name}


class CentreDirectoryMatcher:
    """Link merchant-directory centre headings to NEA hawker centres.

    A merchant directory names a centre the way the traders use it, which is
    usually the street and the block rather than the official name, and the
    heading often starts with the block number. A heading is linked by, in
    order of strength:

    1. an exact match on a name the centre is known by;
    2. the block number, when exactly one centre carries it;
    3. a match on every word of the heading;
    4. a match on the distinctive words of the heading.

    A link is only returned when it is unique. An ambiguous heading stays
    unlinked so a merchant listing is never attached to the wrong centre.
    """

    # Confidence per method: how sure the link is, from 1 (certain) down.
    CONFIDENCE = {
        "exact_name": 1.0,
        "block_number": 0.95,
        "full_words": 0.9,
        "distinctive_words": 0.75,
    }

    # A distinctive-word match must reach this share of the heading's
    # distinctive words before it is accepted.
    MIN_WORD_SHARE = 0.6

    def __init__(self, centres: Iterable[dict[str, Any]]):
        self.centres = list(centres)
        self._entries: list[dict[str, Any]] = []
        for centre in self.centres:
            names = centre_names(centre)
            tokens: set[str] = set()
            for name in names:
                tokens |= name_token_set(name)
            self._entries.append({
                "centre": centre,
                "names": names,
                "tokens": tokens,
                "distinctive": tokens - CENTRE_NAME_STOP_WORDS,
            })

    def match(
        self, heading: Optional[str]
    ) -> tuple[Optional[dict[str, Any]], Optional[str], float]:
        """Return the centre a directory heading belongs to.

        The method and the confidence of the link are returned alongside the
        centre. When the heading cannot be linked, the centre and the method
        are ``None`` and the confidence is ``0.0``.
        """
        if not heading:
            return None, None, 0.0
        wanted = normalise_name(heading)
        numbers = numeric_tokens(heading)

        pool = [entry for entry in self._entries if numbers & entry["tokens"]]
        if not pool:
            pool = list(self._entries)

        exact = [
            entry for entry in pool
            if any(normalise_name(name) == wanted for name in entry["names"])
        ]
        if len(exact) == 1:
            return exact[0]["centre"], "exact_name", self.CONFIDENCE["exact_name"]
        if len(exact) > 1:
            return None, None, 0.0

        if len(pool) == 1:
            return pool[0]["centre"], "block_number", self.CONFIDENCE["block_number"]

        wanted_all = name_token_set(heading)
        full = [
            entry for entry in pool
            if wanted_all and wanted_all <= entry["tokens"]
        ]
        if len(full) == 1:
            return full[0]["centre"], "full_words", self.CONFIDENCE["full_words"]
        if len(full) > 1:
            return None, None, 0.0

        wanted_distinctive = name_token_set(heading) - CENTRE_NAME_STOP_WORDS
        if not wanted_distinctive:
            return None, None, 0.0
        scored = []
        for entry in pool:
            overlap = wanted_distinctive & entry["distinctive"]
            if not overlap:
                continue
            scored.append((len(overlap) / len(wanted_distinctive), len(overlap), entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not scored or scored[0][0] < self.MIN_WORD_SHARE:
            return None, None, 0.0
        best = scored[0]
        if len(scored) > 1:
            second = scored[1]
            if best[2] is not second[2] and best[0] - second[0] <= 0.05:
                return None, None, 0.0
        return best[2]["centre"], "distinctive_words", self.CONFIDENCE["distinctive_words"]


class StallMatcher:
    """Link external records to catalogue stalls.

    A record is linked by unit number and postcode first, and by the
    normalised outlet name and centre name second. Only unique matches are
    returned.
    """

    def __init__(self, stalls: Iterable[dict[str, Any]]):
        self.stalls = [stall for stall in stalls if stall.get("stall_id")]
        self._by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._by_postcode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for stall in self.stalls:
            postcode = extract_postcode(stall.get("address", ""))
            unit = _stall_unit(stall)
            if postcode:
                self._by_postcode[postcode].append(stall)
                if unit:
                    self._by_unit[(postcode, unit)].append(stall)
            centre_key = normalise_name(stall.get("hawker_centre", ""))
            name_key = normalise_name(stall.get("name", ""))
            if centre_key and name_key:
                self._by_name[(centre_key, name_key)].append(stall)

    def match(
        self,
        *,
        postcode: Optional[str],
        unit: Optional[str] = None,
        name: Optional[str] = None,
        centre_name: Optional[str] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Return the catalogue stall for a source record and the match method."""
        if postcode and unit:
            candidates = [
                stall for stall in self._by_postcode.get(postcode, [])
                if unit_matches(unit, _stall_unit(stall))
            ]
            if len(candidates) == 1:
                return candidates[0], "unit_postcode"
        if name and centre_name:
            key = (normalise_name(centre_name), normalise_name(name))
            candidates = list(self._by_name.get(key, []))
            if len(candidates) == 1:
                return candidates[0], "name_centre"
        if name and postcode:
            wanted = normalise_name(name)
            centre_key = normalise_name(centre_name or "")
            candidates = [
                stall for stall in self._by_postcode.get(postcode, [])
                if normalise_name(stall.get("name", "")) == wanted
                and (
                    not centre_key
                    or centre_key == normalise_name(stall.get("hawker_centre", ""))
                )
            ]
            if len(candidates) == 1:
                return candidates[0], "name_postcode"
        return None, None


def _stall_unit(stall: dict[str, Any]) -> Optional[str]:
    """Return the raw unit token of a catalogue stall."""
    location = str(stall.get("location") or "")
    match = re.search(r"#\s*([A-Za-z0-9/-]+)", location)
    if match:
        return match.group(1)
    address = str(stall.get("address") or "")
    match = re.search(r"#\s*([A-Za-z0-9/-]+)", address)
    return match.group(1) if match else None
