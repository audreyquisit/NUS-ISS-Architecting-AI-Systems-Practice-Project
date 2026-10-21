"""Read the NEA hawker-centre list PDF with the standard library only.

The NEA publishes "List of Hawker Centres" as a PDF table. The file is a
Flate-compressed PDF whose page content streams hold positioned text runs, so
the table can be read without a PDF dependency: objects are located by their
``N G obj`` header, page content streams are concatenated per page, and the
text runs are grouped into rows by their vertical position.

Only what the table needs is implemented: positioned text, the four table
columns, and the row/table split. Anything the reader cannot interpret is
skipped rather than guessed.
"""

from __future__ import annotations

import re
import zlib
from typing import Any, Iterable

# Column boundaries in PDF user-space units. The list renders the serial
# number, the centre name, the address, and the managing entity in four
# fixed columns; a run left of a boundary belongs to the column on its right.
_COLUMN_BOUNDS = (75.0, 205.0, 400.0)

# Rows are ~11 units apart, but a wrapped cell can sit a few units below the
# rest of its row. Two and a half units is enough to regroup those cells
# without ever merging two consecutive rows.
_ROW_TOLERANCE = 2.5

_OBJECT_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj")
_STREAM_RE = re.compile(rb"stream\r\n|stream\n|stream\r")
_TEXT_MATRIX_RE = re.compile(rb"1 0 0 1 ([\d.+-]+) ([\d.+-]+) Tm")
_SHOW_TEXT_RE = re.compile(
    rb"\[((?:[^\[\]\\]|\\.)*)\]\s*TJ|\((?:\\.|[^\\()])*\)\s*Tj", re.S
)
_STRING_RE = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)

_ESCAPES = {
    ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12,
    ord("("): 40, ord(")"): 41, ord("\\"): 92,
}


class PdfTableError(RuntimeError):
    """The PDF could not be read as a table."""


def _decode_string(raw: bytes) -> str:
    """Decode a PDF literal string body into text."""
    out = bytearray()
    index = 0
    while index < len(raw):
        char = raw[index]
        if char != 0x5C or index + 1 >= len(raw):
            out.append(char)
            index += 1
            continue
        marker = raw[index + 1]
        if marker in _ESCAPES:
            out.append(_ESCAPES[marker])
            index += 2
            continue
        if 0x30 <= marker <= 0x37:
            digits = b""
            cursor = index + 1
            while cursor < len(raw) and len(digits) < 3 and 0x30 <= raw[cursor] <= 0x37:
                digits += bytes([raw[cursor]])
                cursor += 1
            out.append(int(digits, 8) & 0xFF)
            index = cursor
            continue
        if marker in (10, 13):
            index += 2
            continue
        out.append(marker)
        index += 2
    return out.decode("latin-1")


def _objects(raw: bytes) -> dict[int, tuple[bytes, bytes | None]]:
    """Index the indirect objects of a PDF file."""
    found: dict[int, tuple[bytes, bytes | None]] = {}
    for match in _OBJECT_RE.finditer(raw):
        number = int(match.group(1))
        start = match.end()
        end = raw.find(b"endobj", start)
        body = raw[start:end] if end != -1 else raw[start:]
        stream = _STREAM_RE.search(body)
        if not stream:
            found[number] = (body, None)
            continue
        header = body[:stream.start()]
        payload_start = stream.end()
        length_match = re.search(rb"/Length\s+(\d+)", header)
        if length_match:
            length = int(length_match.group(1))
            payload = body[payload_start:payload_start + length]
        else:
            payload = body[payload_start:]
        found[number] = (header, payload)
    return found


def _stream_bytes(header: bytes, payload: bytes | None) -> bytes:
    if payload is None:
        return b""
    if b"/FlateDecode" in header:
        try:
            return zlib.decompress(payload)
        except zlib.error:
            return b""
    return payload


def _reference_list(value: bytes) -> list[int]:
    return [int(item) for item in re.findall(rb"(\d+)\s+\d+\s+R", value)]


def _page_numbers(objects: dict[int, tuple[bytes, bytes | None]]) -> list[int]:
    """Return the page object numbers in document order."""
    catalog = next(
        (number for number, (header, _) in objects.items() if b"/Type/Catalog" in header),
        None,
    )
    if catalog is None:
        raise PdfTableError("The PDF has no document catalog.")
    pages_match = re.search(rb"/Pages\s+(\d+)\s+\d+\s+R", objects[catalog][0])
    if not pages_match:
        raise PdfTableError("The PDF catalog does not point at a page tree.")
    pages: list[int] = []

    def walk(number: int) -> None:
        header = objects.get(number, (b"", None))[0]
        if b"/Type/Page" in header and b"/Type/Pages" not in header:
            pages.append(number)
            return
        kids = re.search(rb"/Kids\s*\[(.*?)\]", header, re.S)
        for kid in _reference_list(kids.group(1)) if kids else []:
            walk(kid)

    walk(int(pages_match.group(1)))
    return pages


def _page_content(
    objects: dict[int, tuple[bytes, bytes | None]], page_number: int
) -> bytes:
    """Return the concatenated content streams of one page."""
    header = objects.get(page_number, (b"", None))[0]
    contents = re.search(rb"/Contents\s*(\[[^\]]*\]|\d+\s+\d+\s+R)", header, re.S)
    if not contents:
        return b""
    raw = contents.group(1)
    if raw.startswith(b"["):
        references = _reference_list(raw)
    else:
        references = [int(re.search(rb"(\d+)", raw).group(1))]
    chunks = []
    for reference in references:
        object_header, payload = objects.get(reference, (b"", None))
        chunks.append(_stream_bytes(object_header, payload))
    return b"".join(chunks)


def _text_runs(content: bytes) -> list[tuple[float, float, str]]:
    """Return (y, x, text) for every positioned text run in a content stream."""
    runs: list[tuple[float, float, str]] = []
    for block in re.finditer(rb"BT(.*?)ET", content, re.S):
        body = block.group(1)
        matrix = _TEXT_MATRIX_RE.search(body)
        if not matrix:
            continue
        x = float(matrix.group(1))
        y = float(matrix.group(2))
        pieces: list[str] = []
        for show in _SHOW_TEXT_RE.finditer(body):
            if show.group(0).rstrip().endswith(b"TJ"):
                for literal in _STRING_RE.finditer(show.group(1) or b""):
                    pieces.append(_decode_string(literal.group(0)[1:-1]))
            else:
                literal = show.group(0)
                pieces.append(_decode_string(literal[:literal.rindex(b")") + 1][1:-1]))
        text = "".join(pieces)
        if text.strip():
            runs.append((round(y, 1), x, text))
    return runs


def _column_of(x: float) -> int:
    for index, boundary in enumerate(_COLUMN_BOUNDS):
        if x < boundary:
            return index
    return len(_COLUMN_BOUNDS)


def _group_rows(runs: Iterable[tuple[float, float, str]]) -> list[dict[str, str]]:
    """Group text runs into table rows keyed by their column."""
    ordered = sorted(runs, key=lambda item: (-item[0], item[1]))
    clusters: list[list[Any]] = []
    for y, x, text in ordered:
        if clusters and abs(clusters[-1][0] - y) <= _ROW_TOLERANCE:
            clusters[-1][1].append((x, text))
        else:
            clusters.append([y, [(x, text)]])
    rows = []
    for _, cells in clusters:
        columns: dict[int, list[str]] = {}
        for x, text in sorted(cells):
            columns.setdefault(_column_of(x), []).append(text)
        rows.append({
            "serial": " ".join(columns.get(0, [])).strip(),
            "name": " ".join(columns.get(1, [])).strip(),
            "address": " ".join(columns.get(2, [])).strip(),
            "entity": " ".join(columns.get(3, [])).strip(),
        })
    return rows


def extract_centre_rows(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Return the centre rows of the NEA hawker-centre list.

    Each row carries the serial number, the centre name, the address, and the
    entity that manages the common property. Rows without a numeric serial
    number (headers, footnotes, page furniture) are dropped.
    """
    objects = _objects(pdf_bytes)
    rows: list[dict[str, Any]] = []
    table = 0
    previous_serial = 0
    for page_number in _page_numbers(objects):
        content = _page_content(objects, page_number)
        if not content:
            continue
        for row in _group_rows(_text_runs(content)):
            serial = row["serial"]
            if not serial.isdigit():
                continue
            number = int(serial)
            # The list holds more than one table; the serial restarts at 1
            # for each new table.
            if number == 1 and previous_serial != 0:
                table += 1
            previous_serial = number
            rows.append({
                "table": table,
                "serial": number,
                "name": row["name"],
                "address": row["address"],
                "managing_entity": row["entity"],
            })
    if not rows:
        raise PdfTableError("No centre rows were found in the PDF table.")
    return rows
