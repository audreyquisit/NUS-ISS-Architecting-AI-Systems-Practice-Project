"""Evaluation-only response observation and hypothetical route fixtures.

No live runner, credentials, persistence or production networking changes.
"""

from models.location_models import CentreLogistics, TravelEstimate


# Centre ID, total minutes, measured walking metres, known transfers.
# Hypothetical PT routes, not real OneMap measurements. Maxwell costs four
# extra minutes versus Newton; Tekka minimizes walking but costs ten extra.
ROUTE_FIXTURES = (
    ("newton_food_centre", 18, 600, 1),
    ("buffalo_road_blk_665_tekka_centre_zhu_jiao_market", 28, 100, 0),
    ("maxwell_food_centre", 22, 200, 0),
)
MENU_ITEM_IDS = (
    "wakwak_37_item_005", "wakwak_34_item_000", "wakwak_448_item_000",
    "wakwak_398_item_000", "wakwak_588_item_004", "wakwak_881_item_000",
    "wakwak_863_item_000",
)


def build_routes(catalogue):
    """Use public destination identity and synthetic route measurements only."""
    by_id = {centre["id"]: centre for centre in catalogue}
    routes = []
    for centre_id, minutes, metres, transfers in ROUTE_FIXTURES:
        centre = by_id[centre_id]
        travel = TravelEstimate(mode="pt", source="onemap", confidence=0.5,
            total_duration_min=minutes, walking_distance_m=metres, transfers=transfers)
        if travel.walking_time_min > minutes:
            raise ValueError("Synthetic walking time exceeds total duration")
        routes.append(CentreLogistics(centre_id=centre_id, centre_name=centre["name"],
            address=centre["address"], coordinates={"lat": centre["latitude"], "lng": centre["longitude"]},
            travel=travel, evidence=["Hypothetical evaluation route; not live OneMap evidence."]))
    return routes


def response_observer(records):
    """Return an httpx response hook that never reconstructs a response.

    httpx caches the decoded body on the ORIGINAL Response. The SDK then reads
    that same object normally. Retaining Content-Encoding on a NEW Response
    containing decoded bytes would instead attempt decompression twice.
    Record bounded usage metadata only, without headers, prompts or raw prose.
    """
    def observe(response):
        response.read()
        record = {"status_code": response.status_code, "model": None, "usage": None}
        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            payload = None  # Leave malformed content for the SDK to reject.
        if isinstance(payload, dict):
            model = payload.get("model")
            record["model"] = model[:100] if isinstance(model, str) else None
            usage = payload.get("usage")
            if isinstance(usage, dict):
                record["usage"] = {name: value for name in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                ) if isinstance((value := usage.get(name)), int) and not isinstance(value, bool) and value >= 0}
        records.append(record)
    return observe
