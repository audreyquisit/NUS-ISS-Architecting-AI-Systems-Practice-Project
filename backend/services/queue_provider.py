import os
from typing import Callable, Optional

from services.google_places_queue_feed import get_live_queue_feed as get_google_places_feed
from services.live_queue_feed import get_live_queue_feed as get_mock_feed


ProviderFn = Callable[[Optional[str]], list[dict]]


def get_live_queue_feed(task: Optional[str] = None) -> list[dict]:
    provider = (os.getenv("QUEUE_FEED_PROVIDER") or "mock").strip().lower()
    if provider == "google_places":
        return get_google_places_feed(task)
    if provider == "live" or provider == "google":
        return get_google_places_feed(task)
    return get_mock_feed(task)
