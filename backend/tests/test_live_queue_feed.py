import json
import unittest
from pathlib import Path

from services.live_queue_feed import get_live_queue_feed


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class LiveQueueFeedTests(unittest.TestCase):
    def test_records_cover_source_and_synthetic_stalls_at_known_centres(self):
        source_payload = json.loads(
            (DATA_DIR / "hawker_stall.json").read_text(encoding="utf-8")
        )
        mock_payload = json.loads(
            (DATA_DIR / "hawker_stall_mock.json").read_text(encoding="utf-8")
        )
        centres = json.loads(
            (DATA_DIR / "hawker_centres.json").read_text(encoding="utf-8")
        )
        stalls_by_id = {
            stall["stall_id"]: stall for stall in source_payload["stalls"]
        }
        stalls_by_id.update({
            stall["stall_id"]: stall for stall in mock_payload["stalls"]
        })
        centre_ids = {centre["id"] for centre in centres}
        stalls_by_id = {
            stall_id: stall for stall_id, stall in stalls_by_id.items()
            if stall.get("hawker_centre_id") in centre_ids
        }
        records = {item["stall_id"]: item for item in get_live_queue_feed()}

        self.assertEqual(set(records), set(stalls_by_id))
        self.assertTrue(all(
            record["hawker_centre_id"] in centre_ids
            and record["hawker_centre_id"] == stalls_by_id[stall_id]["hawker_centre_id"]
            and record["queue_is_mock"] is True
            and record["source"] == "synthetic_queue_mock"
            and "not a live" in record["queue_mock_notice"]
            for stall_id, record in records.items()
        ))
        self.assertTrue(any(not record["is_mock"] for record in records.values()))
        self.assertTrue(any(record["is_mock"] for record in records.values()))

    def test_newton_food_centre_has_source_backed_queue_records(self):
        records = get_live_queue_feed("Newton Food Centre")

        self.assertEqual(len(records), 72)
        self.assertEqual(
            {item["hawker_centre_id"] for item in records},
            {"newton_food_centre"},
        )
        self.assertTrue(all(item["queue_is_mock"] for item in records))
        self.assertTrue(all(not item["is_mock"] for item in records))

    def test_mock_queue_values_are_stable_and_copies_are_isolated(self):
        first = get_live_queue_feed()
        first[0]["queue_minutes"] = -1
        second = get_live_queue_feed()

        self.assertGreaterEqual(second[0]["queue_minutes"], 4)
        self.assertEqual(
            [item["queue_minutes"] for item in second],
            [item["queue_minutes"] for item in get_live_queue_feed()],
        )

    def test_filters_by_specific_centre_and_longest_region_alias(self):
        centre_records = get_live_queue_feed("Commonwealth Crescent Market")
        northeast_records = get_live_queue_feed("north-east")

        self.assertEqual(len(centre_records), 3)
        self.assertEqual(
            {item["hawker_centre"] for item in centre_records},
            {"Commonwealth Crescent Market"},
        )
        self.assertTrue(northeast_records)
        self.assertTrue(all(item["region"] == "north_east" for item in northeast_records))


if __name__ == "__main__":
    unittest.main()