"""The data every source holds must be present and traceable."""

import json
import os
import unittest
from pathlib import Path

from services.coverage_service import get_coverage
from services.hawker_service import _menu_catalogue, load_hawker_centres
from services.sources_registry import get_source, get_sources_registry


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CENTRES_FILE = DATA_DIR / "hawker_centres.json"
STALLS_FILE = DATA_DIR / "hawker_stall.json"
DIRECTORY_FILE = DATA_DIR / "hawker_stall_directory_linked.json"


class DataSourceTests(unittest.TestCase):
    def test_every_centre_carries_its_provenance(self):
        centres = json.loads(CENTRES_FILE.read_text(encoding="utf-8"))
        self.assertTrue(centres)
        for centre in centres:
            self.assertTrue(
                centre.get("source_as_of"), f"{centre.get('id')} has no source_as_of"
            )
            self.assertTrue(centre.get("source_url"), f"{centre.get('id')} has no source_url")

    def test_every_catalogue_record_carries_its_provenance(self):
        records = _menu_catalogue()
        self.assertTrue(records)
        checked = 0
        for record in records:
            if record.get("is_mock"):
                # Synthetic demo records carry no source provenance.
                continue
            checked += 1
            self.assertTrue(
                record.get("data_as_of"),
                f"{record.get('menu_item_id')} has no data_as_of",
            )
            self.assertTrue(
                record.get("centre_match_method")
                in {"exact_postcode", "name_unique", "unit_postcode", "street_unique"},
                f"{record.get('menu_item_id')} has no centre_match_method",
            )
        self.assertTrue(checked, "every catalogue record was a mock record")

    def test_sources_registry_holds_the_sources_in_use(self):
        keys = {entry["key"] for entry in get_sources_registry()}
        for expected in (
            "nea_hawker_centres",
            "wak_wak_hawker",
            "dbs_paylah_directory",
            "hpb_healthier_dining",
            "muis_halal",
            "mom_public_holidays",
        ):
            self.assertIn(expected, keys)

    def test_registry_entries_state_where_the_data_came_from(self):
        for entry in get_sources_registry():
            self.assertTrue(entry.get("name"), f"{entry.get('key')} has no name")
            self.assertTrue(entry.get("kind"), f"{entry.get('key')} has no kind")
            self.assertTrue(
                entry.get("url") or entry.get("dataset_id"),
                f"{entry.get('key')} has no url or dataset_id",
            )
            self.assertTrue(entry.get("licence"), f"{entry.get('key')} has no licence")

    def test_a_source_that_is_not_held_is_not_listed(self):
        self.assertIsNone(get_source("a_source_that_does_not_exist"))

    def test_merchant_directory_is_linked_to_centres(self):
        directory = json.loads(DIRECTORY_FILE.read_text(encoding="utf-8"))
        self.assertTrue(directory["linked_record_count"] > 0)
        self.assertGreaterEqual(
            directory["heading_link_rate"], 0.9,
            "the merchant directory link rate fell below 90%",
        )
        for record in directory["records"]:
            self.assertTrue(record.get("centre_id"))
            self.assertIn("source_page", record)

    def test_halal_and_healthier_claims_name_their_source(self):
        records = _menu_catalogue()
        halal = [r for r in records if r.get("halal_certified")]
        healthier = [r for r in records if r.get("healthier_dining")]
        for record in halal:
            self.assertTrue(record.get("halal_source"))
            self.assertTrue(record.get("halal_as_of"))
            self.assertTrue(record.get("halal_certificate_no"))
        for record in healthier:
            self.assertTrue(record.get("healthier_source"))
            self.assertTrue(record.get("healthier_as_of"))
            self.assertTrue(record.get("healthier_criteria"))
        self.assertTrue(halal, "no halal certification was linked to the catalogue")
        self.assertTrue(healthier, "no Healthier Dining endorsement was linked")

    def test_centres_expose_their_merchant_list(self):
        centres = load_hawker_centres()
        listed = [c for c in centres if c.get("merchant_stalls")]
        self.assertTrue(listed)
        for centre in listed:
            self.assertTrue(centre.get("merchant_source"))
            self.assertTrue(centre.get("merchant_as_of"))
            for merchant in centre["merchant_stalls"]:
                # The directory lists some units without a merchant name and
                # some merchants without a unit. Every row carries at least
                # one of the two.
                self.assertTrue(
                    merchant.get("name") or merchant.get("unit_number"),
                    f"{centre.get('id')} has an empty merchant row",
                )
                if merchant.get("name") is not None:
                    self.assertTrue(str(merchant["name"]).strip())

    def test_merchant_entries_without_a_name_are_counted(self):
        from services.coverage_service import coverage_summary

        summary = coverage_summary()
        self.assertGreater(summary["merchant_listed_stalls"], 0)
        self.assertGreaterEqual(
            summary["merchant_entries_without_name"], 0,
            "a missing count means the nameless merchant rows are hidden",
        )

    def test_every_ingest_output_parses(self):
        for name in (
            "hawker_centres.json",
            "hawker_centres.meta.json",
            "hawker_stall.json",
            "hawker_stall_directory.json",
            "hawker_stall_directory_linked.json",
            "hawker_stall_mock.json",
            "hpb_healthier_dining.json",
            "muis_halal.json",
            "public_holidays.json",
        ):
            path = DATA_DIR / name
            self.assertTrue(path.exists(), f"{name} is missing")
            json.loads(path.read_text(encoding="utf-8"))

    def test_dietary_flags_are_booleans_or_absent(self):
        for record in _menu_catalogue():
            for field in ("halal_certified", "healthier_dining"):
                value = record.get(field)
                self.assertIn(
                    value, (True, False, None),
                    f"{record.get('menu_item_id')} has a free-text {field}",
                )
            source = record.get("crowdsourced_source")
            self.assertIn(
                source, (True, False, None),
                f"{record.get('menu_item_id')} has a free-text crowdsourced_source",
            )

    def test_crowdsourced_fields_are_absent_when_the_source_is_off(self):
        if os.getenv("ENABLE_CROWDSOURCED_SOURCE", "").casefold() == "true":
            self.skipTest("the crowdsourced source is enabled")
        for record in _menu_catalogue():
            for field in (
                "crowdsourced_menu",
                "crowdsourced_price",
                "crowdsourced_cuisine",
                "crowdsourced_opening",
                "crowdsourced_source",
                "crowdsourced_as_of",
            ):
                self.assertIsNone(
                    record.get(field),
                    f"{record.get('menu_item_id')} carries {field} while the source is off",
                )


if __name__ == "__main__":
    unittest.main()
