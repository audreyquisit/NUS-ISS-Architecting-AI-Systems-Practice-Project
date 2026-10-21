"""The community listing only contributes records at a NEA hawker centre."""

import unittest
from unittest.mock import patch

import scripts.fetch_hawkercentral as hawkercentral


CENTRE_POSTCODES = {"149644", "123456"}


def _record(**overrides):
    record = {
        "name": "Ah Cheng Lim Chee Cheong Fun",
        "hawker_centre": "Commonwealth Crescent Market",
        "postal": "149644",
        "unit": "#01-12",
        "street": "31 Commonwealth Crescent",
        "location": {"lat": "1.3069", "lng": "103.8003"},
        "cuisine": "Dessert",
        "categories": "Dessert",
        "tags": "Dessert",
        "opening": "9am - 9pm",
        "contact": "",
        "menu": [],
        "price": None,
        "source_doc_id": "abc",
        "source_last_modified": "2020-06-03T19:50:04.455Z",
        "inperson": True,
        "delivery_option": False,
        "regions": "",
        "custom": False,
    }
    record.update(overrides)
    return record


class CrowdsourcedFilterTests(unittest.TestCase):
    def test_a_record_at_a_nea_centre_is_kept(self):
        kept, dropped = hawkercentral.filter_records(
            [_record()], CENTRE_POSTCODES
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped["postal_not_a_nea_centre"], 0)

    def test_a_record_outside_a_nea_centre_is_dropped(self):
        kept, dropped = hawkercentral.filter_records(
            [_record(postal="999999")], CENTRE_POSTCODES
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped["postal_not_a_nea_centre"], 1)

    def test_a_record_without_a_postal_is_dropped(self):
        kept, dropped = hawkercentral.filter_records(
            [_record(postal="")], CENTRE_POSTCODES
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped["no_postal"], 1)

    def test_a_grocer_is_not_a_hawker_stall(self):
        kept, dropped = hawkercentral.filter_records(
            [_record(categories="Grocery Shopping", cuisine="Grocery Shopping")],
            CENTRE_POSTCODES,
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped["not_a_hawker_stall"], 1)

    def test_a_delivery_only_outlet_is_dropped(self):
        kept, dropped = hawkercentral.filter_records(
            [_record(inperson=False, delivery_option=True, regions="Islandwide")],
            CENTRE_POSTCODES,
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped["delivery_only"], 1)

    def test_a_fixture_with_one_valid_and_one_invalid_record_gives_the_expected_counts(self):
        records = [
            _record(name="Good Stall"),
            _record(name="Wrong Postcode", postal="000000"),
        ]
        kept, dropped = hawkercentral.filter_records(records, CENTRE_POSTCODES)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["name"], "Good Stall")
        self.assertEqual(dropped["postal_not_a_nea_centre"], 1)

    def test_a_source_document_is_read_into_the_catalogue_shape(self):
        document = {
            "name": "projects/x/databases/(default)/documents/hawkers/doc-1",
            "fields": {
                "name": {"stringValue": "Penyet King"},
                "postal": {"stringValue": "486038"},
                "unit": {"stringValue": "#02-11"},
                "categories": {"arrayValue": {"values": [
                    {"stringValue": "Indonesian food"},
                ]}},
                "lastmodified": {"stringValue": "2020-06-03T19:50:04.455Z"},
                "menu_combined": {"arrayValue": {"values": [
                    {"mapValue": {"fields": {
                        "name": {"stringValue": "Ayam Penyet"},
                        "price": {"stringValue": "7.00"},
                        "available": {"booleanValue": True},
                        "description": {"stringValue": "Fried chicken."},
                        "variants": {"arrayValue": {}},
                        "section": {"stringValue": ""},
                        "image": {"stringValue": "https://example.test/a.jpg"},
                    }}},
                ]}},
            },
        }
        record = hawkercentral.to_record(document)
        self.assertEqual(record["name"], "Penyet King")
        self.assertEqual(record["postal"], "486038")
        self.assertEqual(record["source_doc_id"], "doc-1")
        self.assertEqual(record["categories"], "Indonesian food")
        self.assertEqual(len(record["menu"]), 1)
        self.assertEqual(record["menu"][0]["name"], "Ayam Penyet")
        self.assertEqual(record["menu"][0]["price"], "7.00")
        self.assertTrue(record["menu"][0]["available"])

    def test_a_document_without_a_name_is_not_a_record(self):
        self.assertIsNone(hawkercentral.to_record({"fields": {}}))

    def test_the_centre_postcodes_come_from_the_nea_catalogue(self):
        with patch(
            "services.location_service.load_hawker_centres",
            return_value=[{"postal_code": "149644", "address": "31, Commonwealth Crescent, Singapore 149644"}],
        ):
            postcodes = hawkercentral.load_centre_postcodes()
        self.assertIn("149644", postcodes)


if __name__ == "__main__":
    unittest.main()
