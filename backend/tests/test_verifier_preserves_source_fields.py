"""The verifier keeps the source-owned fields and drops injected ones."""

import unittest
from unittest.mock import patch

from agents.verifier import verify_results


SOURCE_RECORD = {
    "menu_item_id": "stall-a-item-1",
    "stall_id": "stall-a",
    "stall_name": "Verified Stall",
    "hawker_centre_id": "centre-a",
    "hawker_centre": "Verified Centre",
    "price_label": "$4.00",
    "price_sgd_min": 4.0,
    "price_sgd_max": 4.0,
    "categories": ["Chinese"],
    "dietary_tags": ["halal"],
    "data_as_of": "2026-10-02",
    "centre_match_method": "exact_postcode",
    "halal_certified": True,
    "halal_certificate_no": "MUIS-12345",
    "halal_source": "MUIS halal e-Service establishment directory",
    "halal_as_of": "2026-09-30",
    "healthier_dining": False,
    "healthier_criteria": None,
    "healthier_source": None,
    "healthier_as_of": None,
    "merchant_list_name": "VERIFIED STALL",
    "merchant_unit_number": "#01-01",
    "merchant_source": "DBS PayLah! Saturdays participating merchants",
    "crowdsourced_source": None,
    "crowdsourced_as_of": None,
}


def _worker_result(**overrides):
    candidate = {
        "menu_item_id": SOURCE_RECORD["menu_item_id"],
        "stall_id": SOURCE_RECORD["stall_id"],
        "stall_name": "A name the worker made up",
        "hawker_centre_id": SOURCE_RECORD["hawker_centre_id"],
        "hawker_centre": "A centre the worker made up",
        "price_label": "$99.00",
        "reason": "Looks good to me.",
        "dietary_suitable": True,
        "dietary_confidence": 0.9,
    }
    candidate.update(overrides)
    return {
        "agent": "dietary",
        "status": "ok",
        "confidence": 0.8,
        "candidates": [candidate],
    }


class VerifierSourceFieldTests(unittest.TestCase):
    def _verify(self, result=None, request=None):
        with patch("agents.verifier.get_menu_item_record") as get_record:
            get_record.return_value = SOURCE_RECORD
            return verify_results(
                [result or _worker_result()],
                request or {"dietary_preferences": ["halal"]},
            )

    def test_the_source_owned_fields_survive_verification(self):
        verified = self._verify()
        candidate = verified[0]["candidates"][0]
        self.assertEqual(candidate["halal_certified"], True)
        self.assertEqual(candidate["halal_certificate_no"], "MUIS-12345")
        self.assertEqual(candidate["halal_as_of"], "2026-09-30")
        self.assertEqual(candidate["healthier_dining"], False)
        self.assertEqual(candidate["merchant_list_name"], "VERIFIED STALL")
        self.assertEqual(candidate["merchant_unit_number"], "#01-01")
        self.assertEqual(candidate["data_as_of"], "2026-10-02")
        self.assertEqual(candidate["centre_match_method"], "exact_postcode")

    def test_a_worker_cannot_inject_a_halal_claim(self):
        verified = self._verify(
            _worker_result(
                halal_certified=True,
                halal_certificate_no="MADE-UP-999",
                halal_source="somewhere",
                halal_as_of="2020-01-01",
            )
        )
        candidate = verified[0]["candidates"][0]
        # The source record holds the truth; the worker's own claim is gone.
        self.assertEqual(candidate["halal_certificate_no"], "MUIS-12345")
        self.assertEqual(candidate["halal_as_of"], "2026-09-30")

    def test_a_worker_cannot_inject_a_crowdsourced_claim(self):
        verified = self._verify(
            _worker_result(
                crowdsourced_source="hawkercentral",
                crowdsourced_as_of="2026-10-20",
                crowdsourced_menu=["Nasi goreng $6.00"],
                crowdsourced_price="$6.00",
            )
        )
        candidate = verified[0]["candidates"][0]
        self.assertIsNone(candidate.get("crowdsourced_source"))
        self.assertIsNone(candidate.get("crowdsourced_menu"))
        self.assertNotIn("crowdsourced_price", candidate)

    def test_the_source_identity_replaces_the_worker_identity(self):
        verified = self._verify()
        candidate = verified[0]["candidates"][0]
        self.assertEqual(candidate["stall_name"], "Verified Stall")
        self.assertEqual(candidate["hawker_centre"], "Verified Centre")
        self.assertEqual(candidate["price_label"], "$4.00")

    def test_a_candidate_with_no_source_record_is_dropped(self):
        with patch("agents.verifier.get_menu_item_record", return_value=None):
            verified = verify_results(
                [_worker_result()],
                {"dietary_preferences": ["halal"]},
            )
        self.assertEqual(verified[0]["candidates"], [])

    def test_a_candidate_pointing_at_another_stall_is_dropped(self):
        with patch("agents.verifier.get_menu_item_record", return_value=SOURCE_RECORD):
            verified = verify_results(
                [_worker_result(stall_id="stall-b")],
                {"dietary_preferences": ["halal"]},
            )
        self.assertEqual(verified[0]["candidates"], [])

    def test_the_worker_may_only_add_its_own_explanation_fields(self):
        verified = self._verify()
        candidate = verified[0]["candidates"][0]
        self.assertTrue(candidate["dietary_suitable"])
        self.assertEqual(
            candidate["reason"],
            "Matched verified catalogue fields and request constraints.",
        )


if __name__ == "__main__":
    unittest.main()
