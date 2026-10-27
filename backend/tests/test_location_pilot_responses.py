"""Captured pilot failures and corrected response shapes, entirely offline."""

import copy
import unittest

from agents import location_agent as agent
from evaluations.pilot_support import ROUTE_FIXTURES, build_routes
from models.location_models import LocationQuery
from models.orchestration import ParsedRequest
from test_location_grounded_reasoning import comparative_reason, grounded_ranking
import test_location_grounded_reasoning as location_fixtures


NEWTON, TEKKA, MAXWELL = (row[0] for row in ROUTE_FIXTURES)
TRADEOFF_REQUEST = "Find chicken rice. Fastest matters, but I would accept about four extra minutes for much less walking."


def captured_response(tradeoff=False):
    """Retain the pilot's actual interpretation and incomplete evidence shapes."""
    ids = [MAXWELL, NEWTON, TEKKA] if tradeoff else [TEKKA, MAXWELL, NEWTON]
    response = grounded_ranking(ids, confidence=0.85).model_dump()
    response["preference_interpretation"] = (
        {"focus": "balanced", "walking": "flexible",
         "source_quote": "I would accept about four extra minutes for much less walking."}
        if tradeoff else {"focus": "unspecified", "walking": "minimize",
                            "source_quote": "proximity_preference: convenient"})
    for choice in response["selected_centres"]:
        centre = choice["centre_id"]
        if centre == NEWTON:
            choice["logistics_fit"] = 0.7 if tradeoff else 0.5
        elif centre == MAXWELL:
            choice.update(logistics_fit=0.9 if tradeoff else 0.7, reasons=[{
                "code": "BALANCED_TRADEOFF", "comparison_centre_id": NEWTON if tradeoff else TEKKA,
                "evidence_refs": [f"{centre}:primary:total_duration_min", f"{centre}:primary:walking_distance_m"],
            }])
        else:
            choice.update(logistics_fit=0.6 if tradeoff else 0.9, reasons=[{
                "code": "LESS_MEASURED_WALKING", "comparison_centre_id": NEWTON,
                "evidence_refs": [f"{centre}:primary:walking_distance_m"],
            }])
            if not tradeoff:
                choice["reasons"].append({"code": "FEWER_KNOWN_TRANSFERS", "comparison_centre_id": NEWTON,
                    "evidence_refs": [f"{centre}:primary:transfers"]})
    return response


class LocationPilotResponseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = location_fixtures.GroundedLocationReasoningTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.routes = build_routes([{"id": centre, "name": centre, "address": "Synthetic fixture",
            "latitude": 1.33, "longitude": 103.85} for centre, *_ in ROUTE_FIXTURES])

    @staticmethod
    def query(tradeoff=False):
        return LocationQuery(request=TRADEOFF_REQUEST if tradeoff else
            "Find laksa under S$8 from the synthetic central Singapore origin.",
            parsed_request=ParsedRequest(intent="food_discovery", travel_mode="pt",
                priorities=["fastest", "least walking"] if tradeoff else []).model_dump())

    def rank(self, response, tradeoff=False):
        result = self.fixture.rank(response=agent._GroundedRanking.model_validate(response),
            query=self.query(tradeoff), routes=self.routes)
        self.fixture.response.assert_called_once()
        return result

    def test_captured_invented_quote_still_triggers_fallback(self):
        result = self.rank(captured_response())
        self.fixture.assert_fallback(result)
        self.assertIn("invalid_preference_quote", str(result.limitations))
        self.assertEqual(result.selected_centres[0].centre_id, NEWTON)

    def test_captured_conflicting_walking_reading_still_triggers_fallback(self):
        result = self.rank(captured_response(tradeoff=True), tradeoff=True)
        self.fixture.assert_fallback(result)
        self.assertIn("conflicting_walking_preference", str(result.limitations))

    def test_incomplete_captured_comparison_refs_remain_rejected(self):
        for code, metrics in (("LESS_MEASURED_WALKING", ["walking_distance_m"]),
                              ("FEWER_KNOWN_TRANSFERS", ["transfers"]),
                              ("BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"])):
            with self.subTest(code=code):
                response = grounded_ranking([TEKKA]).model_dump()
                response["selected_centres"][0]["reasons"] = [{"code": code,
                    "comparison_centre_id": NEWTON,
                    "evidence_refs": [f"{TEKKA}:primary:{metric}" for metric in metrics]}]
                self.fixture.response.reset_mock()
                result = self.rank(response)
                self.fixture.assert_fallback(result)
                self.assertIn("no_valid_model_selections", str(result.limitations))

    def test_corrected_no_preference_reading_is_accepted_without_invented_quote(self):
        result = self.rank(grounded_ranking([NEWTON, MAXWELL, TEKKA], confidence=0.85).model_dump())
        self.assertNotIn("Deterministic fallback", result.reasoning)
        self.assertEqual([row.centre_id for row in result.selected_centres], [NEWTON, MAXWELL, TEKKA])
        self.assertEqual(result.confidence, 0.85)

    def test_supported_non_fastest_tradeoff_survives_with_complete_evidence(self):
        response = copy.deepcopy(captured_response(tradeoff=True))
        response["preference_interpretation"]["walking"] = "minimize"
        response["selected_centres"][0]["reasons"] = [comparative_reason(
            MAXWELL, NEWTON, "BALANCED_TRADEOFF", ["total_duration_min", "walking_distance_m"]).model_dump()]
        response["selected_centres"][2]["reasons"] = [comparative_reason(
            TEKKA, NEWTON, "LESS_MEASURED_WALKING", ["walking_distance_m"]).model_dump()]
        result = self.rank(response, tradeoff=True)
        self.assertNotIn("Deterministic fallback", result.reasoning)
        self.assertEqual([row.centre_id for row in result.selected_centres], [MAXWELL, NEWTON, TEKKA])
        self.assertIn("Trade-off versus", result.selected_centres[0].rationale)
        self.assertIn("22 min", result.selected_centres[0].rationale)
        self.assertIn("18 min", result.selected_centres[0].rationale)
        self.assertEqual(result.selected_centres[0].logistics_fit, 0.9)
        self.assertEqual(result.confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
