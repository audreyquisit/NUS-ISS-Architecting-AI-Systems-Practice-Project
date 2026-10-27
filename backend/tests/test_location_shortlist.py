"""Offline shortlist policy tests with synthetic routes and model responses."""

import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda

from agents import location_agent
from models.location_models import CentreLogistics, LocationQuery, TravelEstimate
from models.orchestration import ParsedRequest
from test_location_grounded_reasoning import grounded_ranking


def route(centre_id, minutes, walking=None, transfers=None, source="onemap"):
    return CentreLogistics(
        centre_id=centre_id, centre_name=centre_id,
        address="Synthetic test address", coordinates={"lat": 1.33, "lng": 103.8},
        travel=TravelEstimate(
            total_duration_min=minutes, walking_distance_m=walking,
            transfers=transfers, source=source, confidence=0.9, mode="pt",
        ),
    )


class LocationShortlistTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch.object(
            location_agent, "get_route", side_effect=AssertionError("No live routing"),
        ))
        self.llm = self.patches.enter_context(patch.object(
            location_agent, "_llm", side_effect=AssertionError("No live models"),
        ))
        self.intent = location_agent.LocationIntent(
            request_intent="find_nearby_centres", proximity_preference="convenient",
            interpretation="Compare measured routes.",
        )
        self.routes = [
            route("fastest", 5, 2000, 3),
            route("least_walking", 45, 10, 2),
            route("fewest_transfers", 40, 1700, 0),
        ] + [route(f"balanced_{i}", 10 + i, 100 + i * 100, 1) for i in range(8)]

    def shortlist(self, message="Find food nearby", routes=None, limit=8, **fields):
        query = LocationQuery(request=message, **fields)
        return location_agent._shortlist_for_reasoning(
            query, self.intent, self.routes if routes is None else routes, limit=limit,
        )

    @staticmethod
    def ids(items):
        return [item.centre_id for item in items]

    def rank_with_mock(self, query, routes, selected_ids):
        response = grounded_ranking(selected_ids, confidence=0.9)
        model_response = Mock(return_value=response)
        model = Mock()
        model.with_structured_output.return_value = RunnableLambda(model_response)
        self.llm.side_effect = None
        self.llm.return_value = model
        # Keep this test focused on the evidence/model boundary, independently
        # of prompt rendering. The production prompt is not changed.
        with patch.object(location_agent, "RANK_PROMPT", RunnableLambda(lambda data: data)):
            result = location_agent._rank_centres(query, self.intent, routes)
        self.llm.assert_called_once()
        model.with_structured_output.assert_called_once_with(
            location_agent._GroundedRanking, method="function_calling", include_raw=True,
        )
        model_response.assert_called_once()
        payload = model_response.call_args.args[0]
        return result, json.loads(payload["evidence"]), payload["intent"]

    def test_permissive_walking_phrases_do_not_change_the_shortlist_policy(self):
        neutral = self.ids(self.shortlist())
        for message in (
            "I don't mind walking",
            "Find the fastest option; I don't mind walking.",
            "Fastest option; walking is fine",
            "I do not need the least walking option",
        ):
            with self.subTest(message=message):
                shortlist = self.ids(self.shortlist(message))
                self.assertEqual(shortlist, neutral)
                self.assertEqual(shortlist[0], "fastest")
        self.llm.assert_not_called()

    def test_old_walking_sort_excludes_fastest_but_corrected_policy_retains_it(self):
        # All ten competitors walk less. The old raw 'walking' substring
        # branch therefore omitted the fastest route from its eight slots.
        old_shortlist = sorted(self.routes, key=lambda item: (
            item.travel.walking_distance_m, item.travel.total_duration_min,
        ))[:8]
        self.assertNotIn("fastest", self.ids(old_shortlist))
        shortlist = self.ids(self.shortlist(
            "Find the fastest option; I don't mind walking.",
        ))
        self.assertIn("fastest", shortlist)
        self.assertEqual(len(shortlist), 8)

    def test_shortest_duration_requests_preserve_fastest(self):
        for message in ("Shortest travel time", "Fastest option", "Quickest destination"):
            with self.subTest(message=message):
                self.assertEqual(self.ids(self.shortlist(message))[0], "fastest")

    def test_explicit_least_walking_requests_preserve_measured_minimum(self):
        for message in ("Least walking possible", "Avoid long walks"):
            with self.subTest(message=message):
                shortlist = self.ids(self.shortlist(message))
                self.assertIn("least_walking", shortlist)
                self.assertIn("fastest", shortlist)

    def test_avoiding_transfers_preserves_transfer_minimum(self):
        shortlist = self.ids(self.shortlist("I would rather avoid transfers"))
        self.assertIn("fewest_transfers", shortlist)
        self.assertIn("fastest", shortlist)

    def test_mixed_preferences_preserve_distinct_metric_extremes(self):
        shortlist = self.ids(self.shortlist(
            "Fastest, least walking, and fewest transfers if possible",
            priorities=["fastest", "least walking", "few transfers"],
        ))
        self.assertTrue({"fastest", "least_walking", "fewest_transfers"} <= set(shortlist))
        self.assertEqual(len(shortlist), 8)
        self.assertEqual(len(set(shortlist)), 8)

    def test_missing_walking_distance_is_not_treated_as_zero(self):
        routes = [route("fast", 1, 1000), route("unknown", 10), route("least", 20, 10)]
        shortlist = self.shortlist("Least walking possible", routes)
        self.assertEqual(self.ids(shortlist)[:2], ["fast", "least"])
        self.assertIsNone(routes[1].travel.walking_distance_m)
        self.assertIsNone(routes[1].travel.walking_time_min)

    def test_all_optional_metrics_missing_uses_duration_only(self):
        routes = [route(f"route_{i}", i) for i in range(12, 0, -1)]
        shortlist = self.shortlist("Avoid long walks and transfers", routes)
        self.assertEqual([item.travel.total_duration_min for item in shortlist], list(range(1, 9)))
        self.assertTrue(all(item.travel.walking_distance_m is None for item in shortlist))
        self.assertTrue(all(item.travel.transfers is None for item in shortlist))

    def test_measured_zero_walking_distance_is_valid_evidence(self):
        routes = self.routes + [route("zero_walking", 50, 0, 2)]
        self.assertIn("zero_walking", self.ids(self.shortlist("Least walking possible", routes)))

    def test_missing_transfers_are_not_treated_as_zero(self):
        routes = [route("fast_unknown", 1), route("one_transfer", 10, transfers=1),
                  route("zero_transfers", 20, transfers=0)]
        self.assertEqual(self.ids(self.shortlist("Avoid transfers", routes))[:2],
                         ["fast_unknown", "zero_transfers"])

    def test_unavailable_sources_do_not_supply_walking_or_transfer_extremes(self):
        routes = self.routes + [route("unavailable", 30, 0, 0, source="unavailable")]
        shortlist = self.ids(self.shortlist("Least walking and transfers", routes))
        self.assertEqual(shortlist[:3], ["fastest", "least_walking", "fewest_transfers"])

    def test_equal_durations_have_deterministic_order_independent_of_input_order(self):
        routes = [route(f"centre_{i:02d}", 10) for i in range(12)]
        forward = self.ids(self.shortlist(routes=routes))
        reverse = self.ids(self.shortlist(routes=list(reversed(routes))))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, [f"centre_{i:02d}" for i in range(8)])

    def test_missing_durations_are_excluded_and_limit_is_bounded(self):
        routes = self.routes + [route("no_duration", None, 0, 0)]
        for limit in (0, 1, 2, 8, 20):
            with self.subTest(limit=limit):
                shortlist = self.ids(self.shortlist(routes=routes, limit=limit))
                self.assertEqual(len(shortlist), min(limit, len(self.routes)))
                self.assertNotIn("no_duration", shortlist)
                if limit:
                    self.assertEqual(shortlist[0], "fastest")

    def test_duplicate_centre_ids_do_not_consume_multiple_slots(self):
        shortlist = self.ids(self.shortlist(routes=self.routes + [self.routes[0]]))
        self.assertEqual(len(shortlist), 8)
        self.assertEqual(len(set(shortlist)), 8)

    def test_hard_time_limit_is_enforced_before_ranking_and_result_validation(self):
        query = LocationQuery(request="Least walking possible", max_travel_time_min=12)
        routes = self.routes + [route("unknown_duration", None, 0, 0)]
        result, evidence, _ = self.rank_with_mock(
            query, routes, ["least_walking", "unknown_duration", "fastest"],
        )
        ids = {item["centre_id"] for item in evidence}
        self.assertEqual(ids, {"fastest", "balanced_0", "balanced_1", "balanced_2"})
        self.assertTrue(all(item["travel"]["minutes"] <= 12 for item in evidence))
        self.assertEqual([item.centre_id for item in result.selected_centres], ["fastest"])

    def test_no_route_meets_hard_limit_skips_model(self):
        result = location_agent._rank_centres(
            LocationQuery(max_travel_time_min=0), self.intent, self.routes,
        )
        self.assertEqual(result.selected_centres, [])
        self.llm.assert_not_called()

    def test_structured_preferences_reach_ranking_without_raw_text_override(self):
        parsed = ParsedRequest(
            intent="food_discovery", priorities=["fastest", "few transfers"],
        ).model_dump()
        query = LocationQuery(
            request="Least walking possible", priorities=parsed["priorities"],
            parsed_request=parsed,
        )
        self.intent = self.intent.model_copy(update={
            "proximity_preference": "fastest", "inferred_priorities": ["few transfers"],
        })
        before = (query.model_dump(), self.intent.model_dump())
        _, evidence, ranking_intent = self.rank_with_mock(query, self.routes, ["fastest"])
        self.assertEqual(before, (query.model_dump(), self.intent.model_dump()))
        self.assertEqual(ranking_intent["priorities"], parsed["priorities"])
        self.assertEqual(ranking_intent["proximity_preference"], "fastest")
        self.assertEqual(ranking_intent["inferred_priorities"], ["few transfers"])
        self.assertIn("fastest", {item["centre_id"] for item in evidence})

    def test_ranking_receives_eight_tradeoffs_with_one_call_and_three_selections(self):
        query = LocationQuery(request="Fastest option; walking is fine")
        result, evidence, _ = self.rank_with_mock(
            query, self.routes,
            ["fastest", "least_walking", "fewest_transfers", "balanced_0"],
        )
        self.assertEqual(len(evidence), 8)
        self.assertTrue({"fastest", "least_walking", "fewest_transfers"} <= {
            item["centre_id"] for item in evidence
        })
        self.assertEqual(len(result.selected_centres), 3)

    def test_missing_walking_evidence_remains_missing_in_model_input(self):
        query = LocationQuery(request="Least walking possible")
        _, evidence, _ = self.rank_with_mock(query, [route("only_route", 5)], ["only_route"])
        self.assertIsNone(evidence[0]["travel"]["walking_distance"])
        self.assertIsNone(evidence[0]["travel"]["walking_time"])
        self.assertIsNone(evidence[0]["travel"]["transfers"])


if __name__ == "__main__":
    unittest.main()
