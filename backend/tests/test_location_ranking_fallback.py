"""Offline ranking validation/fallback tests; no persistence or report hooks."""

import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from openai import (
    APIConnectionError, APIResponseValidationError, APITimeoutError,
    AuthenticationError, BadRequestError, InternalServerError,
    LengthFinishReasonError, NotFoundError, PermissionDeniedError, RateLimitError,
)
from pydantic import ValidationError

from agents import location_agent
from models.location_models import CentreLogistics, LocationDecision, LocationQuery, TravelEstimate
from models.orchestration import ParsedRequest
from services.activity_log import begin_activity_capture, end_activity_capture


def route(centre_id, minutes, walking=None, transfers=None, source="onemap"):
    return CentreLogistics(
        centre_id=centre_id, centre_name=f"Fixture {centre_id} centre",
        address="Synthetic test address", coordinates={"lat": 1.33, "lng": 103.8},
        travel=TravelEstimate(
            total_duration_min=minutes, walking_distance_m=walking,
            transfers=transfers, source=source, confidence=0.85, mode="pt",
        ),
    )


def ranking(ids, confidence=0.97):
    return location_agent.CentreRanking(
        selected_centres=[{
            "centre_id": centre_id, "rank": index, "logistics_fit": 0.9,
            "rationale": "Mock ranking grounded in route evidence.",
        } for index, centre_id in enumerate(ids, start=1)],
        reasoning="Original model reasoning.", confidence=confidence,
        limitations=["Original model limitation."],
    )


class LocationRankingFallbackTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch(
            "services.location_service._get", side_effect=AssertionError("No live OneMap"),
        ))
        self.routes = [
            route("fastest", 5, 1800, 3), route("least_walking", 20, 20, 2),
            route("fewest_transfers", 15, 700, 0), route("balanced", 8, 400, 1),
            route("other", 10, 500, 1),
        ]
        self.query = LocationQuery(request_id="ranking-request", request="Find food nearby")
        self.intent = location_agent.LocationIntent(
            request_intent="find_nearby_centres", proximity_preference="convenient",
            interpretation="Compare verified routes.",
        )
        self.response = Mock(return_value=ranking(["other", "fastest"]))
        self.model = Mock()
        self.model.with_structured_output.return_value = RunnableLambda(self.response)
        self.llm = self.patches.enter_context(patch.object(
            location_agent, "_llm", return_value=self.model,
        ))
        self.sdk_request = httpx.Request("POST", "https://example.invalid/ranking")

    def rank(self, *, query=None, routes=None, outcome=None, error=None):
        self.llm.reset_mock()
        self.response.reset_mock()
        self.response.side_effect = error
        if outcome is not None:
            self.response.return_value = outcome
        return location_agent._rank_centres(
            self.query if query is None else query, self.intent,
            self.routes if routes is None else routes,
        )

    @staticmethod
    def ids(result):
        return [item.centre_id for item in result.selected_centres]

    def assert_fallback(self, result):
        self.assertIn("Deterministic fallback", result.reasoning)
        self.assertTrue(any("fallback" in item.casefold() for item in result.limitations))
        self.assertLessEqual(result.confidence, 0.5)
        self.assertLessEqual(len(result.selected_centres), 3)
        self.assertEqual([item.rank for item in result.selected_centres],
                         list(range(1, len(result.selected_centres) + 1)))
        self.assertLessEqual(len({item.centre_id for item in result.selected_centres}), 3)

    def test_valid_model_ranking_and_normal_call_count_are_preserved(self):
        result = self.rank()
        self.assertEqual(self.ids(result), ["other", "fastest"])
        self.assertEqual(result.confidence, 0.97)
        self.assertEqual(result.reasoning, "Original model reasoning.")
        self.assertEqual(result.limitations, ["Original model limitation."])
        self.llm.assert_called_once()
        self.model.with_structured_output.assert_called_once_with(
            location_agent.CentreRanking, method="function_calling",
        )
        self.response.assert_called_once()

    def test_typed_transport_timeout_and_service_errors_use_fallback(self):
        errors = [
            TimeoutError("private timeout context"), ConnectionError("private context"),
            APITimeoutError(self.sdk_request), APIConnectionError(request=self.sdk_request),
            RateLimitError("rate limited", response=httpx.Response(429, request=self.sdk_request), body=None),
            InternalServerError("unavailable", response=httpx.Response(503, request=self.sdk_request), body=None),
            APIResponseValidationError(httpx.Response(200, request=self.sdk_request), {}),
            LengthFinishReasonError(completion=Mock(usage=None)),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                result = self.rank(error=error)
                self.assert_fallback(result)
                self.assertEqual(self.ids(result), ["fastest", "balanced", "other"])
                self.response.assert_called_once()
                self.llm.assert_called_once()

    def test_typed_parser_and_schema_exceptions_use_fallback(self):
        try:
            location_agent.CentreRanking.model_validate({"confidence": 2})
        except ValidationError as error:
            schema_error = error
        for error in (OutputParserException("private parser context"),
                      json.JSONDecodeError("bad JSON", "", 0), schema_error):
            with self.subTest(error=type(error).__name__):
                result = self.rank(error=error)
                self.assert_fallback(result)
                self.assertEqual(self.ids(result)[0], "fastest")

    def test_malformed_outputs_are_revalidated(self):
        valid = ranking(["fastest"]).model_dump()
        invalid = ["not structured JSON", {}, {**valid, "confidence": 2},
                   {**valid, "selected_centres": [{"centre_id": "fastest", "rank": 0}]}]
        for outcome in invalid:
            with self.subTest(outcome=outcome):
                result = self.rank(outcome=outcome)
                self.assert_fallback(result)
                self.assertEqual(self.ids(result)[0], "fastest")

    def test_missing_parsed_output_uses_fallback(self):
        self.response.return_value = None
        self.assert_fallback(self.rank())

    def test_valid_mapping_output_is_validated_without_extra_calls(self):
        result = self.rank(outcome=ranking(["balanced"]).model_dump())
        self.assertEqual(self.ids(result), ["balanced"])
        self.assertEqual(result.confidence, 0.97)
        self.response.assert_called_once()

    def test_model_instances_are_revalidated_without_mutating_rejected_output(self):
        outcome = ranking(["fastest"]).model_copy(update={"confidence": 2.0})
        result = self.rank(outcome=outcome)
        self.assert_fallback(result)
        self.assertEqual(outcome.confidence, 2.0)
        self.assertEqual(outcome.limitations, ["Original model limitation."])

    def test_invalid_ids_and_empty_selections_share_the_fallback_policy(self):
        for ids in (["unknown"], []):
            with self.subTest(ids=ids):
                result = self.rank(outcome=ranking(ids, confidence=1.0))
                self.assert_fallback(result)
                self.assertEqual(self.ids(result), ["fastest", "balanced", "other"])
                self.assertNotEqual(result.reasoning, "Original model reasoning.")
                self.assertNotIn("Original model limitation.", result.limitations)

    def test_model_ids_outside_the_shortlist_are_not_accepted(self):
        routes = self.routes + [route(f"extra_{i}", 30 + i, 1000 + i, 4) for i in range(10)]
        shortlist = location_agent._shortlist_for_reasoning(self.query, self.intent, routes)
        allowed = {item.centre_id for item in shortlist}
        excluded = next(item.centre_id for item in routes if item.centre_id not in allowed)
        result = self.rank(routes=routes, outcome=ranking([excluded]))
        self.assert_fallback(result)
        self.assertTrue(set(self.ids(result)) <= allowed)

    def test_duplicates_are_removed_and_partial_valid_selections_are_retained(self):
        result = self.rank(outcome=ranking(["unknown", "balanced", "balanced", "fastest"]))
        self.assertEqual(self.ids(result), ["balanced", "fastest"])
        self.assertEqual([item.rank for item in result.selected_centres], [1, 2])
        self.assertEqual(result.confidence, 0.97)
        self.assertTrue(any("removed" in item for item in result.limitations))
        self.assertNotIn("Deterministic fallback", result.reasoning)

    def test_successful_model_selections_remain_capped_at_three(self):
        result = self.rank(outcome=ranking([item.centre_id for item in self.routes]))
        self.assertEqual(self.ids(result), ["fastest", "least_walking", "fewest_transfers"])

    def test_explicit_structured_priorities_control_fallback(self):
        for priority, expected in (
            ("fastest", "fastest"), ("least walking", "least_walking"),
            ("avoid long walks", "least_walking"), ("fewest_transfers", "fewest_transfers"),
        ):
            with self.subTest(priority=priority):
                query = self.query.model_copy(update={
                    "request": "I don't mind walking", "priorities": [priority],
                })
                result = self.rank(query=query, error=TimeoutError())
                self.assertEqual(self.ids(result)[0], expected)
                self.assert_fallback(result)

    def test_parsed_priorities_are_used_when_query_priorities_are_empty(self):
        query = self.query.model_copy(update={"parsed_request": ParsedRequest(
            intent="food_discovery", priorities=["least walking"],
        ).model_dump()})
        result = self.rank(query=query, error=TimeoutError())
        self.assertEqual(self.ids(result)[0], "least_walking")

    def test_structured_request_without_priority_does_not_use_conflicting_inference(self):
        self.intent = self.intent.model_copy(update={"inferred_priorities": ["least walking"]})
        query = self.query.model_copy(update={"parsed_request": ParsedRequest(
            intent="food_discovery",
        ).model_dump()})
        result = self.rank(query=query, error=TimeoutError())
        self.assertEqual(self.ids(result)[0], "fastest")

    def test_standalone_structured_intent_priority_is_supported(self):
        self.intent = self.intent.model_copy(update={"proximity_preference": "fewest transfers"})
        self.assertEqual(self.ids(self.rank(error=TimeoutError()))[0], "fewest_transfers")

    def test_multiple_priorities_are_lexicographic_in_declared_order(self):
        routes = [route("a", 10, 100, 2), route("b", 15, 100, 0), route("c", 5, 500, 1)]
        query = self.query.model_copy(update={"priorities": ["least walking", "fewest transfers"]})
        self.assertEqual(self.ids(self.rank(query=query, routes=routes, error=TimeoutError())),
                         ["b", "a", "c"])

    def test_unsupported_and_permissive_priorities_default_to_duration(self):
        query = self.query.model_copy(update={"priorities": ["scenic", "I don't mind walking"]})
        result = self.rank(query=query, error=TimeoutError())
        self.assertEqual(self.ids(result), ["fastest", "balanced", "other"])
        self.assertTrue(any("unsupported" in item.casefold() for item in result.limitations))

    def test_unknown_metrics_are_not_zero_and_known_zero_is_valid(self):
        routes = [route("unknown", 1), route("known", 8, 300, 2), route("zero", 12, 0, 0)]
        for priority in ("least walking", "fewest transfers"):
            with self.subTest(priority=priority):
                query = self.query.model_copy(update={"priorities": [priority]})
                self.assertEqual(self.ids(self.rank(query=query, routes=routes, error=TimeoutError())),
                                 ["zero", "known", "unknown"])

    def test_absent_metric_evidence_defaults_to_duration_with_limitation(self):
        query = self.query.model_copy(update={"priorities": ["least walking", "fewest transfers"]})
        result = self.rank(query=query, routes=[route("b", 8), route("a", 5)], error=TimeoutError())
        self.assertEqual(self.ids(result), ["a", "b"])
        self.assertTrue(any("unavailable" in item.casefold() for item in result.limitations))

    def test_hard_time_limit_applies_to_failure_and_invalid_selection_fallbacks(self):
        query = self.query.model_copy(update={"max_travel_time_min": 10, "priorities": ["least walking"]})
        for error, outcome in ((TimeoutError(), None), (None, ranking(["least_walking"]))):
            with self.subTest(error=error):
                result = self.rank(query=query, error=error, outcome=outcome)
                self.assertEqual(self.ids(result), ["balanced", "other", "fastest"])
                self.assertTrue(set(self.ids(result)) <= {"fastest", "balanced", "other"})

    def test_no_usable_routes_preserves_no_selection_without_calling_model(self):
        for routes in ([], [route("unknown", None)], [route("unavailable", 5, source="unavailable")]):
            with self.subTest(routes=routes):
                result = self.rank(routes=routes)
                self.assertEqual(result.selected_centres, [])
                self.llm.assert_not_called()

    def test_no_routes_meet_time_limit_does_not_call_model(self):
        result = self.rank(query=self.query.model_copy(update={"max_travel_time_min": 0}))
        self.assertEqual(result.selected_centres, [])
        self.llm.assert_not_called()

    def test_fallback_is_deterministic_when_durations_tie(self):
        routes = [route("b", 5), route("a", 5), route("c", 5)]
        first = self.rank(routes=routes, error=TimeoutError())
        second = self.rank(routes=list(reversed(routes)), error=TimeoutError())
        self.assertEqual(first.model_dump(), second.model_dump())
        self.assertEqual(self.ids(first), ["a", "b", "c"])

    def test_fallback_confidence_and_provenance_do_not_reuse_rejected_output(self):
        token, events = begin_activity_capture()
        try:
            result = self.rank(error=OutputParserException("SECRET precise-origin-context"))
        finally:
            end_activity_capture(token)
        self.assert_fallback(result)
        self.assertEqual([event["event"] for event in events], [
            "location.centre_ranking.started", "location.centre_ranking.failed",
            "location.centre_ranking.completed",
        ])
        self.assertTrue(all(event["request_id"] == self.query.request_id for event in events))
        self.assertEqual(events[-1]["ranking_source"], "deterministic_fallback")
        self.assertEqual(events[-1]["fallback_reason"], "OutputParserException")
        self.assertNotIn("SECRET", json.dumps([result.model_dump(), events]))

    def test_configuration_and_programming_errors_still_propagate(self):
        for error in (RuntimeError("configuration"), TypeError("bug"), KeyError("bug"),
                      AttributeError("bug"), ValueError("unclassified error")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as raised:
                    self.rank(error=error)
                self.assertIs(raised.exception, error)

    def test_successful_ranking_preserves_existing_completion_event_payload(self):
        token, events = begin_activity_capture()
        try:
            self.rank()
        finally:
            end_activity_capture(token)
        self.assertEqual([event["event"] for event in events], [
            "location.centre_ranking.started", "location.centre_ranking.completed",
        ])
        self.assertEqual(set(events[-1]), {
            "time", "request_id", "event", "model", "selected_centres", "llm_called",
        })
        self.assertEqual(events[-1]["request_id"], self.query.request_id)
        self.assertEqual(events[-1]["selected_centres"], 2)

    def test_authentication_permissions_bad_request_and_model_errors_propagate(self):
        for cls, status in ((AuthenticationError, 401), (PermissionDeniedError, 403),
                            (BadRequestError, 400), (NotFoundError, 404)):
            error = cls("configuration", response=httpx.Response(status, request=self.sdk_request), body=None)
            with self.subTest(error=cls.__name__), self.assertRaises(cls):
                self.rank(error=error)

    def test_model_construction_and_structured_output_setup_errors_propagate(self):
        self.llm.side_effect = RuntimeError("OPENAI_API_KEY is required")
        with self.assertRaises(RuntimeError):
            self.rank()
        self.response.assert_not_called()
        self.llm.side_effect = None
        self.model.with_structured_output.side_effect = ValueError("Unsupported model configuration")
        with self.assertRaises(ValueError):
            self.rank()
        self.response.assert_not_called()
        try:
            location_agent.CentreRanking.model_validate({})
        except ValidationError as error:
            self.model.with_structured_output.side_effect = error
        with self.assertRaises(ValidationError):
            self.rank()
        self.response.assert_not_called()

    def test_invalid_input_and_missing_origin_still_propagate(self):
        with self.assertRaises(ValidationError):
            location_agent.run({"user_location": {"lat": 3, "lng": 101}})
        with patch.object(location_agent, "load_hawker_centres", return_value=[]):
            with self.assertRaises(location_agent.MissingOriginError):
                location_agent.run(LocationQuery(request="Find food nearby", parsed_request=ParsedRequest(
                    intent="food_discovery",
                ).model_dump()))
        self.llm.assert_not_called()

    def test_fallback_flows_through_run_and_standalone_public_contract(self):
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        catalogue = [{"id": item.centre_id, "name": item.centre_name,
                      "address": item.address} for item in self.routes]
        self.response.side_effect = TimeoutError()
        with patch.object(location_agent, "load_hawker_centres", return_value=catalogue), \
             patch.object(location_agent, "_collect_logistics", return_value=self.routes), \
             patch.object(location_agent, "_resolve_origin", return_value=(
                 {"lat": 1.33, "lng": 103.8, "label": "Test origin"}, "provided_coordinates",
             )), patch.object(main, "save_trace", side_effect=AssertionError("No trace writes")), \
             patch.object(main, "_resolve_session", side_effect=AssertionError("No session writes")):
            response = main.app.test_client().post("/api/location", json={
                "request": "Find food nearby", "parsed_request": ParsedRequest(
                    intent="food_discovery",
                ).model_dump(),
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        decision = LocationDecision.model_validate(response.get_json())
        self.assertEqual(set(response.get_json()), set(LocationDecision.model_fields))
        self.assertEqual([item.centre_id for item in decision.selected_centres], ["fastest", "balanced", "other"])
        self.assertIn("Deterministic fallback", decision.reasoning)
        self.assertLessEqual(decision.confidence, 0.5)
        self.response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
