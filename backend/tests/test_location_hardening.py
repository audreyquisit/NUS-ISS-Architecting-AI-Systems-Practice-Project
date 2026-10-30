"""Offline regressions for directions limits, content filtering and route evidence."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI
from openai import ContentFilterFinishReasonError

from agents import location_agent as agent, location_grounding, verifier
from models.location_models import LocationDecision, LocationQuery, TravelEstimate
from models.orchestration import ParsedRequest
from services import location_service
from test_location_grounded_reasoning import comparative_reason, grounded_ranking, route
import test_location_grounding_handoff as handoff


class DirectionsLimitTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.centre = next(item for item in location_service.load_hawker_centres()
                           if item["id"] == "newton_food_centre")
        self.origin = {"lat": 1.3, "lng": 103.8, "label": "Synthetic fixture origin"}
        self.patches.enter_context(patch.object(agent, "_resolve_origin", return_value=(
            self.origin, "provided_coordinates")))
        self.patches.enter_context(patch.object(agent, "get_candidate_centres", return_value=[self.centre]))
        self.routing = self.patches.enter_context(patch.object(agent, "get_route"))
        self.model = self.patches.enter_context(patch.object(agent, "_llm",
            side_effect=AssertionError("Structured directions need no model")))
        self.patches.enter_context(patch.object(location_service, "_get",
            side_effect=AssertionError("No live OneMap")))

    def query(self, maximum=5, intent="directions"):
        return LocationQuery(request="Directions to Newton Food Centre", travel_mode="pt",
            max_travel_time_min=maximum, parsed_request=ParsedRequest(intent=intent,
                target_centre_text="Newton Food Centre", max_travel_time_min=maximum).model_dump())

    def execute(self, minutes, maximum=5):
        self.routing.return_value = {"total_duration_min": minutes, "mode": "pt",
                                     "source": "onemap", "confidence": 0.85}
        return LocationDecision.model_validate(agent.run(self.query(maximum)))

    def test_over_limit_directions_have_no_eligible_selection(self):
        result = self.execute(18)
        self.assertEqual(result.selected_centres, [])
        self.assertEqual(result.confidence, 0)
        self.assertTrue(any("limit" in item for item in result.limitations))
        self.assertEqual(result.origin.model_dump(), {"lat": 1.3, "lng": 103.8})
        self.routing.assert_called_once()
        self.model.assert_not_called()

    def test_duration_equal_to_limit_is_eligible(self):
        result = self.execute(5)
        self.assertEqual([item.centre_id for item in result.selected_centres], [self.centre["id"]])
        self.assertEqual(result.logistics[0].travel.total_duration_min, 5)
        self.assertEqual(result.logistics[0].travel.source, "onemap")
        self.routing.assert_called_once()

    def test_missing_and_invalid_duration_cannot_satisfy_limit(self):
        for minutes in (None, -1, float("nan"), float("inf"), True, "4", 4.5):
            with self.subTest(minutes=minutes):
                self.assertEqual(self.execute(minutes).selected_centres, [])
        self.model.assert_not_called()

    def test_unlimited_directions_preserve_existing_result(self):
        result = self.execute(18, maximum=None)
        self.assertEqual(len(result.selected_centres), 1)
        self.assertEqual(result.logistics[0].travel.total_duration_min, 18)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.selected_centres[0].logistics_fit, 1)
        self.routing.assert_called_once()
        self.model.assert_not_called()

    def test_zero_duration_at_zero_limit_is_eligible(self):
        self.assertEqual(len(self.execute(0, maximum=0).selected_centres), 1)

    def test_named_catalogue_lookup_cannot_establish_hard_travel_eligibility(self):
        query = self.query(intent="centre_details").model_copy(update={"request": "Food at Newton Food Centre"})
        result = LocationDecision.model_validate(agent.run(query))
        self.assertEqual(result.selected_centres, [])
        self.assertTrue(any("route" in item.lower() for item in result.limitations))
        self.routing.assert_not_called()
        self.model.assert_not_called()

    def test_named_catalogue_lookup_without_limit_remains_available(self):
        query = self.query(maximum=None, intent="centre_details").model_copy(
            update={"request": "Food at Newton Food Centre"})
        result = LocationDecision.model_validate(agent.run(query))
        self.assertEqual(len(result.selected_centres), 1)
        self.assertEqual(result.logistics, [])
        self.routing.assert_not_called()

    def test_standalone_endpoint_enforces_limit_without_changing_public_fields(self):
        with patch("dotenv.load_dotenv", return_value=False):
            import main
        self.routing.return_value = {"total_duration_min": 18, "mode": "pt",
                                     "source": "onemap", "confidence": 0.85}
        response = main.app.test_client().post("/api/location", json=self.query().model_dump())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["selected_centres"], [])
        self.assertEqual(set(response.get_json()), set(LocationDecision.model_fields))
        self.routing.assert_called_once()


class LocationContentFilterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = handoff.LocationGroundingHandoffTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_actual_sdk_content_filter_falls_back_through_location_execution_once(self):
        attempts = []

        def filtered(request):
            attempts.append(request)
            return httpx.Response(200, json={
                "id": "chatcmpl-offline-location", "object": "chat.completion",
                "created": 1, "model": "gpt-4o", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter"}],
            })

        self.fixture.query = self.fixture.query.model_copy(update={"max_travel_time_min": 10})
        with httpx.Client(transport=httpx.MockTransport(filtered), trust_env=False) as client:
            model = ChatOpenAI(model="gpt-4o", api_key="offline-test", http_client=client, max_retries=0)
            self.fixture.llm.return_value = model
            decision, bundle = self.fixture.execute()
        self.assertEqual([item["centre_id"] for item in decision["selected_centres"]], ["fast", "unknown"])
        self.assertTrue(all(item["logistics_fit"] == 0.5 for item in decision["selected_centres"]))
        # This SDK's function-calling path returns no parsed selection on a
        # filtered completion; other supported SDK paths raise the typed error.
        self.assertIn(bundle.fallback_reason, {"ValidationError", "ContentFilterFinishReasonError"})
        self.assertEqual(bundle.decision_source, "deterministic_fallback")
        self.assertEqual(bundle.confidence_basis, "deterministic_policy")
        self.assertIn(bundle.fallback_reason, " ".join(decision["limitations"]))
        self.assertEqual(len(attempts), 1)
        self.fixture.llm.assert_called_once()
        self.fixture.collect.assert_called_once()
        with patch.object(verifier, "load_hawker_centres", return_value=self.fixture.catalogue):
            checked = verifier._verified_location({"payload": decision},
                {"intent": "food_discovery", "max_travel_time_min": 10},
                grounding=bundle, request_id=self.fixture.query.request_id)["payload"]
        self.assertEqual([item["centre_id"] for item in checked["selected_centres"]], ["fast", "unknown"])

    def test_typed_content_filter_exception_retains_preference_aware_fallback(self):
        self.fixture.response.side_effect = ContentFilterFinishReasonError()
        self.fixture.query = self.fixture.query.model_copy(update={"priorities": ["fewest transfers"]})
        decision, bundle = self.fixture.execute()
        self.assertEqual(decision["selected_centres"][0]["centre_id"], "direct")
        self.assertEqual(bundle.fallback_reason, "ContentFilterFinishReasonError")
        self.assertEqual(bundle.fallback_order, ("transfers", "total_duration_min"))
        self.fixture.response.assert_called_once()


class TransferEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.http = self.patches.enter_context(patch.object(location_service, "_get"))

    def estimate(self, legs=None, *, omit=False):
        itinerary = {"duration": 600}
        if not omit:
            itinerary["legs"] = legs
        self.http.return_value = {"plan": {"itineraries": [itinerary]}}
        return TravelEstimate.model_validate(location_service.get_route(
            {"lat": 1.3, "lng": 103.8}, {"lat": 1.31, "lng": 103.81}, "pt"))

    def candidate(self, centre="unknown"):
        item = route(centre, 10)
        item.travel = self.estimate(omit=True)
        return item

    def test_duration_survives_missing_or_incomplete_legs_with_unknown_transfers(self):
        for legs in (None, [], {}, "missing", [None], [{}], [{"mode": ""}],
                     [{"mode": "BUS"}, {}]):
            with self.subTest(legs=legs):
                estimate = self.estimate(legs)
                self.assertEqual(estimate.total_duration_min, 10)
                self.assertIsNone(estimate.transfers)
        self.assertIsNone(self.estimate(omit=True).transfers)

    def test_verified_direct_and_transfer_itineraries_keep_measured_counts(self):
        for legs, expected in (([{"mode": "BUS"}], 0),
                ([{"mode": "WALK", "distance": 100}, {"mode": "BUS"}], 0),
                ([{"mode": "BUS"}, {"mode": "WALK"}, {"mode": "SUBWAY"}], 1),
                ([{"mode": "WALK", "distance": 100}], 0)):
            with self.subTest(legs=legs):
                self.assertEqual(self.estimate(legs).transfers, expected)

    def test_unknown_transfer_candidate_keeps_duration_but_not_transfer_priority(self):
        unknown = self.candidate()
        known = route("known", 12, transfers=0)
        intent = agent.LocationIntent(request_intent="find_nearby_centres",
            proximity_preference="convenient", interpretation="Compare routes")
        query = LocationQuery(priorities=["fewest transfers"])
        shortlist = agent._shortlist_for_reasoning(query, intent, [unknown, known])
        self.assertEqual([item.centre_id for item in shortlist], ["unknown", "known"])
        fallback = agent._deterministic_ranking(query, intent, shortlist, "offline")
        self.assertEqual([item.centre_id for item in fallback.selected_centres], ["known", "unknown"])

    def test_grounded_unknown_reason_cannot_become_zero_transfer_comparison(self):
        unknown, known = self.candidate(), route("known", 12, transfers=1)
        _, facts = agent._primary_evidence([unknown, known])
        fact = facts["unknown:primary:transfers"]
        self.assertIsNone(fact["value"])
        self.assertEqual(fact["provenance"], "unavailable")
        response = grounded_ranking(["unknown"])
        response.selected_centres[0].reasons = [agent._RouteReason(
            code="METRIC_UNAVAILABLE", evidence_refs=[fact["id"]])]
        intent = agent.LocationIntent(request_intent="find_nearby_centres",
            proximity_preference="convenient", interpretation="Compare routes")
        query = LocationQuery()
        profile = agent._normalize_preferences(query, intent)
        ranking = agent._render_grounded_ranking(response, profile, query, [unknown, known], facts)
        self.assertIn("Transfer count unavailable", ranking.selected_centres[0].rationale)
        response.selected_centres[0].reasons = [comparative_reason(
            "unknown", "known", "FEWER_KNOWN_TRANSFERS", ["transfers"])]
        with self.assertRaises(location_grounding._RankingOutputError):
            agent._render_grounded_ranking(response, profile, query, [unknown, known], facts)

    def test_adapter_unknown_survives_private_grounding_verifier_and_rendering(self):
        estimate = self.estimate(omit=True)
        fixture = handoff.LocationGroundingHandoffTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.routes[0].travel = estimate
        fixture.response.return_value = grounded_ranking(["fast"])
        decision, bundle = fixture.execute()
        fact = next(fact for snapshot in bundle.routes if snapshot.centre_id == "fast"
                    for fact in snapshot.facts if fact.metric == "transfers")
        self.assertIsNone(fact.value)
        self.assertEqual(fact.provenance, "unavailable")
        with patch.object(verifier, "load_hawker_centres", return_value=fixture.catalogue):
            checked, context = verifier._verify_results_with_context(
                [{"agent": "location", "payload": decision}], {"intent": "food_discovery"},
                location_grounding=bundle, request_id=fixture.query.request_id)
        self.assertIsNone(checked[0]["payload"]["logistics"][0]["travel"]["transfers"])
        self.assertIsNone(context.centres[0].primary_route.transfers.value)
        from agents.recommendation_rendering import _render
        text = _render(context, [], directions=True)
        self.assertIn("Transfer count unknown", text)
        self.assertNotIn("Known transfers: 0", text)
