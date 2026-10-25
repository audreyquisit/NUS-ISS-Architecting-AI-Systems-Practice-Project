"""Offline candidate-scope regressions; no report or persistence hooks."""

import os
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda

from agents import location_agent
from models.location_models import LocationDecision, LocationQuery, RankedCentre
from models.orchestration import ParsedRequest
from services.activity_log import begin_activity_capture, end_activity_capture


class LocationCandidateScopeTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.dict(os.environ, {
            "LOCATION_ROUTE_CANDIDATE_LIMIT": "30",
            "LOCATION_ROUTE_WORKERS": "6",
            "LOCATION_SHORT_WALK_THRESHOLD_MIN": "8",
        }))
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch(
            "services.location_service._get",
            side_effect=AssertionError("Location tests must not call OneMap"),
        ))
        self.llm = self.patches.enter_context(patch.object(
            location_agent, "_llm",
            side_effect=AssertionError("Candidate collection must not call an LLM"),
        ))
        self.catalogue_mock = self.patches.enter_context(patch.object(
            location_agent, "load_hawker_centres",
        ))
        self.candidates_mock = self.patches.enter_context(patch.object(
            location_agent, "get_candidate_centres",
        ))
        self.route = self.patches.enter_context(patch.object(
            location_agent, "get_route", side_effect=self._route_response,
        ))
        self.request_id = "27a71492-81b6-44a7-8715-c91e0aab4140"
        self._set_origin(1.3331, 103.7423)

    def _set_origin(self, lat, lng):
        self.origin = {"lat": lat, "lng": lng}
        # Each cluster exceeds the normal routing cap. Put origin candidates
        # last so selecting the first catalogue entries cannot pass the tests.
        self.catalogue = []
        for prefix, point in (
            ("east", (1.350, 103.955)),
            ("north", (1.435, 103.790)),
            ("origin", (lat, lng)),
        ):
            self.catalogue.extend({
                "id": f"{prefix}_{index:02d}",
                "name": f"Fixture {prefix} centre {index:02d}",
                "address": "Synthetic test address",
                "latitude": point[0] + index * 0.00001,
                "longitude": point[1],
            } for index in range(35))
        self.catalogue_mock.return_value = self.catalogue
        self.candidates_mock.return_value = self.catalogue

    @staticmethod
    def _route_response(_start, _destination, mode, *, request_id):
        return {
            "total_duration_min": 15 if mode == "walk" else 10,
            "mode": mode,
            "source": "onemap",
            "confidence": 0.9,
        }

    def _query(self, request="Find food near Jurong East MRT", **fields):
        payload = {
            "request_id": self.request_id,
            "request": request,
            "travel_mode": "pt",
            "parsed_request": ParsedRequest(
                intent="food_discovery", search_scope_mode="nearby",
            ).model_dump(),
        }
        payload.update(fields)
        return LocationQuery.model_validate(payload)

    def _collect(self, query, *, allow_short_walk_fallback=False):
        return location_agent._collect_logistics(
            query, self.origin,
            allow_short_walk_fallback=allow_short_walk_fallback,
        )

    def _assert_cluster(self, logistics, prefix, count=30):
        self.assertEqual(
            [item.centre_id for item in logistics],
            [f"{prefix}_{index:02d}" for index in range(count)],
        )

    def test_jurong_east_without_regional_scope_uses_resolved_origin(self):
        self._assert_cluster(self._collect(self._query()), "origin")

    def test_north_bridge_road_without_regional_scope_uses_resolved_origin(self):
        self._set_origin(1.301, 103.856)
        self._assert_cluster(self._collect(self._query(
            "Find food near North Bridge Road",
        )), "origin")

    def test_east_coast_park_without_regional_scope_uses_resolved_origin(self):
        self._set_origin(1.302, 103.912)
        self._assert_cluster(self._collect(self._query(
            "Find food near East Coast Park",
        )), "origin")

    def test_ordinary_place_without_compass_words_uses_resolved_origin(self):
        self._set_origin(1.315, 103.765)
        self._assert_cluster(self._collect(self._query(
            "Find food near Clementi MRT",
        )), "origin")

    def test_populated_structured_request_without_area_key_uses_origin(self):
        parsed = ParsedRequest(intent="food_discovery").model_dump(exclude_none=True)
        self.assertNotIn("requested_area", parsed)
        self._assert_cluster(self._collect(self._query(parsed_request=parsed)), "origin")

    def test_explicit_structured_east_overrides_raw_north_word(self):
        parsed = ParsedRequest(
            intent="food_discovery", requested_area="east",
        ).model_dump()
        self._assert_cluster(self._collect(self._query(
            "Find food from North Bridge Road", parsed_request=parsed,
        )), "east")

    def test_missing_or_empty_parse_preserves_raw_region_fallback(self):
        for fields in ({}, {"parsed_request": None}, {"parsed_request": {}}):
            with self.subTest(fields=fields):
                query = LocationQuery(
                    request="Find food near Jurong East MRT", travel_mode="pt",
                    **fields,
                )
                self._assert_cluster(self._collect(query), "east")

    def test_legacy_request_without_region_uses_origin(self):
        query = LocationQuery(request="Find food near Clementi MRT", travel_mode="pt")
        self._assert_cluster(self._collect(query), "origin")

    def test_candidate_and_worker_limits_are_preserved(self):
        for limit, workers, expected_count, expected_workers in (
            (30, 6, 30, 6), (3, 6, 3, 3), (0, 6, 1, 1),
            (3, 2, 3, 2), (3, 0, 3, 1),
        ):
            with self.subTest(limit=limit, workers=workers), patch.dict(os.environ, {
                "LOCATION_ROUTE_CANDIDATE_LIMIT": str(limit),
                "LOCATION_ROUTE_WORKERS": str(workers),
            }), patch.object(
                location_agent, "ThreadPoolExecutor",
                wraps=location_agent.ThreadPoolExecutor,
            ) as executor:
                self.route.reset_mock()
                logistics = self._collect(self._query())
                self._assert_cluster(logistics, "origin", expected_count)
                self.assertEqual(self.route.call_count, expected_count)
                executor.assert_called_once_with(max_workers=expected_workers)
        self.llm.assert_not_called()

    def test_default_pt_checks_one_walking_route_per_candidate(self):
        logistics = self._collect(self._query(), allow_short_walk_fallback=True)
        self._assert_cluster(logistics, "origin")
        self.assertEqual(self.route.call_count, 60)
        modes = [call.args[2] for call in self.route.call_args_list]
        self.assertEqual(modes.count("pt"), 30)
        self.assertEqual(modes.count("walk"), 30)
        self.llm.assert_not_called()

    def test_explicit_modes_use_one_route_per_candidate(self):
        for mode in ("pt", "walk", "drive", "cycle"):
            with self.subTest(mode=mode):
                self.route.reset_mock()
                logistics = self._collect(self._query(travel_mode=mode))
                self._assert_cluster(logistics, "origin")
                self.assertEqual(self.route.call_count, 30)
                for call in self.route.call_args_list:
                    self.assertEqual(call.args[0], self.origin)
                    self.assertEqual(call.args[2], mode)
                    self.assertEqual(call.kwargs, {"request_id": self.request_id})
                self.assertCountEqual(
                    [call.args[1] for call in self.route.call_args_list],
                    [item.coordinates.model_dump() for item in logistics],
                )
        self.llm.assert_not_called()

    def test_candidate_trace_payload_and_request_correlation_are_preserved(self):
        for area in (None, "east"):
            with self.subTest(area=area):
                query = self._query(parsed_request=ParsedRequest(
                    intent="food_discovery", requested_area=area,
                ).model_dump())
                token, events = begin_activity_capture()
                try:
                    self._collect(query)
                finally:
                    end_activity_capture(token)
                self.assertEqual([event["event"] for event in events], [
                    "location.candidate_pool.ready",
                    "location.onemap_routes.started",
                    "location.onemap_routes.completed",
                ])
                for event in events:
                    self.assertEqual(event["request_id"], self.request_id)
                    self.assertIsInstance(event["time"], str)
                pool = events[0]
                self.assertEqual(set(pool), {
                    "time", "request_id", "event", "candidate_source",
                    "catalogue_centres", "candidates_for_routing", "candidate_limit",
                    "filter_basis", "filter_method",
                })
                self.assertEqual(pool["candidate_source"], "NEA catalogue (local filtering)")
                self.assertEqual(pool["catalogue_centres"], len(self.catalogue))
                self.assertEqual(pool["candidates_for_routing"], 30)
                self.assertEqual(pool["candidate_limit"], 30)
                self.assertEqual(pool["filter_basis"], area or "route origin")
                self.assertEqual(pool["filter_method"], (
                    "broad-region anchor" if area else "route-origin coordinates"
                ))
                self.assertEqual(events[1]["route_candidates"], 30)
                self.assertEqual(events[2]["centres_submitted"], 30)
                self.assertEqual(events[2]["routes_with_duration"], 30)
                self.assertFalse(events[2]["llm_called"])

    def test_standalone_endpoint_preserves_contract_and_model_call_count(self):
        # Importing main normally loads .env; this isolated endpoint test must
        # not load credentials or touch trace/session persistence.
        with patch("dotenv.load_dotenv", return_value=False):
            import main

        intent = location_agent.LocationIntent(
            request_intent="find_nearby_centres", origin_place=None,
            proximity_preference="convenient", interpretation="Find nearby food.",
        )
        structured = ParsedRequest(
            intent="food_discovery", search_scope_mode="nearby",
        ).model_dump()
        for parsed, cluster in ((None, "east"), ({}, "east"), (structured, "origin")):
            with self.subTest(parsed=parsed), ExitStack() as patches:
                patches.enter_context(patch.object(
                    location_agent, "is_singapore_location", return_value=True,
                ))
                patches.enter_context(patch.object(
                    location_agent, "reverse_geocode_location", return_value=None,
                ))
                patches.enter_context(patch.object(
                    main, "save_trace", side_effect=AssertionError("No trace writes"),
                ))
                patches.enter_context(patch.object(
                    main, "_resolve_session", side_effect=AssertionError("No session access"),
                ))
                ranking = location_agent.CentreRanking(
                    selected_centres=[RankedCentre(
                        centre_id=f"{cluster}_00", rank=1, logistics_fit=0.9,
                        rationale="Mock ranking grounded in supplied routes.",
                    )], reasoning="Offline ranking response.", confidence=0.9,
                )
                intent_response = Mock(return_value=intent)
                ranking_response = Mock(return_value=ranking)
                model = Mock()
                model.with_structured_output.side_effect = lambda schema, **_: RunnableLambda(
                    intent_response if schema is location_agent.LocationIntent
                    else ranking_response
                )
                self.llm.side_effect = None
                self.llm.return_value = model
                self.llm.reset_mock()
                self.route.reset_mock()
                payload = {
                    "request_id": self.request_id,
                    "request": "Find food near Jurong East MRT",
                    "user_location": self.origin,
                    "travel_mode": "pt",
                }
                if parsed is not None:
                    payload["parsed_request"] = parsed
                response = main.app.test_client().post("/api/location", json=payload)
                self.assertEqual(response.status_code, 200, response.get_json())
                body = response.get_json()
                decision = LocationDecision.model_validate(body)
                self.assertEqual(set(body), set(LocationDecision.model_fields))
                self._assert_cluster(decision.logistics, cluster)
                self.assertEqual(decision.selected_centres, ranking.selected_centres)
                self.assertEqual(decision.origin.model_dump(), self.origin)
                self.assertEqual(self.route.call_count, 30)
                self.assertEqual(intent_response.call_count, 0 if parsed else 1)
                ranking_response.assert_called_once()
                self.assertEqual(self.llm.call_count, 1 if parsed else 2)


if __name__ == "__main__":
    unittest.main()
