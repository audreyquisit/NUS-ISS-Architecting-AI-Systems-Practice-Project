"""Session search boundaries through Flask, SQLite and the production graph."""
import copy
import json
import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import main
import test_session_store as session_fixtures
from agents import orchestrator as orch, location_agent as loc, recommendation_agent as rec, dietary_agent
from evaluations.pilot_support import build_routes, MENU_ITEM_IDS
from models.orchestration import ParsedRequest, OrchestrationPlan
from services import session_store, trace_store, location_service, hawker_service
from test_location_grounded_reasoning import grounded_ranking


ORIGIN = "Synthetic central Singapore origin"
NEWTON = "newton_food_centre"
MAXWELL = "maxwell_food_centre"
TEKKA = "buffalo_road_blk_665_tekka_centre_zhu_jiao_market"


class SessionSearchLifecycleTests(session_fixtures.SessionStoreTestCase):
    def setUp(self):
        super().setUp()
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        directory = os.path.join(self._temp_dir.name, "traces")
        self.patches.enter_context(patch.object(trace_store, "_TRACE_DIR", directory))
        self.patches.enter_context(patch.object(trace_store, "_DB_PATH", os.path.join(directory, "traces.db")))
        self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch.dict("os.environ", {"OPENAI_API_KEY": "", "SESSION_RETENTION_DAYS": "30", "TRACE_RETENTION_DAYS": "30"}))
        self.patches.enter_context(patch.dict(main.app.config, {"TESTING": True}))
        self.client = main.app.test_client()
        self.sid = None
        self.turns = []
        self.menu = tuple(copy.deepcopy(hawker_service.get_menu_item_record(mid)) for mid in MENU_ITEM_IDS)
        self.routes = build_routes(location_service.load_hawker_centres())
        self.routes[0].travel.total_duration_min = 12
        self.patches.enter_context(patch.object(hawker_service, "_menu_catalogue", return_value=self.menu))
        self.patches.enter_context(patch.object(location_service, "_get", side_effect=AssertionError("No live OneMap")))
        self.patches.enter_context(patch.object(dietary_agent, "_llm_assess", side_effect=AssertionError("No live Dietary")))
        self.patches.enter_context(patch("langchain_openai.ChatOpenAI", side_effect=AssertionError("No unexpected model")))
        planner = type("Planner", (), {"with_structured_output": lambda _, schema, **kwargs: RunnableLambda(self.plan)})()
        self.patches.enter_context(patch.object(orch, "build_model", side_effect=lambda state: {**state, "model": planner}))
        self.patches.enter_context(patch.object(loc, "_resolve_origin", side_effect=self.origin))
        self.patches.enter_context(patch.object(loc, "_collect_logistics", return_value=self.routes))
        rank = loc._rank_centres
        def capture(query, intent, logistics, **kwargs):
            self.active["location_requests"].append(query.model_dump())
            return rank(query, intent, logistics, **kwargs)
        self.patches.enter_context(patch.object(loc, "_rank_centres", side_effect=capture))
        ranker = type("Ranker", (), {"with_structured_output": lambda _, *args, **kwargs: RunnableLambda(self.ranking)})()
        self.patches.enter_context(patch.object(loc, "_llm", return_value=ranker))
        recommender = type("Recommender", (), {"with_structured_output": lambda _, schema, **kwargs: RunnableLambda(
            lambda prompt: schema.model_validate({"choices": [{"menu_item_id": row["menu_item_id"], "reason_refs": ["eligible_menu"]}
                for row in json.JSONDecoder().raw_decode(prompt.messages[-1].content)[0]]}))})()
        self.patches.enter_context(patch.object(rec, "ChatOpenAI", return_value=recommender))
        self.reuse = self.patches.enter_context(patch.object(orch, "reuse_location", side_effect=AssertionError("Do not enable cache reuse")))

    def plan(self, prompt):
        self.active["planner_prompt"] = prompt.to_string()
        return OrchestrationPlan.model_validate({"request": self.active["planned"], "tasks": []})

    def ranking(self, prompt):
        query = self.active["location_requests"][-1]
        eligible = [route for route in self.routes if query["max_travel_time_min"] is None
                    or route.travel.total_duration_min <= query["max_travel_time_min"]]
        metric = "walking_distance_m" if query["priorities"] and query["priorities"][0] == "least walking" else "total_duration_min"
        eligible.sort(key=lambda route: getattr(route.travel, metric))
        return grounded_ranking([route.centre_id for route in eligible[:self.active["selection_limit"]]])

    @staticmethod
    def origin(query, intent):
        if not query.location_text and not query.conversation_context.get("state", {}).get("location_text"):
            raise loc.MissingOriginError("Where are you starting from?")
        return {"lat": 1.33333337, "lng": 103.88888889, "label": ORIGIN}, "free_text_place"

    def chat(self, message, selection_limit=3, **fields):
        values = {"intent": "food_discovery", "origin_text": ORIGIN, "travel_mode": "pt",
                  "menu_item_preferences": ["chicken rice"]}
        values.update(fields)
        self.active = {"planned": ParsedRequest(**values).model_dump(), "location_requests": [], "selection_limit": selection_limit}
        payload = {"message": message}
        if self.sid:
            payload["session_id"] = self.sid
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.sid = body["session_id"]
        self.active["body"] = body
        self.active["state"] = self.state()
        self.turns.append(self.active)
        self.reuse.assert_not_called()
        return body

    def state(self):
        return session_store.get_context(self.sid)["state"]

    @staticmethod
    def centres(body):
        return [row["centre_id"] for row in (body.get("location_data") or {}).get("selected_centres", [])]

    @staticmethod
    def menus(body):
        joined = next((row for row in body.get("verified_results", []) if row["agent"] == "candidate_join"), {})
        return [row["menu_item_id"] for row in joined.get("candidates", [])]

    def first(self, **fields):
        return self.chat("Find chicken rice from " + ORIGIN + ".", **fields)

    def test_independent_unlimited_replaces_old_limit_and_preserves_origin(self):
        first = self.first(max_travel_time_min=15)
        self.assertEqual(self.centres(first), [NEWTON])
        second = self.chat("Start a new independent chicken rice search from " + ORIGIN + "; no travel-time limit.")
        self.assertEqual(len(self.menus(second)), 3)
        self.assertIsNone(self.turns[-1]["location_requests"][0]["max_travel_time_min"])
        self.assertIsNone(self.state()["max_travel_time_min"])
        self.assertEqual(self.state()["priorities"], [])
        self.assertEqual(self.state()["location_text"], ORIGIN)
        self.assertEqual(len(session_store.get_context(self.sid)["turns"]), 4)

    def test_independent_fastest_replaces_walking_order(self):
        self.first(priorities=["least walking"])
        second = self.chat("Start a new independent chicken rice search; fastest possible.", priorities=["fastest"])
        self.assertEqual(self.centres(second), [NEWTON, MAXWELL, TEKKA])
        self.assertEqual(self.state()["priorities"], ["fastest"])

    def test_completed_zero_approval_clears_shortlist_and_menu_ids(self):
        self.first()
        second = self.chat("Find chicken rice from " + ORIGIN + " within five minutes.", max_travel_time_min=5)
        self.assertEqual(self.centres(second), [])
        self.assertEqual(self.menus(second), [])
        self.assertEqual(self.state()["location_result"], {"selected_centres": [], "logistics": []})
        self.assertEqual(self.state()["previous_menu_item_ids"], [])
        third = self.chat("Start another independent chicken rice search; no travel-time limit.")
        self.assertEqual(len(self.menus(third)), 3)
        self.assertIsNone(self.state()["max_travel_time_min"])

    def test_clarification_and_answer_retain_pending_context(self):
        first = self.chat("Find chicken rice within 30 minutes.", origin_text=None,
            max_travel_time_min=30, clarification_question="Where are you starting from?")
        self.assertEqual(first["reply"], "Where are you starting from?")
        self.assertEqual(self.turns[-1]["location_requests"], [])
        second = self.chat(ORIGIN, max_travel_time_min=30,
            field_sources={"origin_text": "current_message", "max_travel_time_min": "conversation_context"})
        self.assertIn("Find chicken rice within 30 minutes.", self.turns[-1]["planner_prompt"])
        self.assertEqual(len(self.menus(second)), 3)
        self.assertEqual(self.state()["max_travel_time_min"], 30)

    def test_food_followup_inherits_intended_preferences_without_wrong_exclusions(self):
        self.first(priorities=["least walking"], max_travel_time_min=30)
        second = self.chat("What about laksa?", menu_item_preferences=["laksa"], priorities=["least walking"],
            max_travel_time_min=30, field_sources={"max_travel_time_min": "conversation_context", "priorities": "conversation_context"})
        self.assertEqual(self.centres(second), [TEKKA, MAXWELL, NEWTON])
        self.assertEqual(len(self.menus(second)), 3)
        self.assertEqual(second["parsed_request"]["excluded_menu_item_ids"], [])
        self.assertEqual(self.state()["max_travel_time_min"], 30)
        self.assertEqual(self.state()["priorities"], ["least walking"])

    def test_new_and_another_independent_do_not_import_menu_exclusions(self):
        self.first()
        for wording in ("Start a new independent", "Start another independent"):
            second = self.chat(wording + " chicken rice search.")
            self.assertEqual(second["parsed_request"]["excluded_menu_item_ids"], [])
            self.assertEqual(len(self.menus(second)), 3)

    def test_another_option_excludes_previous_relevant_menu(self):
        first = self.first(selection_limit=1)
        self.assertEqual(self.menus(first), ["wakwak_37_item_005"])
        second = self.chat("Show me another chicken rice option.")
        self.assertEqual(second["parsed_request"]["excluded_menu_item_ids"], ["wakwak_37_item_005"])
        self.assertNotIn("wakwak_37_item_005", self.menus(second))
        self.assertEqual(set(self.menus(second)), {"wakwak_881_item_000", "wakwak_448_item_000"})

    def test_alternative_ids_are_relevant_and_explicit_exclusions_preserved(self):
        self.chat("Find laksa from " + ORIGIN + ".", menu_item_preferences=["laksa"])
        second = self.chat("Show me another chicken rice option.", excluded_menu_item_ids=["wakwak_37_item_005"])
        self.assertEqual(second["parsed_request"]["excluded_menu_item_ids"], ["wakwak_37_item_005"])
        self.assertEqual(len(self.menus(second)), 2)

    def test_explicit_unlimited_overrides_mocked_historical_promotion(self):
        self.first(max_travel_time_min=15)
        second = self.chat("Start another independent chicken rice search; no travel-time limit.",
            max_travel_time_min=15, field_sources={"max_travel_time_min": "conversation_context"},
            uncertain_fields=["max_travel_time_min"])
        request = second["parsed_request"]
        self.assertIsNone(request["max_travel_time_min"])
        self.assertNotIn("max_travel_time_min", request["field_sources"])
        self.assertNotIn("max_travel_time_min", request["uncertain_fields"])
        self.assertIsNone(self.turns[-1]["location_requests"][0]["max_travel_time_min"])
        self.assertEqual(len(self.menus(second)), 3)
        self.assertIsNone(self.state()["max_travel_time_min"])

    def test_contradictory_unlimited_and_absolute_limit_clarifies(self):
        self.first(max_travel_time_min=15)
        second = self.chat("Find chicken rice; no travel-time limit, but within 20 minutes.", max_travel_time_min=20)
        self.assertIn("no travel-time limit", second["reply"])
        self.assertEqual(self.turns[-1]["location_requests"], [])
        self.assertIsNotNone(second["parsed_request"]["clarification_question"])

    def test_ambiguous_hour_limit_is_not_silently_erased(self):
        self.first(max_travel_time_min=15)
        second = self.chat("Find chicken rice; no travel-time limit, but within half an hour.", max_travel_time_min=30)
        self.assertEqual(second["parsed_request"]["max_travel_time_min"], 30)
        self.assertIsNotNone(second["parsed_request"]["clarification_question"])
        self.assertEqual(self.turns[-1]["location_requests"], [])

    def test_incomplete_turn_does_not_clear_previous_shortlist(self):
        self.first(priorities=["least walking"])
        before = self.state()
        self.chat("Find chicken rice.", origin_text=None, clarification_question="Where are you starting from?")
        self.assertEqual(self.state()["location_result"], before["location_result"])
        self.assertEqual(self.state()["priorities"], ["least walking"])

    def test_router_failure_preserves_previous_state(self):
        self.first(max_travel_time_min=15)
        before = copy.deepcopy(self.state())
        with patch.object(loc, "_collect_logistics", side_effect=location_service.LocationServiceError("Fixture unavailable")):
            self.chat("Find chicken rice from " + ORIGIN + ".", priorities=["fastest"])
        self.assertEqual(self.state(), before)

    def test_application_failure_preserves_context_and_user_turn(self):
        self.first(max_travel_time_min=15)
        before = copy.deepcopy(self.state())
        count = len(session_store.get_context(self.sid)["turns"])
        with patch.object(main, "run_orchestrator", side_effect=RuntimeError("Fixture failure")), self.assertRaises(RuntimeError):
            self.client.post("/api/chat", json={"message": "Find food", "session_id": self.sid})
        self.assertEqual(self.state(), before)
        self.assertEqual(len(session_store.get_context(self.sid)["turns"]), count + 1)

    def test_generic_sqlite_shallow_merge_is_unchanged(self):
        self.first()
        session_store.update_state(self.sid, {"unrelated": "retained", "priorities": ["least walking"]})
        session_store.update_state(self.sid, {"travel_mode": "pt"})
        self.assertEqual(self.state()["unrelated"], "retained")
        self.assertEqual(self.state()["priorities"], ["least walking"])

    def test_bare_alternative_words_are_insufficient(self):
        request = ParsedRequest(intent="food_discovery", menu_item_preferences=["chicken rice"])
        for message in ("another", "other", "different", "Find different chicken rice", "Start another independent chicken rice search"):
            self.assertFalse(orch._requests_alternative_options(message, request))
        self.assertTrue(orch._requests_alternative_options("Show another option", request))

    def test_unrecognized_negation_is_not_guessed(self):
        # Deliberately outside the narrow direct assertion guard; a Planner
        # misinterpretation of this double negation remains a residual risk.
        self.first(max_travel_time_min=15)
        second = self.chat("Find chicken rice. I don't want no travel-time limit.", max_travel_time_min=15)
        self.assertEqual(second["parsed_request"]["max_travel_time_min"], 15)
        self.assertEqual(self.state()["max_travel_time_min"], 15)

    def test_historical_promotion_without_explicit_negation_remains_a_planner_risk(self):
        self.first(max_travel_time_min=15)
        # Without an unambiguous current override, normalization cannot decide
        # whether this historical constraint is intentional or a Planner error.
        second = self.chat("Start a new independent chicken rice search.", max_travel_time_min=15,
            field_sources={"max_travel_time_min": "conversation_context"})
        self.assertEqual(second["parsed_request"]["max_travel_time_min"], 15)
        self.assertEqual(self.centres(second), [NEWTON])

    def test_unlimited_does_not_invent_an_absolute_limit_from_extra_minutes(self):
        self.first(max_travel_time_min=15)
        second = self.chat("Find chicken rice with no travel-time limit; four extra minutes for less walking is fine.",
            max_travel_time_min=15, field_sources={"max_travel_time_min": "conversation_context"})
        self.assertIsNone(second["parsed_request"]["max_travel_time_min"])
        self.assertIsNone(second["parsed_request"]["clarification_question"])
        self.assertEqual(len(self.menus(second)), 3)


if __name__ == "__main__":
    unittest.main()

