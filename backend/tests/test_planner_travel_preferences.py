"""Captured Planner semantic failures, replayed without models or services."""
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda
from agents import location_agent, orchestrator
from models.orchestration import OrchestrationPlan

# Actual four-case live outputs, before production task/provenance normalization.
CAPTURED = json.loads(r'''[
  {
    "message": "Find chicken rice. Fastest possible.",
    "plan": {
      "request": {
        "intent": "food_discovery",
        "origin_text": "Synthetic central Singapore origin",
        "target_centre_text": null,
        "requested_area": null,
        "search_scope_mode": null,
        "travel_mode": null,
        "dietary_preferences": [],
        "cuisine_preferences": [],
        "menu_item_preferences": [
          "chicken rice"
        ],
        "excluded_menu_item_ids": [],
        "stall_name_preferences": [],
        "budget_amount": null,
        "available_time_min": null,
        "max_travel_time_min": null,
        "max_queue_min": null,
        "time_period": null,
        "priorities": [
          "fastest"
        ],
        "clarification_question": null,
        "field_sources": {
          "origin_text": "conversation context",
          "menu_item_preferences": "current message",
          "priorities": "current message"
        },
        "uncertain_fields": []
      },
      "tasks": [
        {
          "agent": "queue",
          "task": "Find chicken rice with the shortest wait time from Synthetic central Singapore origin."
        }
      ]
    }
  },
  {
    "message": "Find chicken rice. Least walking possible.",
    "plan": {
      "request": {
        "intent": "food_discovery",
        "origin_text": "Synthetic central Singapore origin",
        "target_centre_text": null,
        "requested_area": null,
        "search_scope_mode": null,
        "travel_mode": null,
        "dietary_preferences": [],
        "cuisine_preferences": [],
        "menu_item_preferences": [
          "chicken rice"
        ],
        "excluded_menu_item_ids": [],
        "stall_name_preferences": [],
        "budget_amount": null,
        "available_time_min": null,
        "max_travel_time_min": null,
        "max_queue_min": null,
        "time_period": null,
        "priorities": [
          "least walking"
        ],
        "clarification_question": null,
        "field_sources": {
          "origin_text": "conversation context",
          "menu_item_preferences": "current message",
          "priorities": "current message"
        },
        "uncertain_fields": []
      },
      "tasks": [
        {
          "agent": "queue",
          "task": "Find chicken rice stalls with minimal walking from Synthetic central Singapore origin."
        }
      ]
    }
  },
  {
    "message": "Find chicken rice. Fastest matters, but I'd accept about four extra minutes for much less walking.",
    "plan": {
      "request": {
        "intent": "food_discovery",
        "origin_text": "Synthetic central Singapore origin",
        "target_centre_text": null,
        "requested_area": null,
        "search_scope_mode": null,
        "travel_mode": null,
        "dietary_preferences": [],
        "cuisine_preferences": [],
        "menu_item_preferences": [
          "chicken rice"
        ],
        "excluded_menu_item_ids": [],
        "stall_name_preferences": [],
        "budget_amount": null,
        "available_time_min": null,
        "max_travel_time_min": 4,
        "max_queue_min": null,
        "time_period": null,
        "priorities": [
          "fastest"
        ],
        "clarification_question": null,
        "field_sources": {
          "origin_text": "conversation context",
          "menu_item_preferences": "current message",
          "priorities": "current message",
          "max_travel_time_min": "current message"
        },
        "uncertain_fields": []
      },
      "tasks": [
        {
          "agent": "location",
          "task": "Find chicken rice stalls near Synthetic central Singapore origin with the fastest route, allowing up to 4 extra minutes for less walking."
        }
      ]
    }
  },
  {
    "message": "Find chicken rice. I don't mind a slightly longer trip if I can avoid walking.",
    "plan": {
      "request": {
        "intent": "food_discovery",
        "origin_text": "Synthetic central Singapore origin",
        "target_centre_text": null,
        "requested_area": null,
        "search_scope_mode": null,
        "travel_mode": "drive",
        "dietary_preferences": [],
        "cuisine_preferences": [],
        "menu_item_preferences": [
          "chicken rice"
        ],
        "excluded_menu_item_ids": [],
        "stall_name_preferences": [],
        "budget_amount": null,
        "available_time_min": null,
        "max_travel_time_min": null,
        "max_queue_min": null,
        "time_period": null,
        "priorities": [],
        "clarification_question": null,
        "field_sources": {
          "origin_text": "conversation context",
          "menu_item_preferences": "current message",
          "travel_mode": "current message"
        },
        "uncertain_fields": []
      },
      "tasks": [
        {
          "agent": "location",
          "task": "Resolve the origin 'Synthetic central Singapore origin' to coordinates."
        }
      ]
    }
  }
]''')


class PlannerTravelPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("services.activity_log.log_activity"))
        self.patches.enter_context(patch("services.location_service._get",
            side_effect=AssertionError("No live routing")))
        self.context = {"state": {}, "turns": [
            {"role": "user", "content": "My starting point is Synthetic central Singapore origin."},
            {"role": "assistant", "content": "I'll use Synthetic central Singapore origin as your starting point."},
        ]}

    def parse(self, case=0, *, message=None, fields=None, context=None):
        captured = copy.deepcopy(CAPTURED[case])
        message = message or captured["message"]
        captured["plan"]["request"].update(fields or {})
        model = Mock()
        self.prompts = []
        def reply(prompt):
            self.prompts.append(prompt.to_string())
            return OrchestrationPlan.model_validate(captured["plan"])
        model.with_structured_output.return_value = RunnableLambda(reply)
        result = orchestrator.parse_request({"user_request": message, "model": model,
            "conversation_context": self.context if context is None else context})
        model.with_structured_output.assert_called_once_with(OrchestrationPlan, method="function_calling")
        self.assertEqual(len(self.prompts), 1)
        self.assertIn(message, self.prompts[0])
        return result

    def handoff(self, parsed, message):
        with patch.object(location_agent, "_run_with_grounding", return_value=({}, None)) as run:
            orchestrator.location_worker({"user_request": message, **parsed})
        run.assert_called_once()
        sent = run.call_args.args[0]
        self.assertEqual(sent["request"], message)
        self.assertEqual(sent["parsed_request"], parsed["parsed_request"])
        return sent["parsed_request"]

    def test_captured_fastest_routes_to_location_not_queue(self):
        result = self.parse(0)
        self.assertEqual(result["parsed_request"]["priorities"], ["fastest"])
        self.assertIn("location", [t.agent for t in result["tasks"]])
        self.assertNotIn("queue", [t.agent for t in result["tasks"]])
        self.assertIsNone(result["parsed_request"]["max_travel_time_min"])

    def test_captured_least_walking_routes_to_location_not_queue(self):
        result = self.parse(1)
        self.assertEqual(result["parsed_request"]["priorities"], ["least walking"])
        self.assertNotIn("queue", [t.agent for t in result["tasks"]])
        self.assertIn("location", [t.agent for t in result["tasks"]])

    def test_captured_relative_allowance_cannot_be_absolute_limit(self):
        result = self.parse(2)
        request = self.handoff(result, CAPTURED[2]["message"])
        self.assertIsNone(request["max_travel_time_min"])
        self.assertNotIn("max_travel_time_min", request["field_sources"])
        self.assertIn("relative_travel_time_tolerance", request["uncertain_fields"])
        self.assertIsNone(request["clarification_question"])
        # Missing walking objective in the captured output is not fabricated here.
        self.assertEqual(request["priorities"], ["fastest"])

    def test_captured_walking_avoidance_does_not_authorize_driving(self):
        result = self.parse(3)
        request = self.handoff(result, CAPTURED[3]["message"])
        self.assertIsNone(request["travel_mode"])
        self.assertNotIn("travel_mode", request["field_sources"])
        self.assertIn("travel_mode", request["uncertain_fields"])
        self.assertEqual(request["priorities"], [])

    def test_real_within_four_minutes_is_preserved(self):
        result = self.parse(2, message="Find chicken rice within four minutes.", fields={
            "max_travel_time_min": 4})
        request = self.handoff(result, "Find chicken rice within four minutes.")
        self.assertEqual(request["max_travel_time_min"], 4)
        self.assertEqual(request["field_sources"]["max_travel_time_min"], "current_message")
        self.assertEqual(request["uncertain_fields"], [])

    def test_combined_absolute_limit_and_relative_allowance_preserved(self):
        for limit in ("within 30 minutes", "under 30 minutes", "at most 30 minutes", "no more than 30 minutes"):
            with self.subTest(limit=limit):
                message = "Find chicken rice " + limit + "; I accept four extra minutes for less walking."
                result = self.parse(2, message=message, fields={"max_travel_time_min": 30})
                request = self.handoff(result, message)
                self.assertEqual(request["max_travel_time_min"], 30)
                self.assertIsNone(request["clarification_question"])
                self.assertEqual(request["field_sources"]["max_travel_time_min"], "current_message")

    def test_combined_limit_mismatch_requires_clarification_without_erasing_limit(self):
        result = self.parse(2, message="Find chicken rice within 30 minutes; accept four extra minutes for less walking.")
        request = result["parsed_request"]
        self.assertEqual(request["max_travel_time_min"], 4)
        self.assertIn("max_travel_time_min", request["uncertain_fields"])
        self.assertEqual(orchestrator.route_after_parse(result), "clarification_response")

    def test_ambiguous_other_timing_evidence_clarifies(self):
        result = self.parse(2, message="Find chicken rice; four extra minutes is fine, but I have an hour.")
        self.assertEqual(orchestrator.route_after_parse(result), "clarification_response")
        self.assertIsNotNone(result["parsed_request"]["clarification_question"])

    def test_limit_with_omitted_units_is_not_erased(self):
        result = self.parse(2, message="Find chicken rice; at most 30 total, with four extra minutes for less walking.",
            fields={"max_travel_time_min": 30})
        self.assertEqual(result["parsed_request"]["max_travel_time_min"], 30)
        self.assertEqual(orchestrator.route_after_parse(result), "clarification_response")

    def test_queue_limit_cannot_prove_absolute_journey_limit(self):
        result = self.parse(2, message="Find chicken rice; queue within four minutes; accept four extra minutes for less walking.")
        self.assertEqual(orchestrator.route_after_parse(result), "clarification_response")

    def test_explicit_driving_is_preserved(self):
        message = "Find chicken rice by car; I don't mind a longer trip to avoid walking."
        result = self.parse(3, message=message)
        request = self.handoff(result, message)
        self.assertEqual(request["travel_mode"], "drive")
        self.assertEqual(request["field_sources"]["travel_mode"], "current_message")
        self.assertEqual(request["uncertain_fields"], [])

    def test_other_explicit_transport_choice_is_unchanged(self):
        for mode, wording in (("pt", "public transport"), ("walk", "on foot"), ("cycle", "by bicycle")):
            with self.subTest(mode=mode):
                result = self.parse(3, message="Find chicken rice " + wording + "; avoid long walks.",
                    fields={"travel_mode": mode})
                self.assertEqual(result["parsed_request"]["travel_mode"], mode)

    def test_user_history_supports_existing_mode_and_absolute_limit(self):
        context = copy.deepcopy(self.context)
        context["turns"].append({"role": "user", "content": "Use my car; keep the journey within 30 minutes."})
        result = self.parse(2, fields={"travel_mode": "drive", "max_travel_time_min": 30,
            "field_sources": {"travel_mode": "conversation_context", "max_travel_time_min": "conversation_context"}},
            message="Find chicken rice; four extra minutes for less walking is fine.", context=context)
        request = self.handoff(result, "Find chicken rice; four extra minutes for less walking is fine.")
        self.assertEqual(request["travel_mode"], "drive")
        self.assertEqual(request["max_travel_time_min"], 30)
        self.assertIsNone(request["clarification_question"])

    def test_assistant_prose_does_not_establish_mode_or_limit(self):
        context = copy.deepcopy(self.context)
        context["turns"].append({"role": "assistant", "content": "Use a car within four minutes."})
        result = self.parse(3, context=context)
        self.assertIsNone(result["parsed_request"]["travel_mode"])
        result = self.parse(2, context=context)
        self.assertIsNone(result["parsed_request"]["max_travel_time_min"])

    def test_both_objectives_and_declared_order_survive(self):
        for priorities in (["fastest", "least walking"], ["least walking", "fastest"]):
            result = self.parse(2, fields={"max_travel_time_min": None, "priorities": priorities})
            self.assertEqual(result["parsed_request"]["priorities"], priorities)
            self.assertIn("relative_travel_time_tolerance", result["parsed_request"]["uncertain_fields"])

    def test_existing_valid_fields_provenance_and_uncertainty_unchanged(self):
        fields = {"priorities": ["fastest"], "budget_amount": 8, "dietary_preferences": ["vegetarian"],
            "travel_mode": "pt", "max_travel_time_min": 30, "uncertain_fields": ["cuisine_preferences"],
            "field_sources": {"travel_mode": "current_message", "max_travel_time_min": "current_message",
                "budget_amount": "current_message", "dietary_preferences": "current_message"}}
        result = self.parse(0, message="Find vegetarian chicken rice by public transport within 30 minutes under $8.",
            fields=fields)
        for key, value in fields.items():
            self.assertEqual(result["parsed_request"][key], value)

    def test_shortest_queue_remains_queue_request(self):
        result = self.parse(0, message="Find chicken rice with the shortest queue, no more than five minutes waiting.",
            fields={"max_queue_min": 5})
        self.assertIn("queue", [t.agent for t in result["tasks"]])
        self.assertEqual(result["parsed_request"]["max_queue_min"], 5)

    def test_prompt_distinguishes_relative_limits_modes_and_existing_fallback(self):
        self.parse(2)
        for guidance in ("shortest journey duration", "minimizing walking distance",
            "relative to another option", "never authorizes driving", "lexicographically",
            "relative_travel_time_tolerance", "original current message is passed unchanged"):
            self.assertIn(guidance, self.prompts[0])

    def test_relative_uncertainty_is_not_duplicated(self):
        result = self.parse(2, fields={"uncertain_fields": ["relative_travel_time_tolerance"]})
        self.assertEqual(result["parsed_request"]["uncertain_fields"], ["relative_travel_time_tolerance"])


if __name__ == "__main__":
    unittest.main()

