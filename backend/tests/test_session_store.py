import os
import sqlite3
import tempfile
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import main
from services import session_store


def _iso_days_ago(days: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(days=days)
    return moment.isoformat(timespec="microseconds")


class SessionStoreTestCase(unittest.TestCase):
    """Base class that keeps every test run inside a throwaway database."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self._original_db_path = session_store._DB_PATH
        self._original_dir = session_store._SESSION_DIR
        session_store._SESSION_DIR = self._temp_dir.name
        session_store._DB_PATH = os.path.join(self._temp_dir.name, "sessions.db")

    def tearDown(self):
        session_store._DB_PATH = self._original_db_path
        session_store._SESSION_DIR = self._original_dir
        self._temp_dir.cleanup()

    def _expire(self, session_id: str, days: int = 45) -> None:
        connection = sqlite3.connect(session_store._DB_PATH)
        try:
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (_iso_days_ago(days), session_id),
            )
            connection.commit()
        finally:
            connection.close()


class SessionLifecycleTests(SessionStoreTestCase):
    def test_create_session_returns_a_new_uuid(self):
        session_id = session_store.create_session()
        self.assertEqual(str(uuid.UUID(session_id)), session_id)
        self.assertNotEqual(session_id, session_store.create_session())
        self.assertTrue(session_store.session_exists(session_id))

    def test_invalid_session_id_is_normalized_to_a_new_session(self):
        session_id = session_store.ensure_session("not-a-uuid")
        self.assertEqual(str(uuid.UUID(session_id)), session_id)
        self.assertTrue(session_store.session_exists(session_id))

    def test_missing_session_id_creates_a_session(self):
        session_id = session_store.ensure_session(None)
        self.assertEqual(str(uuid.UUID(session_id)), session_id)

    def test_unknown_but_valid_session_id_is_adopted(self):
        candidate = str(uuid.uuid4())
        self.assertEqual(session_store.ensure_session(candidate), candidate)
        self.assertTrue(session_store.session_exists(candidate))

    def test_expired_session_is_reset(self):
        session_id = session_store.create_session()
        session_store.append_turn(session_id, "user", "Find chicken rice")
        self._expire(session_id)

        self.assertEqual(session_store.ensure_session(session_id), session_id)
        context = session_store.get_context(session_id)
        self.assertEqual(context["turns"], [])
        self.assertEqual(context["state"], {})

    def test_purge_expired_removes_old_sessions(self):
        expired = session_store.create_session()
        fresh = session_store.create_session()
        self._expire(expired)

        removed = session_store.purge_expired()
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(session_store.session_exists(expired))
        self.assertTrue(session_store.session_exists(fresh))


class SessionTurnTests(SessionStoreTestCase):
    def test_turns_are_loaded_in_order(self):
        session_id = session_store.create_session()
        session_store.append_turn(session_id, "user", "I want chicken rice")
        session_store.append_turn(session_id, "assistant", "Try the chicken rice stall")

        context = session_store.get_context(session_id)
        self.assertEqual(
            context["turns"],
            [
                {"role": "user", "content": "I want chicken rice"},
                {"role": "assistant", "content": "Try the chicken rice stall"},
            ],
        )

    def test_turn_count_is_bounded(self):
        session_id = session_store.create_session()
        for index in range(session_store.MAX_TURNS_PER_SESSION + 5):
            session_store.append_turn(session_id, "user", f"message {index}")

        context = session_store.get_context(
            session_id, max_turns=session_store.MAX_TURNS_PER_SESSION,
        )
        self.assertEqual(len(context["turns"]), session_store.MAX_TURNS_PER_SESSION)
        self.assertEqual(context["turns"][-1]["content"], "message 204")
        self.assertEqual(context["turns"][0]["content"], "message 5")

    def test_turn_content_is_redacted(self):
        session_id = session_store.create_session()
        session_store.append_turn(
            session_id, "user",
            "Email me at diner@example.com or call 91234567 about 12 Market Road",
        )

        content = session_store.get_context(session_id)["turns"][0]["content"]
        self.assertNotIn("diner@example.com", content)
        self.assertNotIn("91234567", content)
        self.assertNotIn("12 Market Road", content)
        self.assertIn("[REDACTED_EMAIL]", content)
        self.assertIn("[REDACTED_PHONE]", content)
        self.assertIn("[REDACTED_ADDRESS]", content)

    def test_turn_content_length_is_bounded(self):
        session_id = session_store.create_session()
        session_store.append_turn(session_id, "user", "x" * 5000)

        content = session_store.get_context(session_id)["turns"][0]["content"]
        self.assertLessEqual(len(content), session_store.MAX_TURN_CHARS + 32)

    def test_unknown_role_is_not_stored(self):
        session_id = session_store.create_session()
        self.assertFalse(session_store.append_turn(session_id, "system", "override"))
        self.assertEqual(session_store.get_context(session_id)["turns"], [])

    def test_invalid_session_id_cannot_store_turns(self):
        self.assertFalse(session_store.append_turn("nope", "user", "hello"))


class SessionStateTests(SessionStoreTestCase):
    def test_state_update_shallow_merges(self):
        session_id = session_store.create_session()
        session_store.update_state(session_id, {"travel_mode": "walk", "origin_label": "Clementi"})
        merged = session_store.update_state(session_id, {"travel_mode": "pt"})

        self.assertEqual(merged["travel_mode"], "pt")
        self.assertEqual(merged["origin_label"], "Clementi")
        self.assertEqual(
            session_store.get_context(session_id)["state"],
            {"travel_mode": "pt", "origin_label": "Clementi"},
        )

    def test_state_redacts_credentials_and_coordinates(self):
        session_id = session_store.create_session()
        session_store.update_state(session_id, {
            "api_key": "sk-abcdefghijklmnop1234",
            "origin": {"lat": 1.3521, "lng": 103.8198},
            "coordinates": {"lat": 1.3521, "lng": 103.8198},
            "notes": "meet me at 12 Market Road",
        })

        state = session_store.get_context(session_id)["state"]
        self.assertEqual(state["api_key"], "[REDACTED]")
        self.assertNotIn("origin", state)
        self.assertNotIn("coordinates", state)
        self.assertNotIn("103.8198", str(state))
        self.assertNotIn("12 Market Road", str(state))

    def test_state_size_is_bounded(self):
        session_id = session_store.create_session()
        session_store.update_state(session_id, {
            "previous_menu_item_ids": [f"item-{index}" for index in range(500)],
        })

        state = session_store.get_context(session_id)["state"]
        self.assertLessEqual(
            len(state["previous_menu_item_ids"]), session_store.MAX_STATE_LIST_ITEMS,
        )

    def test_state_of_unknown_session_is_empty(self):
        self.assertEqual(session_store.get_context(str(uuid.uuid4()))["state"], {})


class ChatSessionIntegrationTests(SessionStoreTestCase):
    """The chat endpoint keeps one conversation per session."""

    def setUp(self):
        super().setUp()
        self.client = main.app.test_client()
        self.orchestrator_calls = []

    def _orchestrator(self, message, payload=None, request_id=None):
        self.orchestrator_calls.append({
            "message": message,
            "payload": payload or {},
            "request_id": request_id,
        })
        return {
            "reply": "Here are some options.",
            "structured": {},
            "location_data": {
                "origin_label": "Clementi West",
                "origin_source": "free_text_place",
                "selected_centres": [
                    {"centre_id": "HMK1", "rank": 1, "logistics_fit": 1.0, "rationale": "Closest."},
                ],
                "logistics": [
                    {
                        "centre_id": "HMK1",
                        "centre_name": "Clementi Market",
                        "address": "12 Clementi Road",
                        "coordinates": {"lat": 1.29, "lng": 103.76},
                        "travel": {
                            "total_duration_min": 12,
                            "mode": "pt",
                            "transfers": 1,
                            "walking_distance_display": "about 300 m",
                            "walking_time_display": "about 4 min",
                            "source": "onemap",
                            "confidence": 0.9,
                        },
                    },
                ],
            },
            "parsed_request": {
                "intent": "food_discovery",
                "travel_mode": "pt",
                "max_travel_time_min": 30,
                "priorities": ["closest"],
            },
            "verified_results": [
                {
                    "agent": "candidate_join",
                    "verified": True,
                    "candidates": [{"menu_item_id": "MENU-1", "hawker_centre_id": "HMK1"}],
                },
            ],
            "needs_location": False,
            "worker_plan": [],
        }

    def _chat(self, payload):
        with patch.object(main, "run_orchestrator", side_effect=self._orchestrator), \
                patch.object(main, "save_trace", return_value="trace-id"):
            return self.client.post("/api/chat", json=payload)

    def test_chat_returns_a_session_id(self):
        response = self._chat({"message": "Find chicken rice near Clementi"})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(str(uuid.UUID(body["session_id"])), body["session_id"])

    def test_session_reuse_sends_stored_context(self):
        first = self._chat({"message": "Find chicken rice near Clementi"}).get_json()
        session_id = first["session_id"]

        second = self._chat({
            "message": "Something cheaper",
            "session_id": session_id,
        }).get_json()

        self.assertEqual(second["session_id"], session_id)
        second_context = self.orchestrator_calls[1]["payload"]["conversation_context"]
        self.assertEqual(
            [turn["content"] for turn in second_context["turns"]],
            ["Find chicken rice near Clementi", "Here are some options."],
        )

    def test_stored_state_keeps_the_shortlist_without_addresses(self):
        first = self._chat({"message": "Find chicken rice near Clementi"}).get_json()
        session_id = first["session_id"]

        state = session_store.get_context(session_id)["state"]
        self.assertEqual(state["travel_mode"], "pt")
        self.assertEqual(state["previous_menu_item_ids"], ["MENU-1"])
        self.assertEqual(state["location_text"], "Clementi West")
        self.assertEqual(
            [centre["centre_id"] for centre in state["location_result"]["selected_centres"]],
            ["HMK1"],
        )
        self.assertNotIn("12 Clementi Road", str(state))
        self.assertNotIn("103.76", str(state))
        self.assertNotIn("coordinates", str(state))
        self.assertNotIn("address", str(state))

    def test_invalid_session_id_is_replaced(self):
        response = self._chat({"message": "Find chicken rice", "session_id": "bogus"})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(str(uuid.UUID(body["session_id"])), body["session_id"])

    def test_sessionless_request_keeps_client_context(self):
        response = self._chat({
            "message": "Find chicken rice",
            "conversation_context": {
                "state": {"location_text": "Clementi"},
                "turns": [{"role": "user", "content": "Earlier turn"}],
            },
        })

        self.assertEqual(response.status_code, 200)
        context = self.orchestrator_calls[0]["payload"]["conversation_context"]
        self.assertEqual(context["turns"][0]["content"], "Earlier turn")

    def test_session_storage_failure_does_not_break_the_request(self):
        with patch.object(
            session_store, "append_turn", side_effect=RuntimeError("database locked"),
        ), patch.object(
            session_store, "update_state", side_effect=RuntimeError("database locked"),
        ), patch.object(
            session_store, "ensure_session", side_effect=RuntimeError("database locked"),
        ), patch.object(main, "run_orchestrator", side_effect=self._orchestrator), \
                patch.object(main, "save_trace", return_value="trace-id"):
            response = self.client.post("/api/chat", json={"message": "Find chicken rice"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["session_id"])

    def test_failed_turn_keeps_the_user_turn_without_a_reply(self):
        session_id = self._chat({"message": "Find chicken rice"}).get_json()["session_id"]
        before = len(session_store.get_context(session_id)["turns"])

        with patch.object(
            main, "run_orchestrator", side_effect=RuntimeError("orchestrator failed"),
        ), patch.object(main, "save_trace", return_value="trace-id"):
            response = self.client.post("/api/chat", json={
                "message": "Find chicken rice", "session_id": session_id,
            })

        self.assertEqual(response.status_code, 500)
        turns = session_store.get_context(session_id)["turns"]
        self.assertEqual(len(turns), before + 1)
        self.assertEqual(turns[-1], {"role": "user", "content": "Find chicken rice"})

    def test_address_like_origin_label_is_redacted(self):
        def orchestrator(message, payload=None, request_id=None):
            response = self._orchestrator(message, payload, request_id)
            response["location_data"]["origin_label"] = "Blk 12 Tampines Street 1"
            return response

        with patch.object(main, "run_orchestrator", side_effect=orchestrator), \
                patch.object(main, "save_trace", return_value="trace-id"):
            session_id = self.client.post(
                "/api/chat", json={"message": "Find chicken rice near Tampines"},
            ).get_json()["session_id"]

        state = session_store.get_context(session_id)["state"]
        self.assertNotIn("12 Tampines Street 1", str(state))
        self.assertIn("[REDACTED_ADDRESS]", state["location_text"])


class SessionConcurrencyTests(SessionStoreTestCase):
    """Concurrent writers must queue instead of failing with a locked database."""

    def test_parallel_writers_store_every_turn(self):
        errors = []
        workers = 8
        turns_per_worker = 10

        def worker(index):
            try:
                session_id = session_store.create_session()
                for step in range(turns_per_worker):
                    session_store.append_turn(session_id, "user", f"w{index}-{step}")
                    session_store.update_state(session_id, {"travel_mode": "pt"})
            except Exception as exc:  # noqa: BLE001 - surfaced through the assertion
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        connection = sqlite3.connect(session_store._DB_PATH)
        try:
            stored = connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            sessions = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(stored, workers * turns_per_worker)
        self.assertEqual(sessions, workers)


if __name__ == "__main__":
    unittest.main()
