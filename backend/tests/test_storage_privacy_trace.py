"""Real disposable persistence and Flask boundaries; no live agents/services."""

import json
import os
import sqlite3
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import main
import test_session_store as session_fixtures
from services import session_store, trace_store


class StoragePrivacyTraceTests(session_fixtures.SessionStoreTestCase):
    def setUp(self):
        super().setUp()
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        directory = os.path.join(self._temp_dir.name, "traces")
        self.patches.enter_context(patch.object(trace_store, "_TRACE_DIR", directory))
        self.patches.enter_context(patch.object(trace_store, "_DB_PATH", os.path.join(directory, "traces.db")))
        self.logger = self.patches.enter_context(patch("services.activity_log._LOGGER.info"))
        self.patches.enter_context(patch.dict("os.environ", {
            "SESSION_RETENTION_DAYS": "30", "TRACE_RETENTION_DAYS": "30"}))
        self.client = main.app.test_client()

    @staticmethod
    def response():
        return {"reply": "Fixture food options.", "structured": {"fixture": True},
                "parsed_request": {"intent": "food_discovery", "travel_mode": "pt"},
                "location_data": {}, "verified_results": [], "needs_location": False, "worker_plan": []}

    def persist_text(self, text):
        session_id = session_store.create_session()
        session_store.append_turn(session_id, "user", text)
        session_store.append_turn(session_id, "assistant", text)
        # Inspect actual SQLite rows, not just the sanitizer's return value.
        with sqlite3.connect(session_store._DB_PATH) as db:
            turns = [row[0] for row in db.execute("SELECT content FROM turns WHERE session_id=?", (session_id,))]
        request_id = trace_store.save_trace({"request_id": str(uuid.uuid4()), "user_query": text,
            "response": {"reply": text, "parsed_request": {"intent": "food_discovery"}}})
        loaded = trace_store.load_trace(request_id)
        with sqlite3.connect(trace_store._DB_PATH) as db:
            relative = db.execute("SELECT file_path FROM traces WHERE request_id=?", (request_id,)).fetchone()[0]
        disk_trace = json.loads((Path(trace_store._TRACE_DIR) / relative).read_text())
        self.assertEqual(loaded, disk_trace)
        return [*turns, disk_trace["input"]["user_query"], disk_trace["output"]["final_response"]]

    def test_labelled_coordinates_roundtrip_all_persisted_text(self):
        texts = [
            "latitude=1.234567 longitude=103.123456",
            "lat: 1.234567, lon: 103.123456",
            '\"latitude\": 1.234567, \"longitude\": 103.123456',
            "longitude=103.123456 latitude=1.234567",
            '\"longitude\": 103.123456, \"latitude\": 1.234567',
            "LAT = + 1.234567\n LNG : - 103.123456",
            "'latitude' : '-1.234567', 'longitude': '+103.123456'",
            "lat=.234567 lon=+103.123456",
            "latitude=1.234567e0 longitude=1.03123456e2",
        ]
        for text in texts:
            with self.subTest(text=text):
                for stored in self.persist_text(text):
                    for exact in ("1.234567", "103.123456", ".234567", "1.03123456"):
                        self.assertNotIn(exact, stored)
                    self.assertIn("[REDACTED_COORDINATE]", stored)

    def test_partial_and_malformed_coordinate_tokens_roundtrip(self):
        for text in ("latitude=1.234567", "lon=103.123456.78", '\"lat\": \"1.234567',
                     "lat=-1.234567 longitude=unavailable"):
            with self.subTest(text=text):
                for stored in self.persist_text(text):
                    self.assertNotIn("1.234567", stored)
                    self.assertNotIn("103.123456", stored)
                    self.assertIn("[REDACTED_COORDINATE]", stored)

    def test_prices_durations_and_menu_ids_are_preserved(self):
        ordinary = "Laksa $5.50; 30 minutes; MENU-12345; ratio 1.25; latency=12.34567"
        for stored in self.persist_text(ordinary + "; lat=1.234567 lon=103.123456"):
            self.assertIn(ordinary, stored)
            self.assertNotIn("1.234567", stored)
            self.assertNotIn("103.123456", stored)

    def test_existing_comma_coordinate_pair_is_still_redacted(self):
        for stored in self.persist_text("Fixture point 1.234567, 103.123456"):
            self.assertIn("[REDACTED_COORDINATES]", stored)
            self.assertNotIn("1.234567", stored)
            self.assertNotIn("103.123456", stored)

    def test_structured_coordinate_rounding_and_public_origin_unchanged(self):
        origin = {"lat": 1.234567, "lng": 103.123456}
        self.assertEqual(trace_store._safe_value(origin), {"lat": 1.235, "lng": 103.123})
        data = self.response()
        data["location_data"] = {"origin": origin}
        with patch.object(main, "run_orchestrator", return_value=data):
            response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["location_data"]["origin"], origin)
        self.assertNotIn("origin", session_store.get_context(response.get_json()["session_id"])["state"])

    def assert_success_and_session(self, response):
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for field in ("reply", "structured", "location_data", "parsed_request", "verified_results", "worker_plan"):
            self.assertEqual(body[field], self.response()[field])
        context = session_store.get_context(body["session_id"])
        self.assertEqual([turn["role"] for turn in context["turns"]], ["user", "assistant"])
        self.assertEqual(context["state"]["travel_mode"], "pt")
        events = [json.loads(call.args[0]) for call in self.logger.call_args_list]
        failed = [event for event in events if event["event"] == "trace.save_failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(set(failed[0]), {"time", "request_id", "event", "error_type"})
        logged = json.dumps(events)
        for secret in ("private-storage-message", "1.234567", "sk-private-fixture", "private_context"):
            self.assertNotIn(secret, logged)

    def test_expected_storage_exceptions_preserve_success_and_session(self):
        for error in (PermissionError("private-storage-message"), FileNotFoundError("private-storage-message"),
                      sqlite3.OperationalError("database is locked; private-storage-message"),
                      sqlite3.DatabaseError("private-storage-message")):
            with self.subTest(error=type(error).__name__):
                self.logger.reset_mock()
                with patch.object(main, "run_orchestrator", return_value=self.response()), \
                     patch.object(main, "save_trace", side_effect=error) as save:
                    response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
                save.assert_called_once()
                self.assert_success_and_session(response)

    def test_unavailable_trace_directory_real_storage(self):
        blocked = Path(self._temp_dir.name) / "file-instead-of-directory"
        blocked.write_text("fixture")
        with patch.object(trace_store, "_TRACE_DIR", str(blocked)), \
             patch.object(main, "run_orchestrator", return_value=self.response()):
            response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
        self.assert_success_and_session(response)

    def test_unavailable_trace_database_real_storage(self):
        Path(trace_store._TRACE_DIR).mkdir()
        with patch.object(trace_store, "_DB_PATH", trace_store._TRACE_DIR), \
             patch.object(main, "run_orchestrator", return_value=self.response()):
            response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
        self.assert_success_and_session(response)

    def test_locked_trace_database_real_storage(self):
        trace_store._init_db()
        connection = sqlite3.connect(trace_store._DB_PATH)
        original_connect = sqlite3.connect

        def short_timeout(*args, **kwargs):
            kwargs.setdefault("timeout", 0.02)
            return original_connect(*args, **kwargs)

        try:
            connection.execute("BEGIN IMMEDIATE")
            with patch.object(sqlite3, "connect", side_effect=short_timeout), \
                 patch.object(main, "run_orchestrator", return_value=self.response()):
                response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
            self.assert_success_and_session(response)
        finally:
            connection.rollback()
            connection.close()

    def test_trace_programming_errors_propagate_without_storage_diagnostic(self):
        for error in (RuntimeError("application bug"), ValueError("invalid application data"), TypeError("application bug"),
                      sqlite3.ProgrammingError("bad SQL bindings"), sqlite3.InterfaceError("invalid interface"),
                      sqlite3.IntegrityError("application constraint violation"), sqlite3.NotSupportedError("unsupported API")):
            with self.subTest(error=type(error).__name__):
                self.logger.reset_mock()
                with patch.object(main, "run_orchestrator", return_value=self.response()), \
                     patch.object(main, "save_trace", side_effect=error), self.assertRaises(type(error)):
                    main._execute_and_trace("Find chicken rice", {})
                events = [json.loads(call.args[0]) for call in self.logger.call_args_list]
                self.assertFalse(any(event["event"] == "trace.save_failed" for event in events))

    def test_orchestrator_error_is_not_misclassified_or_suppressed(self):
        with patch.object(main, "run_orchestrator", side_effect=ValueError("original application error")), \
             patch.object(main, "save_trace", side_effect=sqlite3.OperationalError("trace unavailable")), \
             self.assertRaisesRegex(ValueError, "original application error"):
            main._execute_and_trace("Find chicken rice", {})
        events = [json.loads(call.args[0]) for call in self.logger.call_args_list]
        self.assertTrue(any(event["event"] == "request.failed" and event["error_type"] == "ValueError" for event in events))
        self.assertFalse(any(event["event"] == "request.completed" for event in events))

    def test_intended_clarification_status_survives_trace_failure(self):
        data = self.response()
        data["needs_location"] = True
        data["location_data"] = {"location_required": True, "error": "Starting point required"}
        with patch.object(main, "run_orchestrator", return_value=data), \
             patch.object(main, "save_trace", side_effect=OSError("private-storage-message")):
            response = self.client.post("/api/chat", json={"message": "Find chicken rice"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["code"], "location_required")
        self.assertIsNotNone(response.get_json()["session_id"])


if __name__ == "__main__":
    unittest.main()
