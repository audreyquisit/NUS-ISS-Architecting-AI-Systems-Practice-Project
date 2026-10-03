import unittest
from unittest.mock import patch

from agents.location_agent import (
    LocationIntent,
    LocationResolutionError,
    _resolve_origin,
)
from models.location_models import LocationQuery
from services.location_service import is_singapore_location


def _intent():
    return LocationIntent(
        request_intent="find_nearby_centres",
        origin_place=None,
        proximity_preference="convenient",
        interpretation="Find nearby centres.",
    )


class LocationScopeTests(unittest.TestCase):
    @patch("agents.location_agent.is_singapore_location", return_value=True)
    def test_arbitrary_singapore_coordinates_are_accepted(self, _is_singapore):
        query = LocationQuery(
            request="Recommend food near my current location",
            user_location={"lat": 1.3521, "lng": 103.8198},
            user_location_source="browser_current_location",
        )

        origin, source = _resolve_origin(query, _intent())

        self.assertEqual(source, "browser_current_location")
        self.assertEqual(origin["label"], "your current location")
        self.assertEqual(origin["lat"], 1.3521)

    @patch("agents.location_agent.is_singapore_location", return_value=False)
    def test_coordinates_outside_singapore_are_rejected(self, _is_singapore):
        query = LocationQuery(
            request="Recommend food near my current location",
            user_location={"lat": 3.139, "lng": 101.6869},
            user_location_source="browser_current_location",
        )

        with self.assertRaisesRegex(LocationResolutionError, "outside Singapore"):
            _resolve_origin(query, _intent())

    @patch("services.location_service._get")
    def test_reverse_geocode_evidence_controls_singapore_validation(self, get):
        point = {"lat": 1.3521, "lng": 103.8198}
        get.return_value = {"GeocodeInfo": [{"POSTALCODE": "123456"}]}
        self.assertTrue(is_singapore_location(point))
        self.assertEqual(
            get.call_args.args[0],
            "https://www.onemap.gov.sg/api/public/revgeocode",
        )
        self.assertEqual(get.call_args.kwargs["params"]["buffer"], 500)

        get.return_value = {"GeocodeInfo": []}
        self.assertFalse(is_singapore_location({"lat": 3.139, "lng": 101.6869}))


if __name__ == "__main__":
    unittest.main()