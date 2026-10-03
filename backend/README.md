# Smart Hawker AI Backend

A simple Flask backend scaffold for the Smart Hawker AI project.

## Run locally

1. Create and activate a Python environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate (macOS/Linux)
   .venv\Scripts\Activate.ps1 (Windows) 
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Update your OpenAI API Key


   Create file .env under backend directory as such
   ```bash
   OPENAI_API_KEY=[your-secret-key]
   ```
   *Note: do not push your .env file

   
4. Start the backend
   ```bash
   python main.py
   ```
5. Interacting with the agents - if you'd like to test out the agents alone.
   ```bash
   python terminal.py
   ```
## Endpoints

- `GET /health` - health check
- `POST /api/chat` - parse a chat request and run the orchestrated agent workflow
- `POST /api/recommendation` - get a stall recommendation
- `POST /api/location` - run the Location & Logistics agent directly
- `GET /api/stalls?limit=50&offset=0` - list stalls with source-backed menu data
- `GET /api/menu?q=laksa&budget=6` - search menu items and apply an SGD item-price ceiling

### Orchestration flow

The chat workflow parses each message and recent turns into a typed request and
worker plan. Location runs first for food discovery, named-centre, and directions
requests. Relevant dietary, budget, queue, and weather workers then receive the
parsed request and Location's selected-centre scope. The verifier checks centre
IDs against the NEA catalogue and applies parsed dietary, budget, and queue limits
before the recommendation model sees the results. `/api/chat` includes the parsed
request, worker plan, and verified results for tracing. `/api/location` remains a
direct worker endpoint and performs its own intent interpretation when no parsed
request is supplied.

### Location & Logistics request

The location endpoint accepts a natural-language request and either a named place
in that request or browser coordinates. A named place takes precedence over the
coordinates. Public transport is the default; `travel_mode` can be `walk`, `drive`,
or `cycle` when the user specifies another mode. If no mode was specified, the
agent checks a real walking route too and can select walking when the route is eight
minutes or less and no slower than public transport. Configure the threshold with
`LOCATION_SHORT_WALK_THRESHOLD_MIN`. Centre coordinates are used internally to
route only to a broad nearby candidate pool (30 centres by default), reducing
OneMap requests. An explicit regional request such as “food options in the East”
selects candidates around that region even when the user's origin is elsewhere.
This coordinate filter is not shown as distance or travel time; displayed route
metrics come from OneMap. Set `LOCATION_ROUTE_CANDIDATE_LIMIT` to change the pool
size.

```json
{
  "request": "Find food near Yio Chu Kang; fastest public transport route",
  "user_location": {"lat": 1.35, "lng": 103.82},
  "user_location_source": "browser_current_location"
}
```

The endpoint uses OneMap to resolve places, load NEA centre coordinates, and
request route evidence. In the orchestrated flow, the planner supplies the
structured intent and Location ranks only verified route candidates. Direct
`/api/location` calls use the Location agent's own intent prompt. Set `OPENAI_API_KEY`,
`ONEMAP_EMAIL`, and `ONEMAP_EMAIL_PASSWORD` in `backend/.env`. Never commit that
file.

Stall and menu candidates now come from `data/hawker_stall.json`. Stable stall
and menu-item IDs are attached, and stalls are eligible only when their source
postcode uniquely matches a centre in the NEA catalogue. The Dietary worker
matches requested dish, cuisine, stall, and centre; the Budget worker filters
individual menu items by their listed SGD starting price. The verifier joins
workers on menu-item ID before the recommendation model sees the shortlist.
Multiple listed prices are presented as portion tiers when the source uses an
unlabelled two- or three-price portion format, or as separate options when the source
indicates variants. Ingredient-choice items with explicit source notes are
described as variable-price rather than assigned a guessed amount. The source
has no structured ingredients, allergens, or dietary certification, so dietary
suitability is left unknown and restrictive diet requests cannot be verified.
A budget applies to one listed item, not an entire meal. Source listing
freshness and many serving bases are unknown. Queue and weather data remain
mocked.

For food discovery, the orchestrator first checks the nearest route-ranked
centres against the menu catalogue. If those have no matching items, it searches
the other OneMap-routed candidates. The default nearby search horizon is 30
minutes (`LOCATION_DEFAULT_SEARCH_HORIZON_MIN`); an explicit travel-time limit
takes precedence. If a matching dish is found only beyond the horizon, the app
asks before recommending that longer trip. Requests that explicitly allow
farther travel can use the full route-checked candidate pool.

For a small local demonstration, run `python main.py`, then query for example:

```bash
curl 'http://localhost:8000/api/menu?q=laksa&budget=6&limit=5'
curl 'http://localhost:8000/api/stalls?limit=2'
```

## Notes

- The backend uses Flask for API routing and Pydantic for payload validation.
- Replace the stubbed logic in `services.py` with real hawker recommendation and agent orchestration.
- The backend runs on port `8000` by default.
