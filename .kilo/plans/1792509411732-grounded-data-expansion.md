# Grounded data expansion for the Smart Hawker chatbot

## Goal

Give the chatbot more **real, source-backed** facts to return (stall names, prices, halal
status, healthier-dining criteria, centre details, real weather, real crowd context) without
letting it invent facts. The chosen approach is **data expansion + prompt rules only**: no new
grounding engine, no deterministic fact renderer, no post-generation audit.

## Decisions locked with the user

| Decision | Choice |
| --- | --- |
| Data sources | Government / open data only: data.gov.sg, NEA, HPB, MUIS, MOM, LTA DataMall, plus the DBS PayLah directory already in `backend/data/`. **Opt-in exception:** `hawkercentral` (limyifan1, MIT-licensed, crowdsourced) is allowed behind `ENABLE_CROWDSOURCED_SOURCE=1` because it is the only public source with structured stall + menu + price + cuisine data. No Google Places, no other community blog scraping. |
| Anti-hallucination mechanism | Data + prompt rules only. Every new fact must arrive as a source-owned field the model can copy verbatim; the prompt forbids asserting absent fields. |
| Weather | Replace the hardcoded mock with the real NEA/MSS forecast. |
| Queue | No official per-stall queue feed exists. Keep the synthetic feed but relabel it as a simulated estimate; optionally add LTA real-time **station crowd density** as clearly-labelled area busyness (not a stall wait). |

## Current state (measured, for grounding the plan)

- `backend/data/hawker_centres.json`: 123 NEA centres, 6,625 cooked-food stalls, 3 with aliases.
  The repo keeps only a **subset** of the official attributes (drops `DESCRIPTION`, `PHOTOURL`,
  `INFO_ON_CO_LOCATORS`, `EST_ORIGINAL_COMPLETION_DATE`, `HUP_COMPLETION_DATE`, postal/street/building
  fields, `ADDRESS_MYENV`, award/implementation dates).
- `backend/data/hawker_stall.json` (Wak-Wak): 1,248 stalls, 5,648 menu items, 1,128 linked to a centre,
  **120 dropped** because the postcode did not match a centre exactly (`hawker_service.py:56`).
  Price status: 4,801 fixed / 622 options / 2 variable / 222 unavailable / 1 unparsed.
- `backend/data/hawker_stall_mock.json`: 318 synthetic stalls across ~106 centres — i.e. **only ~17 of
  123 centres have any real menu data**; the rest is demo data.
- `backend/data/hawker_stall_directory.json` (DBS PayLah): 3,199 real merchant records across 112 centre
  headings with unit numbers — **present in the repo but never read by any code**.
- Healthier Dining: `scripts/enrich_dietary_tags.py` targets the stale HPB GeoJSON
  (`d_2925c2ccf75d1c135c2d469e0de3cee6`) and matched **0 stalls**; the current CSV dataset
  (`d_c475b9170c08f90580f4041bffd51639`, 3.8k rows, updated 2026-10-02) is unused.
- MUIS halal: the prompt demands MUIS-verified halal evidence (`recommendation_agent.py:142`) but no MUIS
  data is ever ingested, so every halal claim today is name-heuristic inference.
- Weather: `services/weather_service.py:20` hardcodes `"condition": "rainy"` — the bot can assert rain on
  a dry day. Queue: `services/live_queue_feed.py` is a hash-derived simulation.
- The anti-hallucination backbone that already works: `agents/verifier.py:180-248` re-derives every
  candidate from the source catalogue and strips worker-added fields that are not source-owned.

## Work items

Each item: source → script/field changes → prompt rule → verification. All scripts are idempotent
offline snapshots under `backend/scripts/`, following the existing `enrich_dietary_tags.py` pattern
(poll-download → normalise → merge → write `data/*.json`).

### W1 — NEA centre attributes (all official fields) — high value, low effort
- Source: `d_4a086da0a5553be1d89383cd90d07ecd` (Hawker Centres GEOJSON), updated 2026-10-20.
- New script `scripts/refresh_hawker_centres.py`: keep every current field **plus**
  `official_description` (e.g. "HUP Rebuilding"), `heritage_year` (`EST_ORIGINAL_COMPLETION_DATE`),
  `hup_completion_date`, `photo_url`, `co_locators` (split `INFO_ON_CO_LOCATORS`),
  `postal_code`, `building_name`, `street_name`, `block_number`, `myenv_address`,
  `awarded_date`, `implementation_date`, and a file-level `source_as_of` (`FMEL_UPD_D`).
- Preserve the existing pilot `id`/`name`/`aliases` values (README documents them as stable).
- Prompt rule: cite centre facts only from these fields, and always give `source_as_of`.

### W2 — HPB Healthier Dining (switch to the current CSV dataset) — fixes a 0-result pipeline
- Source: `d_c475b9170c08f90580f4041bffd51639` (`Healthier Dining Partners - F&B Partners`, CSV):
  `Company_Name, Brand_Name, FnB_Setting, Dining_Concept, Category, Name_of_Outlet, Outlet_Address,
  Regions, Menu_Item_Endorsement_Criteria`.
- Rewrite `scripts/enrich_dietary_tags.py` to read the CSV (`poll-download` → zip/CSV parse) and match to
  catalogue stalls by unit number + postcode, falling back to normalised outlet name + centre name.
  Keep the old GeoJSON path as a fallback dataset.
- New source-owned stall fields: `healthier_dining` (bool|None), `healthier_criteria` (verbatim string),
  `healthier_source`, `healthier_as_of`.
- Script must **fail loudly** (non-zero exit) when the tag count is 0, instead of silently writing nothing
  (the current `enrich_dietary_tags.py:269` behaviour that produced 0 tags).

### W3 — MUIS halal certification snapshot — enables the halal claim the prompt already requires
- Source: MUIS public halal e-Service directory (`https://halal.muis.gov.sg/halal/establishments`).
  Not a data.gov.sg dataset and not covered by the Open Data Licence, so treat it as a **periodic,
  polite, rate-limited snapshot** with an explicit provenance note in `backend/data/README.md`.
- New script `scripts/snapshot_muis_halal.py`: fetch the public establishment search (session + CSRF as
  the public e-Service requires), store `backend/data/muis_halal.json` with
  `{name, address, unit_number, postal_code, certificate_no, scheme, retrieved_at}`.
  Config-gated via `ENABLE_MUIS_HALAL=1`; on failure or missing file the pipeline degrades to
  "halal certification could not be verified" and never guesses.
- Match to catalogue stalls by unit + postcode, then name (reuse the HPB matcher's logic).
- New source-owned stall fields: `halal_certified` (bool|None), `halal_certificate_no`,
  `halal_source`, `halal_as_of`.

### W4 — Link the DBS PayLah directory (3,199 real stalls, already in the repo)
- Match `hawker_stall_directory.json` headings to NEA centres by normalised name (+ postcode from the
  centre address). Store `backend/data/hawker_stall_directory_linked.json`:
  `{centre_id, stall_name, unit_number, source_page}` per record, plus a `match_confidence` and
  `match_method` per centre.
- New source-owned fields on catalogue stalls when a directory record joins an existing stall:
  `merchant_list_name`, `merchant_unit_number`, `merchant_source`.
- New source-owned centre-level list: `merchant_stalls: [{name, unit_number}]` — lets the bot answer
  "which stalls are in this centre" for centres that have no menu data.
- Hard rule: the directory carries **no** menu, price, dietary, or hours fields; those must never be
  inferred from a merchant name (README already states this).

### W5 — Recover the 120 unlinked real stalls
- `hawker_service.py:56` discards any stall whose postcode does not match exactly one centre.
- Add a matcher: unique normalised centre-name match in `stall.hawker_centre` → centre; else postcode
  + unit + street similarity. Accept only a **unique** match; otherwise keep the stall excluded but
  count it in the coverage report so the gap is visible rather than silent.
- New field `centre_match_method` (`exact_postcode` | `name_unique` | `unit_postcode`) so the model can
  describe provenance.

### W10 — Crowdsourced hawkercentral catalogue (opt-in) — fills the stalls/menu/price gap
- Source: `hawkercentral` (limyifan1), MIT licensed, data via the public, unauthenticated Firestore
  REST endpoint `https://firestore.googleapis.com/v1/projects/hawkercentral/databases/(default)/documents/hawkers`.
  Not a government source, so it is **opt-in only** (`ENABLE_CROWDSOURCED_SOURCE=1`); unset by default.
- New script `scripts/snapshot_hawkercentral.py`: paginate the `hawkers` collection, store
  `backend/data/hawkercentral_stalls.json` with per-stall `{name, postal, unit, street, location{lat,lng},
  cuisine, categories, tags, opening, contact, source_doc_id, source_last_modified}` and each
  `menu_combined` entry as `{name, price, description, available, variants, section, image}`.
- Filter: keep only records whose `postal` matches a NEA centre postcode exactly (same rule as W5),
  and drop records that are clearly not hawker stalls (delivery-only restaurants / food-court outlets).
  Records that pass the filter are tagged `data_origin: "crowdsourced_unverified"`.
- Match to catalogue stalls by unit + postcode, then normalised name (reuse the HPB matcher's logic).
  New source-owned stall fields: `crowdsourced_menu`, `crowdsourced_price`, `crowdsourced_cuisine`,
  `crowdsourced_opening`, `crowdsourced_source`, `crowdsourced_as_of`.
- Prompt rule: any menu/price/cuisine fact from this source must be introduced with "according to a
  crowdsourced listing (unverified)". Never state it as an observed or official fact. The existing
  `is_mock`/`data_origin` labelling already distinguishes it from `wak_wak_source`.
- Hard constraint: **no new dependencies** — `requests` only. Do not scrape the React app or any
  non-public endpoint; the Firestore REST endpoint above is public and unauthenticated.
- Coverage expectation: report the number of stalls pulled, the number that passed the NEA-postcode
  filter, and the number with at least one menu item + price. A 0-result run must exit non-zero.

### W11 — NEA hawker-centre list refresh (PDF) — authoritative centre list, no new stall data
- Source: `https://www.nea.gov.sg/docs/default-source/hawker-centres-documents/list-of-hcs_-23-july-2026.pdf`
  (and the same file at the 25-July-2025 path), under the Singapore Open Data Licence.
- The PDF is **centre-level only**: `S/No | Name of Hawker Centre | Address | Common Property Managed
  & Maintained by`. It carries no stalls, units, menus, prices, or dietary fields, so it adds nothing
  to the dietary/stalls/prices goal on its own.
- Use it as the authoritative refresh for W1: extract name + address + managing entity, reconcile
  against `hawker_centres.json` (add new centres, retire renamed/merged ones, update the managing
  entity), and record `source_as_of` from the filename date. Fold the change into
  `scripts/refresh_hawker_centres.py` rather than a separate script.
- Prompt rule: centre-level facts may cite this list, always with `source_as_of`.

### W6 — Real weather (replace the hardcoded mock)
- Source: NEA/MSS via data.gov.sg real-time API —
  `https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast` and
  `.../two-hr-forecast` (collection 1456; also available via `api.weather.gov.sg`). No key required;
  an optional `x_api_key` raises rate limits.
- Rewrite `services/weather_service.py`: map the centre's region to the NEA forecast region
  (West / East / Central / North / South), return
  `{available, condition, region, valid_from, valid_to, source, retrieved_at, mock: false}`;
  on any failure return `{available: False, mock: False, condition: None, source: "nea_mss", error: ...}`.
- `services/weather_service.py:20` must no longer return a default condition.
- `agents/weather_agent.py:16-28`: keep confidence 0.0 and the limitation text when `available` is False,
  so the verifier's "no weather evidence" path (`verifier.py:327`) still rejects it.
- Cache the response in-process for ~10 minutes to keep latency and rate limits sane.

### W7 — Queue: honest labelling + optional real area busyness
- **Fact to state plainly to the user: there is no official per-stall or per-centre queue feed.**
  Nothing in data.gov.sg, NEA or OneMap publishes hawker wait times. Any number the bot states as a
  live wait would be fabricated.
- Default behaviour: keep `live_queue_feed` but treat every value as a **simulated estimate**. Add
  `queue_value_kind: "simulated_estimate"` and `queue_data_source` to every queue candidate, and make
  the prompt state the estimate once per reply (the existing mock notices at
  `queue_agent.py:446-448` and `orchestrator.py:1404` are kept).
- Optional real signal (needs a free LTA DataMall account key, `LTA_DATAMALL_KEY`):
  - `Station Crowd Density Real-time` (`PCDRealTime`, 10-minute refresh) and
    `Station Crowd Density Forecast` (`PCDForecast`) for the nearest MRT/LRT station to the centre;
  - Bus Arrival v3 `Load` for the nearest bus stop.
  - Expose as `area_busyness: {station, line, crowd_level: l|m|h, observed_at, source: "lta_datamall"}`
    and describe it as **transit crowding context**, never as a stall queue.
- Hard constraint: queue values must never remove a candidate on their own (already true in
  `orchestrator.py:830-836`); keep the synthetic feed out of the search-expansion decision.

### W8 — Public holidays (MOM) — cheap context, real data
- Source: MOM public-holiday CSVs (2026 `d_149b61ad0a22f61c09dc80f2df5bbec8`,
  2027 `d_0ba69fc6d56717ff0bf8083c6af3bb84`).
- New snapshot `backend/data/public_holidays.json` (`date, name`) and a `services/holiday_service.py`.
- Use: today-is-a-public-holiday disclosure, and to warn that some hawker stalls operate reduced hours
  on public holidays. Phrase as "some hawker stalls close or run reduced hours on public holidays" —
  never as a per-stall claim.

### W9 — Coverage report (the anti-hallucination disclosure, from data already owned)
- For every centre compute and expose:
  `{nea_cooked_food_stalls, catalogue_stalls, menu_items, priced_items, halal_certified_stalls,
  healthier_dining_stalls, merchant_listed_stalls, sources: [...]}`.
- This lets the bot state a true fact when it has no menu match ("NEA records 39 cooked-food stalls at
  Commonwealth Crescent Market; I have menu details for 12 of them"), instead of hedging or inventing.
- Wire it into the location evidence payload and the recommendation compact payload.

## Data model: new source-owned fields

All new fields are set in `services/hawker_service.py:88-121` (catalogue record) or on the centre record
in `services/location_service.py:60-63`, so `agents/verifier.py:180-248` keeps them and cannot have them
overwritten by a worker. Do **not** add any of them to the `worker_fields` whitelist at
`verifier.py:203-208` unless a worker must compute them.

Stall-level: `halal_certified`, `halal_certificate_no`, `halal_source`, `halal_as_of`,
`healthier_dining`, `healthier_criteria`, `healthier_source`, `healthier_as_of`,
`merchant_list_name`, `merchant_unit_number`, `merchant_source`, `centre_match_method`, `data_as_of`,
`crowdsourced_menu`, `crowdsourced_price`, `crowdsourced_cuisine`, `crowdsourced_opening`,
`crowdsourced_source`, `crowdsourced_as_of`.

Centre-level: `official_description`, `heritage_year`, `hup_completion_date`, `photo_url`,
`co_locators`, `postal_code`, `building_name`, `street_name`, `block_number`, `myenv_address`,
`awarded_date`, `implementation_date`, `source_as_of`, `merchant_stalls`, `coverage`.

Weather/queue: `available`, `condition`, `valid_from`, `valid_to`, `retrieved_at`, `source`,
`queue_value_kind`, `area_busyness`.

## Prompt rule changes (the anti-hallucination half)

Append to the system prompt in `agents/recommendation_agent.py:68-164` (keep the existing rules; do not
remove the mock-label, price and no-match rules):

1. **Fact whitelist.** You may state only facts present in the evidence payload. Never supply a missing
   field from general knowledge. If a field is absent, say the source does not list it.
2. **Halal.** Claim halal only when `halal_certified` is true; include `halal_certificate_no` and
   `halal_as_of`. Otherwise say certification could not be verified from the MUIS directory. Never infer
   halal from a stall name, category or cuisine.
3. **Healthier dining.** Claim it only when `healthier_dining` is true and quote `healthier_criteria`
   verbatim; do not restate a criterion as a medical or nutritional claim.
4. **Weather.** Use only `condition` with its `valid_from`/`valid_to`. If `available` is false, say the
   weather could not be retrieved — never assume rain or sun.
5. **Queue.** Present queue minutes only as a simulated estimate (state this once per reply). Never
   describe an estimate as a live or observed wait, and never drop a candidate because of a queue value.
6. **Prices.** Use `price_label` verbatim. For `unavailable` or `variable` prices, say no fixed amount is
   listed. Never total a meal or convert a variable price into a number.
7. **Centre details.** Centre facts must come from the centre record; give `source_as_of` when you state
   a count (e.g. cooked-food stalls) and use `coverage` to state what the catalogue actually holds.
8. **Coverage honesty.** When a requested dish or centre has no catalogue match, report the coverage gap
   and the nearest verified alternative, using the existing `no_match_in_routed_pool` /
   `farther_match_requires_approval` / `dietary_verification_unavailable` wording rules.
9. **Provenance line.** End with one `Sources:` line listing only the sources present in the evidence
   payload, each with its `as_of` / `retrieved_at` date. If a source is absent, do not name it.
10. **Crowdsourced data.** Any menu, price, cuisine, or opening fact from `hawkercentral` must be
    introduced as a crowdsourced, unverified listing ("according to a community-submitted listing"). Never
    present it as observed, official, or NEA-verified. Only state it when `crowdsourced_source` is present
    in the payload; otherwise say the source does not list it.

Supporting changes:
- `agents/recommendation_agent.py:21-66`: include the new source-owned fields in the compact payload
  (the `candidate_join` branch already forwards full candidate dicts — confirm the new keys survive the
  JSON round trip) and add a `sources` block plus the centre `coverage` block.
- `agents/dietary_agent.py:227-230`: update the limitation text — dietary suitability is now verifiable
  for halal (MUIS) and healthier dining (HPB); it is still unverifiable for vegetarian/vegan/allergen
  (no ingredient data), so keep the short-circuit in `hawker_service.py:232-234` for those.
- `agents/weather_agent.py`: the limitation string must state the source and retrieval time, not
  "returns mock data".

## Provenance surface

- Add `backend/services/sources_registry.py`: for each source, `{name, kind, url, dataset_id, licence,
  as_of, record_count}`.
- `backend/main.py`: add `GET /api/sources` returning the registry, and include the registry in the
  `/api/chat` response (`structured.sources`). Note `backend/main.py` already has uncommitted
  rate-limiting work in the working tree — merge, do not overwrite.
- `frontend/templates/index.html`: a small "Sources & freshness" strip under the reply, rendered from
  `structured.sources` only (no new text invented client-side).

## Order of implementation

1. **Phase 1 — data plumbing, no LLM change.** W1 (incl. W11 centre-list refresh), W4, W5, W9 + sources
   registry. Regenerate `backend/data/*.json`; update `backend/data/README.md` with each dataset's id,
   URL, licence and `as_of`.
2. **Phase 2 — dietary truth.** W2 (HPB CSV) then W3 (MUIS snapshot). Re-run the enrichment scripts and
   record the matched counts.
3. **Phase 3 — signals.** W6 (real weather), W7 (queue labelling + optional LTA busyness), W8 (holidays).
4. **Phase 4 — optional coverage.** W10 (`hawkercentral`, opt-in) — run only when
   `ENABLE_CROWDSOURCED_SOURCE=1`; if disabled, the pipeline degrades cleanly and the crowdsourced
   fields are absent.
5. **Phase 5 — prompt rules + payload.** Apply the ten rules, extend the compact payload, add the
   sources line, then re-run the full test suite.
6. **Phase 6 — surface.** `/api/sources` + frontend strip.

## Validation

Commands (from `backend/`): `python -m pytest` for the suite; `python scripts/<name>.py` for each
ingest script; `python -m flask` (or the existing compose stack) for the app.

New tests to add under `backend/tests/`:
- `test_data_sources.py` — every ingest script's output parses; every catalogue record carries
  `data_as_of`; `halal_certified`/`healthier_dining`/`crowdsourced_source` are `None` or bool (never free
  text); with `ENABLE_CROWDSOURCED_SOURCE` unset, no catalogue record carries a `crowdsourced_*` field.
- `test_verifier_preserves_source_fields.py` — after `verify_results`, the new source-owned fields survive
  and a worker cannot inject a `halal_certified` or `crowdsourced_source` value that is not in the source
  record.
- `test_weather_service.py` — with the network unavailable, `get_weather()` returns `available: False`
  and **no** condition; with a stubbed NEA payload it returns the condition and the validity window.
- `test_coverage_report.py` — coverage counts equal the catalogue contents, mock stalls are counted
  separately from real ones, and crowdsourced stalls are counted in a third bucket.
- `test_prompt_contract.py` — the compact payload passed to the recommendation chain contains the new
  fields and the `sources` block (mock the model, assert on the rendered prompt).
- `test_crowdsourced_filter.py` — `snapshot_hawkercentral.py` drops records whose postal does not match a
  NEA centre and drops non-hawker records; a fixture with one valid + one invalid record yields the
  expected counts.

Acceptance targets (report these numbers in the PR description):
- 100% of the 123 centres carry the full NEA attribute set plus `source_as_of` (refreshed from W11).
- HPB: tagged stalls > 0 (report matched / total with menus); script exits non-zero on 0.
- MUIS: report matched catalogue stalls and the snapshot date; degrade cleanly when disabled.
- DBS directory: ≥ 90% of the 112 centre headings matched to a `centre_id`; ≥ 90% of 3,199 records linked.
- Wak-Wak: number of the 120 previously-dropped stalls recovered (report matched / unmatched).
- Weather: no code path can return a condition without an NEA response (test-enforced).
- Crowdsourced (only when enabled): report stalls pulled / passed the NEA-postcode filter / with ≥1
  menu item + price; 0-result run exits non-zero.
- Every chat reply carries a `Sources:` line naming only sources present in the evidence payload.
- Existing suite (`backend/tests/`) stays green; `test_security_evaluations.py` and
  `test_queue_guardrails.py` must still pass unchanged.

## Risks and rollback

- **Crowdsourced quality / acceptable use** → keep `hawkercentral` behind `ENABLE_CROWDSOURCED_SOURCE=1`,
  snapshot with a visible `as_of`, never scrape the React app or non-public endpoints, and make every
  `crowdsourced_*` field optional. Rollback = unset the flag; the prompt then reports that the source
  is absent. Every crowdsourced fact is prefixed "unverified community listing".
- **Crowdsourced staleness** — records were last modified 2020; surface `crowdsourced_as_of` and treat
  any listing older than 90 days as a limitation in the evidence payload.
- **MUIS scraping** may break or be considered outside acceptable use → keep it behind
  `ENABLE_MUIS_HALAL`, snapshot with a visible `as_of`, and make every MUIS field optional. Rollback =
  unset the flag; the prompt then reports "certification could not be verified".
- **Dataset schema drift** (HPB CSV columns, NEA attributes, Firestore field names) → each script
  validates required columns/fields and exits non-zero with a clear message instead of writing a
  half-populated file.
- **Staleness** → every dataset stores `as_of`; the reply shows it; a snapshot older than 90 days is
  surfaced as a limitation in the evidence payload.
- **Prompt bloat** — the recommendation prompt is already long; keep the new rules to the ten bullets
  and drop nothing, but move the per-field detail into the payload (field names) rather than prose.
- **Working tree conflict** — `backend/main.py`, `backend/services/rate_limit.py`,
  `backend/tests/conftest.py`, `backend/tests/test_rate_limit.py` are uncommitted rate-limiting work.
  Touch `backend/main.py` minimally (one new route + one response key) and rebase on that work.
- **No new dependencies** — everything above uses `requests` (already in `backend/requirements.txt`).

## Out of scope

- Any embedding/vector store or RAG retriever: the current deterministic filters are the grounded path;
  the gap is coverage, not retrieval.
- Google Places, community food blogs, Hawkerpedia, or any non-government source — **except**
  `hawkercentral`, which is opt-in only (`ENABLE_CROWDSOURCED_SOURCE=1`) and always labelled unverified.
- A deterministic fact renderer or a post-generation grounding audit (explicitly declined by the user).
- Per-stall live queue data (no such source exists).
- NEA tender notices as a stall/dietary/price source: they list stalls *available for rental* (vacant
  units), so their `HALAL COOKED FOOD` trade flag cannot certify an occupied, recommended stall. Their
  only legitimate use is centre-level vacancy context, which is not worth a work item.
- Changing the LangGraph topology, the planner prompt, or the session/trace stores.
