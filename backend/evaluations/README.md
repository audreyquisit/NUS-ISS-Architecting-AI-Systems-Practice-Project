# Controlled pilot: offline corrections and future setup

This support code makes no live calls and writes no evaluation reports. A future
paid run requires separate authorization. The first pilot's transport wrapper was
ephemeral; it is not repository production code.

## Response observation

The old wrapper did `data = response.read()` and then constructed a NEW
`httpx.Response(..., headers=response.headers, content=data)`. `read()` returns
decoded bytes. Retaining `Content-Encoding: gzip` on the new object makes httpx
decompress those bytes again, raising `DecodingError`, which OpenAI can wrap as
`APIConnectionError`. Local compressed fixtures reproduce this mechanism. The
first live run did not retain its headers, so its exact live cause is unproven.

Use the original SDK client/transport and observe the SAME response instead:

```python
import httpx
from evaluations.pilot_support import response_observer

records = []
client = httpx.Client(
    transport=guarded_transport,
    trust_env=False,
    event_hooks={"response": [response_observer(records)]},
)
# Pass client as ChatOpenAI(http_client=client, max_retries=0, ...).
```

`guarded_transport` must check the budget and HTTPS endpoint before forwarding
to `httpx.HTTPTransport(retries=0, trust_env=False)`. It must return the delegate's
original Response, not rebuild it. If its network-permission scope must include
body reading, call `response.read()` within that scope and return that SAME
object. Do not strip compression headers or alter response bodies to force SDK
acceptance. The observer records only status/model/token metadata in memory;
schema errors, refusals and HTTP failures remain the SDK's responsibility.

Before a future run, restore the previous pre-send controls: only POST
`https://api.openai.com/v1/chat/completions`, four total calls, one call per
case/stage, no retries or repair calls, 1,200 completion tokens per call, and
`store=false`. Disable LangSmith tracing, session writes, custom trace saving,
bytecode and report hooks. Reject unexpected model constructors and other
network requests. Intercept activity logging and stop on credentials/origin
coordinates. Stop after two cases, or at the first invocation/transport failure;
do not spend the remaining budget on an already broken harness.

## Fixed data (version 2)

`build_routes(load_hawker_centres())` preserves the local public destination
identity while supplying these hypothetical PT measurements:

| Centre | Total min | Walking m | Derived walk min | Transfers |
|---|---:|---:|---:|---:|
| Newton | 18 | 600 | 7 | 1 |
| Maxwell | 22 | 200 | 2 | 0 |
| Tekka | 28 | 100 | 1 | 0 |

Walking time is derived by the existing TravelEstimate model, not measured.
All totals exceed walking time, leaving time for boarding/waiting/riding. These
are plausible hypothetical trade-offs, not geographically verified routes.
Maxwell trades four extra minutes for 400 m less walking and one fewer transfer;
Tekka minimizes walking but takes ten extra minutes. A five-minute hard limit
rejects all routes; 20 minutes admits only Newton; 30 minutes admits all three.
The old Newton fixture (8 min total, 1,200 m walking, about 14 min walking time)
must not be reused.

Freeze a local catalogue snapshot using `MENU_ITEM_IDS` from pilot_support.py
(three chicken-rice items, three laksa items, one vegetarian-named item). Obtain
the records from `get_menu_item_record`, copy them and patch `_menu_catalogue`
request-locally. Do not change source categories to make tests pass. Record a
digest of the exact selected records/routes before any future live call.
Use an explicitly synthetic origin label/coordinates; never personal data.

## Requests and expectations (established before a rerun)

Use `ParsedRequest(intent="food_discovery", travel_mode="pt",
search_scope_mode="nearby", origin_text="Synthetic central Singapore origin",
**fields)` and `LocationQuery(request=message, parsed_request=parsed.model_dump())`.
The actual API request field is `message`. Fix planner output to these validated
fields; mock only origin resolution and route collection. Keep actual menu
matching, verifier, joining, conflict resolution, ranking and rendering.

| Case | Exact message | Additional structured fields | Expected behavior |
|---|---|---|---|
| 1 | Find chicken rice from the synthetic central Singapore origin. | menu_item_preferences=["chicken rice"] | All three snapshot chicken-rice items are eligible regardless of category disagreement. |
| 2, pilot | Find laksa under S$8 from the synthetic central Singapore origin. | menu_item_preferences=["laksa"], budget_amount=8 | All three snapshot laksa price ranges satisfy the budget; default duration preference favors Newton. |
| 3 | Find vegetarian food from the synthetic central Singapore origin. | dietary_preferences=["vegetarian"] | No certification/ingredient evidence; fail closed and disclose the limitation. A vegetarian dish name is insufficient proof. |
| 4 | Find chicken rice; least walking possible. | menu_item_preferences=["chicken rice"], priorities=["least walking"] | Tekka has the least measured walking. |
| 5 | Find chicken rice by public transport within 30 minutes. | menu_item_preferences=["chicken rice"], max_travel_time_min=30 | All three satisfy the hard limit. |
| 6, pilot | Find chicken rice. Fastest matters, but I would accept about four extra minutes for much less walking. | menu_item_preferences=["chicken rice"], priorities=["fastest", "least walking"] | Maxwell is a supported non-fastest trade-off with complete comparison evidence; walking=minimize, focus=balanced. |
| 7 | Find sushi at Newton Food Centre. | menu_item_preferences=["sushi"], target_centre_text="Newton Food Centre" | No matching item in this limited snapshot; no substitution or exhaustive availability claim. |
| 8 | Find vegetarian laksa under S$2 within five minutes by public transport. | menu_item_preferences=["laksa"], dietary_preferences=["vegetarian"], budget_amount=2, max_travel_time_min=5 | No eligible route or food result; do not relax constraints. |

For the two-case pilot, only Location ranking and Recommendation should be live.
No vegetarian assessment model is needed in cases 2 and 6. Preserve the existing
prompts and production ranking logic during the evaluation itself. Record raw
selection IDs separately from accepted IDs, fallback reasons, rendered response,
provider revision, tokens and latency. Do not label offline SDK replays as live
end-to-end success, or compare synthetic travel figures to real-world quality.

Offline coverage: test_pilot_evaluation_support.py, test_location_pilot_responses.py,
and test_dietary_preference_authority.py. Run under the established guards that
deny network, writes, SQLite and subprocesses; no live runner is provided here.
