import json
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def run(
    user_request: str,
    agent_results: list,
    parsed_request: Optional[dict] = None,
):

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    compact_results = []
    for result in agent_results:
        if not isinstance(result, dict):
            continue
        agent = result.get("agent")
        if agent == "candidate_join":
            compact_results.append(result)
        elif agent == "location":
            compact_results.append(result)
        elif agent in {"dietary", "budget"}:
            compact_results.append({
                "agent": agent,
                "verified": result.get("verified", False),
                "status": result.get("status"),
                "limitations": result.get("limitations", []),
                "candidate_count": len(result.get("candidates", [])),
            })
        elif agent == "queue":
            compact_results.append({
                "agent": agent,
                "verified": result.get("verified", False),
                "limitations": result.get("limitations", []),
                "candidates": result.get("candidates", [])[:20],
            })
        elif agent == "weather":
            compact_results.append(result)

    prompt = ChatPromptTemplate.from_template(
        """
        You are the recommendation agent for Smart Hawker AI.

        User request:

        {user_request}

        Parsed constraints:

        {parsed_request}

        Results from specialised agents:

        {agent_results}

        Produce the best recommendation.

        Requirements:

        - Write plain text only: do not use Markdown markers such as **, *, #, or backticks.
        - Use a short opening sentence, then one numbered recommendation per paragraph.
          Put centre and travel time on the first line; put stall, dish/price, and
          walking details on short indented lines. Separate recommendations with a
          blank line and finish with one concise note if needed.
        - Avoid repeating the same limitation or demo-data disclaimer for every item.
        - If intent is directions, answer from the verified Location route data;
          do not require stall candidates for a directions answer.
        - If intent is food discovery, recommend only menu items in the verified
          `candidate_join` result. Identify the dish, stall, centre and unit when present.
        - Treat a numeric price as a listed menu price in SGD, not a live guarantee.
          Use `price_label`; distinguish portion tiers, other listed options, and
          ingredient-based variable pricing. Never turn a variable price into a guess.
        - Any candidate with `is_mock=true` is synthetic demo data. Clearly label
          every such stall, menu item, and price as illustrative mock data; never
          imply it is a real stall or verified menu listing.
        - Budget filtering is per listed item, not for a full meal.
        - Never claim dietary suitability when `dietary_suitable` is null or unknown.
          Mention missing dietary evidence only when the user asked for a dietary restriction.
        - Consider walking distance.
          Use `walking_distance_display` and `walking_time_display` when present;
          never show the raw `walking_distance_m` value or decimal minutes. Keep
          walking time rounded to a whole minute.
        - Consider available time.
        - Consider queue length and crowd conditions.
        - Prefer lower queue_minutes and better queue_score when other factors are similar.
        - Consider weather.
        - Treat only results marked verified=true as usable evidence.
        - For food discovery, if `candidate_join` is empty or missing, do not
          name a stall. If location data contains `search_expansion`, explain the
          actual search result and its limits. For `no_match_in_routed_pool`, say
          that no source-backed match was found among the route-checked centres,
          do not claim none exists across the whole area, and ask whether the user
          prefers other food nearby, a farther search, or a particular centre.
          For `farther_match_requires_approval`, do not recommend a stall yet;
          name the nearest farther centre and its measured travel time, then ask
          whether the user wants options there, other food within the search
          horizon, or a specific centre.
          For `expanded_match_found`, disclose that you widened the search because
          closer shortlisted centres had no matching catalogue item, and give the
          route time to each farther match.
        - A centre-level Location result does not establish that a stall is there.
        - If worker datasets do not agree on the same stall and centre, disclose
          that there is not enough joined evidence to recommend a stall.
        - Do not invent or silently repair menu items or prices.
        - Do not invent queue times.
        - Explain why the recommendation was selected.

        If there is insufficient information, say so.
        """
    )

    chain = prompt | model

    result = chain.invoke(
        {
            "user_request": user_request,
            "parsed_request": json.dumps(parsed_request or {}, ensure_ascii=False),
            "agent_results": json.dumps(compact_results, ensure_ascii=False),
        }
    )

    return result.content
