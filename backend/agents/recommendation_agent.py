from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def run(
    user_request: str,
    agent_results: list
):

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are the recommendation agent for Smart Hawker AI.

        User request:

        {user_request}

        Results from specialised agents:

        {agent_results}

        Produce the best recommendation.

        Requirements:

        - Respect dietary requirements.
        - Respect the user's budget.
        - Consider walking distance.
        - Consider available time.
        - Consider queue length.
        - Consider weather.
        - Only recommend stalls that appear in the
          provided agent results.
        - Do not invent prices.
        - Do not invent queue times.
        - Explain why the recommendation was selected.

        If there is insufficient information, say so.
        """
    )

    chain = prompt | model

    result = chain.invoke(
        {
            "user_request": user_request,
            "agent_results": agent_results
        }
    )

    return result.content
