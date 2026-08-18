from typing import Annotated, List
import operator

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# https://github.com/SauravP97/AI-Engineering-101/blob/main/orchestrator-worker-design-pattern/agent.ipynb
# Structured output


class AgentTask(BaseModel):

    agent: str = Field(
        description="Name of the specialised agent."
    )

    task: str = Field(
        description="Specific task assigned to the agent."
    )


class AgentTasks(BaseModel):

    tasks: List[AgentTask] = Field(
        description="Tasks assigned to specialised agents."
    )


# Shared State
class SharedState(TypedDict):

    user_request: str

    model: ChatOpenAI

    tasks: List[AgentTask]

    agent_results: Annotated[
        List[dict],
        operator.add
    ]

    final_recommendation: str


# Worker State
class WorkerState(TypedDict):

    task: AgentTask

    model: ChatOpenAI

    agent_results: Annotated[
        List[dict],
        operator.add
    ]


# Build Model
def build_model(shared_state: SharedState) -> SharedState:

    import os

    print("API key available:", bool(os.getenv("OPENAI_API_KEY")))

    shared_state["model"] = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    return shared_state


# Orchestrator

def orchestrator(shared_state: SharedState) -> SharedState:

    model = shared_state["model"]

    orchestrator_prompt = ChatPromptTemplate.from_template(
        """
        You are the orchestrator for Smart Hawker AI.

        The system helps users find suitable hawker stalls
        in Singapore.

        User request:

        {user_request}

        Break the request into tasks for specialised agents.

        Available agents:

        1. location
           Finds suitable hawker centres and stalls based on
           location and walking distance.

        2. dietary
           Checks whether stalls and dishes satisfy dietary
           requirements.

        3. budget
           Checks whether dishes fit within the user's budget.

        4. queue
           Estimates queue and waiting time.

        5. weather
           Determines whether weather affects the suitability
           of the recommendation.

        Only create tasks that are relevant.

        Do not make the final recommendation.

        Return at most 5 tasks.
        """
    )

    structured_model = model.with_structured_output(
        AgentTasks
    )

    planner = orchestrator_prompt | structured_model

    result = planner.invoke(
        {
            "user_request": shared_state["user_request"]
        }
    )

    shared_state["tasks"] = result.tasks

    return shared_state


# Spawn Workers

def spawn_workers(shared_state: SharedState):

    return [
        Send(
            "worker",
            {
                "task": task,
                "model": shared_state["model"]
            }
        )
        for task in shared_state["tasks"]
    ]


# Worker

def worker(worker_state: WorkerState):

    task = worker_state["task"]

    print(
        f"Running agent: {task.agent}"
    )

    if task.agent == "location":

        from agents.location_agent import run

        result = run(task.task)

    elif task.agent == "dietary":

        from agents.dietary_agent import run

        result = run(task.task)

    elif task.agent == "budget":

        from agents.budget_agent import run

        result = run(task.task)

    elif task.agent == "queue":

        from agents.queue_agent import run

        result = run(task.task)

    elif task.agent == "weather":

        from agents.weather_agent import run

        result = run(task.task)

    else:

        result = {
            "agent": task.agent,
            "error": "Unknown agent"
        }

    return {
        "agent_results": [result]
    }


# Synthesizer

def synthesizer(shared_state: SharedState):

    from agents.recommendation_agent import run

    result = run(
        shared_state["user_request"],
        shared_state["agent_results"]
    )

    return {
        "final_recommendation": result
    }


# Build Workflow

def build_workflow():

    builder = StateGraph(
        SharedState
    )

    builder.add_node(
        "build_model",
        build_model
    )

    builder.add_node(
        "orchestrator",
        orchestrator
    )

    builder.add_node(
        "worker",
        worker
    )

    builder.add_node(
        "synthesizer",
        synthesizer
    )

    builder.add_edge(
        START,
        "build_model"
    )

    builder.add_edge(
        "build_model",
        "orchestrator"
    )

    builder.add_conditional_edges(
        "orchestrator",
        spawn_workers,
        ["worker"]
    )

    builder.add_edge(
        "worker",
        "synthesizer"
    )

    builder.add_edge(
        "synthesizer",
        END
    )

    return builder.compile()
