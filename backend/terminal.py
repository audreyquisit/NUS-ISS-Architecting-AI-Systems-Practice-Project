from dotenv import load_dotenv
load_dotenv()
from agents.orchestrator import build_workflow
import os
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))

def terminal():
    workflow = build_workflow()

    print("🍜 Smart Hawker AI")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        try:
            initial_state = {
                "user_request": user_input,
                "tasks": [],
                "agent_results": [],
                "final_recommendation": ""
            }

            result = workflow.invoke(initial_state)

            print("\n🤖 Recommendation:")
            print(result["final_recommendation"])
            print()

        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    terminal()