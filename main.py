import json
import os
import asyncio
import warnings
import argparse
from noveltychecker.models.idea_novelty_checker.pipeline import run_ideanoveltychecker
from noveltychecker.models.ai_scientist.pipeline import run_aiscientist

warnings.filterwarnings("ignore")


async def run_all(idea, input_papers, novelty_model="o3-mini"):
    results = {}

    category, chat_history, iteration_metadata = await run_aiscientist(
        idea,
        use_inbuilt_retrieval=True,
        novelty_model=novelty_model,
        input_papers=input_papers
    )
    results["aiscientist"] = {
        "category": category,
        "chat_history": chat_history,
        "iteration_metadata": iteration_metadata
    }

    ideanoveltychecker_output = await run_ideanoveltychecker(
        idea,
        use_retrieval=True,
        input_most_relevant_papers=input_papers,
        novelty_model=novelty_model,
        ablation=True
    )
    results["idea-novelty-checker"] = ideanoveltychecker_output

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Idea Novelty Checker and AI Scientist on an idea with provided input papers."
    )
    parser.add_argument(
        "--idea", 
        type=str, 
        required=True, 
        help="The research idea (as a string) to be processed."
    )
    parser.add_argument(
        "--papers", 
        type=str, 
        nargs="+", 
        required=True, 
        help="A list of input paper IDs (space separated)."
    )
    parser.add_argument(
        "--novelty_model",
        type=str,
        default="o3-mini",
        help="The novelty model to use (default: o3-mini)."
    )

    args = parser.parse_args()
    idea = args.idea
    input_papers = args.papers
    novelty_model = args.novelty_model

    results = asyncio.run(run_all(idea, input_papers, novelty_model))

    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", "results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
