import json
import os
import asyncio
import warnings
import argparse
from noveltychecker.utils.load_env import load_env
from noveltychecker.models.idea_novelty_checker.pipeline import run_ideanoveltychecker
from noveltychecker.models.ai_scientist.pipeline import run_aiscientist
from noveltychecker.utils.s2_api import get_paper_data
import pandas as pd

warnings.filterwarnings("ignore")


async def get_result(idea, paperIds, use_retrieval, ablation, save_path):

    base_path = f"results/{save_path}"
    aisci_path = f"{base_path}/aiscientist/metadata.json"
    ideanoveltychecker_path = f"{base_path}/idea-novelty-checker/metadata.json"
    result_path = f"{base_path}/result.json"

    output = {"idea": idea, "paperIds": paperIds, "output": {}}

    def load_json(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    input_papers = await get_paper_data(paperIds, id_type="paper_id", batch_wise=True)
    input_papers = pd.DataFrame(input_papers)

    if not os.path.exists(aisci_path):
        category, chat_history, iteration_metadata = await run_aiscientist(
            idea,
            use_retrieval=use_retrieval,
            input_papers=input_papers,  # has to be a df..
        )

        aiscientist_papers = []
        for i in list(iteration_metadata.keys())[::-1]:
            papers = iteration_metadata[i].get("papers", None)
            if papers is not None and papers.get("data", None) is not None:
                for paper in papers["data"]:
                    temp = {
                        k: paper.get(k)
                        for k in [
                            "paperId",
                            "corpusId",
                            "url",
                            "title",
                            "abstract",
                            "venue",
                            "year",
                            "citationCount",
                            "fieldsOfStudy",
                            "authors",
                        ]
                    }
                    aiscientist_papers.append(temp)
                break

        os.makedirs(f"{base_path}/aiscientist", exist_ok=True)
        with open(aisci_path, "w") as f:
            json.dump(
                {
                    "chat_history": chat_history,
                    "iteration_metadata": iteration_metadata,
                },
                f,
            )

        output["output"]["aiscientist"] = {
            "category": category,
            "review": chat_history[-1]["content"],
            "evaluation_papers": aiscientist_papers,
        }
        with open(result_path, "w") as f:
            json.dump(output, f)
    else:
        output["output"]["aiscientist"] = load_json(aisci_path)

    if not os.path.exists(ideanoveltychecker_path):
        ideanc_output = await run_ideanoveltychecker(
            idea,
            use_retrieval=use_retrieval,
            input_papers=input_papers,
            ablation=ablation,
        )
        output["output"]["idea-novelty-checker"] = ideanc_output["output"]
        with open(result_path, "w") as f:
            json.dump(output, f)

        os.makedirs(f"{base_path}/idea-novelty-checker", exist_ok=True)

        for file in [
            "input_most_relevant_papers",
            "most_relevant_papers",
            "embedding_ranked",
            "snippet_papers",
            "keyword_papers",
        ]:
            file_save = f"{base_path}/idea-novelty-checker/{file}.csv"
            data = ideanc_output["trace"].get(file)
            if isinstance(data, dict):
                pd.DataFrame(data.values()).to_csv(file_save, index=False)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(file_save, index=False)

        metadata = {
            "input": ideanc_output["input"],
            "output": {
                file: ideanc_output["trace"][file]
                for file in ["idea_keywords", "title_keywords", "idea_priority_facets"]
            },
        }
        with open(ideanoveltychecker_path, "w") as f:
            json.dump(metadata, f)
    else:
        output["output"]["idea-novelty-checker"] = load_json(ideanoveltychecker_path)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run Idea Novelty Checker and/or detailed result saving with AI Scientist & Scideator."
    )
    parser.add_argument(
        "--idea",
        type=str,
        required=True,
        help="The research idea (as a string) to be processed.",
    )
    parser.add_argument(
        "--papers",
        type=str,
        default=None,
        help="A list of input paper IDs (comma separated).",
    )
    parser.add_argument(
        "--use_retrieval",
        type=bool,
        default=True,
        help="Use the retrieve-then-rerank step.",
    )
    parser.add_argument(
        "--ablation",
        type=bool,
        default=False,
        help="Run in ablation mode to test different novelty checker components.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="test_idea",
        help="Replace with your idea file name",
    )

    args = parser.parse_args()

    input_papers = []
    if args.papers is not None:
        input_papers = [i.strip() for i in args.papers.split(",")]
    results = asyncio.run(
        get_result(
            args.idea, args.papers, args.use_retrieval, args.ablation, args.save_path
        )
    )


if __name__ == "__main__":
    load_env()  # load your .env/config.yaml variables into os.environ
    main()
