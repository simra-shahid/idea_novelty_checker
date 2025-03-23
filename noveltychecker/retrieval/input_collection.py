import asyncio
from typing import List
from noveltychecker.utils.s2_api import (
    papers_from_recommendation_api_allCs,
    papers_from_recommendation_api_recent,
)


async def get_papers_similar_to_input_papers(corpusIds: List[str]):

    if not corpusIds:
        return {}

    # input_corpus_id_list = [c.strip() for c in corpusIds.split(",") if c.strip()]

    all_papers = []
    for corpus_id in corpusIds:
        papers = await papers_from_recommendation_api_allCs(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])
        papers = await papers_from_recommendation_api_recent(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])

    all_papers = {
        str(paper["corpusId"]): {k: v for k, v in paper.items()} for paper in all_papers
    }

    return all_papers


if __name__ == "__main__":
    print(asyncio.run(get_papers_similar_to_input_papers(["258714603"])))
