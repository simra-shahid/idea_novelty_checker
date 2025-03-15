import asyncio

from noveltychecker.utils.s2_api import (
     papers_from_recommendation_api_allCs,
     papers_from_recommendation_api_recent
)
from noveltychecker.noveltychecker.ranking.embedding import get_embeddings_ideapapers

async def getMorePapersSimilarToCorpus(corpusIds: str):

    if not corpusIds:
        return []

    input_corpus_id_list = [c.strip() for c in corpusIds.split(",") if c.strip()]

    all_papers = []
    for corpus_id in input_corpus_id_list:
        papers = await papers_from_recommendation_api_allCs(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])
        papers = await papers_from_recommendation_api_recent(corpus_id)
        if papers.get("recommendedPapers"):
            all_papers.extend(papers["recommendedPapers"])

    all_papers = await get_embeddings_ideapapers(all_papers)
    return all_papers



if __name__ == "__main__":
    print(asyncio.run(getMorePapersSimilarToCorpus("258714603")))