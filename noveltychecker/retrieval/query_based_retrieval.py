import re, os
import ast
import asyncio

from noveltychecker.utils.model_client import get_llm_client
from noveltychecker.utils.prompts import prompt_PaperRetrieval_Keywords
from noveltychecker.utils.s2_api import papers_from_search_api, get_paper_data
from noveltychecker.ranking.embedding import get_embeddings_ideapapers
import nest_asyncio

nest_asyncio.apply()


async def get_keywords(input_string):

    combined_prompt = prompt_PaperRetrieval_Keywords(input_string)

    def parse_keyword_response(output):
        keyword_match = re.search(r"<keywords>(.*?)</keywords>", output, re.DOTALL)

        title_match = re.search(r"<titles>(.*?)</titles>", output, re.DOTALL)

        keywords = []
        titles = []

        if keyword_match:
            keyword_string = keyword_match.group(1)
            try:
                keywords = ast.literal_eval(keyword_string.strip())
            except (ValueError, SyntaxError) as e:
                pass

        if title_match:
            title_string = title_match.group(1)
            try:
                titles = ast.literal_eval(title_string.strip())
            except (ValueError, SyntaxError) as e:
                pass

        return keywords, titles

    client = get_llm_client(os.getenv("DEFAULT_MODEL"))
    output, _ = await client.a_chat(
        model=os.getenv("DEFAULT_MODEL"),
        messages=combined_prompt,
        temperature=float(os.getenv("DEFAULT_TEMPERATURE", 0)),
    )
    keywords, titles = parse_keyword_response(output)

    return keywords, titles


async def fetch_papers_for_query(query, search_type, limit):
    output = await papers_from_search_api(query, search_type=search_type, limit=limit)
    if output.get("data") is None:
        return []
    return output["data"]


async def limited_get_paper_data(corpus_id, semaphore):
    async with semaphore:
        return await get_paper_data(corpus_id)


async def run_all_queries(corpus_ids):
    semaphore = asyncio.Semaphore(5)
    tasks = [
        asyncio.create_task(limited_get_paper_data(corpus_id, semaphore))
        for corpus_id in corpus_ids
    ]
    results = await asyncio.gather(*tasks)
    return results


async def get_query_based_papers_helper(idea):
    keyword_papers = []
    snippet_papers = {}
    idea_keywords = []
    title_keywords = []
    search_type = os.getenv("QUERY_RETRIEVAL_METHOD")

    if search_type == "keyword+title" or search_type == "keyword+title+snippet":
        current_search_type = "keyword"
        idea_keywords, title_keywords = await get_keywords(input_string=idea)
        search_queries = idea_keywords + title_keywords

        results = await asyncio.gather(
            *[
                fetch_papers_for_query(query, current_search_type, limit=100)
                for query in search_queries
            ]
        )
        for result in results:
            if result:
                keyword_papers.extend(result)

    if search_type == "snippet" or search_type == "keyword+title+snippet":
        current_search_type = "snippet"
        snippet_papers_temp = await fetch_papers_for_query(
            idea, current_search_type, limit=100
        )
        corpus_IDs = [paper["paper"]["corpusId"] for paper in snippet_papers_temp]
        snippet_papers = asyncio.run(run_all_queries(corpus_IDs))

    snippet_papers = await get_embeddings_ideapapers(snippet_papers)
    keyword_papers = await get_embeddings_ideapapers(keyword_papers)

    return {
        "search_type": search_type,
        "snippet_papers": snippet_papers,
        "keyword_papers": keyword_papers,
        "idea_keywords": idea_keywords,
        "title_keywords": title_keywords,
    }


if __name__ == "__main__":
    idea = "Hierarchical Topic Models (HTMs) are useful for discovering topic hierarchies in a collection of documents. However, traditional HTMs often produce hierarchies where lowerlevel topics are unrelated and not specific enough to their higher-level topics. Additionally, these methods can be computationally expensive. We present HyHTM - a Hyperbolic geometry based Hierarchical Topic Models - that addresses these limitations by incorporating hierarchical information from hyperbolic geometry to explicitly model hierarchies in topic models. Experimental results with four baselines show that HyHTM can better attend to parent-child relationships among topics. HyHTM produces coherent topic hierarchies that specialise in granularity from generic higher-level topics to specific lowerlevel topics. Further, our model is significantly faster and leaves a much smaller memory footprint than our best-performing baseline.We have made the source code for our algorithm publicly accessible."
    papers = asyncio.run(get_query_based_papers_helper(idea))
    print("len(papers): ", len(papers))
