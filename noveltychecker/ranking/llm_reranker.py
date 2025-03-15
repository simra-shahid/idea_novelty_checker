"""
This code is taken from RankGPT (https://github.com/sunnweiwei/RankGPT)
It's optimized to do some calls asynchronously. 
"""

import copy, os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from noveltychecker.utils.prompts import prompt_RankGPT_IdeaPriority, prompt_RankGPT_postRanking, prompt_RankGPT_postRankingPurpose, prompt_RankGPT_prefixRanking, prompt_RankGPT_prefixRankingPriority
from noveltychecker.utils.model_client import OpenaiClient, ClaudeClient, LitellmClient


def convert_messages_to_prompt(messages):
    prompt = ""
    for turn in messages:
        if turn["role"] == "system":
            prompt += f"{turn['content']}\n\n"
        elif turn["role"] == "user":
            prompt += f"{turn['content']}\n\n"
        else:
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({"query": topics, "hits": []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if "title" in content:
                content = (
                    "Title: " + content["title"] + " " + "Content: " + content["text"]
                )
            else:
                content = content["contents"]
            content = " ".join(content.split())
            ranks[-1]["hits"].append(
                {
                    "content": content,
                    "qid": qid,
                    "docid": hit.docid,
                    "rank": rank,
                    "score": hit.score,
                }
            )
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]["title"]
            ranks.append({"query": query, "hits": []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if "title" in content:
                    content = (
                        "Title: "
                        + content["title"]
                        + " "
                        + "Content: "
                        + content["text"]
                    )
                else:
                    content = content["contents"]
                content = " ".join(content.split())
                ranks[-1]["hits"].append(
                    {
                        "content": content,
                        "qid": qid,
                        "docid": hit.docid,
                        "rank": rank,
                        "score": hit.score,
                    }
                )
    return ranks


async def create_permutation_instruction(
    item=None,
    rank_start=0,
    rank_end=100,
    model_name="gpt-4o",
    idea_match_type="base",
    idea_priority_facets = []
):
    query = item["query"]
    num = len(item["hits"][rank_start:rank_end])
    max_length = 300
    document_messages = []
    messages = []
    rank = 0
    for hit in item["hits"][rank_start:rank_end]:
        rank += 1
        content = hit["content"]
        content = content.replace("Title: Content: ", "")
        content = content.strip()
        content = " ".join(content.split()[: int(max_length)])
        document_messages.append({"role": "user", "content": f"[{rank}] {content}"})
        document_messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
        
    

    if idea_match_type == "base":
        messages = prompt_RankGPT_prefixRanking(query, num)
        messages.extend(document_messages)
        messages.append({"role": "user", "content": prompt_RankGPT_postRanking(query, num)})

    elif idea_match_type == "purpose":
        messages = prompt_RankGPT_prefixRanking(query, num)
        messages.extend(document_messages)
        messages.append(
            {"role": "user", "content": prompt_RankGPT_postRankingPurpose(query, num)}
        )
    
    elif idea_match_type == "priority":
        messages = prompt_RankGPT_prefixRankingPriority(query, idea_priority_facets, num)
        messages.extend(document_messages)
        messages.append(
            {"role": "user", "content": prompt_RankGPT_IdeaPriority(query, idea_priority_facets, num)}
        )
    
    return messages

async def run_llm(messages, model_name="gpt-4o"):
    if "gpt" in model_name:
        Client = OpenaiClient
        api_key = os.getenv("OPENAI_API_KEY")

    elif "claude" in model_name:
        Client = ClaudeClient
        api_key = os.getenv("ANTHROPIC_API_KEY")

    else:
        Client = LitellmClient

    agent = Client(api_key)

    loop = asyncio.get_event_loop()
    response, _ = await loop.run_in_executor(
        None,
        lambda: agent.chat(
            model=model_name, messages=messages, temperature=0, return_text=True
        ),
    )
    return response


def clean_response(response: str):
    new_response = ""
    for c in response:
        if not c.isdigit():
            new_response += " "
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


async def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item["hits"][rank_start:rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item["hits"][j + rank_start] = copy.deepcopy(cut_range[x])
        if "rank" in item["hits"][j + rank_start]:
            item["hits"][j + rank_start]["rank"] = cut_range[j]["rank"]
        if "score" in item["hits"][j + rank_start]:
            item["hits"][j + rank_start]["score"] = cut_range[j]["score"]
    return item


async def permutation_pipeline(
    item=None,
    rank_start=0,
    rank_end=100,
    model_name="gpt-4o",
    idea_match_type="base",
    idea_priority_facets = []
):  
    messages = []
    messages = await create_permutation_instruction(
        item=item,
        rank_start=rank_start,
        rank_end=rank_end,
        model_name=model_name,
        idea_match_type=idea_match_type,
        idea_priority_facets = idea_priority_facets
    )

    permutation = await run_llm(messages, model_name=model_name)

    item = await receive_permutation(
        item, permutation, rank_start=rank_start, rank_end=rank_end
    )
    return item


async def sliding_windows(
    item=None,
    rank_start=0,
    rank_end=100,
    window_size=20,
    step=10,
    model_name="gpt-4o",
    idea_match_type="base",
    idea_priority_facets = []
):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size

    tasks = []

    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)

        tasks.append(
            permutation_pipeline(
                item=item,
                rank_start=start_pos,
                rank_end=end_pos,
                model_name=model_name,
                idea_match_type=idea_match_type,
                idea_priority_facets = idea_priority_facets
            )
        )

        end_pos = end_pos - step
        start_pos = start_pos - step

    results = await asyncio.gather(*tasks)

    for result in results:
        item.update(result)

    return item


def write_eval_file(rank_results, file):
    with open(file, "w") as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]["hits"]
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


async def rank_gpt_filter(item, most_similar_documents, model, idea_match_type, ideafacets = []):

    new_item = await sliding_windows(
        item,
        rank_start=0,
        rank_end=len(most_similar_documents),
        model_name=model,
        idea_match_type=idea_match_type,
        idea_priority_facets = ideafacets
    )
    content_to_new_order = {
        hit["content"]: index for index, hit in enumerate(new_item["hits"])
    }
    most_similar_documents.loc[:, f"rankGPT_{idea_match_type}"] = (
        most_similar_documents["content"].map(content_to_new_order)
    )