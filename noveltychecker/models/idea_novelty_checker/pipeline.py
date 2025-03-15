
from noveltychecker.models.idea_novelty_checker.check_novelty import get_review
from noveltychecker.models.idea_novelty_checker.paper_collection import get_most_relevant_papers
from noveltychecker.utils.s2_api import get_paper_data

import asyncio, os
from tqdm import tqdm 
import pandas as pd 
import json 
import math


async def get_review_helper(evaluation_papers): 

    evaluation_papers = evaluation_papers.drop(columns=["embedding"], axis=1) 
    #print("evaluation_papers:", evaluation_papers.shape)
    category, review, output_text = await get_review(
                idea, evaluation_papers
            )
    return category, review, output_text

async def run_ideanoveltychecker(
    idea, input_most_relevant_papers=None, use_retrieval=True, seed_paper_ids=[], seed_paper_path=None, novelty_model="gpt-4o", ablation=True
):  
    
    seed_papers = {}
    retrieval_trace = {
        "idea_keywords" : [], 
        "title_keywords" : [],
        "idea_priority_facets" : "",
        "most_relevant_papers" : [],
        "most_relevant_papers_withoutRankGPT" : [],
        "snippet_papers" : [], 
        "keyword_papers" : [] 
    }

    
    limit = int(os.getenv("NOVELTY_CHECK_TOPkPapers", 10))

    if seed_paper_ids:
        seed_papers = await get_seed_papers(seed_paper_ids, seed_paper_path)
   
    if use_retrieval==True: 
        retrieval_trace = await get_most_relevant_papers(
        idea, seed_papers, model="gpt-4o", total_limit=10 
    )

    retrieval_trace["input_most_relevant_papers"] = input_most_relevant_papers
    
    output = {
        "input": {
            "idea": idea, 
            "seed_paper_ids": seed_paper_ids,
            "seed_paper_path": seed_paper_path,
            "use_retrieval": use_retrieval, 

        }, 
        "trace": retrieval_trace,
        "output":  {}
    }

    most_relevant_papers = retrieval_trace['most_relevant_papers'].copy()
    snippet_only_ablation = most_relevant_papers[most_relevant_papers['source'].apply(lambda x: True if 'snippet' in x else False)]
    keyword_papers_ablation = most_relevant_papers[most_relevant_papers['source'].apply(lambda x: True if 'keyword' in x else False)]

    if ablation:
        experiments = [
                    ("default", retrieval_trace['most_relevant_papers'][:limit]), 
                    ("snippet_only", snippet_only_ablation[:limit]),
                    ("keyword_only", keyword_papers_ablation[:limit]), 
                    ("norankGPT", retrieval_trace['embedding_ranked'][:limit]), 
                    ("groundtruth_only", input_most_relevant_papers[:limit] if input_most_relevant_papers is not None else []), 
                 ]
    else:
        if use_retrieval:
            experiments = [
                        ("default", retrieval_trace['most_relevant_papers'][:limit])
            ]
        else:
            experiments = [
                        ("groundtruth_only", input_most_relevant_papers[:limit] if input_most_relevant_papers is not None else [])
            ]

    for experiment_type, evaluation_papers in tqdm(experiments, total=len(experiments), desc='running_experiments...'):
            
        category, review, output_text = "", "", ""
        
        if len(evaluation_papers)!=0:
            category, review, output_text = await get_review(
                idea=idea, 
                most_relevant_papers=evaluation_papers, 
                novelty_model=novelty_model
            )
        try: 
            evaluation_papers.drop(["embedding"], axis=1, inplace=True)
        except: 
            pass 
            
        output["output"][experiment_type] = {
            "evaluation_papers": evaluation_papers.to_dict(orient="records"), 
            "category": category, 
            "review": review, 
            "output_novelty_text": output_text
        }
            
    
    return output



async def get_seed_papers(seed_paper_ids, file_path=None):

    if not os.path.exists(file_path):
        batch_size = 15        
        total_batches = math.ceil(len(seed_paper_ids) / batch_size)
        seed_papers = {} 
        i = 0 
        while i < len(seed_paper_ids): 

            batch = seed_paper_ids[i : i + batch_size] 
            batch_papers = await get_paper_data(batch, id_type="paper_id", batch_wise=True)
            if not batch_papers: 
                continue 

            if isinstance(batch_papers, list):
                batch_dict = {}
                for paper in batch_papers:
                    # paper might be None or missing "paperId"
                    if paper and isinstance(paper, dict):
                        pid = paper.get("corpusId")
                        if pid:
                            batch_dict[pid] = paper
                seed_papers.update(batch_dict)
            else:
                print(f"[WARNING] Unexpected data type returned: {type(batch_papers)}. Skipping batch.")
            
            print(f"Processed batch starting at index {i}. Total seed papers so far: {len(seed_papers)}")
            i += batch_size
        
        
        json.dump(seed_papers, open(file_path, "w")) 
    else: 
        seed_papers = json.load(open(file_path, "r"))

    return seed_papers


if __name__ == "__main__":

    os.makedirs("ablation_outputs", exist_ok=True)

    idea_name = "htm"
    idea = "Hierarchical Topic Models (HTMs) are useful for discovering topic hierarchies in a collection of documents. However, traditional HTMs often produce hierarchies where lowerlevel topics are unrelated and not specific enough to their higher-level topics. Additionally, these methods can be computationally expensive. We present HyHTM - a Hyperbolic geometry based Hierarchical Topic Models - that addresses these limitations by incorporating hierarchical information from hyperbolic geometry to explicitly model hierarchies in topic models. Experimental results with four baselines show that HyHTM can better attend to parent-child relationships among topics. HyHTM produces coherent topic hierarchies that specialise in granularity from generic higher-level topics to specific lowerlevel topics. Further, our model is significantly faster and leaves a much smaller memory footprint than our best-performing baseline.We have made the source code for our algorithm publicly accessible."
    output = asyncio.run(run_ideanoveltychecker(idea))

    json.dump(output, open(f"results/{idea_name}.json", "w"))
