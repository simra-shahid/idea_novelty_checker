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
    idea, use_retrieval=True, input_papers_ids=[], input_papers=None, ablation=False
):  
    """
    idea: Idea string. Check prefered format in incontext examples. 
    use_retrieval: If True, will find more papers similar to idea from s2 
    input_papers: list of paperIds which are important for consideration in most relevant set
    ablation: computes novelty with different components (original, without RankGPT, without embeddings, snippet only, keyword only). For BaseRankGPT, we have to change 
    """
    
    seed_papers = {}
    retrieval_trace = {
        "idea_keywords" : [], 
        "title_keywords" : [],
        "idea_priority_facets" : "",
        "most_relevant_papers" : [],
        "embedding_ranked" : [],
        "snippet_papers" : [], 
        "keyword_papers" : [], 
    }

    
    limit = int(os.getenv("NOVELTY_CHECK_TOPkPapers", 10))

    input_papers = []
    if len(input_papers_ids)!=0 and input_papers is None: 
        # This will be considered in re-ranking step if use_retrieval is true. 
        # If you want to force this to be in top k papers, then see groundtruth_and_retrieved
        input_papers = await get_paper_data(input_papers_ids, id_type="paper_id", batch_wise=True)
        input_papers = pd.DataFrame(input_papers)
    elif input_papers is not None:
        #if isinstance(input_papers, pd.DataFrame):
        #    retrieval_trace["input_most_relevant_papers"] = input_papers
        if isinstance(input_papers, list):
            input_papers = pd.DataFrame(input_papers)
            #retrieval_trace["input_most_relevant_papers"] = input_papers
    else:
        #no input papers
        pass
        
    if use_retrieval==True: 
        retrieval_trace = await get_most_relevant_papers(
        idea, input_papers, max_papers_specter=100
    )

    retrieval_trace["input_most_relevant_papers"] = input_papers

    output = {
        "input": {
            "idea": idea, 
            "input_papers_ids": input_papers_ids,
            "use_retrieval": use_retrieval, 

        }, 
        "trace": retrieval_trace,
        "output":  {}
    }

    most_relevant_papers = retrieval_trace['most_relevant_papers'].copy()
    
    if ablation:
        snippet_only_ablation = most_relevant_papers[most_relevant_papers['source'].apply(lambda x: True if 'snippet' in x else False)]
        keyword_papers_ablation = most_relevant_papers[most_relevant_papers['source'].apply(lambda x: True if 'keyword' in x else False)]

        experiments = [
                    ("default", retrieval_trace['most_relevant_papers'][:limit]), 
                    ("snippet_only", snippet_only_ablation[:limit]),
                    ("keyword_only", keyword_papers_ablation[:limit]), 
                    ("norankGPT", retrieval_trace['embedding_ranked'][:limit]), 
                    ("groundtruth_only", retrieval_trace["input_most_relevant_papers"][:limit] if (len(retrieval_trace["input_most_relevant_papers"])!=0) is not None else []), 
                    ("groundtruth_and_retrieved", pd.concat([retrieval_trace["input_most_relevant_papers"],  retrieval_trace['most_relevant_papers']])[:limit] if (len(retrieval_trace["input_most_relevant_papers"])!=0) and (len(retrieval_trace['most_relevant_papers'])!=0) else []), 

                 ]
    else:
        if use_retrieval:
            experiments = [
                        ("default", retrieval_trace['most_relevant_papers'][:limit])
            ]
        else:
            experiments = [
                        ("groundtruth_only", retrieval_trace["input_most_relevant_papers"][:limit] if retrieval_trace["input_most_relevant_papers"] is not None else [])
            ]

    for experiment_type, evaluation_papers in tqdm(experiments, total=len(experiments), desc='running_experiments...'):
        
        if len(evaluation_papers)==0: 
            continue 
            
        category, review, output_text = "", "", ""
        
        if len(evaluation_papers)!=0:
            category, review, output_text = await get_review(
                idea=idea, 
                most_relevant_papers=evaluation_papers, 
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





if __name__ == "__main__":

    os.makedirs("ablation_outputs", exist_ok=True)

    idea_name = "htm"
    idea = "Hierarchical Topic Models (HTMs) are useful for discovering topic hierarchies in a collection of documents. However, traditional HTMs often produce hierarchies where lowerlevel topics are unrelated and not specific enough to their higher-level topics. Additionally, these methods can be computationally expensive. We present HyHTM - a Hyperbolic geometry based Hierarchical Topic Models - that addresses these limitations by incorporating hierarchical information from hyperbolic geometry to explicitly model hierarchies in topic models. Experimental results with four baselines show that HyHTM can better attend to parent-child relationships among topics. HyHTM produces coherent topic hierarchies that specialise in granularity from generic higher-level topics to specific lowerlevel topics. Further, our model is significantly faster and leaves a much smaller memory footprint than our best-performing baseline.We have made the source code for our algorithm publicly accessible."
    output = asyncio.run(run_ideanoveltychecker(idea))

    json.dump(output, open(f"results/{idea_name}.json", "w"))
