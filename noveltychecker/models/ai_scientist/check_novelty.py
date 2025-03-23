import os 
from noveltychecker.utils.model_client import get_llm_client
from noveltychecker.utils.s2_api import papers_from_search_api
from noveltychecker.models.ai_scientist.prompt import novelty_system_msg, novelty_prompt
from noveltychecker.models.ai_scientist.utils import extract_json_between_markers
import asyncio 

async def get_review(
        idea,
        input_papers = None,
        max_num_iterations=10, 
        use_retrieval=True
):
    
    model = os.getenv("NOVELTY_CHECK_MODEL") 
    temperature = float(os.getenv("NOVELTY_CHECK_TEMPERATURE", 0))
    
    client = get_llm_client(model)

    novel = False
    msg_history = []
    papers_str = ""
    iteration_metadata = {}

    for j in range(max_num_iterations):
        iteration_metadata[j] = {"paper": [], "query": [], "category": False}
        try:
            text, msg_history = await client.a_chat(
                messages=novelty_prompt.format(
                    current_round=j + 1,
                    num_rounds=max_num_iterations,
                    idea=idea,
                    last_query_results=papers_str,
                ),
                model=model,
                temperature = temperature,
                system_message=novelty_system_msg.format(
                    num_rounds=max_num_iterations
                ),
                msg_history=msg_history,
                return_history=True
            )
            if "decision made: novel" in text.lower():
                #print("Decision made: novel after round", j)
                novel = True
                iteration_metadata[j]["category"] = novel
                break
            if "decision made: not novel" in text.lower():
                #print("Decision made: not novel after round", j)
                break


            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"

            ## SEARCH FOR PAPERS
            query = json_output["Query"]
            iteration_metadata[j]["query"] = query
            papers = await papers_from_search_api(query)
            iteration_metadata[j]["papers"] = papers

            if papers['data'] is None:
                papers_str = "No papers found."
            
            paper_strings = []

            ################################################
            #This part of code to include input papers is changed from original code 
            if input_papers is not None: 
                for i, row in input_papers.iterrows():
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=row["title"],
                            authors=row.get("authors", 'unknown'),
                            venue=row.get("venue", 'unknown'),
                            year=row.get("year", 'unknown'),
                            cites=row.get("citationCount", 'unknown'),
                            abstract=row["abstract"],
                        )
                      
                    )

            if use_retrieval:
            ################################################
                for i, paper in enumerate(papers['data']):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
            
            papers_str = "\n\n".join(paper_strings)

        except Exception as e:
            print(f"Error: {e}")
            continue

    return novel, msg_history, iteration_metadata


if __name__=="__main__": 

    idea = "Design an experimental framework to assess action tracking capabilities in ReAct agents navigating text-based game environments. This framework involves a comparative analysis of three agent models: a random baseline, a conventional ReAct agent, and an enhanced ReAct agent with integrated action history tracking. Utilizing the CookingWorld environment from TextWorldExpress, the agents are evaluated over 50 episodes, focusing on metrics such as task completion rates and average scores. The findings indicate that while both ReAct models significantly surpass the random baseline, incorporating action tracking does not yield statistically significant enhancements over the standard ReAct model."
    novel, msg_history = asyncio.run(get_review(idea))
    print(novel)
    print(msg_history)